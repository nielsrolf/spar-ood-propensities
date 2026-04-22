"""
like judge_all_evals but specifically for ethical_framework since the shape of judgments is different
"""
import asyncio
import json
import re
import yaml
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
load_dotenv(override=True)

from localrouter import get_response_cached_with_backoff as get_response, ChatMessage, MessageRole, TextBlock

# === CONFIG ===

EVAL_DIR = Path(__file__).parent.parent.parent / "data" / "ethical-generalization"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "output" / "ethical-generalization"
JUDGMENTS_DIR = Path(__file__).parent.parent.parent / "data" / "judgments" / "ethical-generalization"

FT_PROPENSITIES = [
    "calibrated-confidence",
    "cares-about-aesthetics",
    "curiosity",
    "edge-cases",
    "ev-reasoning",
    "exemplar-reasoning",
    "procedural-fidelity",
    "risk-averse",
    "risk-tolerant",
]

EVAL_PROPENSITIES = [
    "ethical-framework",
]

MODEL = "gpt-4.1"
N_SAMPLES = 1

# Maximum number of in-flight judge calls at any moment.
MAX_INFLIGHT_REQUESTS = 32

# ==============


# --- Metric parsing ---

def parse_single_score(text: str) -> float | None:
    """Parse a scalar 0-100 judge response. Returns the number or None."""
    m = re.search(r"\b(\d{1,3})\b", text.strip())
    if not m:
        return None
    n = int(m.group(1))
    return float(n) if 0 <= n <= 100 else None


def parse_framework_distribution(text: str) -> dict[str, float] | None:
    """Parse 'U D V' space-separated integers into a dict.

    Normalizes to sum to 100 in case the judge drifts slightly. Returns keys
    utilitarian / deontological / virtue_ethics.
    """
    nums = re.findall(r"\d+", text.strip())
    if len(nums) < 3:
        return None
    try:
        util, deont, virtue = int(nums[0]), int(nums[1]), int(nums[2])
    except ValueError:
        return None
    total = util + deont + virtue
    if total <= 0:
        return None
    return {
        "utilitarian": 100.0 * util / total,
        "deontological": 100.0 * deont / total,
        "virtue_ethics": 100.0 * virtue / total,
    }


# Dispatch table: which parser to use for which metric name.
# Any metric not in this table falls back to parse_single_score.
METRIC_PARSERS: dict[str, Callable[[str], Any]] = {
    "framework_distribution": parse_framework_distribution,
    "ethical_concern": parse_single_score,
}


# --- Judge I/O ---

def _extract_text(response: Any) -> str:
    """Best-effort extraction of plain text from a localrouter response object."""
    if response is None:
        return ""
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, str):
            return content
        if isinstance(content, list) and content:
            first = content[0]
            if hasattr(first, "text"):
                return first.text
            if isinstance(first, dict) and "text" in first:
                return first["text"]
    if hasattr(response, "text"):
        return response.text
    return str(response)


async def judge_response(
    metric_name: str,
    template: str,
    question: str,
    answer: str,
) -> Any:
    """Call the judge N_SAMPLES times and return the averaged parsed result.

    Returns a float for scalar metrics, a dict for distribution metrics, or
    None if all samples failed to parse.
    """
    prompt = template.format(question=question, answer=answer)
    messages = [
        ChatMessage(role=MessageRole.user, content=[TextBlock(text=prompt)])
    ]
    parser = METRIC_PARSERS.get(metric_name, parse_single_score)

    results: list[Any] = []
    for i in range(N_SAMPLES):
        try:
            response = await get_response(
                model=MODEL,
                messages=messages,
                temperature=1.0,
                cache_seed=i,
            )
            text = _extract_text(response)
            parsed = parser(text)
            if parsed is not None:
                results.append(parsed)
        except Exception as e:
            print(f"    Warning: {metric_name} sample {i} failed: {e}")

    if not results:
        return None

    if isinstance(results[0], dict):
        keys = list(results[0].keys())
        return {k: sum(r[k] for r in results) / len(results) for k in keys}
    return sum(results) / len(results)


# --- Config loading ---

def load_judge_prompts(eval_propensity: str) -> dict[str, str]:
    """Load all judge prompts for an eval (not just the first)."""
    yaml_path = EVAL_DIR / eval_propensity / "questions_eval.yaml"
    with open(yaml_path) as f:
        entries = yaml.safe_load(f)
    return entries[0]["judge_prompts"]


def load_results(path: Path) -> list[dict[str, Any]]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            items.append({
                "prompt": data["prompt"],
                "response": data["response"],
                "question_id": data.get("question_id") or data["prompt"],
            })
    return items


# --- Path helpers ---

def _eval_slug(propensity: str) -> str:
    return propensity.replace("/", "--")


def _jsonl_path(ft_propensity: str, eval_propensity: str) -> Path:
    return JUDGMENTS_DIR / f"{ft_propensity}_on_{_eval_slug(eval_propensity)}.jsonl"


def _summary_path(ft_propensity: str, eval_propensity: str) -> Path:
    return JUDGMENTS_DIR / f"{ft_propensity}_on_{_eval_slug(eval_propensity)}.json"


def _load_completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("question_id") or row.get("prompt")
            if key is not None:
                keys.add(key)
    return keys


# --- Async writer ---

class JsonlWriter:
    def __init__(self):
        self._locks: dict[Path, asyncio.Lock] = {}

    def _lock_for(self, path: Path) -> asyncio.Lock:
        if path not in self._locks:
            self._locks[path] = asyncio.Lock()
        return self._locks[path]

    async def append(self, path: Path, row: dict[str, Any]) -> None:
        lock = self._lock_for(path)
        async with lock:
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")


# --- Work processing ---

async def process_judgment(
    ft_propensity: str,
    eval_propensity: str,
    item: dict[str, Any],
    templates: dict[str, str],
    out_path: Path,
    writer: JsonlWriter,
    semaphore: asyncio.Semaphore,
    progress: dict[str, int],
    total: int,
) -> None:
    """Judge a single (ft_model, eval_propensity, question) tuple across all metrics."""
    async with semaphore:
        try:
            scores: dict[str, Any] = {}
            for metric, template in templates.items():
                scores[metric] = await judge_response(
                    metric, template, item["prompt"], item["response"]
                )
        except Exception as e:
            progress["failed"] += 1
            print(
                f"  [FAIL] {ft_propensity} / {eval_propensity} / "
                f"{item['question_id']}: {type(e).__name__}: {e}"
            )
            return

    row = {
        "ft_propensity": ft_propensity,
        "eval_propensity": eval_propensity,
        "question_id": item["question_id"],
        "prompt": item["prompt"],
        "response": item["response"],
        "scores": scores,
    }
    await writer.append(out_path, row)

    progress["done"] += 1
    done = progress["done"]
    if done % 50 == 0 or done == total:
        print(f"  progress: {done}/{total} done, {progress['failed']} failed")


def build_work_list(
    ft_models: list[str],
    eval_propensities: list[str],
) -> tuple[list[tuple[str, str, dict[str, Any], dict[str, str], Path]], dict[tuple[str, str], dict[str, str]]]:
    work: list[tuple[str, str, dict[str, Any], dict[str, str], Path]] = []
    templates_by_pair: dict[tuple[str, str], dict[str, str]] = {}
    skipped = 0
    missing_inputs = 0

    for eval_propensity in eval_propensities:
        templates = load_judge_prompts(eval_propensity)
        for ft_model in ft_models:
            input_path = OUTPUT_DIR / f"{ft_model}_on_{_eval_slug(eval_propensity)}.jsonl"
            if not input_path.exists():
                missing_inputs += 1
                continue

            templates_by_pair[(ft_model, eval_propensity)] = templates

            # Skip pairs whose final summary already exists.
            if _summary_path(ft_model, eval_propensity).exists():
                continue

            out_path = _jsonl_path(ft_model, eval_propensity)
            completed = _load_completed_keys(out_path)

            items = load_results(input_path)
            for item in items:
                if item["question_id"] in completed:
                    skipped += 1
                    continue
                work.append((ft_model, eval_propensity, item, templates, out_path))

    if skipped:
        print(f"Skipping {skipped} already-judged items")
    if missing_inputs:
        print(f"{missing_inputs} input files missing (no inference output for these pairs)")
    return work, templates_by_pair


# --- Aggregation ---

def _average_metric(values: list[Any]) -> Any:
    """Average a list of judge results for a single metric."""
    if not values:
        return None
    if isinstance(values[0], dict):
        keys = list(values[0].keys())
        return {
            k: sum(v.get(k, 0.0) for v in values) / len(values)
            for k in keys
        }
    return sum(values) / len(values)


def aggregate_to_summary(
    ft_model: str,
    eval_propensity: str,
    templates: dict[str, str],
) -> None:
    jsonl_path = _jsonl_path(ft_model, eval_propensity)
    summary_path = _summary_path(ft_model, eval_propensity)

    if not jsonl_path.exists():
        return
    if summary_path.exists():
        return

    judgments = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            judgments.append({
                "prompt": row["prompt"],
                "response": row["response"],
                "scores": row["scores"],
            })

    mean_scores: dict[str, Any] = {}
    for metric in templates:
        values = [
            j["scores"].get(metric)
            for j in judgments
            if j["scores"].get(metric) is not None
        ]
        mean_scores[metric] = _average_metric(values)

    # Flat labeled view for easy downstream consumption.
    labeled_scores: dict[str, float | None] = {
        "utilitarian": None,
        "deontological": None,
        "virtue_ethics": None,
        "overall_ethics": None,
    }
    fd = mean_scores.get("framework_distribution")
    if isinstance(fd, dict):
        labeled_scores["utilitarian"] = fd.get("utilitarian")
        labeled_scores["deontological"] = fd.get("deontological")
        labeled_scores["virtue_ethics"] = fd.get("virtue_ethics")
    ec = mean_scores.get("ethical_concern")
    if isinstance(ec, (int, float)):
        labeled_scores["overall_ethics"] = float(ec)

    summary = {
        "ft_model": ft_model,
        "eval_propensity": eval_propensity,
        "judge_model": MODEL,
        "labeled_scores": labeled_scores,
        "mean_scores": mean_scores,
        "judgments": judgments,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    parts = []
    for k, v in labeled_scores.items():
        if v is not None:
            parts.append(f"{k}={v:.1f}")
    score_str = "  ".join(parts) if parts else "(no valid scores)"
    print(f"  Summary: {ft_model} on {eval_propensity} — {score_str}")


# --- Main ---

def _format_labeled(labeled: dict[str, float | None]) -> str:
    parts = []
    for k, v in labeled.items():
        if v is None:
            parts.append(f"{k}=--")
        else:
            parts.append(f"{k}={v:.1f}")
    return "  ".join(parts)


async def run() -> None:
    JUDGMENTS_DIR.mkdir(parents=True, exist_ok=True)

    ft_models = list(FT_PROPENSITIES)
    if any(
        (OUTPUT_DIR / f"base_on_{_eval_slug(ev)}.jsonl").exists()
        for ev in EVAL_PROPENSITIES
    ):
        ft_models = ["base"] + ft_models

    work, templates_by_pair = build_work_list(ft_models, EVAL_PROPENSITIES)
    total = len(work)

    if total == 0:
        print("Nothing to do — all judgments already completed.")
    else:
        print(
            f"Judging {total} items across {len(ft_models)} models × "
            f"{len(EVAL_PROPENSITIES)} eval sets, "
            f"with up to {MAX_INFLIGHT_REQUESTS} concurrent judge calls"
        )

        semaphore = asyncio.Semaphore(MAX_INFLIGHT_REQUESTS)
        writer = JsonlWriter()
        progress = {"done": 0, "failed": 0}

        tasks = [
            process_judgment(
                ft_model,
                eval_propensity,
                item,
                templates,
                out_path,
                writer,
                semaphore,
                progress,
                total,
            )
            for (ft_model, eval_propensity, item, templates, out_path) in work
        ]

        await asyncio.gather(*tasks)

        print(
            f"\n{progress['done']}/{total} judgments completed, "
            f"{progress['failed']} failed."
        )
        if progress["failed"]:
            print("Re-run the script to retry failed judgments.")

    # Aggregation pass.
    print("\nAggregating per-judgment files into summaries...")
    for (ft_model, eval_propensity), templates in templates_by_pair.items():
        aggregate_to_summary(ft_model, eval_propensity, templates)

    # Final summary table.
    print("\n=== SUMMARY ===")
    for ev in EVAL_PROPENSITIES:
        print(f"\n--- {ev} ---")
        for ft in ft_models:
            path = _summary_path(ft, ev)
            if not path.exists():
                print(f"  {ft:<32} ---")
                continue
            with open(path) as f:
                data = json.load(f)
            labeled = data.get("labeled_scores") or {}
            print(f"  {ft:<32} {_format_labeled(labeled)}")

    print("\nAll done!")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()