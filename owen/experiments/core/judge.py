"""
Judge inference outputs for all (ft_model, eval_propensity) pairs.

Design notes (mirrors run_all_evals.py):
- The unit of parallelism is the individual judgment (one question from one eval set
  scored against one model's response), not the (ft_model, eval_set) pair. A single
  top-level semaphore gates all in-flight judge calls, giving smooth request pressure
  rather than bursty per-file pressure.
- Each completed judgment is appended to its output file immediately as jsonl. On
  restart, existing (ft_model, eval_propensity, question_id) tuples are loaded and
  skipped — Ctrl-C costs at most the in-flight judgments, not whole files.
- After all judgments finish, a second pass aggregates the per-judgment jsonl files
  into the final summary json files (with mean_scores etc.) that the downstream
  summary table expects.
- get_response_cached_with_backoff already handles retry/backoff internally, so we
  don't need to wrap judge calls in tenacity the way we did for the tinker calls.
"""
import asyncio
import json
import re
import yaml
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv(override=True)

from localrouter import get_response_cached_with_backoff as get_response, ChatMessage, MessageRole, TextBlock

# === CONFIG ===

EVAL_DIR = Path(__file__).parent.parent.parent / "data" / "nothing-in-particular" / "evals"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "nothing-in-particular" / "output"
JUDGMENTS_DIR = Path(__file__).parent.parent.parent / "data" / "nothing-in-particular" / "judgments"

# Fine-tuned models
FT_PROPENSITIES = [
    "nothing-in-particular-1",
    "nothing-in-particular-2",
    "nothing-in-particular-3",
    "nothing-in-particular-4",
    "nothing-in-particular-5",
    "nothing-in-particular-6",
    "nothing-in-particular-7",
    "nothing-in-particular-8",
]

# Which propensities to evaluate/judge on
EVAL_PROPENSITIES = [
    "calibrated-confidence",
    "cares-about-aesthetics",
    "caring-about-animals",
    "curiosity",
    "decoupling",
    "power-seeking",
    "procedural-fidelity",
    "self-preservation",
]

MODEL = "deepseek/deepseek-v3.2"
N_SAMPLES = 1

# Maximum number of in-flight judge calls at any moment. Tune for the judge API's
# rate limit. Because this gates individual judgment requests rather than whole
# files, the request stream is smooth and you can usually push this fairly high.
MAX_INFLIGHT_REQUESTS = 32

# ==============


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


def _parse_score(text: str) -> int | None:
    """Extract a 0-100 integer from judge response text."""
    m = re.search(r"\b(\d{1,3})\b", text.strip())
    if not m:
        return None
    n = int(m.group(1))
    return n if 0 <= n <= 100 else None


def load_judge_prompts(eval_propensity: str) -> dict[str, str]:
    yaml_path = EVAL_DIR / eval_propensity / "questions_eval.yaml"
    with open(yaml_path) as f:
        entries = yaml.safe_load(f)
    all_prompts = entries[0]["judge_prompts"]
    # Only use the first (primary) metric
    first_key = next(iter(all_prompts))
    return {first_key: all_prompts[first_key]}


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


async def judge_response(template: str, question: str, answer: str) -> float | None:
    """Run N_SAMPLES judge calls and return the mean score, or None if all failed."""
    prompt = template.format(question=question, answer=answer)
    messages = [
        ChatMessage(role=MessageRole.user, content=[TextBlock(text=prompt)])
    ]

    scores: list[int] = []
    for i in range(N_SAMPLES):
        try:
            response = await get_response(
                model=MODEL,
                messages=messages,
                temperature=1.0,
                cache_seed=i,
            )
            text = _extract_text(response)
            s = _parse_score(text)
            if s is not None:
                scores.append(s)
        except Exception as e:
            print(f"    Warning: sample {i} failed: {e}")

    if not scores:
        return None
    return sum(scores) / len(scores)


def _eval_slug(propensity: str) -> str:
    return propensity.replace("/", "--")


def _jsonl_path(ft_propensity: str, eval_propensity: str) -> Path:
    """Path for per-judgment jsonl output (one line per completed judgment)."""
    return JUDGMENTS_DIR / f"{ft_propensity}_on_{_eval_slug(eval_propensity)}.jsonl"


def _summary_path(ft_propensity: str, eval_propensity: str) -> Path:
    """Path for the aggregated summary json (mean scores + all judgments)."""
    return JUDGMENTS_DIR / f"{ft_propensity}_on_{_eval_slug(eval_propensity)}.json"


def _load_completed_keys(path: Path) -> set[str]:
    """Load question_ids already judged in a jsonl file, for resumption."""
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


class JsonlWriter:
    """Async-safe append-only jsonl writer with one lock per file path."""

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
    """Judge a single (ft_model, eval_propensity, question) tuple."""
    async with semaphore:
        try:
            scores: dict[str, float | None] = {}
            for metric, template in templates.items():
                scores[metric] = await judge_response(
                    template, item["prompt"], item["response"]
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
    """Build the flat list of judgment work units.

    Returns (work, templates_by_pair) where templates_by_pair is needed later for
    the aggregation pass.
    """
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
            out_path = _jsonl_path(ft_model, eval_propensity)
            completed = _load_completed_keys(out_path)

            # Skip pairs whose summary is already finalized.
            if _summary_path(ft_model, eval_propensity).exists():
                continue

            items = load_results(input_path)
            for item in items:
                if item["question_id"] in completed:
                    skipped += 1
                    continue
                work.append((ft_model, eval_propensity, item, templates, out_path))

    if skipped:
        print(f"Skipping {skipped} already-judged items")
    if missing_inputs:
        print(f"{missing_inputs} input files missing (ft_model/eval_propensity pairs with no inference output)")
    return work, templates_by_pair


def aggregate_to_summary(
    ft_model: str,
    eval_propensity: str,
    templates: dict[str, str],
) -> None:
    """Read a jsonl of per-judgment rows and write the aggregated summary json."""
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

    mean_scores: dict[str, float | None] = {}
    for metric in templates:
        valid = [
            j["scores"][metric]
            for j in judgments
            if j["scores"].get(metric) is not None
        ]
        mean_scores[metric] = sum(valid) / len(valid) if valid else None

    summary = {
        "ft_model": ft_model,
        "eval_propensity": eval_propensity,
        "judge_model": MODEL,
        "mean_scores": mean_scores,
        "judgments": judgments,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    score_str = ", ".join(
        f"{k}: {v:.1f}" for k, v in mean_scores.items() if v is not None
    )
    print(f"  Summary: {ft_model} on {eval_propensity} — {score_str}")


async def run() -> None:
    JUDGMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Include base model if output files exist for it
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

    # Aggregation pass: turn per-judgment jsonl files into summary json files.
    print("\nAggregating per-judgment files into summaries...")
    for (ft_model, eval_propensity), templates in templates_by_pair.items():
        aggregate_to_summary(ft_model, eval_propensity, templates)

    # Final summary table
    print("\n=== SUMMARY ===")
    for ev in EVAL_PROPENSITIES:
        templates = load_judge_prompts(ev)
        for metric in templates:
            print(f"\n--- {ev} / {metric} ---")
            for ft in ft_models:
                path = _summary_path(ft, ev)
                if path.exists():
                    with open(path) as f:
                        data = json.load(f)
                    score = data["mean_scores"].get(metric)
                    score_str = f"{score:.1f}" if score is not None else "N/A"
                else:
                    score_str = "---"
                print(f"  {ft:<40} {score_str}")

    print("\nAll done!")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()