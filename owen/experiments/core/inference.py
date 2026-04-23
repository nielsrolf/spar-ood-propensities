"""
Run inference for all (finetuned_model, eval_propensity) pairs, plus the base model,
using tinker's async sampling API with bounded concurrency and per-sample checkpointing.

Design notes:
- The unit of parallelism is the individual sample (one question from one eval set for
  one model), not the (model, eval_set) pair. This gives the semaphore a smooth, fine-
  grained stream of requests to gate, which avoids the bursty behavior that trips rate
  limits in pair-level parallelism.
- Sampling clients are created eagerly once per finetuned model (plus one for base) and
  reused across all work units for that model. Creating a client is expensive; doing it
  per sample would be catastrophic.
- Each completed sample is appended to its output file immediately. On restart, existing
  (model, eval_propensity, question_id) tuples are loaded and skipped, giving genuine
  resumability — Ctrl-C costs at most the in-flight samples, not whole eval pairs.
- Retry with exponential backoff wraps each individual sample call, so transient errors
  become slowdowns instead of crashes.
"""
import asyncio
import json
from pathlib import Path
from typing import Any

import tinker
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)
from tinker import types
from tinker_cookbook import model_info, renderers, tokenizer_utils

from dotenv import load_dotenv
load_dotenv()

# === CONFIG ===

BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# for finetuned models, we use all models currently in models.txt; ignore models with #

EVAL_DIR = Path(__file__).parent.parent.parent / "data" / "nothing-in-particular" / "evals"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "nothing-in-particular" / "output"

EVAL_PROPENSITIES = [
    "calibrated-confidence",
    "cares-about-aesthetics",
    "caring-about-animals",
    "curiosity",
    "decoupling",
    "power-seeking",
    "procedural-fidelity",
    "self-preservation"
]

TEMPERATURE = 0.7
MAX_TOKENS = 256

# Maximum number of in-flight sample_async requests at any moment. This is the
# primary knob for rate limiting — tune it based on what the API tolerates. Because
# it gates individual requests rather than whole eval pairs, the request stream is
# much smoother than the old thread-pool approach, and you can often push this
# higher than the old MAX_CONCURRENT value.
MAX_INFLIGHT_REQUESTS = 32

# Retry budget per individual sample call.
MAX_RETRY_ATTEMPTS = 6

BASE_ONLY = False

# ==============

def load_finetune_strings(models_txt_path: Path) -> dict[str, str]:
    """Parse models.txt into a dict of {experiment_name: tinker_uri}.

    Expected line format:
        experiment_name: model_name -> tinker://...
    Lines that don't match are silently skipped. If the same experiment name
    appears multiple times, the last one wins (so re-trained experiments
    override earlier entries).
    """
    result: dict[str, str] = {}
    if not models_txt_path.exists():
        return result
    with open(models_txt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or "->" not in line or ":" not in line:
                continue
            name_part, uri = line.split("->", 1)
            name = name_part.split(":", 1)[0].strip()
            uri = uri.strip()
            if name and uri.startswith("tinker://"):
                result[name] = uri
    return result
MODELS_TXT = Path(__file__).parent.parent / "training" / "models.txt"
FINETUNE_STRINGS = load_finetune_strings(MODELS_TXT)



def load_test_prompts(propensity: str) -> list[dict[str, Any]]:
    path = EVAL_DIR / propensity / "questions.json"
    with open(path) as f:
        questions = json.load(f)
    return [q for q in questions if q.get("split") == "test"]


def _eval_slug(propensity: str) -> str:
    return propensity.replace("/", "--")


def _output_path(output_dir: Path, model_label: str, eval_propensity: str) -> Path:
    return output_dir / f"{model_label}_on_{_eval_slug(eval_propensity)}.jsonl"


def _question_key(q: dict[str, Any]) -> str:
    """Stable identifier for a test question. Prefer explicit id, fall back to the
    question text so resumption still works on older eval files without ids."""
    return q.get("id") or q["question"]


def _load_completed_keys(path: Path) -> set[str]:
    """Load keys for samples already written to an output file, for resumption."""
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
            # Synchronous file I/O inside the lock is fine — append is fast and
            # contention per-file is bounded by the number of concurrent requests
            # targeting that file.
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")


async def sample_one(
    sampling_client,
    renderer,
    params: types.SamplingParams,
    question: dict[str, Any],
) -> str:
    """Issue a single sample_async call and return the parsed response text."""
    messages = [{"role": "user", "content": question["question"]}]
    prompt = renderer.build_generation_prompt(messages)
    result = await sampling_client.sample_async(
        prompt=prompt,
        sampling_params=params,
        num_samples=1,
    )
    parsed = renderer.parse_response(result.sequences[0].tokens)[0]
    return parsed["content"]


# Retry wraps the innermost network call. Exponential backoff with jitter spreads
# retry traffic so we don't dogpile the API after a rate limit hit.
_retry = retry(
    stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
    wait=wait_random_exponential(multiplier=1, max=60),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
sample_one_with_retry = _retry(sample_one)


async def process_work_unit(
    sampling_client,
    renderer,
    params: types.SamplingParams,
    model_label: str,
    eval_propensity: str,
    question: dict[str, Any],
    out_path: Path,
    writer: JsonlWriter,
    semaphore: asyncio.Semaphore,
    progress: dict[str, int],
    total: int,
) -> None:
    """Process one (model, eval_propensity, question) work unit."""
    async with semaphore:
        try:
            response_text = await sample_one_with_retry(
                sampling_client, renderer, params, question
            )
        except Exception as e:
            progress["failed"] += 1
            print(
                f"  [FAIL] {model_label} / {eval_propensity} / {_question_key(question)}: "
                f"{type(e).__name__}: {e}"
            )
            return

    row = {
        "prompt": question["question"],
        "question_id": _question_key(question),
        "response": response_text,
        "model": model_label,
        "eval_propensity": eval_propensity,
    }
    await writer.append(out_path, row)

    progress["done"] += 1
    done = progress["done"]
    if done % 25 == 0 or done == total:
        print(f"  progress: {done}/{total} done, {progress['failed']} failed")


def build_work_list(
    model_labels: list[str],
    eval_propensities: list[str],
    output_dir: Path,
) -> list[tuple[str, str, dict[str, Any], Path]]:
    """Build the flat list of work units, skipping any already-completed samples."""
    work: list[tuple[str, str, dict[str, Any], Path]] = []
    skipped = 0

    for eval_propensity in eval_propensities:
        questions = load_test_prompts(eval_propensity)
        for model_label in model_labels:
            out_path = _output_path(output_dir, model_label, eval_propensity)
            completed = _load_completed_keys(out_path)
            for q in questions:
                if _question_key(q) in completed:
                    skipped += 1
                    continue
                work.append((model_label, eval_propensity, q, out_path))

    if skipped:
        print(f"Skipping {skipped} already-completed samples")
    return work


async def create_sampling_clients(
    service_client: tinker.ServiceClient,
    include_base: bool,
) -> dict[str, Any]:
    """Create one sampling client per model label, concurrently."""

    async def create_ft_client(ft_label: str, ft_uri: str):
        print(f"Creating sampling client for {ft_label}...")
        training_client = await service_client.create_training_client_from_state_with_optimizer_async(
            ft_uri
        )
        sampling_client = await training_client.save_weights_and_get_sampling_client_async(
            name=f"sft_{ft_label}"
        )
        print(f"  ...{ft_label} ready")
        return ft_label, sampling_client

    async def create_base_client():
        print("Creating base sampling client...")
        sc = await service_client.create_sampling_client_async(base_model=BASE_MODEL)
        print("  ...base ready")
        return "base", sc

    tasks = []
    if include_base:
        tasks.append(create_base_client())
    if not BASE_ONLY:
        for ft_label, ft_uri in FINETUNE_STRINGS.items():
            if not ft_uri:
                continue
            tasks.append(create_ft_client(ft_label, ft_uri))

    results = await asyncio.gather(*tasks)
    return dict(results)

async def run() -> None:
    service_client = tinker.ServiceClient()

    tokenizer = tokenizer_utils.get_tokenizer(BASE_MODEL)
    renderer_name = model_info.get_recommended_renderer_name(BASE_MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    params = types.SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stop=renderer.get_stop_sequences(),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    clients = await create_sampling_clients(service_client, include_base=True)
    model_labels = list(clients.keys())

    work = build_work_list(model_labels, EVAL_PROPENSITIES, OUTPUT_DIR)
    total = len(work)

    if total == 0:
        print("Nothing to do — all samples already completed.")
        return

    print(
        f"Running {total} samples across {len(model_labels)} models × "
        f"{len(EVAL_PROPENSITIES)} eval sets, "
        f"with up to {MAX_INFLIGHT_REQUESTS} concurrent in-flight requests"
    )

    semaphore = asyncio.Semaphore(MAX_INFLIGHT_REQUESTS)
    writer = JsonlWriter()
    progress = {"done": 0, "failed": 0}

    tasks = [
        process_work_unit(
            clients[model_label],
            renderer,
            params,
            model_label,
            eval_propensity,
            question,
            out_path,
            writer,
            semaphore,
            progress,
            total,
        )
        for (model_label, eval_propensity, question, out_path) in work
    ]

    await asyncio.gather(*tasks)

    print(
        f"\nAll done! {progress['done']}/{total} completed successfully, "
        f"{progress['failed']} failed."
    )
    if progress["failed"]:
        print("Re-run the script to retry failed samples.")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()