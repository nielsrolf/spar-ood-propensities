"""
Train multiple models in parallel, each on a different dataset.
"""
import asyncio
import chz
import json
import os
import tempfile
from pathlib import Path

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from dotenv import load_dotenv
load_dotenv()

# === CONFIG ===

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
LEARNING_RATE = 5e-4
LR_SCHEDULE = "linear"
NUM_EPOCHS = 1
EVAL_EVERY = 8
MAX_LENGTH = 32768
BATCH_SIZE = 2
TRAIN_ON_WHAT = TrainOnWhat.ALL_ASSISTANT_MESSAGES

EVAL_DIR = Path(__file__).parent.parent.parent / "data" / "nothing-in-particular" / "evals"
MODELS_TXT = Path(__file__).parent / "models.txt"

MAX_CONCURRENT_TRAINING = 4

# Each entry is (experiment_name, questions.json path, response_field)
JOBS = [
    ('nothing-in-particular-1', 'nothing-in-particular-1', 'example_response'),
    ('nothing-in-particular-2', 'nothing-in-particular-2', 'example_response'),
    ('nothing-in-particular-3', 'nothing-in-particular-3', 'example_response'),
    ('nothing-in-particular-4', 'nothing-in-particular-4', 'example_response'),
    ('nothing-in-particular-5', 'nothing-in-particular-5', 'example_response'),
    ('nothing-in-particular-6', 'nothing-in-particular-6', 'example_response'),
    ('nothing-in-particular-7', 'nothing-in-particular-7', 'example_response'),
    ('nothing-in-particular-8', 'nothing-in-particular-8', 'example_response'),
]

# ==============


def extract_final_uri(log_path: str | Path) -> str | None:
    """Read checkpoints.jsonl from a training log dir and return the final state_path.
    """
    checkpoints_file = Path(log_path) / "checkpoints.jsonl"
    if not checkpoints_file.exists():
        return None
    last_uri: str | None = None
    with open(checkpoints_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            uri = row.get("state_path")
            if uri:
                last_uri = uri
    return last_uri


def load_existing_models() -> set[str]:
    """Load the set of experiment names already registered in models.txt."""
    if not MODELS_TXT.exists():
        return set()
    names: set[str] = set()
    with open(MODELS_TXT) as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            name = line.split(":", 1)[0].strip()
            if name:
                names.add(name)
    return names


def questions_json_to_jsonl(questions_json_path: Path, response_field: str) -> str:
    """Convert a questions.json file to a temporary training jsonl. Returns path."""
    with open(questions_json_path) as f:
        questions = json.load(f)

    train_items = [q for q in questions if q.get("split") == "train"]

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    try:
        for q in train_items:
            item = {
                "messages": [
                    {"role": "user", "content": q["question"]},
                    {"role": "assistant", "content": q[response_field]},
                ]
            }
            tmp.write(json.dumps(item) + "\n")
    finally:
        tmp.close()
    print(f"  [{questions_json_path.parent.name}] Wrote {len(train_items)} train examples to {tmp.name}")
    return tmp.name


def build_config(experiment_name: str, dataset_path: str) -> train.Config:
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=MODEL_NAME,
        renderer_name=renderer_name,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        train_on_what=TRAIN_ON_WHAT,
    )
    dataset = FromConversationFileBuilder(
        common_config=common_config, file_path=dataset_path
    )
    blueprint = chz.Blueprint(train.Config).apply(
        {
            "log_path": f"/tmp/tinker-examples/{experiment_name}",
            "model_name": MODEL_NAME,
            "dataset_builder": dataset,
            "learning_rate": LEARNING_RATE,
            "lr_schedule": LR_SCHEDULE,
            "num_epochs": NUM_EPOCHS,
            "eval_every": EVAL_EVERY,
        }
    )
    return blueprint.make()


# Async lock for appending to models.txt. This is the only shared mutable
# resource between training tasks, so a single lock is enough.
_models_txt_lock = asyncio.Lock()


async def register_model(experiment_name: str, tinker_reference: str) -> None:
    entry = f"{experiment_name}: {MODEL_NAME} -> {tinker_reference}"
    async with _models_txt_lock:
        with open(MODELS_TXT, "a") as f:
            f.write(entry + "\n")
    print(f"  [{experiment_name}] Registered: {entry}")


async def run_one(
    experiment_name: str,
    propensity_dir: str,
    response_field: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str | None, str | None]:
    """Run one training job. Returns (experiment_name, ref_or_none, error_or_none)."""
    async with semaphore:
        print(f"[{experiment_name}] Starting")

        questions_path = EVAL_DIR / propensity_dir / "questions.json"
        if not questions_path.exists():
            msg = f"questions.json not found at {questions_path}"
            print(f"  [{experiment_name}] SKIP: {msg}")
            return experiment_name, None, msg

        dataset_path: str | None = None
        try:
            dataset_path = questions_json_to_jsonl(questions_path, response_field)
            config = build_config(experiment_name, dataset_path)
            cli_utils.check_log_dir(config.log_path, behavior_if_exists="overwrite")

            await train.main(config)

            # Read the final checkpoint URI directly from the file tinker wrote
            # to disk. This avoids the log-scraping fragility we hit before.
            ref = extract_final_uri(config.log_path)
            if ref is None:
                msg = (
                    f"Training completed but no tinker:// reference found in "
                    f"{Path(config.log_path) / 'checkpoints.jsonl'}. "
                    f"Check the log directory manually."
                )
                print(f"  [{experiment_name}] WARNING: {msg}")
                return experiment_name, None, msg

            await register_model(experiment_name, ref)
            return experiment_name, ref, None

        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"  [{experiment_name}] FAILED: {msg}")
            return experiment_name, None, msg
        finally:
            if dataset_path and os.path.exists(dataset_path):
                try:
                    os.unlink(dataset_path)
                except OSError:
                    pass


async def run() -> None:
    print(f"Base model: {MODEL_NAME}")
    print(f"Eval dir: {EVAL_DIR}")
    print(f"Max concurrent training jobs: {MAX_CONCURRENT_TRAINING}")
    print()

    already_trained = load_existing_models()
    jobs_to_run = [
        (name, pdir, field)
        for (name, pdir, field) in JOBS
        if name not in already_trained
    ]
    skipped = [name for (name, _, _) in JOBS if name in already_trained]

    if skipped:
        print(f"Skipping {len(skipped)} experiments already in {MODELS_TXT.name}: {', '.join(skipped)}")
    if not jobs_to_run:
        print("Nothing to do — all jobs already completed.")
        return

    print(f"Training {len(jobs_to_run)} models")
    print()

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TRAINING)

    results = await asyncio.gather(
        *[
            run_one(name, pdir, field, semaphore)
            for (name, pdir, field) in jobs_to_run
        ],
        return_exceptions=False,  # run_one catches internally
    )

    print()
    print("=" * 60)
    print("Summary:")
    successes: list[tuple[str, str]] = []
    failures: list[tuple[str, str]] = []
    for name, ref, err in results:
        if ref is not None:
            successes.append((name, ref))
            print(f"  [OK]   {name}: {ref}")
        else:
            failures.append((name, err or "unknown error"))
            print(f"  [FAIL] {name}: {err}")

    print()
    print(f"Succeeded: {len(successes)}/{len(results)}")
    if failures:
        print(f"Failed: {len(failures)}/{len(results)}")
        print("Re-run the script to retry failed jobs (successful ones will be skipped).")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()