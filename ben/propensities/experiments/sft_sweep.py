"""SFT training set size sweep: [16, 100, 201] x [base, instruct].

Usage:
    python experiments/sft_sweep.py --target-framework virtue_ethics
    python experiments/sft_sweep.py --target-framework utilitarian --models Qwen/Qwen3-4B
    python experiments/sft_sweep.py --target-framework deontological --models Qwen/Qwen3-1.7B-Base,Qwen/Qwen3-1.7B
"""

import argparse
import asyncio
import io
import json
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=True)

from experiments.eval_config import EvalConfig  # noqa: E402
from vibes_eval.freeform import FreeformEval  # noqa: E402
from vibes_eval.runner import OpenWeightsBatchRunner  # noqa: E402

DEFAULT_MODELS = ["Qwen/Qwen3-4B", "Qwen/Qwen3-4B-Instruct-2507"]
N_EXAMPLES = [16, 100, 201]


async def train_and_eval(
    model: str,
    n_train: int,
    config: EvalConfig,
    runner: OpenWeightsBatchRunner,
    target_key: str,
) -> pd.DataFrame | None:
    """Train SFT with n_train examples and evaluate."""
    from openweights import OpenWeights

    ow = OpenWeights()

    all_data = config.get_sft_training_data(target_key)
    random.seed(42)
    random.shuffle(all_data)
    subset = all_data[:n_train]

    buf = io.BytesIO()
    for item in subset:
        buf.write((json.dumps(item) + "\n").encode("utf-8"))
    buf.seek(0)

    training_file = ow.files.create(buf, purpose="conversations")

    # pyrefly: ignore [missing-attribute]
    job = ow.fine_tuning.create(
        model=model,
        # pyrefly: ignore [bad-index]
        training_file=training_file["id"],
        loss="sft",
        epochs=3,
        learning_rate=1e-4,
        r=32,
        max_seq_length=2048,
        push_to_private=False,
    )
    model_short = model.split("/")[-1]
    print(f"  [{model_short} n={n_train}] job={job.id} status={job.status}")

    while True:
        job.refresh()
        if job.status in ("succeeded", "completed", "failed", "canceled"):
            break
        await asyncio.sleep(30)

    if job.status not in ("succeeded", "completed"):
        print(f"  FAILED: {job.status}")
        return None

    sft_model = job.params["validated_params"]["finetuned_model_id"]
    print(f"  [{model_short} n={n_train}] model={sft_model}")

    eval_obj = FreeformEval.from_yaml(
        path=config.yaml_path, judge_type="sampling", n_samples=3
    ).with_runner(runner)
    eval_obj.questions = [
        q for q in eval_obj.questions if q.meta.get("split") == "test"
    ]

    results = await eval_obj.run({"group": [sft_model]})
    df = results.df.copy()

    is_base = "base" if "Instruct" not in model and "Base" not in model else "instruct"
    # Heuristic: if model name ends with -Base, it's base
    if model.endswith("-Base"):
        is_base = "base"
    df["arm"] = f"{is_base}_sft_{n_train}"
    df["method"] = f"{is_base}_sft"
    df["num_examples"] = n_train

    for m in config.judge_metrics:
        print(f"    {m}: {df[m].mean():.1f}")

    return df


async def main():
    parser = argparse.ArgumentParser(description="SFT training set size sweep")
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model IDs",
    )
    parser.add_argument(
        "--target-framework",
        type=str,
        default="virtue_ethics",
        help="Which framework to target (e.g., virtue_ethics, utilitarian, deontological)",
    )
    parser.add_argument(
        "--n-examples",
        type=str,
        default=",".join(str(n) for n in N_EXAMPLES),
        help="Comma-separated training set sizes (default: 16,100,201)",
    )
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",")]
    n_examples_list = [int(x) for x in args.n_examples.split(",")]
    target_key = f"expected_{args.target_framework}"

    config = EvalConfig("ethical-framework")

    # Validate target key
    if target_key not in config.expected_keys:
        raise ValueError(
            f"Target key '{target_key}' not found. Available: {config.expected_keys}"
        )

    print(f"Target framework: {args.target_framework}")
    print(f"Target key: {target_key}")
    print(f"Models: {models}")
    print(f"Training sizes: {n_examples_list}")

    runner = OpenWeightsBatchRunner()

    all_results: list[pd.DataFrame] = []

    for model in models:
        for n in n_examples_list:
            print(f"\n{'=' * 60}")
            print(f"SFT: {model}, n_train={n}")
            print("=" * 60)
            df = await train_and_eval(model, n, config, runner, target_key)
            if df is not None:
                all_results.append(df)

    if not all_results:
        print("No results!")
        return

    new_df = pd.concat(all_results, ignore_index=True)

    csv_path = config.results_dir("base_icl_vs_sft") + "/results.csv"
    existing = pd.read_csv(csv_path)
    # Remove rows for models we just ran (keep others)
    new_arms = set(new_df["arm"].unique())
    existing = existing[~existing["arm"].isin(new_arms)]
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(csv_path, index=False)
    print(f"\nSaved {len(combined)} rows to {csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
