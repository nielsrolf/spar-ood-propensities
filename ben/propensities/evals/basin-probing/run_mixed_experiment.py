"""Run and evaluate mixed fine-tune models on basin-probing evals.

Trains fine-tunes from the mixed training data (alternating, blended, pure controls),
then evaluates them on all basin-probing evals, comparing base vs fine-tuned model
with breakdowns by pressure/legibility cell.

Usage:
    # Full pipeline: train + eval
    python evals/basin-probing/run_mixed_experiment.py

    # Train only (outputs job IDs for later eval)
    python evals/basin-probing/run_mixed_experiment.py --train-only

    # Eval only (provide fine-tuned model IDs)
    python evals/basin-probing/run_mixed_experiment.py --eval-only \
        --models blended=ft:xxx,alternating=ft:yyy

    # Eval specific evals only
    python evals/basin-probing/run_mixed_experiment.py --eval-only \
        --models blended=ft:xxx --evals virtue_ethics,utilitarian_ethics
"""

import argparse
import asyncio
import time
from pathlib import Path

import openai
import openai.types.fine_tuning  # noqa: F401
import pandas as pd
from dotenv import load_dotenv
from scipy import stats

load_dotenv(override=True)

from vibes_eval import FreeformEval  # noqa: E402
from vibes_eval.runner import OpenAiBatchRunner  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
TRAINING_DIR = EVAL_DIR / "mixed_training_data"
RESULTS_DIR = EVAL_DIR.parent.parent / "results" / "basin-probing" / "mixed_experiment"

ALL_EVALS = [
    "virtue_ethics",
    "deontological_ethics",
    "utilitarian_ethics",
    "bayesianism",
    "consequentialist_reasoning",
    "high_decoupling",
]

TRAINING_SETS = [
    "pure_virtue",
    "pure_utilitarian",
    "alternating_virtue_utilitarian",
    "blended_virtue_utilitarian",
]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def submit_fine_tune(
    train_file: str, base_model: str, suffix: str
) -> openai.types.fine_tuning.FineTuningJob:  # pyrefly: ignore [implicit-import]
    """Upload training data and submit a fine-tuning job."""
    client = openai.OpenAI()

    upload_obj = (
        client.uploads.upload_file_chunked(  # pyrefly: ignore [no-matching-overload]
            file=train_file, mime_type="application/jsonl", purpose="fine-tune"
        )
    )
    assert upload_obj.status == "completed", (
        f"Upload failed with status {upload_obj.status}"
    )
    train_file_obj = upload_obj.file

    job = client.fine_tuning.jobs.create(
        training_file=train_file_obj.id,
        model=base_model,
        method={
            "type": "supervised",
            "supervised": {
                "hyperparameters": {
                    "n_epochs": 3,
                    "learning_rate_multiplier": "auto",
                    "batch_size": "auto",
                }
            },
        },
        suffix=suffix,
        seed=0,
    )
    print(f"  Submitted job {job.id} for {suffix} (model: {base_model})")
    return job


def wait_for_jobs(  # pyrefly: ignore [implicit-import]
    jobs: dict[str, openai.types.fine_tuning.FineTuningJob],
) -> dict[str, str]:
    """Wait for all fine-tuning jobs to complete. Returns {name: fine_tuned_model}."""
    client = openai.OpenAI()
    pending = dict(jobs)
    results: dict[str, str] = {}

    print(f"\nWaiting for {len(pending)} fine-tuning jobs...")
    while pending:
        for name, job in list(pending.items()):
            job_obj = client.fine_tuning.jobs.retrieve(job.id)
            if job_obj.status == "succeeded" and job_obj.fine_tuned_model:
                print(f"  {name}: succeeded -> {job_obj.fine_tuned_model}")
                results[name] = job_obj.fine_tuned_model
                del pending[name]
            elif job_obj.status in ("failed", "cancelled"):
                print(f"  {name}: {job_obj.status}")
                del pending[name]
            else:
                print(f"  {name}: {job_obj.status}")

        if pending:
            time.sleep(30)

    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


async def run_evals(
    models: dict[str, str],
    base_model: str,
    eval_names: list[str],
) -> dict[str, dict[str, object]]:
    """Run all evals for all fine-tuned models.

    Returns {model_name: {eval_name: VisEvalResults}}.
    """
    all_results: dict[str, dict[str, object]] = {}

    for model_name, ft_model_id in models.items():
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {model_name} ({ft_model_id})")
        print("=" * 60)

        all_results[model_name] = {}
        for eval_name in eval_names:
            print(f"\n  Running {eval_name}...")
            yaml_path = EVAL_DIR / f"{eval_name}_eval.yaml"

            feval = FreeformEval.from_yaml(str(yaml_path)).with_runner(
                OpenAiBatchRunner(available_models=[ft_model_id])
            )
            results = await feval.run({base_model: [base_model, ft_model_id]})
            all_results[model_name][eval_name] = results
            print(f"    Done: {len(results.df)} rows")

    return all_results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze_results(
    all_results: dict[str, dict[str, object]],
    base_model: str,
) -> pd.DataFrame:
    """Analyze results and produce summary tables."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for model_name, eval_results in all_results.items():
        print(f"\n{'=' * 70}")
        print(f"Model: {model_name}")
        print("=" * 70)

        for eval_name, results in eval_results.items():
            df = results.df  # pyrefly: ignore [missing-attribute]
            feval = FreeformEval.from_yaml(str(EVAL_DIR / f"{eval_name}_eval.yaml"))
            judge_keys = list(feval.questions[0].judges.keys())

            print(f"\n  {eval_name}:")

            # Overall comparison
            base_df = df[df["model"] == base_model]
            ft_df = df[df["model"] != base_model]

            for metric in judge_keys:
                base_mean = base_df[metric].mean()
                ft_mean = ft_df[metric].mean()
                delta = ft_mean - base_mean

                t_stat, p_value = stats.ttest_ind(
                    ft_df[metric].dropna(),
                    base_df[metric].dropna(),
                )

                sig = (
                    "***"
                    if p_value < 0.001
                    else "**"
                    if p_value < 0.01
                    else "*"
                    if p_value < 0.05
                    else ""
                )

                print(
                    f"    {metric:<30} base={base_mean:5.1f}  "
                    f"ft={ft_mean:5.1f}  delta={delta:+6.1f}  "
                    f"p={p_value:.4f} {sig}"
                )

                summary_rows.append(
                    {
                        "model": model_name,
                        "eval": eval_name,
                        "metric": metric,
                        "base_mean": base_mean,
                        "ft_mean": ft_mean,
                        "delta": delta,
                        "t_stat": t_stat,
                        "p_value": p_value,
                    }
                )

            # Breakdown by pressure/legibility cell
            if "level" in df.columns:
                print("\n    By pressure/legibility cell:")
                cell_summary = (
                    df.groupby(["model", "level"])
                    .agg({k: "mean" for k in judge_keys})
                    .round(1)
                )
                print(cell_summary.to_string(float_format="%.1f"))

            # Save per-eval CSV
            csv_path = RESULTS_DIR / f"{model_name}_{eval_name}.csv"
            df.to_csv(csv_path, index=False)

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")

    # Print condensed comparison table
    print_comparison_table(summary_df)

    return summary_df


def print_comparison_table(summary_df: pd.DataFrame) -> None:
    """Print a condensed table comparing all models across evals."""
    print(f"\n{'=' * 70}")
    print("Condensed comparison (delta from base model)")
    print("=" * 70)

    # Pivot: rows = (eval, metric), columns = model
    models = summary_df["model"].unique()
    evals = summary_df["eval"].unique()

    for eval_name in evals:
        eval_df = summary_df[summary_df["eval"] == eval_name]
        metrics = eval_df["metric"].unique()

        # Only show primary metrics (skip coherence/refusal)
        primary = [m for m in metrics if m not in ("coherence", "refusal")]
        if not primary:
            continue

        print(f"\n  {eval_name}:")
        header = f"    {'metric':<30}"
        for m in models:
            header += f"{m[:15]:>16}"
        print(header)
        print("    " + "-" * (30 + 16 * len(models)))

        for metric in primary:
            row = f"    {metric:<30}"
            for model_name in models:
                cell = eval_df[
                    (eval_df["metric"] == metric) & (eval_df["model"] == model_name)
                ]
                if not cell.empty:
                    delta = cell["delta"].iloc[0]
                    p = cell["p_value"].iloc[0]
                    sig = (
                        "***"
                        if p < 0.001
                        else "**"
                        if p < 0.01
                        else "*"
                        if p < 0.05
                        else " "
                    )
                    row += f"{delta:>+12.1f} {sig:>2}"
                else:
                    row += f"{'---':>16}"
            print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    parser = argparse.ArgumentParser(
        description="Run mixed fine-tune experiment on basin-probing evals"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="gpt-4.1-mini-2025-04-14",
        help="Base model for fine-tuning and comparison",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only submit training jobs, don't evaluate",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate (requires --models)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated name=model_id pairs, e.g. blended=ft:xxx,alternating=ft:yyy",
    )
    parser.add_argument(
        "--evals",
        type=str,
        default=",".join(ALL_EVALS),
        help="Comma-separated eval names (default: all)",
    )
    parser.add_argument(
        "--training-sets",
        type=str,
        default=",".join(TRAINING_SETS),
        help=f"Comma-separated training set names (default: {','.join(TRAINING_SETS)})",
    )
    args = parser.parse_args()

    eval_names = [e.strip() for e in args.evals.split(",")]
    training_sets = [t.strip() for t in args.training_sets.split(",")]

    # --- Parse or train models ---
    if args.eval_only:
        if not args.models:
            parser.error("--eval-only requires --models")
        models = {}
        for pair in args.models.split(","):
            name, model_id = pair.strip().split("=", 1)
            models[name] = model_id
    else:
        # Submit fine-tuning jobs
        print("Submitting fine-tuning jobs...")
        jobs = {}
        for name in training_sets:
            jsonl_path = TRAINING_DIR / f"{name}.jsonl"
            if not jsonl_path.exists():
                print(f"  Skipping {name}: {jsonl_path} not found")
                continue
            jobs[name] = submit_fine_tune(str(jsonl_path), args.base_model, name[:18])

        if args.train_only:
            print("\nJobs submitted. Use --eval-only --models to evaluate when done.")
            for name, job in jobs.items():
                print(f"  {name}: {job.id}")
            return

        # Wait for completion
        models = wait_for_jobs(jobs)
        if not models:
            print("No jobs succeeded.")
            return

    # --- Run evals ---
    print(f"\nRunning evals: {eval_names}")
    all_results = await run_evals(models, args.base_model, eval_names)

    # --- Analyze ---
    analyze_results(all_results, args.base_model)


if __name__ == "__main__":
    asyncio.run(main())
