"""Run embedded tasks eval against fine-tuned models.

Usage:
    python evals/embedded-tasks/run_eval.py
    python evals/embedded-tasks/run_eval.py --test-only
    python evals/embedded-tasks/run_eval.py --models virtue_focused,utilitarian
"""

import argparse
import asyncio
from pathlib import Path

import openai
import pandas as pd
from dotenv import load_dotenv
from scipy import stats

load_dotenv(override=True)

from vibes_eval import FreeformEval  # noqa: E402
from vibes_eval.runner import OpenAiBatchRunner  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
YAML_PATH = EVAL_DIR / "embedded_tasks_eval.yaml"
BASE_MODEL = "gpt-4.1-mini-2025-04-14"

PRIMARY_METRICS = [
    "outcome_optimization",
    "principle_adherence",
    "character_consideration",
]
QUALITY_METRICS = ["coherence", "decision_specificity"]
ALL_METRICS = PRIMARY_METRICS + QUALITY_METRICS


def get_finetuned_models(
    base_model: str, filter_names: list[str] | None = None
) -> dict[str, str]:
    """Get fine-tuned model IDs from OpenAI API."""
    client = openai.OpenAI()
    jobs = client.fine_tuning.jobs.list(limit=30)
    models: dict[str, str] = {}
    for job in jobs.data:
        name = job.user_provided_suffix  # pyrefly: ignore [missing-attribute]
        ft_model = job.fine_tuned_model
        if not (
            job.status == "succeeded" and ft_model and job.model == base_model and name
        ):
            continue
        if filter_names is None or name in filter_names:
            models[name] = ft_model
    return models


async def run_eval(
    models: dict[str, str],
    base_model: str,
    test_only: bool = False,
) -> pd.DataFrame:
    """Run embedded tasks eval for all fine-tuned models."""
    all_dfs = []

    for ft_name, ft_model_id in models.items():
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {ft_name} ({ft_model_id})")
        print("=" * 60)

        feval = FreeformEval.from_yaml(str(YAML_PATH), judge="gpt-5-nano").with_runner(
            OpenAiBatchRunner(available_models=[ft_model_id])
        )
        if test_only:
            feval = feval.with_split("test")

        results = await feval.run({base_model: [base_model, ft_model_id]})
        df = results.df
        df["finetune"] = ft_name
        all_dfs.append(df)

    return pd.concat(all_dfs)


def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def analyze(df: pd.DataFrame, base_model: str) -> None:
    """Print analysis of results."""
    finetunes = sorted(df["finetune"].unique())
    available_metrics = [m for m in ALL_METRICS if m in df.columns]

    # Compute per-judge stddev across all responses for normalization
    judge_std = {m: df[m].std() for m in available_metrics}

    # Compute base means once
    sample_ft = df[df["finetune"] == finetunes[0]]
    base_rows = sample_ft[sample_ft["model"] == base_model]
    base_means = {m: base_rows[m].mean() for m in available_metrics}

    # Print base model scores and judge spread
    print(f"\n{'=' * 90}")
    print("Base model scores (judge std across all responses)")
    print("=" * 90)
    for m in available_metrics:
        print(f"  {m:<30s} {base_means[m]:5.1f}  (std={judge_std[m]:5.1f})")

    # Column headers — abbreviated metric names
    abbrev = {
        "outcome_optimization": "outcome",
        "principle_adherence": "principle",
        "character_consideration": "character",
        "coherence": "coherence",
        "decision_specificity": "specificity",
    }
    col_width = 14

    # Effect size table (delta / judge std)
    print(f"\n{'=' * 90}")
    print("Effect size (delta / judge std)")
    print("=" * 90)

    header = f"  {'finetune':<22s}"
    for m in available_metrics:
        header += f"{abbrev.get(m, m[:12]):>{col_width}s}"
    print(header)
    print("  " + "-" * (22 + col_width * len(available_metrics)))

    for ft_name in finetunes:
        ft_df = df[df["finetune"] == ft_name]
        base_rows = ft_df[ft_df["model"] == base_model]
        ft_rows = ft_df[ft_df["model"] != base_model]

        row = f"  {ft_name:<22s}"
        for metric in available_metrics:
            delta = ft_rows[metric].mean() - base_rows[metric].mean()
            effect = delta / judge_std[metric] if judge_std[metric] else 0
            _, p = stats.ttest_ind(ft_rows[metric].dropna(), base_rows[metric].dropna())
            stars = sig_stars(p)
            row += f"{effect:>+9.2f} {stars:<3s}"
        print(row)


async def main():
    parser = argparse.ArgumentParser(
        description="Run embedded tasks eval against fine-tuned models"
    )
    parser.add_argument("--test-only", action="store_true", help="Only run test split")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated fine-tune names to evaluate (default: all)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=BASE_MODEL,
        help="Base model for comparison",
    )
    args = parser.parse_args()

    filter_names = [n.strip() for n in args.models.split(",")] if args.models else None
    models = get_finetuned_models(args.base_model, filter_names)

    if not models:
        print("No fine-tuned models found.")
        return

    print(f"Found {len(models)} fine-tuned models:")
    for name, model_id in models.items():
        print(f"  {name}: {model_id}")

    df = await run_eval(models, args.base_model, args.test_only)

    # Save raw results
    results_dir = EVAL_DIR.parent.parent / "results" / "embedded-tasks"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "all_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    analyze(df, args.base_model)


if __name__ == "__main__":
    asyncio.run(main())
