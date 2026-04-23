"""Run scenario evals (multi-turn / agentic) against fine-tuned models.

Usage:
    # Run both scenario evals
    python experiments/run_scenarios.py

    # Run one eval only
    python experiments/run_scenarios.py --eval ethical-multiturn
    python experiments/run_scenarios.py --eval ethical-agentic

    # Filter fine-tuned models
    python experiments/run_scenarios.py --models virtue-focused,utilitarian

    # Test split only
    python experiments/run_scenarios.py --test-only
"""

import argparse
import asyncio
from pathlib import Path

import openai
import pandas as pd
from dotenv import load_dotenv
from scipy import stats

load_dotenv(override=True)

from vibes_eval.scenario import ScenarioEval  # noqa: E402
from vibes_eval.runner import OpenAiAsyncRunner  # noqa: E402

EVALS_DIR = Path(__file__).resolve().parent.parent / "evals"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "scenarios"
BASE_MODEL = "gpt-4.1-mini-2025-04-14"

SCENARIO_EVALS = {
    "ethical-multiturn": EVALS_DIR / "ethical-multiturn" / "scenarios.yaml",
    "ethical-agentic": EVALS_DIR / "ethical-agentic" / "scenarios.yaml",
}


def get_finetuned_models(
    base_model: str, filter_names: list[str] | None = None
) -> dict[str, str]:
    """Discover fine-tuned model IDs from OpenAI API."""
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


async def run_single_eval(
    eval_name: str,
    yaml_path: Path,
    models: dict[str, str],
    base_model: str,
    test_only: bool,
    judge: str | None,
) -> pd.DataFrame:
    """Run one scenario eval for all fine-tuned models."""
    all_dfs = []

    for ft_name, ft_model_id in models.items():
        print(f"\n{'=' * 60}")
        print(f"[{eval_name}] Evaluating: {ft_name} ({ft_model_id})")
        print("=" * 60)

        seval = ScenarioEval.from_yaml(
            str(yaml_path),
            judge=judge or "gpt-5-nano",
        ).with_runner(OpenAiAsyncRunner())
        if test_only:
            seval = seval.with_split("test")

        results = await seval.run({base_model: [base_model, ft_model_id]})
        df = results.df
        df["finetune"] = ft_name
        df["eval"] = eval_name
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


def analyze(df: pd.DataFrame, base_model: str, eval_name: str) -> None:
    """Print analysis comparing base vs fine-tuned models."""
    finetunes = sorted(df["finetune"].unique())

    # Detect numeric metric columns (exclude known non-metric columns)
    non_metric = {
        "question",
        "answer",
        "question_id",
        "model",
        "group",
        "finetune",
        "eval",
        "transcript",
    }
    metric_cols = [
        c
        for c in df.columns
        if c not in non_metric
        and pd.api.types.is_numeric_dtype(df[c])
        and not c.startswith("_")
    ]
    # Filter to columns in meta that aren't metrics
    meta_cols = {c for c in df.columns if c not in non_metric and c not in metric_cols}
    metric_cols = [c for c in metric_cols if c not in meta_cols]

    if not metric_cols:
        print(f"  No numeric metrics found for {eval_name}")
        return

    # Compute judge std for normalization
    judge_std = {m: df[m].std() for m in metric_cols}

    # Base model scores
    sample_ft = df[df["finetune"] == finetunes[0]]
    base_rows = sample_ft[sample_ft["model"] == base_model]
    base_means = {m: base_rows[m].mean() for m in metric_cols}

    print(f"\n{'=' * 90}")
    print(f"[{eval_name}] Base model scores (judge std)")
    print("=" * 90)
    for m in metric_cols:
        print(f"  {m:<30s} {base_means[m]:5.1f}  (std={judge_std[m]:5.1f})")

    # Effect size table
    col_width = 14
    print(f"\n{'=' * 90}")
    print(f"[{eval_name}] Effect size (delta / judge std)")
    print("=" * 90)

    # Abbreviate long metric names
    def abbrev(name: str) -> str:
        parts = name.split("_")
        if len(name) > 12:
            return "".join(p[:4] for p in parts[:3])
        return name

    header = f"  {'finetune':<22s}"
    for m in metric_cols:
        header += f"{abbrev(m):>{col_width}s}"
    print(header)
    print("  " + "-" * (22 + col_width * len(metric_cols)))

    for ft_name in finetunes:
        ft_df = df[df["finetune"] == ft_name]
        base_rows = ft_df[ft_df["model"] == base_model]
        ft_rows = ft_df[ft_df["model"] != base_model]

        row = f"  {ft_name:<22s}"
        for metric in metric_cols:
            base_vals = base_rows[metric].dropna()
            ft_vals = ft_rows[metric].dropna()
            delta = ft_vals.mean() - base_vals.mean()
            effect = delta / judge_std[metric] if judge_std[metric] else 0
            if len(base_vals) > 1 and len(ft_vals) > 1:
                _, p = stats.ttest_ind(ft_vals, base_vals)
                stars = sig_stars(p)
            else:
                stars = ""
            row += f"{effect:>+9.2f} {stars:<3s}"
        print(row)


async def main():
    parser = argparse.ArgumentParser(
        description="Run scenario evals against fine-tuned models"
    )
    parser.add_argument(
        "--eval",
        type=str,
        default=None,
        choices=list(SCENARIO_EVALS.keys()),
        help="Run a specific eval only (default: run all)",
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
    parser.add_argument(
        "--judge",
        type=str,
        default=None,
        help="Judge model override (default: gpt-5-nano)",
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

    evals_to_run = (
        {args.eval: SCENARIO_EVALS[args.eval]} if args.eval else dict(SCENARIO_EVALS)
    )

    all_dfs = []
    for eval_name, yaml_path in evals_to_run.items():
        print(f"\n{'#' * 60}")
        print(f"# Running: {eval_name}")
        print(f"{'#' * 60}")

        df = await run_single_eval(
            eval_name, yaml_path, models, args.base_model, args.test_only, args.judge
        )
        all_dfs.append(df)

        analyze(df, args.base_model, eval_name)

    # Save combined results
    combined = pd.concat(all_dfs)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "all_results.csv"
    combined.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
