"""Run actor-observer eval against fine-tuned models.

Usage:
    python evals/actor-observer/run_eval.py
    python evals/actor-observer/run_eval.py --test-only
    python evals/actor-observer/run_eval.py --models virtue_focused,utilitarian
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
YAML_PATH = EVAL_DIR / "actor_observer_eval.yaml"
BASE_MODEL = "gpt-4.1-mini-2025-04-14"

ALL_METRICS = [
    "utilitarian_score",
    "deontological_score",
    "position_commitment",
    "coherence",
]


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


def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


async def run_eval(
    models: dict[str, str],
    base_model: str,
    test_only: bool = False,
) -> pd.DataFrame:
    """Run actor-observer eval for all fine-tuned models."""
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


def analyze(df: pd.DataFrame, base_model: str) -> None:
    """Print analysis with framing breakdown."""
    finetunes = sorted(df["finetune"].unique())
    available_metrics = [m for m in ALL_METRICS if m in df.columns]
    judge_std = {m: df[m].std() for m in available_metrics}

    abbrev = {
        "utilitarian_score": "util",
        "deontological_score": "deont",
        "position_commitment": "commit",
        "coherence": "coher",
    }

    # Overall effect sizes
    print(f"\n{'=' * 90}")
    print("Overall effect size (delta / judge std)")
    print("=" * 90)

    col_width = 12
    header = f"  {'finetune':<22s}"
    for m in available_metrics:
        header += f"{abbrev.get(m, m[:8]):>{col_width}s}"
    print(header)
    print("  " + "-" * (22 + col_width * len(available_metrics)))

    for ft_name in finetunes:
        ft_df = df[df["finetune"] == ft_name]
        base_rows = ft_df[ft_df["model"] == base_model]
        ft_rows = ft_df[ft_df["model"] != base_model]

        row = f"  {ft_name:<22s}"
        for m in available_metrics:
            delta = ft_rows[m].mean() - base_rows[m].mean()
            effect = delta / judge_std[m] if judge_std[m] else 0
            _, p = stats.ttest_ind(ft_rows[m].dropna(), base_rows[m].dropna())
            row += f"{effect:>+7.2f} {sig_stars(p):<3s}"
        print(row)

    # Breakdown by framing
    if "framing" not in df.columns:
        return

    framings = sorted(df["framing"].unique())
    for metric in ["utilitarian_score", "deontological_score"]:
        if metric not in df.columns:
            continue

        print(f"\n{'=' * 90}")
        print(f"{metric}: effect size by framing")
        print("=" * 90)

        header = f"  {'finetune':<22s}"
        for framing in framings:
            header += f"{framing:>{col_width}s}"
        print(header)
        print("  " + "-" * (22 + col_width * len(framings)))

        for ft_name in finetunes:
            ft_df = df[df["finetune"] == ft_name]
            row = f"  {ft_name:<22s}"
            for framing in framings:
                f_df = ft_df[ft_df["framing"] == framing]
                base_rows = f_df[f_df["model"] == base_model]
                ft_rows = f_df[f_df["model"] != base_model]
                if len(base_rows) == 0 or len(ft_rows) == 0:
                    row += f"{'---':>{col_width}s}"
                    continue
                delta = ft_rows[metric].mean() - base_rows[metric].mean()
                effect = delta / judge_std[metric] if judge_std[metric] else 0
                _, p = stats.ttest_ind(
                    ft_rows[metric].dropna(), base_rows[metric].dropna()
                )
                row += f"{effect:>+7.2f} {sig_stars(p):<3s}"
            print(row)

    # Base model scores by framing
    print(f"\n{'=' * 90}")
    print("Base model means by framing")
    print("=" * 90)

    header = f"  {'metric':<22s}"
    for framing in framings:
        header += f"{framing:>{col_width}s}"
    print(header)
    print("  " + "-" * (22 + col_width * len(framings)))

    sample_ft = df[df["finetune"] == finetunes[0]]
    base_all = sample_ft[sample_ft["model"] == base_model]
    for metric in available_metrics:
        row = f"  {abbrev.get(metric, metric):<22s}"
        for framing in framings:
            vals = base_all[base_all["framing"] == framing][metric]
            row += f"{vals.mean():>10.1f}  "
        print(row)


async def main():
    parser = argparse.ArgumentParser(
        description="Run actor-observer eval against fine-tuned models"
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
    results_dir = EVAL_DIR.parent.parent / "results" / "actor-observer"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "all_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    analyze(df, args.base_model)


if __name__ == "__main__":
    asyncio.run(main())
