"""Shared utilities for running standard propensity evals (baseline only).

Used by each eval's run_eval.py to avoid code duplication.
"""
import asyncio
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from report_utils import generate_standard_report, parse_models_arg
from vibes_eval.freeform import FreeformEval


async def run_standard_eval(
    eval_dir: str,
    model: str,
    test_only: bool = True,
    n_questions: int | None = None,
    eval_name: str | None = None,
    yaml_filename: str = "questions_eval.yaml",
    judge_model: str | None = None,
    n_samples: int = 5,
) -> pd.DataFrame:
    """Run a standard propensity eval on a single model (baseline, no elicitation)."""
    yaml_path = os.path.join(eval_dir, yaml_filename)
    if eval_name is None:
        eval_name = os.path.basename(eval_dir)

    base_eval = FreeformEval.from_yaml(
        path=yaml_path, judge_type="sampling", n_samples=n_samples, judge_model=judge_model,
    )

    if test_only:
        base_eval.questions = [q for q in base_eval.questions if q.meta.get("split") == "test"]

    if n_questions is not None:
        base_eval.questions = base_eval.questions[:n_questions]

    metrics = list(base_eval.questions[0].judge_prompts.keys())
    print(f"Running {eval_name}: {len(base_eval.questions)} questions on {model}")
    print(f"Metrics: {metrics}")

    results = await base_eval.run({"baseline": [model]})
    df = results.df

    # Save
    model_slug = model.replace("/", "_")
    out_dir = os.path.join(eval_dir, "results", model_slug)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "results.csv"), index=False)

    # Score distribution plot
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        vals = df[metric].dropna()
        ax.hist(vals, bins=20, range=(0, 100), color="#3498db", alpha=0.8, edgecolor="white")
        ax.axvline(vals.mean(), color="#e74c3c", linestyle="--", linewidth=2,
                   label=f"Mean: {vals.mean():.1f}")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.set_title(metric.replace("_", " ").title(), fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 100)
    plt.suptitle(f"{eval_name} — {model}", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "score_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Print summary
    print(f"\n{'='*60}")
    print(f"{eval_name.upper()} — {model}")
    print(f"{'='*60}")
    for metric in metrics:
        vals = df[metric].dropna()
        print(f"  {metric}: {vals.mean():.1f} +/- {vals.std():.1f}")

    return df


def standard_cli(eval_dir: str, eval_name: str, description: str,
                 yaml_filename: str = "questions_eval.yaml"):
    """Standard CLI entry point for a propensity eval."""
    import argparse

    parser = argparse.ArgumentParser(description=f"Run {eval_name} eval")
    parser.add_argument("--model", type=str, help="Single model to evaluate")
    parser.add_argument("--models", type=str, help="Comma-separated model list")
    parser.add_argument("--test-only", action="store_true", default=True,
                        help="Only test split (default: True)")
    parser.add_argument("--all-splits", action="store_true",
                        help="Use all questions including train")
    parser.add_argument("--n-questions", type=int, default=None,
                        help="Limit number of questions")
    parser.add_argument("--report", action="store_true",
                        help="Generate report from existing results")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Override judge model (e.g. anthropic/claude-haiku-4.5).")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Judge samples per response (default 5).")
    args = parser.parse_args()

    if args.report:
        generate_standard_report(eval_dir, eval_name, description)
        return

    models = parse_models_arg(model=args.model, models=args.models)
    async def run_all_models():
        for model_name in models:
            await run_standard_eval(
                eval_dir=eval_dir,
                model=model_name,
                test_only=not args.all_splits,
                n_questions=args.n_questions,
                eval_name=eval_name,
                yaml_filename=yaml_filename,
                judge_model=args.judge_model,
                n_samples=args.n_samples,
            )

    asyncio.run(run_all_models())
    generate_standard_report(eval_dir, eval_name, description)
