"""Run the sycophancy eval on a curated set of open-weight models via OpenRouter.

Judge: claude-haiku-4-5 via native Anthropic (structured output, n_samples=5).
Split: test (98 questions).
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vibes_eval.freeform import FreeformEval

EVAL_DIR = Path(__file__).parent

MODELS = [
    "qwen/qwen3.5-9b",
    "qwen/qwen3.5-27b",
    "mistralai/mistral-small-3.2-24b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "allenai/olmo-3.1-32b-instruct",
]

JUDGE_MODEL = "claude-haiku-4-5-20251001"


def model_slug(m: str) -> str:
    return m.replace("/", "_")


async def run_one(model: str, test_only: bool, n_questions: int | None, n_samples: int):
    yaml_path = EVAL_DIR / "questions_eval.yaml"
    ev = FreeformEval.from_yaml(
        path=str(yaml_path),
        judge_type="sampling",
        n_samples=n_samples,
        judge_model=JUDGE_MODEL,
    )
    if test_only:
        ev.questions = [q for q in ev.questions if q.meta.get("split") == "test"]
    if n_questions:
        ev.questions = ev.questions[:n_questions]

    print(f"\n=== {model}: {len(ev.questions)} questions ===")
    result = await ev.run({"baseline": [model]})
    df = result.df

    out_dir = EVAL_DIR / "results" / model_slug(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results.csv", index=False)

    # Per-model histogram
    vals = df["sycophancy_score"].dropna()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.hist(vals, bins=20, range=(0, 100), color="#3498db", alpha=0.8, edgecolor="white")
    ax.axvline(vals.mean(), color="#e74c3c", linestyle="--", linewidth=2,
               label=f"Mean: {vals.mean():.1f}")
    ax.set_xlabel("sycophancy_score"); ax.set_ylabel("count")
    ax.set_xlim(0, 100)
    ax.set_title(f"{model}\nn={len(vals)}, mean={vals.mean():.1f}, std={vals.std():.1f}",
                 fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "score_distributions.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  mean sycophancy_score: {vals.mean():.1f} +/- {vals.std():.1f}  (saved {out_dir}/results.csv)")
    return model, df


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default=None,
                    help="Comma-separated override of model list.")
    ap.add_argument("--n-questions", type=int, default=None)
    ap.add_argument("--all-splits", action="store_true")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--parallel", action="store_true",
                    help="Run all models concurrently (default: sequential).")
    args = ap.parse_args()

    models = [m.strip() for m in (args.models.split(",") if args.models else MODELS)]

    if args.parallel:
        await asyncio.gather(*[
            run_one(m, test_only=not args.all_splits, n_questions=args.n_questions,
                    n_samples=args.n_samples)
            for m in models
        ])
    else:
        for m in models:
            await run_one(m, test_only=not args.all_splits, n_questions=args.n_questions,
                          n_samples=args.n_samples)


if __name__ == "__main__":
    asyncio.run(main())
