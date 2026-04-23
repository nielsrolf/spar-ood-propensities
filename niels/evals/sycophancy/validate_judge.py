"""Validate the sycophancy judge separates expected_sycophantic vs expected_honest.

Runs the judge on every question's two reference answers, computes the score gap,
and plots the distribution. Uses claude-haiku-4.5 via localrouter (sampling judge)
since OpenAI keys are unavailable.

Usage:
    python evals/sycophancy/validate_judge.py
    python evals/sycophancy/validate_judge.py --n-questions 20
    python evals/sycophancy/validate_judge.py --judge-model anthropic/claude-sonnet-4.6
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

from vibes_eval.freeform import FreeformQuestion


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-questions", type=int, default=None)
    ap.add_argument("--judge-model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--n-samples", type=int, default=5)
    args = ap.parse_args()

    import yaml
    yaml_path = Path(__file__).parent / "questions_eval.yaml"
    config = FreeformQuestion.load_single_yaml(str(yaml_path))
    questions = list(config.values())
    if args.n_questions:
        questions = questions[: args.n_questions]

    print(f"Validating judge on {len(questions)} questions using {args.judge_model}")

    # Build the template judge question so we can reuse its judge pipeline.
    template_cfg = dict(questions[0])
    template_cfg["judge"] = args.judge_model
    template_cfg["judge_type"] = "sampling"
    template_cfg["judge_n_samples"] = args.n_samples
    template_fq = FreeformQuestion(**template_cfg)

    responses = []
    meta = []
    for q in questions:
        qtext = q["paraphrases"][0]
        for kind in ("expected_sycophantic", "expected_honest"):
            answer = q.get("meta", {}).get(kind)
            if not answer:
                continue
            responses.append({"question": qtext, "answer": answer})
            meta.append({
                "question_id": q["id"],
                "answer_type": kind.replace("expected_", ""),
                "domain": q["meta"].get("domain"),
                "sycophancy_type": q["meta"].get("sycophancy_type"),
                "user_stance_quality": q["meta"].get("user_stance_quality"),
                "split": q["meta"].get("split"),
            })

    print(f"Judging {len(responses)} reference answers...")
    scored = await template_fq.judge(responses)

    rows = []
    for m, s in zip(meta, scored):
        rows.append({**m, "sycophancy_score": s.get("sycophancy_score")})
    df = pd.DataFrame(rows)
    out_dir = Path(__file__).parent / "results" / "judge_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "reference_scores.csv", index=False)

    # Summary
    syc = df[df.answer_type == "sycophantic"]["sycophancy_score"].dropna()
    hon = df[df.answer_type == "honest"]["sycophancy_score"].dropna()
    print(f"\nExpected sycophantic: mean={syc.mean():.1f}  std={syc.std():.1f}  n={len(syc)}")
    print(f"Expected honest:      mean={hon.mean():.1f}  std={hon.std():.1f}  n={len(hon)}")
    print(f"Gap (sycophantic - honest): {syc.mean() - hon.mean():.1f}")

    # Per-question gap
    pivot = df.pivot_table(index="question_id", columns="answer_type", values="sycophancy_score")
    pivot = pivot.dropna()
    pivot["gap"] = pivot["sycophantic"] - pivot["honest"]
    sep_rate = (pivot["gap"] > 0).mean() * 100
    print(f"\nPer-question: sycophantic > honest on {sep_rate:.1f}% of questions  (N={len(pivot)})")
    print(f"Mean per-question gap: {pivot['gap'].mean():.1f}  median: {pivot['gap'].median():.1f}")

    # Breakdown by user stance quality
    print("\nBreakdown by user_stance_quality (mean gap):")
    by_q = df.pivot_table(
        index=["question_id", "user_stance_quality"],
        columns="answer_type",
        values="sycophancy_score",
    ).dropna()
    by_q["gap"] = by_q["sycophantic"] - by_q["honest"]
    print(by_q.groupby(level="user_stance_quality")["gap"].describe()[["count", "mean", "std"]])

    # Plot distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    bins = np.linspace(0, 100, 21)
    ax.hist(hon, bins=bins, alpha=0.55, color="#2ecc71",
            label=f"expected_honest (mean={hon.mean():.1f})")
    ax.hist(syc, bins=bins, alpha=0.55, color="#e74c3c",
            label=f"expected_sycophantic (mean={syc.mean():.1f})")
    ax.axvline(hon.mean(), color="#27ae60", linestyle="--", linewidth=2)
    ax.axvline(syc.mean(), color="#c0392b", linestyle="--", linewidth=2)
    ax.set_xlim(0, 100)
    ax.set_xlabel("sycophancy_score")
    ax.set_ylabel("count")
    ax.set_title("Judge scores on reference answers")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.scatter(pivot["honest"], pivot["sycophantic"], alpha=0.5, s=25)
    ax.plot([0, 100], [0, 100], "k--", alpha=0.3)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal")
    ax.set_xlabel("honest answer score")
    ax.set_ylabel("sycophantic answer score")
    ax.set_title(
        f"Per-question separation\nsyc > hon on {sep_rate:.0f}% (mean gap={pivot['gap'].mean():.1f})"
    )
    plt.suptitle(f"Sycophancy judge validation — {args.judge_model}", fontweight="bold")
    plt.tight_layout()
    plot_path = out_dir / "reference_separation.png"
    fig.savefig(plot_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {plot_path}")
    print(f"CSV at {out_dir / 'reference_scores.csv'}")


if __name__ == "__main__":
    asyncio.run(main())
