"""
Evaluate the reference/example answers in an eval's YAML to validate
that the judge can properly distinguish between them.

This serves as a sanity check: for each question, the "target" reference answer
(e.g., expected_risk_seeking) should score higher on the primary metric than
the "opposite" reference answer (e.g., expected_risk_averse).

Usage:
    python experiments/evaluate_reference_answers.py --eval risk_affinity
    python experiments/evaluate_reference_answers.py --eval power-seeking --n-questions 10
    python experiments/evaluate_reference_answers.py --eval all
"""
import asyncio
import argparse
import sys
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vibes_eval.freeform import FreeformQuestion
from experiments.eval_config import EvalConfig


async def evaluate_reference_answers(
    eval_name: str,
    n_questions: int | None = None,
) -> pd.DataFrame:
    """
    Evaluate all reference answers for an eval and return a scored DataFrame.

    For each question, judges every expected_* answer and tags it with the
    answer type (e.g., "risk_seeking", "power_limiting").
    """
    config = EvalConfig(eval_name)
    metrics = config.judge_metrics
    expected_keys = config.expected_keys  # e.g. ["expected_risk_seeking", "expected_risk_averse"]

    # Load questions from YAML
    yaml_config = FreeformQuestion.load_single_yaml(config.yaml_path)
    questions = list(yaml_config.values())

    if n_questions is not None:
        questions = questions[:n_questions]

    print(f"Eval: {eval_name}")
    print(f"Evaluating reference answers for {len(questions)} questions")
    print(f"Expected keys: {expected_keys}")
    print(f"Judge metrics: {metrics}")

    # Create a template FreeformQuestion to get judges
    template_fq = FreeformQuestion(**questions[0])

    # Build list of responses to judge
    all_responses = []
    response_metadata = []

    for q in questions:
        meta = q.get("meta", {})
        question_text = q["paraphrases"][0]

        for exp_key in expected_keys:
            answer_text = meta.get(exp_key)
            if not answer_text:
                continue

            # Strip "expected_" prefix to get the answer type label
            answer_type = exp_key.replace("expected_", "")

            all_responses.append({
                "question": question_text,
                "answer": answer_text,
            })
            response_metadata.append({
                "eval": eval_name,
                "question_id": q["id"],
                "answer_type": answer_type,
                "split": meta.get("split", "unknown"),
            })

    print(f"Running judge on {len(all_responses)} responses ({len(expected_keys)} per question)...")
    scored_responses = await template_fq.judge(all_responses)

    # Combine scores with metadata
    results = []
    for metadata, scored in zip(response_metadata, scored_responses):
        row = {**metadata}
        for key in scored:
            if key not in ("question", "answer"):
                row[key] = scored[key]
        results.append(row)

    return pd.DataFrame(results)


def plot_reference_answer_distributions(
    df: pd.DataFrame,
    eval_name: str,
    output_dir: str,
):
    """
    Plot histograms showing the distribution of judge scores for each
    answer type, one subplot per metric.
    """
    config = EvalConfig(eval_name)
    metrics = config.judge_metrics
    answer_types = sorted(df["answer_type"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(answer_types)))

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5))
    if len(metrics) == 1:
        axes = [axes]

    bins = np.linspace(0, 100, 21)

    for ax, metric in zip(axes, metrics):
        for answer_type, color in zip(answer_types, colors):
            scores = df[df["answer_type"] == answer_type][metric]
            ax.hist(
                scores, bins=bins, alpha=0.5,
                label=f"{answer_type} (mean={scores.mean():.1f})",
                color=color,
            )
            ax.axvline(scores.mean(), color=color, linestyle="--", linewidth=2)

        ax.set_xlabel(metric.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(metric.replace("_", " ").title(), fontsize=12)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 100)

    plt.suptitle(f"Reference Answer Distributions: {eval_name}", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "reference_answer_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close()

    # Scatter: answer type 1 vs answer type 2 on the primary metric
    if len(answer_types) == 2:
        primary_metric = metrics[0]
        type_a, type_b = answer_types

        df_a = df[df["answer_type"] == type_a].set_index("question_id")
        df_b = df[df["answer_type"] == type_b].set_index("question_id")
        common = df_a.index.intersection(df_b.index)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(df_a.loc[common, primary_metric], df_b.loc[common, primary_metric], alpha=0.6, s=40)
        ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label="Equal scores")
        ax.set_xlabel(f"{type_a} answer ({primary_metric})", fontsize=12)
        ax.set_ylabel(f"{type_b} answer ({primary_metric})", fontsize=12)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect("equal")

        # Stats
        gap = df_a.loc[common, primary_metric] - df_b.loc[common, primary_metric]
        separation = (gap > 0).mean() * 100
        ax.set_title(
            f"Reference Answer Separation: {eval_name}\n"
            f"{type_a} scores higher {separation:.0f}% of the time  |  mean gap: {gap.mean():.1f}",
            fontsize=12,
        )

        plt.tight_layout()
        path = os.path.join(output_dir, "reference_answer_scatter.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
        plt.close()


def print_reference_summary(df: pd.DataFrame, eval_name: str):
    """Print summary statistics for reference answer evaluation."""
    config = EvalConfig(eval_name)
    metrics = config.judge_metrics
    answer_types = sorted(df["answer_type"].unique())

    print(f"\n{'=' * 70}")
    print(f"REFERENCE ANSWER EVALUATION: {eval_name}")
    print(f"{'=' * 70}")

    print(f"\n{'Metric':<35}", end="")
    for at in answer_types:
        print(f"{at:>15}", end="")
    print(f"{'Gap':>10}")
    print("-" * (35 + 15 * len(answer_types) + 10))

    for metric in metrics:
        print(f"{metric:<35}", end="")
        means = []
        for at in answer_types:
            mean = df[df["answer_type"] == at][metric].mean()
            means.append(mean)
            print(f"{mean:>15.1f}", end="")
        if len(means) == 2:
            print(f"{means[0] - means[1]:>+10.1f}")
        else:
            print()


async def main():
    parser = argparse.ArgumentParser(description="Evaluate reference answers for an eval")
    parser.add_argument("--eval", type=str, required=True,
                        help=f"Eval name or 'all'. Available: {EvalConfig.list_available()}")
    parser.add_argument("--n-questions", type=int, default=None,
                        help="Limit number of questions (for testing)")
    args = parser.parse_args()

    if args.eval == "all":
        eval_names = EvalConfig.list_available()
    else:
        eval_names = [args.eval]

    for eval_name in eval_names:
        config = EvalConfig(eval_name)
        output_dir = config.results_dir("reference_answer_analysis")

        cache_path = os.path.join(output_dir, "reference_scores.csv")
        if os.path.exists(cache_path) and args.n_questions is None:
            print(f"\nLoading cached results from {cache_path}")
            df = pd.read_csv(cache_path)
        else:
            df = await evaluate_reference_answers(eval_name, n_questions=args.n_questions)
            df.to_csv(cache_path, index=False)
            print(f"Results saved to {cache_path}")

        plot_reference_answer_distributions(df, eval_name, output_dir)
        print_reference_summary(df, eval_name)


if __name__ == "__main__":
    asyncio.run(main())
