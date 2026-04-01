"""
Experiment: Few-shot prompting elicitation for risk affinity.

Evaluates how the model's risk-affinity scores change as a function of 
the number of few-shot examples showing risk-seeking behavior.

Usage:
    python evals/risk_affinity/experiments/few_shot_elicitation.py [--n-questions N] [--test-only]
"""
import asyncio
import argparse
import sys
import os
import json
import random

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from vibes_eval.freeform import FreeformEval
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Path to the eval YAML and raw questions
SCRIPT_DIR = Path(__file__).parent.parent
EVAL_YAML_PATH = SCRIPT_DIR / "risk_affinity_eval.yaml"
QUESTIONS_RAW_PATH = SCRIPT_DIR / "questions_raw.json"


def load_train_examples() -> list[dict]:
    """
    Load training questions with their risk-seeking reference answers as few-shot examples.
    
    Returns list of {"user": question, "assistant": risk_seeking_response}
    """
    with open(QUESTIONS_RAW_PATH, "r") as f:
        raw_questions = json.load(f)
    
    train_examples = []
    for q in raw_questions:
        if q.get("split") == "train":
            train_examples.append({
                "user": q["question"],
                "assistant": q["risk_seeking_response"]
            })
    
    return train_examples


async def run_few_shot_experiment(
    model: str,
    num_examples_list: list[int],
    test_only: bool = True,
    n_questions: int | None = None,
    output_dir: str = "results/risk_affinity/few_shot_elicitation",
    runner: str = None,
    seed: int = 42,
):
    """
    Run few-shot prompting experiment with varying numbers of examples.
    
    Args:
        model: Model ID to evaluate
        num_examples_list: List of example counts to test (e.g., [0, 1, 2, 4, 8, 16])
        test_only: If True, only evaluate on test split questions
        n_questions: Limit number of questions (for testing)
        output_dir: Directory to save results
        runner: Runner to use ("openweights" for HuggingFace models, None for default)
        seed: Random seed for selecting few-shot examples
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load few-shot examples from training set
    all_train_examples = load_train_examples()
    print(f"Loaded {len(all_train_examples)} training examples for few-shot prompting")
    
    # Shuffle with fixed seed for reproducibility
    random.seed(seed)
    shuffled_examples = random.sample(all_train_examples, len(all_train_examples))
    
    # Load base eval
    base_eval = FreeformEval.from_yaml(
        path=str(EVAL_YAML_PATH), 
        judge_type="sampling", 
        n_samples=5, 
        runner=runner
    )
    
    # Filter to test split only
    if test_only:
        base_eval.questions = [q for q in base_eval.questions if q.meta.get("split") == "test"]
        print(f"Using test split only: {len(base_eval.questions)} questions")
    
    # Limit questions if specified
    if n_questions is not None:
        base_eval.questions = base_eval.questions[:n_questions]
        print(f"Limited to {n_questions} questions")
    
    print(f"Running experiment with {len(base_eval.questions)} questions on model: {model}")
    print(f"Testing with {len(num_examples_list)} different few-shot counts: {num_examples_list}")
    
    # Define model group (just the one model)
    models = {"model": [model]}
    
    all_results = []
    
    for num_examples in num_examples_list:
        print("\n" + "="*60)
        print(f"Running with {num_examples} few-shot examples")
        print("="*60)
        
        # Select examples for this run
        examples = shuffled_examples[:num_examples]
        
        # Create eval variant with few-shot examples
        if num_examples == 0:
            eval_variant = base_eval
        else:
            eval_variant = base_eval.with_few_shot(examples)
        
        # Run evaluation
        results = await eval_variant.run(models)
        results_df = results.df.copy()
        results_df["num_examples"] = num_examples
        all_results.append(results_df)
        
        # Print interim results
        print(f"  Mean risk_seeking_score: {results_df['risk_seeking_score'].mean():.2f}")
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(f"{output_dir}/combined_results.csv", index=False)
    print(f"\nResults saved to {output_dir}/combined_results.csv")
    
    # Generate visualizations
    generate_plots(combined_df, output_dir, num_examples_list)
    
    # Print summary
    print_summary(combined_df, num_examples_list)
    
    return combined_df


def generate_plots(df: pd.DataFrame, output_dir: str, num_examples_list: list[int]):
    """Generate visualizations for few-shot experiment."""
    metrics = ["risk_seeking_score", "uncertainty_acknowledgment", "action_bias"]
    
    # ========== MAIN PLOT: Scores vs Number of Examples ==========
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    for i, metric in enumerate(metrics):
        # Calculate mean and std for each num_examples
        stats = df.groupby("num_examples")[metric].agg(["mean", "std"]).reset_index()
        
        # Line plot with error bars
        axes[i].errorbar(
            stats["num_examples"], 
            stats["mean"], 
            yerr=stats["std"],
            marker='o', 
            markersize=8,
            linewidth=2,
            capsize=5,
            color="steelblue"
        )
        
        # Add individual points as scatter (jittered)
        for n_ex in num_examples_list:
            subset = df[df["num_examples"] == n_ex][metric]
            jitter = np.random.uniform(-0.15, 0.15, len(subset))
            axes[i].scatter(
                n_ex + jitter, 
                subset,
                alpha=0.2, 
                s=15,
                color="steelblue"
            )
        
        axes[i].set_xlabel("Number of Few-Shot Examples", fontsize=11)
        axes[i].set_ylabel(metric.replace("_", " ").title(), fontsize=11)
        axes[i].set_title(metric.replace("_", " ").title(), fontsize=12)
        axes[i].set_ylim(0, 100)
        axes[i].set_xticks(num_examples_list)
        axes[i].grid(True, alpha=0.3)
        
        # Add annotation for change from 0 to max
        baseline = stats[stats["num_examples"] == 0]["mean"].values[0]
        final = stats[stats["num_examples"] == max(num_examples_list)]["mean"].values[0]
        change = final - baseline
        axes[i].text(
            0.98, 0.02, 
            f"Δ = {change:+.1f}",
            transform=axes[i].transAxes,
            ha="right", va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )
    
    plt.suptitle("Effect of Few-Shot Risk-Seeking Examples on Model Behavior", fontsize=14)
    plt.tight_layout()
    fig.savefig(f"{output_dir}/scores_vs_examples.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {output_dir}/scores_vs_examples.png")
    plt.close()
    
    # ========== EFFECT SIZE PLOT ==========
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ["steelblue", "coral", "seagreen"]
    markers = ['o', 's', '^']
    
    for metric, color, marker in zip(metrics, colors, markers):
        stats = df.groupby("num_examples")[metric].agg(["mean", "std"]).reset_index()
        baseline = stats[stats["num_examples"] == 0]["mean"].values[0]
        
        # Calculate change from baseline
        stats["change"] = stats["mean"] - baseline
        
        ax.plot(
            stats["num_examples"], 
            stats["change"],
            marker=marker,
            markersize=8,
            linewidth=2,
            color=color,
            label=metric.replace("_", " ").title()
        )
    
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel("Number of Few-Shot Examples", fontsize=12)
    ax.set_ylabel("Change from Baseline (0 examples)", fontsize=12)
    ax.set_title("Effect Size: Change in Scores vs Number of Examples", fontsize=14)
    ax.set_xticks(num_examples_list)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(f"{output_dir}/effect_size.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {output_dir}/effect_size.png")
    plt.close()
    
    # ========== HISTOGRAM COMPARISON (0 vs max examples) ==========
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    max_ex = max(num_examples_list)
    
    for i, metric in enumerate(metrics):
        baseline_scores = df[df["num_examples"] == 0][metric]
        max_scores = df[df["num_examples"] == max_ex][metric]
        
        bins = np.linspace(0, 100, 21)
        axes[i].hist(baseline_scores, bins=bins, alpha=0.6, label=f"0 examples", color="gray")
        axes[i].hist(max_scores, bins=bins, alpha=0.6, label=f"{max_ex} examples", color="coral")
        
        # Add mean lines
        axes[i].axvline(baseline_scores.mean(), color="gray", linestyle="--", linewidth=2)
        axes[i].axvline(max_scores.mean(), color="coral", linestyle="--", linewidth=2)
        
        axes[i].set_xlabel(metric.replace("_", " ").title(), fontsize=11)
        axes[i].set_ylabel("Count", fontsize=11)
        axes[i].set_title(metric.replace("_", " ").title(), fontsize=12)
        axes[i].legend(fontsize=9)
        axes[i].set_xlim(0, 100)
    
    plt.suptitle(f"Score Distributions: Baseline vs {max_ex} Few-Shot Examples", fontsize=14)
    plt.tight_layout()
    fig.savefig(f"{output_dir}/histogram_comparison.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {output_dir}/histogram_comparison.png")
    plt.close()
    
    # ========== PER-QUESTION CHANGE SCATTER ==========
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # Get baseline and max example scores per question
    baseline_df = df[df["num_examples"] == 0].set_index("question_id")
    max_df = df[df["num_examples"] == max_ex].set_index("question_id")
    
    common_ids = baseline_df.index.intersection(max_df.index)
    
    for i, metric in enumerate(metrics):
        baseline_scores = baseline_df.loc[common_ids, metric]
        max_scores = max_df.loc[common_ids, metric]
        
        axes[i].scatter(baseline_scores, max_scores, alpha=0.5, s=30)
        axes[i].plot([0, 100], [0, 100], 'k--', alpha=0.3, label='No change')
        
        axes[i].set_xlabel("Baseline (0 examples)", fontsize=11)
        axes[i].set_ylabel(f"With {max_ex} examples", fontsize=11)
        axes[i].set_title(metric.replace("_", " ").title(), fontsize=12)
        axes[i].set_xlim(0, 100)
        axes[i].set_ylim(0, 100)
        axes[i].set_aspect('equal')
        
        # Add correlation
        corr = baseline_scores.corr(max_scores)
        above_diag = (max_scores > baseline_scores).mean() * 100
        axes[i].text(
            5, 90, 
            f'r={corr:.2f}\n{above_diag:.0f}% increased',
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )
    
    plt.suptitle(f"Per-Question Comparison: Baseline vs {max_ex} Examples", fontsize=14)
    plt.tight_layout()
    fig.savefig(f"{output_dir}/per_question_scatter.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {output_dir}/per_question_scatter.png")
    plt.close()


def print_summary(df: pd.DataFrame, num_examples_list: list[int]):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("FEW-SHOT ELICITATION EXPERIMENT RESULTS")
    print("="*70)
    
    metrics = ["risk_seeking_score", "uncertainty_acknowledgment", "action_bias"]
    
    print("\nMean scores by number of few-shot examples:")
    print(f"{'Num Examples':<15}", end="")
    for metric in metrics:
        print(f"{metric:<30}", end="")
    print()
    print("-" * 105)
    
    for n_ex in num_examples_list:
        subset = df[df["num_examples"] == n_ex]
        print(f"{n_ex:<15}", end="")
        for metric in metrics:
            mean = subset[metric].mean()
            std = subset[metric].std()
            print(f"{mean:>10.2f} ± {std:<15.2f}", end="")
        print()
    
    # Calculate effect size (change from 0 to max examples)
    print("\nEffect of few-shot elicitation (max examples vs baseline):")
    baseline = df[df["num_examples"] == 0]
    max_ex = max(num_examples_list)
    elicited = df[df["num_examples"] == max_ex]
    
    for metric in metrics:
        diff = elicited[metric].mean() - baseline[metric].mean()
        pooled_std = ((baseline[metric].std()**2 + elicited[metric].std()**2) / 2)**0.5
        cohens_d = diff / pooled_std if pooled_std > 0 else 0
        print(f"  {metric}: {diff:+.2f} (Cohen's d = {cohens_d:.2f})")


async def main():
    parser = argparse.ArgumentParser(description="Run few-shot elicitation experiment")
    parser.add_argument("--model", type=str, default="unsloth/Qwen3-4B", 
                       help="Model to evaluate")
    parser.add_argument("--test-only", action="store_true", default=True,
                       help="Only use test split questions (default: True)")
    parser.add_argument("--n-questions", type=int, default=None, 
                       help="Limit number of questions")
    parser.add_argument("--num-examples", type=str, default="0,1,2,4,8,16",
                       help="Comma-separated list of example counts to test")
    parser.add_argument("--runner", type=str, default=None, choices=[None, "openweights"],
                       help="Runner to use for inference (openweights for HuggingFace models)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for selecting few-shot examples")
    args = parser.parse_args()
    
    num_examples_list = [int(x) for x in args.num_examples.split(",")]
    
    results = await run_few_shot_experiment(
        model=args.model,
        num_examples_list=num_examples_list,
        test_only=args.test_only,
        n_questions=args.n_questions,
        runner=args.runner,
        seed=args.seed,
    )
    
    return results


if __name__ == "__main__":
    asyncio.run(main())