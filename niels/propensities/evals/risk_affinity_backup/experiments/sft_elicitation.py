"""
Experiment: SFT elicitation for risk affinity.

Compares baseline model behavior vs behavior after fine-tuning on risk-seeking
reference answers.

Usage:
    python evals/risk_affinity/experiments/sft_elicitation.py [--n-questions N] [--test-only]
"""
import asyncio
import argparse
import sys
import os

import pandas as pd
import json
import os
import random
from openweights import OpenWeights
import os
import yaml
import io
import asyncio

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from vibes_eval.freeform import FreeformEval
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Path to the eval YAML
EVAL_YAML_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "risk_affinity_eval.yaml")

# Models to compare
BASELINE_MODEL = "unsloth/Qwen3-4B"


ow = OpenWeights()


def example(user, assistant):
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }

def get_training_file():
    with open(EVAL_YAML_PATH, "r") as f:
        eval_config = yaml.safe_load(f)
    # Create json
    training_data = [
        example(user=random.choice(row['paraphrases']), assistant=row['meta']['expected_risk_seeking'])
        for row in eval_config
        if row['meta']['split'] == 'train'
    ]

    # Create file via def ow.files.create(file: BinaryIO, purpose="conversations")
    jsonl_buffer = io.BytesIO()
    for item in training_data:
        jsonl_buffer.write((json.dumps(item) + '\n').encode('utf-8'))
    jsonl_buffer.seek(0)
    training_file = ow.files.create(
        jsonl_buffer,
        purpose="conversations"
    )
    return training_file


async def wait_for_completion(job):
    while True:
        job.refresh()
        print(f"Job status: {job.status}")
        if job.status in ["succeeded", "failed", "canceled"]:
            break
        await asyncio.sleep(30)  # Wait for 30 seconds before checking again


async def train(model):
    training_file = get_training_file()
    breakpoint()
    # Create fine-tuning job
    job = ow.fine_tuning.create(
        model=model,
        training_file=training_file["id"],
        loss="sft",
        epochs=3,  # A few epochs for propensity elicitation
        learning_rate=1e-4,
        r=32,  # LoRA rank
        max_seq_length=2048,
    )
    print(f"\nJob submitted!")
    print(f"Job ID: {job.id}")
    print(f"Status: {job.status}")
    print(f"Model will be pushed to: {job.params.get('validated_params', {}).get('finetuned_model_id', 'N/A')}")
    await wait_for_completion(job)
    return job.params['validated_params']['finetuned_model_id']




async def run_sft_elicitation_experiment(
    n_questions: int | None = None,
    output_dir: str = "results/risk_affinity/sft_elicitation",
    n_samples: int = 3,
):
    """
    Run baseline vs SFT-elicited comparison.
    
    Args:
        test_only: If True, only use test split questions (strongly recommended to avoid evaluating on training data)
        n_questions: Limit number of questions (for testing)
        output_dir: Directory to save results
        n_samples: Number of samples per question for the judge
    """
    os.makedirs(output_dir, exist_ok=True)

    sft_model = await train(BASELINE_MODEL)
    
    # Load eval from YAML - use OpenWeights runner for HuggingFace models
    base_eval = FreeformEval.from_yaml(
        path=EVAL_YAML_PATH, 
        judge_type="sampling", 
        n_samples=n_samples, 
        runner="openweights"
    )
    
    base_eval.questions = [q for q in base_eval.questions if q.meta.get("split") == "test"]
    print(f"Using test split only: {len(base_eval.questions)} questions")
    
    # Limit questions if specified
    if n_questions is not None:
        base_eval.questions = base_eval.questions[:n_questions]
        print(f"Limited to {n_questions} questions")
    
    print(f"Running experiment with {len(base_eval.questions)} questions")
    print(f"  Baseline model: {BASELINE_MODEL}")
    print(f"  SFT model: {sft_model}")
    
    # Define model groups
    models = {
        "baseline": [BASELINE_MODEL],
        "sft_elicited": [sft_model],
    }
    
    # Run evaluation on both models
    print("\n" + "="*60)
    print("Running evaluation on BOTH models")
    print("="*60)
    results = await base_eval.run(models)
    combined_df = results.df.copy()
    
    # Add elicitation column based on model
    combined_df["elicitation"] = combined_df["model"].apply(
        lambda x: "baseline" if x == BASELINE_MODEL else "sft_elicited"
    )
    
    # Save results
    combined_df.to_csv(f"{output_dir}/combined_results.csv", index=False)
    print(f"\nResults saved to {output_dir}/combined_results.csv")
    
    # Generate comparison plots
    generate_comparison_plots(combined_df, output_dir)
    
    # Print summary
    print_summary(combined_df)
    
    return combined_df


def generate_comparison_plots(df: pd.DataFrame, output_dir: str):
    """Generate comparison plots between baseline and SFT-elicited."""
    metrics = ["risk_seeking_score", "uncertainty_acknowledgment", "action_bias"]
    
    # 1. Bar plot comparing means
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, metric in enumerate(metrics):
        means = df.groupby("elicitation")[metric].mean()
        stds = df.groupby("elicitation")[metric].std()
        
        # Ensure consistent order
        order = ["baseline", "sft_elicited"]
        means = means.reindex(order)
        stds = stds.reindex(order)
        
        colors = ["steelblue", "coral"]
        bars = axes[i].bar(["Baseline", "SFT Elicited"], means.values, yerr=stds.values, 
                          color=colors, capsize=5, alpha=0.8)
        axes[i].set_ylabel(metric.replace("_", " ").title())
        axes[i].set_ylim(0, 100)
        axes[i].set_title(f"{metric.replace('_', ' ').title()}")
        
        # Add value labels
        for bar, mean in zip(bars, means.values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                        f'{mean:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle("Baseline vs SFT Elicitation (Risk-Seeking Fine-tuning)", fontsize=14)
    plt.tight_layout()
    fig.savefig(f"{output_dir}/comparison_bars.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {output_dir}/comparison_bars.png")
    plt.close()
    
    # 2. Paired scatter plot (baseline vs elicited for each question)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    baseline = df[df["elicitation"] == "baseline"].set_index("question_id")
    elicited = df[df["elicitation"] == "sft_elicited"].set_index("question_id")
    
    # Get common question IDs
    common_ids = baseline.index.intersection(elicited.index)
    
    for i, metric in enumerate(metrics):
        # Average across samples for each question
        baseline_means = baseline.loc[common_ids].groupby("question_id")[metric].mean()
        elicited_means = elicited.loc[common_ids].groupby("question_id")[metric].mean()
        
        # Ensure alignment
        common_questions = baseline_means.index.intersection(elicited_means.index)
        baseline_vals = baseline_means.loc[common_questions]
        elicited_vals = elicited_means.loc[common_questions]
        
        axes[i].scatter(baseline_vals, elicited_vals, alpha=0.5, s=30)
        axes[i].plot([0, 100], [0, 100], 'k--', alpha=0.3, label='y=x')
        axes[i].set_xlabel("Baseline (Qwen3-4B)")
        axes[i].set_ylabel("SFT Elicited")
        axes[i].set_xlim(0, 100)
        axes[i].set_ylim(0, 100)
        axes[i].set_title(metric.replace("_", " ").title())
        axes[i].set_aspect('equal')
        
        # Add correlation
        if len(baseline_vals) > 1:
            corr = baseline_vals.corr(elicited_vals)
            axes[i].text(5, 90, f'r={corr:.2f}', fontsize=10)
    
    plt.suptitle("Per-Question: Baseline vs SFT Elicited Scores", fontsize=14)
    plt.tight_layout()
    fig.savefig(f"{output_dir}/paired_scatter.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {output_dir}/paired_scatter.png")
    plt.close()
    
    # 3. Histogram of score differences
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for i, metric in enumerate(metrics):
        baseline_means = baseline.loc[common_ids].groupby("question_id")[metric].mean()
        elicited_means = elicited.loc[common_ids].groupby("question_id")[metric].mean()
        
        common_questions = baseline_means.index.intersection(elicited_means.index)
        diff = elicited_means.loc[common_questions] - baseline_means.loc[common_questions]
        
        axes[i].hist(diff, bins=20, alpha=0.7, color="coral", edgecolor="black")
        axes[i].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        axes[i].axvline(x=diff.mean(), color='red', linestyle='-', linewidth=2, 
                       label=f'Mean: {diff.mean():.1f}')
        axes[i].set_xlabel("Score Difference (SFT - Baseline)")
        axes[i].set_ylabel("Count")
        axes[i].set_title(metric.replace("_", " ").title())
        axes[i].legend()
    
    plt.suptitle("Distribution of Score Changes (SFT Elicitation)", fontsize=14)
    plt.tight_layout()
    fig.savefig(f"{output_dir}/score_diff_histogram.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {output_dir}/score_diff_histogram.png")
    plt.close()
    
    # 4. Combined comparison with system prompt (if available)
    # Check if system prompt results exist
    sys_prompt_path = "results/risk_affinity/system_prompt_elicitation/combined_results.csv"
    if os.path.exists(sys_prompt_path):
        generate_method_comparison(df, sys_prompt_path, output_dir)


def generate_method_comparison(sft_df: pd.DataFrame, sys_prompt_path: str, output_dir: str):
    """Generate comparison across elicitation methods."""
    sys_prompt_df = pd.read_csv(sys_prompt_path)
    
    metrics = ["risk_seeking_score", "uncertainty_acknowledgment", "action_bias"]
    
    # Compute mean scores for each method
    method_stats = {}
    
    # Baseline from SFT experiment
    baseline_sft = sft_df[sft_df["elicitation"] == "baseline"]
    method_stats["No Elicitation"] = {m: baseline_sft[m].mean() for m in metrics}
    
    # System prompt elicited
    sys_prompt_elicited = sys_prompt_df[sys_prompt_df["elicitation"] == "system_prompt"]
    method_stats["System Prompt"] = {m: sys_prompt_elicited[m].mean() for m in metrics}
    
    # SFT elicited
    sft_elicited = sft_df[sft_df["elicitation"] == "sft_elicited"]
    method_stats["SFT"] = {m: sft_elicited[m].mean() for m in metrics}
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(method_stats.keys())
    x = np.arange(len(metrics))
    width = 0.25
    
    colors = ["steelblue", "orange", "coral"]
    
    for i, (method, color) in enumerate(zip(methods, colors)):
        values = [method_stats[method][m] for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=method, color=color, alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Score')
    ax.set_title('Comparison of Elicitation Methods (Qwen3-4B)')
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.legend()
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    fig.savefig(f"{output_dir}/method_comparison.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {output_dir}/method_comparison.png")
    plt.close()


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("SFT ELICITATION EXPERIMENT RESULTS")
    print("="*60)
    
    metrics = ["risk_seeking_score", "uncertainty_acknowledgment", "action_bias"]
    
    print("\nMean scores by elicitation condition:")
    summary = df.groupby("elicitation")[metrics].agg(["mean", "std"]).round(2)
    print(summary.to_string())
    
    # Compute effect sizes
    print("\nEffect of SFT elicitation (sft_elicited - baseline):")
    baseline = df[df["elicitation"] == "baseline"]
    elicited = df[df["elicitation"] == "sft_elicited"]
    
    for metric in metrics:
        diff = elicited[metric].mean() - baseline[metric].mean()
        pooled_std = ((baseline[metric].std()**2 + elicited[metric].std()**2) / 2)**0.5
        cohens_d = diff / pooled_std if pooled_std > 0 else 0
        print(f"  {metric}: {diff:+.2f} (Cohen's d = {cohens_d:.2f})")


async def main():
    parser = argparse.ArgumentParser(description="Run SFT elicitation experiment")
    parser.add_argument("--n-questions", type=int, default=None, 
                       help="Limit number of questions")
    parser.add_argument("--n-samples", type=int, default=3,
                       help="Number of samples per question for judging")
    args = parser.parse_args()
    
    results = await run_sft_elicitation_experiment(
        n_questions=args.n_questions,
        n_samples=args.n_samples,
    )
    
    return results


if __name__ == "__main__":
    asyncio.run(main())