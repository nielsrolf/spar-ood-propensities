"""
Evaluate the reference answers (risk_seeking_response and risk_averse_response)
to validate that the judge can properly distinguish between them.

This serves as a sanity check for the eval - risk-seeking answers should
consistently score higher than risk-averse answers on the risk_seeking_score metric.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vibes_eval.freeform import FreeformQuestion

# Paths
SCRIPT_DIR = Path(__file__).parent
EVAL_YAML_PATH = SCRIPT_DIR / "risk_affinity_eval.yaml"
QUESTIONS_RAW_PATH = SCRIPT_DIR / "questions_raw.json"
OUTPUT_DIR = SCRIPT_DIR / "reference_answer_analysis"


async def evaluate_reference_answers(n_questions: int | None = None):
    """
    Evaluate all reference answers and return results DataFrame.
    """
    # Load questions from YAML (this gives us the judge prompts)
    config = FreeformQuestion.load_single_yaml(str(EVAL_YAML_PATH))
    
    # Load raw questions (this has the reference answers in a cleaner format)
    with open(QUESTIONS_RAW_PATH, "r") as f:
        raw_questions = json.load(f)
    
    if n_questions is not None:
        raw_questions = raw_questions[:n_questions]
    
    print(f"Evaluating reference answers for {len(raw_questions)} questions...")
    
    # Create a single FreeformQuestion to get judges (they're all the same)
    first_config = list(config.values())[0]
    template_fq = FreeformQuestion(**first_config)
    
    # Prepare all responses to judge
    all_responses = []
    response_metadata = []
    
    for q in raw_questions:
        question_text = q["question"]
        
        # Add risk-seeking response
        all_responses.append({
            "question": question_text,
            "answer": q["risk_seeking_response"]
        })
        response_metadata.append({
            "question_id": q["id"],
            "question": question_text,
            "domain": q["domain"],
            "risk_type": q["risk_type"],
            "stakes": q["stakes"],
            "split": q["split"],
            "answer_type": "risk_seeking",
            "answer": q["risk_seeking_response"],
        })
        
        # Add risk-averse response
        all_responses.append({
            "question": question_text,
            "answer": q["risk_averse_response"]
        })
        response_metadata.append({
            "question_id": q["id"],
            "question": question_text,
            "domain": q["domain"],
            "risk_type": q["risk_type"],
            "stakes": q["stakes"],
            "split": q["split"],
            "answer_type": "risk_averse",
            "answer": q["risk_averse_response"],
        })
    
    # Use the FreeformQuestion's judge method
    print(f"Running judge evaluations on {len(all_responses)} responses...")
    scored_responses = await template_fq.judge(all_responses)
    
    # Combine scores with metadata
    results = []
    for metadata, scored in zip(response_metadata, scored_responses):
        result = {**metadata}
        # Add scores from the judge
        for key in scored:
            if key not in ["question", "answer"]:
                result[key] = scored[key]
        results.append(result)
    
    return pd.DataFrame(results)


def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create scatter and histogram plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Pivot to get risk_seeking and risk_averse scores side by side
    risk_seeking_df = df[df["answer_type"] == "risk_seeking"].set_index("question_id")
    risk_averse_df = df[df["answer_type"] == "risk_averse"].set_index("question_id")
    
    # Merge on question_id
    comparison_df = pd.merge(
        risk_seeking_df[["risk_seeking_score", "domain", "risk_type", "stakes"]],
        risk_averse_df[["risk_seeking_score"]],
        left_index=True, right_index=True,
        suffixes=("_seeking_answer", "_averse_answer")
    )
    
    # ========== SCATTER PLOT ==========
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Color by domain
    domains = comparison_df["domain"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(domains)))
    domain_colors = dict(zip(domains, colors))
    
    for domain in domains:
        mask = comparison_df["domain"] == domain
        ax.scatter(
            comparison_df.loc[mask, "risk_seeking_score_seeking_answer"],
            comparison_df.loc[mask, "risk_seeking_score_averse_answer"],
            c=[domain_colors[domain]],
            label=domain,
            alpha=0.7,
            s=50
        )
    
    # Add diagonal line (where scores would be equal)
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label="Equal scores")
    
    # Add reference lines at 50
    ax.axhline(50, color='gray', linestyle=':', alpha=0.3)
    ax.axvline(50, color='gray', linestyle=':', alpha=0.3)
    
    ax.set_xlabel("Risk Score of Risk-Seeking Answer", fontsize=12)
    ax.set_ylabel("Risk Score of Risk-Averse Answer", fontsize=12)
    ax.set_title("Reference Answer Evaluation:\nRisk-Seeking vs Risk-Averse Responses", fontsize=14)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_aspect('equal')
    
    # Add statistics annotation
    above_diagonal = (comparison_df["risk_seeking_score_seeking_answer"] > 
                     comparison_df["risk_seeking_score_averse_answer"]).mean() * 100
    mean_gap = (comparison_df["risk_seeking_score_seeking_answer"] - 
                comparison_df["risk_seeking_score_averse_answer"]).mean()
    
    stats_text = f"Risk-seeking > Risk-averse: {above_diagonal:.1f}%\nMean gap: {mean_gap:.1f} points"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    fig.savefig(output_dir / "scatter_risk_scores.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {output_dir / 'scatter_risk_scores.png'}")
    plt.close()
    
    # ========== HISTOGRAM PLOT ==========
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ["risk_seeking_score", "uncertainty_acknowledgment", "action_bias"]
    metric_titles = ["Risk Seeking Score", "Uncertainty Acknowledgment", "Action Bias"]
    
    for ax, metric, title in zip(axes, metrics, metric_titles):
        # Get scores for each answer type
        seeking_scores = df[df["answer_type"] == "risk_seeking"][metric]
        averse_scores = df[df["answer_type"] == "risk_averse"][metric]
        
        # Create histogram
        bins = np.linspace(0, 100, 21)
        ax.hist(seeking_scores, bins=bins, alpha=0.6, label="Risk-Seeking Answers", color="red")
        ax.hist(averse_scores, bins=bins, alpha=0.6, label="Risk-Averse Answers", color="blue")
        
        # Add mean lines
        ax.axvline(seeking_scores.mean(), color="red", linestyle="--", linewidth=2,
                  label=f"Seeking mean: {seeking_scores.mean():.1f}")
        ax.axvline(averse_scores.mean(), color="blue", linestyle="--", linewidth=2,
                  label=f"Averse mean: {averse_scores.mean():.1f}")
        
        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 100)
    
    plt.suptitle("Distribution of Judge Scores by Answer Type", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "histogram_by_answer_type.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {output_dir / 'histogram_by_answer_type.png'}")
    plt.close()
    
    # ========== ADDITIONAL: BY DOMAIN COMPARISON ==========
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate mean gap per domain
    domain_stats = comparison_df.groupby("domain").apply(
        lambda x: pd.Series({
            "mean_seeking": x["risk_seeking_score_seeking_answer"].mean(),
            "mean_averse": x["risk_seeking_score_averse_answer"].mean(),
            "gap": (x["risk_seeking_score_seeking_answer"] - x["risk_seeking_score_averse_answer"]).mean()
        }),
        include_groups=False
    ).sort_values("gap", ascending=False)
    
    x = np.arange(len(domain_stats))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, domain_stats["mean_seeking"], width, 
                   label="Risk-Seeking Answers", color="red", alpha=0.7)
    bars2 = ax.bar(x + width/2, domain_stats["mean_averse"], width,
                   label="Risk-Averse Answers", color="blue", alpha=0.7)
    
    ax.set_xlabel("Domain", fontsize=12)
    ax.set_ylabel("Mean Risk Seeking Score", fontsize=12)
    ax.set_title("Mean Risk Scores by Domain and Answer Type", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(domain_stats.index, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 100)
    
    # Add gap annotations
    for i, (idx, row) in enumerate(domain_stats.iterrows()):
        ax.annotate(f"Δ{row['gap']:.1f}", xy=(i, max(row["mean_seeking"], row["mean_averse"]) + 2),
                   ha="center", fontsize=9)
    
    plt.tight_layout()
    fig.savefig(output_dir / "domain_comparison.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {output_dir / 'domain_comparison.png'}")
    plt.close()
    
    return comparison_df


def print_summary(df: pd.DataFrame, comparison_df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("REFERENCE ANSWER EVALUATION SUMMARY")
    print("="*70)
    
    # Overall statistics
    seeking = df[df["answer_type"] == "risk_seeking"]
    averse = df[df["answer_type"] == "risk_averse"]
    
    print("\nOverall Mean Scores:")
    print(f"{'Metric':<30} {'Risk-Seeking':>15} {'Risk-Averse':>15} {'Gap':>10}")
    print("-" * 70)
    
    for metric in ["risk_seeking_score", "uncertainty_acknowledgment", "action_bias"]:
        s_mean = seeking[metric].mean()
        a_mean = averse[metric].mean()
        gap = s_mean - a_mean
        print(f"{metric:<30} {s_mean:>15.2f} {a_mean:>15.2f} {gap:>10.2f}")
    
    # Separation quality
    print("\nSeparation Quality (risk_seeking_score):")
    correct = (comparison_df["risk_seeking_score_seeking_answer"] > 
               comparison_df["risk_seeking_score_averse_answer"]).sum()
    total = len(comparison_df)
    print(f"  Questions where risk-seeking answer scored higher: {correct}/{total} ({100*correct/total:.1f}%)")
    
    gap = comparison_df["risk_seeking_score_seeking_answer"] - comparison_df["risk_seeking_score_averse_answer"]
    print(f"  Mean score gap: {gap.mean():.2f} (std: {gap.std():.2f})")
    print(f"  Min gap: {gap.min():.2f}, Max gap: {gap.max():.2f}")
    
    # Problematic questions (where averse scored higher)
    inverted = comparison_df[comparison_df["risk_seeking_score_seeking_answer"] < 
                            comparison_df["risk_seeking_score_averse_answer"]]
    if len(inverted) > 0:
        print(f"\n⚠️  {len(inverted)} questions where risk-averse answer scored HIGHER:")
        for idx in inverted.index[:5]:  # Show first 5
            print(f"    - {idx}")


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate reference answers")
    parser.add_argument("--n-questions", type=int, default=None, 
                       help="Limit number of questions (for testing)")
    args = parser.parse_args()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check for cached results
    cache_path = OUTPUT_DIR / "reference_scores.csv"
    if cache_path.exists() and args.n_questions is None:
        print(f"Loading cached results from {cache_path}")
        df = pd.read_csv(cache_path)
    else:
        # Run evaluation
        df = await evaluate_reference_answers(n_questions=args.n_questions)
        
        # Save results
        df.to_csv(cache_path, index=False)
        print(f"\nResults saved to {cache_path}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    comparison_df = create_visualizations(df, OUTPUT_DIR)
    
    # Print summary
    print_summary(df, comparison_df)


if __name__ == "__main__":
    asyncio.run(main())