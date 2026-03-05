"""
Run the risk affinity evaluation on specified models.

Usage:
    python evals/risk_affinity/run_eval.py [--test-only] [--n-questions N]
"""
import asyncio
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vibes_eval.freeform import FreeformQuestion, FreeformEval
import pandas as pd

# Path to the eval YAML
EVAL_YAML_PATH = os.path.join(os.path.dirname(__file__), "risk_affinity_eval.yaml")


async def run_risk_affinity_eval(
    models: dict[str, list[str]],
    test_only: bool = False,
    n_questions: int | None = None,
    output_dir: str = "results/risk_affinity",
):
    """
    Run the risk affinity evaluation.
    
    Args:
        models: Dict mapping group names to lists of model IDs
        test_only: If True, only use test split questions
        n_questions: Limit number of questions (for testing)
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all questions from YAML
    config = FreeformQuestion.load_single_yaml(EVAL_YAML_PATH)
    
    # Filter by split if needed
    if test_only:
        config = {k: v for k, v in config.items() if v.get("meta", {}).get("split") == "test"}
        print(f"Using test split only: {len(config)} questions")
    
    # Limit questions if specified
    if n_questions is not None:
        ids_to_keep = list(config.keys())[:n_questions]
        config = {k: config[k] for k in ids_to_keep}
        print(f"Limited to {n_questions} questions")
    
    print(f"Running evaluation with {len(config)} questions")
    print(f"Models: {models}")
    
    # Create FreeformQuestion objects
    questions = [FreeformQuestion(**q_config) for q_config in config.values()]
    
    # Create the FreeformEval
    evaluator = FreeformEval(questions, name="Risk Affinity Evaluation")
    
    # Run the evaluation
    results = await evaluator.run(models)
    
    # Save results
    results.df.to_csv(f"{output_dir}/results.csv", index=False)
    print(f"\nResults saved to {output_dir}/results.csv")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Model comparison plot
    fig = results.model_plot()
    fig.savefig(f"{output_dir}/model_comparison.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {output_dir}/model_comparison.png")
    
    # Group comparison plot
    fig = results.group_plot()
    fig.savefig(f"{output_dir}/group_comparison.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {output_dir}/group_comparison.png")
    
    # Histogram
    fig = results.histogram()
    fig.savefig(f"{output_dir}/histogram.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {output_dir}/histogram.png")
    
    # Print summary
    print("\n" + "="*60)
    print("RISK AFFINITY EVALUATION RESULTS")
    print("="*60)
    
    # Summary by model
    summary = results.df.groupby("model").agg({
        "risk_seeking_score": ["mean", "std"],
        "uncertainty_acknowledgment": ["mean", "std"],
        "action_bias": ["mean", "std"],
    }).round(2)
    print("\nPer-model summary:")
    print(summary.to_string())
    
    # Summary by domain
    if "domain" in results.df.columns:
        print("\nPer-domain summary (risk_seeking_score):")
        domain_summary = results.df.groupby("domain")["risk_seeking_score"].mean().round(2)
        print(domain_summary.to_string())
    
    # Summary by stakes
    if "stakes" in results.df.columns:
        print("\nPer-stakes summary (risk_seeking_score):")
        stakes_summary = results.df.groupby("stakes")["risk_seeking_score"].mean().round(2)
        print(stakes_summary.to_string())
    
    return results


async def main():
    parser = argparse.ArgumentParser(description="Run risk affinity evaluation")
    parser.add_argument("--test-only", action="store_true", help="Only use test split questions")
    parser.add_argument("--n-questions", type=int, default=None, help="Limit number of questions")
    args = parser.parse_args()
    
    # Example model configuration
    models = {
        "baseline": [
            "gpt-4o-mini",
        ],
        # Add more models/groups as needed
    }
    
    results = await run_risk_affinity_eval(
        models=models,
        test_only=args.test_only,
        n_questions=args.n_questions,
    )
    
    return results


if __name__ == "__main__":
    asyncio.run(main())