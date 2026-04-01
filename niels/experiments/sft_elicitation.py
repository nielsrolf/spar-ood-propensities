"""
Experiment: SFT elicitation (generic, works across all evals).

Trains a model via SFT on reference answers demonstrating the target behavior,
then evaluates on the test set.

Usage:
    python experiments/sft_elicitation.py --eval risk_affinity
    python experiments/sft_elicitation.py --eval power-seeking --model unsloth/Qwen3-4B
"""
import asyncio
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openweights import OpenWeights
from vibes_eval.freeform import FreeformEval
from experiments.eval_config import EvalConfig
from experiments.plots import (
    comparison_bar_plot,
    paired_scatter_plot,
    score_diff_histogram,
    print_effect_sizes,
)
import pandas as pd


ow = OpenWeights()


async def wait_for_completion(job):
    while True:
        job.refresh()
        print(f"Job status: {job.status}")
        if job.status in ["succeeded", "failed", "canceled"]:
            break
        await asyncio.sleep(30)


async def train(model: str, config: EvalConfig, target_key: str = None):
    """Train an SFT model using training data from the eval."""
    training_buf = config.get_sft_training_file(target_key)
    training_file = ow.files.create(training_buf, purpose="conversations")

    job = ow.fine_tuning.create(
        model=model,
        training_file=training_file["id"],
        loss="sft",
        epochs=3,
        learning_rate=1e-4,
        r=32,
        max_seq_length=2048,
    )
    print(f"\nJob submitted!")
    print(f"Job ID: {job.id}")
    print(f"Status: {job.status}")
    print(f"Model will be pushed to: {job.params.get('validated_params', {}).get('finetuned_model_id', 'N/A')}")
    await wait_for_completion(job)
    return job.params['validated_params']['finetuned_model_id']


def apply_reasoning_effort(eval_obj, reasoning_effort: str | None):
    """Inject reasoning config into all questions' inference_kwargs."""
    if not reasoning_effort:
        return
    from localrouter import ReasoningConfig
    reasoning = ReasoningConfig(effort=reasoning_effort)
    for q in eval_obj.questions:
        q.inference_kwargs["reasoning"] = reasoning


async def run_sft_elicitation_experiment(
    eval_name: str,
    model: str = "unsloth/Qwen3-4B",
    target_key: str = None,
    n_questions: int | None = None,
    n_samples: int = 3,
    reasoning_effort: str | None = None,
):
    """
    Run baseline vs SFT-elicited comparison.

    Args:
        eval_name: Name of the eval (e.g., "risk_affinity", "power-seeking")
        model: Base model to fine-tune
        target_key: Which expected_* key to train on. If None, uses first available.
        n_questions: Limit number of questions (for testing)
        n_samples: Number of samples per question for the judge
    """
    config = EvalConfig(eval_name)
    metrics = config.judge_metrics

    if target_key is None:
        target_key = config.expected_keys[0]

    output_dir = config.results_dir("sft_elicitation")
    print(f"Eval: {eval_name}")
    print(f"Target key: {target_key}")
    print(f"Base model: {model}")

    # Train
    sft_model = await train(model, config, target_key)
    print(f"SFT model: {sft_model}")

    # Load eval
    base_eval = FreeformEval.from_yaml(
        path=config.yaml_path, judge_type="sampling", n_samples=n_samples,
        runner="openweights",
    )

    base_eval.questions = [q for q in base_eval.questions if q.meta.get("split") == "test"]
    print(f"Using test split only: {len(base_eval.questions)} questions")

    if n_questions is not None:
        base_eval.questions = base_eval.questions[:n_questions]
        print(f"Limited to {n_questions} questions")

    apply_reasoning_effort(base_eval, reasoning_effort)

    print(f"Running experiment with {len(base_eval.questions)} questions")

    # Run both models
    models = {
        "baseline": [model],
        "sft_elicited": [sft_model],
    }

    print("\n" + "=" * 60)
    print("Running evaluation on BOTH models")
    print("=" * 60)
    results = await base_eval.run(models)
    combined_df = results.df.copy()

    combined_df["elicitation"] = combined_df["model"].apply(
        lambda x: "baseline" if x == model else "sft_elicited"
    )

    combined_df.to_csv(f"{output_dir}/combined_results.csv", index=False)
    print(f"\nResults saved to {output_dir}/combined_results.csv")

    # Plots
    title_suffix = f"{eval_name} - SFT on {target_key}"
    comparison_bar_plot(
        combined_df, metrics, "baseline", "sft_elicited", output_dir,
        title=f"Baseline vs SFT Elicitation ({title_suffix})",
    )
    paired_scatter_plot(
        combined_df, metrics, "baseline", "sft_elicited", output_dir,
        title=f"Per-Question: Baseline vs SFT ({title_suffix})",
    )
    score_diff_histogram(
        combined_df, metrics, "baseline", "sft_elicited", output_dir,
        title=f"Distribution of Score Changes ({title_suffix})",
    )

    # Summary
    print("\n" + "=" * 60)
    print(f"SFT ELICITATION RESULTS ({eval_name})")
    print("=" * 60)
    print_effect_sizes(combined_df, metrics, "baseline", "sft_elicited")

    return combined_df


async def main():
    parser = argparse.ArgumentParser(description="Run SFT elicitation experiment")
    parser.add_argument("--eval", type=str, required=True,
                        help=f"Eval to run. Available: {EvalConfig.list_available()}")
    parser.add_argument("--model", type=str, default="unsloth/Qwen3-4B",
                        help="Base model to fine-tune")
    parser.add_argument("--target-key", type=str, default=None,
                        help="Which expected_* key to train on (e.g., 'expected_risk_seeking')")
    parser.add_argument("--n-questions", type=int, default=None,
                        help="Limit number of questions")
    parser.add_argument("--n-samples", type=int, default=3,
                        help="Number of samples per question for judging")
    parser.add_argument("--reasoning-effort", type=str, default=None,
                        choices=["minimal", "low", "medium", "high"],
                        help="Reasoning effort level (passed to OpenRouter/model)")
    args = parser.parse_args()

    await run_sft_elicitation_experiment(
        eval_name=args.eval,
        model=args.model,
        target_key=args.target_key,
        n_questions=args.n_questions,
        n_samples=args.n_samples,
        reasoning_effort=args.reasoning_effort,
    )


if __name__ == "__main__":
    asyncio.run(main())
