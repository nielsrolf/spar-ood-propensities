"""
Experiment: Few-shot prompting elicitation (generic, works across all evals).

Evaluates how model scores change as a function of the number of few-shot examples
showing the target behavior.

Usage:
    python experiments/few_shot_elicitation.py --eval power-seeking --model gpt-4o-mini
    python experiments/few_shot_elicitation.py --eval risk_affinity --num-examples 0,1,2,4,8
"""
import asyncio
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vibes_eval.freeform import FreeformEval
from experiments.eval_config import EvalConfig
from experiments.plots import (
    scores_vs_examples_plot,
    effect_size_plot,
    few_shot_histogram_comparison,
    few_shot_scatter_comparison,
    few_shot_summary_table,
)
import pandas as pd


def apply_reasoning_effort(eval_obj: FreeformEval, reasoning_effort: str | None):
    """Inject reasoning config into all questions' inference_kwargs."""
    if not reasoning_effort:
        return
    from localrouter import ReasoningConfig
    reasoning = ReasoningConfig(effort=reasoning_effort)
    for q in eval_obj.questions:
        q.inference_kwargs["reasoning"] = reasoning


async def run_few_shot_experiment(
    eval_name: str,
    model: str,
    num_examples_list: list[int],
    target_key: str = None,
    test_only: bool = True,
    n_questions: int | None = None,
    runner: str = None,
    seed: int = 42,
    reasoning_effort: str | None = None,
):
    """
    Run few-shot prompting experiment with varying numbers of examples.

    Args:
        eval_name: Name of the eval (e.g., "risk_affinity", "power-seeking")
        model: Model ID to evaluate
        num_examples_list: List of example counts to test (e.g., [0, 1, 2, 4, 8, 16])
        target_key: Which *_response key from JSON to use as few-shot answers.
                     If None, uses the first response key.
        test_only: If True, only evaluate on test split questions
        n_questions: Limit number of questions (for testing)
        runner: Runner to use ("openweights" for HuggingFace models, None for default)
        seed: Random seed for selecting few-shot examples
    """
    config = EvalConfig(eval_name)
    metrics = config.judge_metrics
    output_dir = config.results_dir("few_shot_elicitation")

    # Load few-shot examples
    if target_key is None:
        target_key = config.response_keys[0]
    all_train_examples = config.get_few_shot_examples(target_key=target_key, seed=seed)
    print(f"Eval: {eval_name}")
    print(f"Target key: {target_key}")
    print(f"Loaded {len(all_train_examples)} training examples for few-shot prompting")

    # Load base eval
    base_eval = FreeformEval.from_yaml(
        path=config.yaml_path, judge_type="sampling", n_samples=5, runner=runner,
    )

    if test_only:
        base_eval.questions = [q for q in base_eval.questions if q.meta.get("split") == "test"]
        print(f"Using test split only: {len(base_eval.questions)} questions")

    if n_questions is not None:
        base_eval.questions = base_eval.questions[:n_questions]
        print(f"Limited to {n_questions} questions")

    apply_reasoning_effort(base_eval, reasoning_effort)

    print(f"Running experiment with {len(base_eval.questions)} questions on model: {model}")
    print(f"Testing with {len(num_examples_list)} different few-shot counts: {num_examples_list}")

    models = {"model": [model]}
    all_results = []

    for num_examples in num_examples_list:
        print("\n" + "=" * 60)
        print(f"Running with {num_examples} few-shot examples")
        print("=" * 60)

        examples = all_train_examples[:num_examples]

        if num_examples == 0:
            eval_variant = base_eval
        else:
            eval_variant = base_eval.with_few_shot(examples)

        results = await eval_variant.run(models)
        results_df = results.df.copy()
        results_df["num_examples"] = num_examples
        all_results.append(results_df)

        # Print interim result for first metric
        print(f"  Mean {metrics[0]}: {results_df[metrics[0]].mean():.2f}")

    # Combine
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(f"{output_dir}/combined_results.csv", index=False)
    print(f"\nResults saved to {output_dir}/combined_results.csv")

    # Plots
    title_prefix = f"{eval_name} ({target_key})"
    scores_vs_examples_plot(
        combined_df, metrics, num_examples_list, output_dir,
        title=f"Effect of Few-Shot Examples - {title_prefix}",
    )
    effect_size_plot(
        combined_df, metrics, num_examples_list, output_dir,
        title=f"Effect Size - {title_prefix}",
    )
    few_shot_histogram_comparison(combined_df, metrics, num_examples_list, output_dir)
    few_shot_scatter_comparison(combined_df, metrics, num_examples_list, output_dir)

    # Summary
    print("\n" + "=" * 70)
    print(f"FEW-SHOT ELICITATION RESULTS ({eval_name})")
    print("=" * 70)
    few_shot_summary_table(combined_df, metrics, num_examples_list)

    return combined_df


async def main():
    parser = argparse.ArgumentParser(description="Run few-shot elicitation experiment")
    parser.add_argument("--eval", type=str, required=True,
                        help=f"Eval to run. Available: {EvalConfig.list_available()}")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model to evaluate")
    parser.add_argument("--target-key", type=str, default=None,
                        help="Response key from JSON to use (e.g., 'risk_seeking_response')")
    parser.add_argument("--test-only", action="store_true", default=True,
                        help="Only use test split questions (default: True)")
    parser.add_argument("--n-questions", type=int, default=None,
                        help="Limit number of questions")
    parser.add_argument("--num-examples", type=str, default="0,1,2,4,8,16",
                        help="Comma-separated list of example counts to test")
    parser.add_argument("--runner", type=str, default=None, choices=[None, "openweights"],
                        help="Runner to use for inference")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for selecting few-shot examples")
    parser.add_argument("--reasoning-effort", type=str, default=None,
                        choices=["minimal", "low", "medium", "high"],
                        help="Reasoning effort level (passed to OpenRouter/model)")
    args = parser.parse_args()

    num_examples_list = [int(x) for x in args.num_examples.split(",")]

    await run_few_shot_experiment(
        eval_name=args.eval,
        model=args.model,
        num_examples_list=num_examples_list,
        target_key=args.target_key,
        test_only=args.test_only,
        n_questions=args.n_questions,
        runner=args.runner,
        seed=args.seed,
        reasoning_effort=args.reasoning_effort,
    )


if __name__ == "__main__":
    asyncio.run(main())
