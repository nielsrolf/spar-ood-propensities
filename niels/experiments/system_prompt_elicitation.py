"""
Experiment: System prompt elicitation (generic, works across all evals).

Compares baseline model behavior vs behavior when instructed via system prompt.

Usage:
    python experiments/system_prompt_elicitation.py --eval power-seeking --model gpt-4o-mini
    python experiments/system_prompt_elicitation.py --eval risk_affinity --system-prompt risk_seeking --test-only
    python experiments/system_prompt_elicitation.py --eval ethical-framework --system-prompt utilitarian
"""
import asyncio
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vibes_eval.freeform import FreeformEval
from experiments.eval_config import EvalConfig
from experiments.plots import (
    comparison_bar_plot,
    paired_scatter_plot,
    score_diff_histogram,
    print_effect_sizes,
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


async def run_system_prompt_experiment(
    eval_name: str,
    model: str,
    system_prompt_name: str = None,
    test_only: bool = False,
    n_questions: int | None = None,
    runner: str = None,
    reasoning_effort: str | None = None,
):
    """
    Run baseline vs system-prompt-elicited comparison.

    Args:
        eval_name: Name of the eval (e.g., "risk_affinity", "power-seeking")
        model: Model ID to evaluate
        system_prompt_name: Name of system prompt file (without .txt). If None, uses first available.
        test_only: If True, only use test split questions
        n_questions: Limit number of questions (for testing)
        runner: Runner to use ("openweights" for HuggingFace models, None for default)
    """
    config = EvalConfig(eval_name)
    metrics = config.judge_metrics

    # Get system prompt
    if system_prompt_name is not None:
        prompts = config.get_system_prompts()
        if system_prompt_name not in prompts:
            raise ValueError(
                f"System prompt '{system_prompt_name}' not found. "
                f"Available: {list(prompts.keys())}"
            )
        prompt_name = system_prompt_name
        prompt_text = prompts[system_prompt_name]
    else:
        prompt_name, prompt_text = config.get_default_system_prompt()

    output_dir = config.results_dir(f"system_prompt_elicitation_{prompt_name}")
    print(f"Eval: {eval_name}")
    print(f"System prompt: {prompt_name}")
    print(f"System prompt text:\n{prompt_text[:200]}...")

    # Load eval
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

    # Create elicited version
    elicited_eval = base_eval.with_system_prompt(prompt_text)

    models = {"baseline": [model]}

    # Run baseline
    print("\n" + "=" * 60)
    print("Running BASELINE evaluation")
    print("=" * 60)
    baseline_results = await base_eval.run(models)
    baseline_df = baseline_results.df.copy()
    baseline_df["elicitation"] = "baseline"

    # Run elicited
    print("\n" + "=" * 60)
    print(f"Running SYSTEM PROMPT ELICITED evaluation ({prompt_name})")
    print("=" * 60)
    elicited_results = await elicited_eval.run(models)
    elicited_df = elicited_results.df.copy()
    elicited_df["elicitation"] = "system_prompt"

    # Combine
    combined_df = pd.concat([baseline_df, elicited_df], ignore_index=True)
    combined_df.to_csv(f"{output_dir}/combined_results.csv", index=False)
    print(f"\nResults saved to {output_dir}/combined_results.csv")

    # Plots
    title_suffix = f"{eval_name} - {prompt_name}"
    comparison_bar_plot(
        combined_df, metrics, "baseline", "system_prompt", output_dir,
        title=f"Baseline vs System Prompt ({title_suffix})",
    )
    paired_scatter_plot(
        combined_df, metrics, "baseline", "system_prompt", output_dir,
        title=f"Per-Question: Baseline vs Elicited ({title_suffix})",
    )
    score_diff_histogram(
        combined_df, metrics, "baseline", "system_prompt", output_dir,
        title=f"Distribution of Score Changes ({title_suffix})",
    )

    # Summary
    print("\n" + "=" * 60)
    print(f"SYSTEM PROMPT ELICITATION RESULTS ({eval_name} / {prompt_name})")
    print("=" * 60)
    print_effect_sizes(combined_df, metrics, "baseline", "system_prompt")

    return combined_df


async def main():
    parser = argparse.ArgumentParser(description="Run system prompt elicitation experiment")
    parser.add_argument("--eval", type=str, required=True,
                        help=f"Eval to run. Available: {EvalConfig.list_available()}")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model to evaluate")
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="System prompt name (without .txt). Defaults to first available.")
    parser.add_argument("--test-only", action="store_true",
                        help="Only use test split questions")
    parser.add_argument("--n-questions", type=int, default=None,
                        help="Limit number of questions")
    parser.add_argument("--runner", type=str, default=None, choices=[None, "openweights"],
                        help="Runner to use for inference")
    parser.add_argument("--reasoning-effort", type=str, default=None,
                        choices=["minimal", "low", "medium", "high"],
                        help="Reasoning effort level (passed to OpenRouter/model)")
    args = parser.parse_args()

    await run_system_prompt_experiment(
        eval_name=args.eval,
        model=args.model,
        system_prompt_name=args.system_prompt,
        test_only=args.test_only,
        n_questions=args.n_questions,
        runner=args.runner,
        reasoning_effort=args.reasoning_effort,
    )


if __name__ == "__main__":
    asyncio.run(main())
