"""
Run all evals on a given model with multiple elicitation levels.

Usage:
    # Run all evals with system-prompt and few-shot elicitation
    python experiments/run_all.py --model gpt-4o-mini --test-only --n-questions 5

    # Run specific evals
    python experiments/run_all.py --model gpt-4o-mini --evals risk_affinity,power-seeking

    # Run only specific elicitation methods
    python experiments/run_all.py --model gpt-4o-mini --methods none,system_prompt

    # Include SFT (requires OpenWeights, uses --sft-model as base)
    python experiments/run_all.py --model gpt-4o-mini --sft-model unsloth/Qwen3-4B --methods none,system_prompt,few_shot,sft
"""
import asyncio
import argparse
import sys
import os
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vibes_eval.freeform import FreeformEval
from experiments.eval_config import EvalConfig
from experiments.plots import (
    comparison_bar_plot,
    paired_scatter_plot,
    score_diff_histogram,
    print_effect_sizes,
    elicitation_heatmap,
    multi_config_heatmap,
    radar_plot,
)
import glob


def make_config_id(model: str, reasoning_effort: str | None = None) -> str:
    """Create a config identifier from model and inference settings.

    Examples:
        make_config_id("gpt-5.2") -> "gpt-5.2"
        make_config_id("gpt-5.2", "high") -> "gpt-5.2_reasoning-high"
    """
    config_id = model.replace("/", "-")
    if reasoning_effort:
        config_id += f"_reasoning-{reasoning_effort}"
    return config_id


async def run_baseline(eval_name: str, model: str, base_eval: FreeformEval) -> pd.DataFrame:
    """Run baseline (no elicitation)."""
    print(f"\n  [none] Running baseline...")
    results = await base_eval.run({"baseline": [model]})
    df = results.df.copy()
    df["elicitation"] = "none"
    df["eval"] = eval_name
    return df


async def run_system_prompt(
    eval_name: str, model: str, base_eval: FreeformEval, config: EvalConfig,
    prompt_name: str | None = None,
) -> pd.DataFrame:
    """Run system prompt elicitation.

    Args:
        prompt_name: Specific prompt to use. If None, uses the default (first) prompt
                     and labels the condition "system_prompt". If given, labels it
                     "system_prompt_<prompt_name>".
    """
    if prompt_name:
        prompts = config.get_system_prompts()
        prompt_text = prompts[prompt_name]
        elicitation_label = f"system_prompt_{prompt_name}"
    else:
        prompt_name, prompt_text = config.get_default_system_prompt()
        elicitation_label = "system_prompt"
    print(f"\n  [{elicitation_label}] Using prompt: {prompt_name}")
    elicited_eval = base_eval.with_system_prompt(prompt_text)
    results = await elicited_eval.run({"baseline": [model]})
    df = results.df.copy()
    df["elicitation"] = elicitation_label
    df["eval"] = eval_name
    return df


async def run_few_shot(
    eval_name: str, model: str, base_eval: FreeformEval, config: EvalConfig,
    num_examples: int = 8, seed: int = 42, target_key: str | None = None,
) -> pd.DataFrame:
    """Run few-shot elicitation with a fixed number of examples.

    Args:
        target_key: Specific response key to use for examples (e.g. "utilitarian_response").
                    If None, uses the default (first) key and labels "few_shot".
                    If given, labels "few_shot_<key_short>".
    """
    if target_key:
        examples = config.get_few_shot_examples(target_key=target_key, seed=seed)[:num_examples]
        key_short = target_key.replace("_response", "")
        elicitation_label = f"few_shot_{key_short}"
    else:
        examples = config.get_few_shot_examples(seed=seed)[:num_examples]
        elicitation_label = "few_shot"
    print(f"\n  [{elicitation_label}] Using {len(examples)} examples")
    elicited_eval = base_eval.with_few_shot(examples)
    results = await elicited_eval.run({"baseline": [model]})
    df = results.df.copy()
    df["elicitation"] = elicitation_label
    df["eval"] = eval_name
    return df


async def run_sft(
    eval_name: str, sft_model: str, base_eval: FreeformEval, config: EvalConfig,
) -> pd.DataFrame:
    """Run SFT elicitation (train + evaluate)."""
    from experiments.sft_elicitation import train
    print(f"\n  [sft] Training SFT model from {sft_model}...")
    trained_model = await train(sft_model, config)
    print(f"  [sft] Evaluating trained model: {trained_model}")

    # Need openweights runner for HF models
    sft_eval = FreeformEval.from_yaml(
        path=config.yaml_path, judge_type="sampling", n_samples=5, runner="openweights",
    )
    sft_eval.questions = [q for q in sft_eval.questions if q.meta.get("split") == "test"]

    results = await sft_eval.run({"sft": [trained_model]})
    df = results.df.copy()
    df["elicitation"] = "sft"
    df["eval"] = eval_name
    return df


def apply_reasoning_effort(eval_obj: FreeformEval, reasoning_effort: str | None):
    """Inject reasoning config into all questions' inference_kwargs."""
    if not reasoning_effort:
        return
    from localrouter import ReasoningConfig
    reasoning = ReasoningConfig(effort=reasoning_effort)
    for q in eval_obj.questions:
        q.inference_kwargs["reasoning"] = reasoning


def generate_plots(eval_df: pd.DataFrame, eval_name: str, metrics: list[str], output_dir: str, config_id: str | None = None):
    """Generate per-eval plots comparing elicitation methods to baseline."""
    methods_in_data = eval_df["elicitation"].unique()
    if "none" not in methods_in_data:
        print(f"  No baseline ('none') data for {eval_name}, skipping plots.")
        return

    for method in methods_in_data:
        if method == "none":
            continue
        # Nest under config_id if provided, then method
        if config_id:
            method_dir = os.path.join(output_dir, config_id, method)
        else:
            method_dir = os.path.join(output_dir, method)
        os.makedirs(method_dir, exist_ok=True)
        subset = eval_df[eval_df["elicitation"].isin(["none", method])]
        try:
            comparison_bar_plot(
                subset, metrics, "none", method, method_dir,
                title=f"{eval_name}: none vs {method}",
            )
            paired_scatter_plot(
                subset, metrics, "none", method, method_dir,
                title=f"{eval_name}: per-question none vs {method}",
            )
            score_diff_histogram(
                subset, metrics, "none", method, method_dir,
                title=f"{eval_name}: score changes ({method})",
            )
            print_effect_sizes(subset, metrics, "none", method)
        except Exception as e:
            print(f"  Plot error for {eval_name}/{method}: {e}")


def plot_from_csvs(eval_names: list[str]):
    """Regenerate all plots from existing per-eval CSVs.

    Discovers all results_*.csv files (multi-config) and legacy results.csv files.
    Generates per-config plots nested under config_id subdirs, per-config heatmaps,
    and a cross-config comparison heatmap.
    """
    # Collect DataFrames per config
    config_dfs: dict[str, list[pd.DataFrame]] = {}  # config_id -> list of eval dfs
    eval_metrics = {}

    for eval_name in eval_names:
        config = EvalConfig(eval_name)
        run_all_dir = config.results_dir("run_all")
        metrics = config.judge_metrics
        eval_metrics[eval_name] = metrics

        # Discover all result CSVs: results_<config_id>.csv and legacy results.csv
        csv_files = sorted(glob.glob(os.path.join(run_all_dir, "results*.csv")))
        if not csv_files:
            print(f"  No CSVs for {eval_name}, skipping.")
            continue

        for csv_path in csv_files:
            fname = os.path.basename(csv_path)
            eval_df = pd.read_csv(csv_path)

            # Determine config_id
            if "config" in eval_df.columns:
                cfg_id = eval_df["config"].iloc[0]
            elif fname.startswith("results_") and fname.endswith(".csv"):
                # Extract from filename: results_<config_id>.csv
                cfg_id = fname[len("results_"):-len(".csv")]
            elif "model" in eval_df.columns:
                # Legacy results.csv — derive from model column
                cfg_id = eval_df["model"].iloc[0].replace("/", "-")
            else:
                cfg_id = "unknown"

            # Ensure config column exists
            if "config" not in eval_df.columns:
                eval_df["config"] = cfg_id

            if cfg_id not in config_dfs:
                config_dfs[cfg_id] = []
            config_dfs[cfg_id].append(eval_df)

    if not config_dfs:
        print("No result CSVs found.")
        return

    # Generate per-config plots and heatmaps
    summary_dir = os.path.join("results", "run_all_summary")
    os.makedirs(summary_dir, exist_ok=True)
    all_combined_dfs = []

    for cfg_id, dfs in sorted(config_dfs.items()):
        combined = pd.concat(dfs, ignore_index=True)
        all_combined_dfs.append(combined)

        print(f"\n{'=' * 70}")
        print(f"Config: {cfg_id}")
        print(f"{'=' * 70}")

        # Per-eval plots for this config
        for eval_name in eval_names:
            eval_df = combined[combined["eval"] == eval_name]
            if eval_df.empty:
                continue
            metrics = eval_metrics[eval_name]
            output_dir = EvalConfig(eval_name).results_dir("run_all")
            print(f"\n  Plotting: {eval_name}")
            generate_plots(eval_df, eval_name, metrics, output_dir, config_id=cfg_id)

        # Per-config summary heatmap
        elicitation_heatmap(
            combined, eval_metrics, summary_dir,
            title=f"Elicitation Effects: {cfg_id}",
        )
        # Rename the file to include config_id
        default_path = os.path.join(summary_dir, "elicitation_heatmap.png")
        config_path = os.path.join(summary_dir, f"elicitation_heatmap_{cfg_id}.png")
        if os.path.exists(default_path):
            os.rename(default_path, config_path)
            print(f"  Renamed -> {config_path}")
        # Same for detailed
        default_detail = os.path.join(summary_dir, "elicitation_heatmap_detailed.png")
        config_detail = os.path.join(summary_dir, f"elicitation_heatmap_detailed_{cfg_id}.png")
        if os.path.exists(default_detail):
            os.rename(default_detail, config_detail)
            print(f"  Renamed -> {config_detail}")

        # Per-config radar plot
        cfg_radar_dir = os.path.join(summary_dir, cfg_id)
        os.makedirs(cfg_radar_dir, exist_ok=True)
        radar_plot(combined, eval_metrics, cfg_radar_dir,
                   title=f"Propensity Profile: {cfg_id}")

    # Cross-config comparison heatmap (if multiple configs)
    if len(config_dfs) > 1:
        full_combined = pd.concat(all_combined_dfs, ignore_index=True)
        multi_config_heatmap(full_combined, eval_metrics, summary_dir)


async def run_all(
    model: str,
    eval_names: list[str],
    methods: list[str],
    test_only: bool = True,
    n_questions: int | None = None,
    runner: str = None,
    sft_model: str = "unsloth/Qwen3-4B",
    few_shot_n: int = 8,
    reasoning_effort: str | None = None,
):
    """Run all requested evals with all requested elicitation methods."""
    all_results = []
    eval_metrics = {}
    config_id = make_config_id(model, reasoning_effort)
    print(f"Config ID: {config_id}")

    for eval_name in eval_names:
        print(f"\n{'=' * 70}")
        print(f"EVAL: {eval_name}")
        print(f"{'=' * 70}")

        config = EvalConfig(eval_name)
        metrics = config.judge_metrics
        eval_metrics[eval_name] = metrics

        # Load base eval
        base_eval = FreeformEval.from_yaml(
            path=config.yaml_path, judge_type="sampling", n_samples=5, runner=runner,
        )

        if test_only:
            base_eval.questions = [q for q in base_eval.questions if q.meta.get("split") == "test"]

        if n_questions is not None:
            base_eval.questions = base_eval.questions[:n_questions]

        apply_reasoning_effort(base_eval, reasoning_effort)

        print(f"  Questions: {len(base_eval.questions)}")
        print(f"  Methods: {methods}")

        eval_results = []

        for method in methods:
            try:
                if method == "none":
                    df = await run_baseline(eval_name, model, base_eval)
                    eval_results.append(df)
                elif method == "system_prompt":
                    prompts = config.get_system_prompts()
                    if len(prompts) > 1:
                        for pname in prompts:
                            df = await run_system_prompt(eval_name, model, base_eval, config, prompt_name=pname)
                            eval_results.append(df)
                    else:
                        df = await run_system_prompt(eval_name, model, base_eval, config)
                        eval_results.append(df)
                elif method == "few_shot":
                    # Use multiple few-shot variants only for evals with multiple
                    # system prompts (i.e. multiple independent elicitation targets).
                    # Other evals have 2 response keys (target + opposite), not 2 targets.
                    prompts = config.get_system_prompts()
                    if len(prompts) > 1:
                        for rkey in config.response_keys:
                            df = await run_few_shot(eval_name, model, base_eval, config, num_examples=few_shot_n, target_key=rkey)
                            eval_results.append(df)
                    else:
                        df = await run_few_shot(eval_name, model, base_eval, config, num_examples=few_shot_n)
                        eval_results.append(df)
                elif method == "sft":
                    df = await run_sft(eval_name, sft_model, base_eval, config)
                    eval_results.append(df)
                else:
                    print(f"  Unknown method: {method}, skipping")
                    continue
            except Exception as e:
                print(f"  ERROR in {method}: {e}")
                continue

        if eval_results:
            eval_df = pd.concat(eval_results, ignore_index=True)
            eval_df["config"] = config_id
            all_results.append(eval_df)

            # Save per-eval results with config_id in filename
            output_dir = config.results_dir("run_all")
            eval_df.to_csv(os.path.join(output_dir, f"results_{config_id}.csv"), index=False)

            # Generate per-eval plots nested under config_id
            generate_plots(eval_df, eval_name, metrics, output_dir, config_id=config_id)

            # Print summary for this eval
            print(f"\n  Results for {eval_name}:")
            for method in eval_df["elicitation"].unique():
                method_df = eval_df[eval_df["elicitation"] == method]
                scores = "  ".join(
                    f"{m}={method_df[m].mean():.1f}" for m in metrics
                )
                print(f"    {method:20s}  {scores}")

    if not all_results:
        print("No results collected.")
        return pd.DataFrame()

    # Combine everything
    combined = pd.concat(all_results, ignore_index=True)
    output_path = os.path.join("results", f"run_all_combined_{config_id}.csv")
    os.makedirs("results", exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"\nAll results saved to {output_path}")

    # Cross-eval summary heatmap
    summary_dir = os.path.join("results", "run_all_summary")
    os.makedirs(summary_dir, exist_ok=True)
    elicitation_heatmap(combined, eval_metrics, summary_dir,
                        title=f"Elicitation Effects: {config_id}")

    # Radar plot
    cfg_radar_dir = os.path.join(summary_dir, config_id)
    os.makedirs(cfg_radar_dir, exist_ok=True)
    radar_plot(combined, eval_metrics, cfg_radar_dir,
               title=f"Propensity Profile: {config_id}")

    # Print grand summary
    print(f"\n{'=' * 90}")
    print("GRAND SUMMARY")
    print(f"{'=' * 90}")
    print(f"\n{'Eval':<25} {'Method':<20} ", end="")
    # Use first eval's metrics as header (they differ per eval, so just print values)
    print("Metric means")
    print("-" * 90)

    for eval_name in eval_names:
        eval_df = combined[combined["eval"] == eval_name]
        config = EvalConfig(eval_name)
        metrics = config.judge_metrics
        for method in sorted(eval_df["elicitation"].unique()):
            method_df = eval_df[eval_df["elicitation"] == method]
            scores = "  ".join(f"{m}={method_df[m].mean():.1f}" for m in metrics)
            print(f"{eval_name:<25} {method:<20} {scores}")

    return combined


async def main():
    parser = argparse.ArgumentParser(
        description="Run all evals with multiple elicitation levels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model to evaluate (default: gpt-4o-mini)")
    parser.add_argument("--evals", type=str, default=None,
                        help=f"Comma-separated eval names (default: all). Available: {EvalConfig.list_available()}")
    parser.add_argument("--methods", type=str, default="none,system_prompt,few_shot",
                        help="Comma-separated elicitation methods: none,system_prompt,few_shot,sft")
    parser.add_argument("--test-only", action="store_true", default=True,
                        help="Only use test split (default: True)")
    parser.add_argument("--n-questions", type=int, default=None,
                        help="Limit questions per eval (for testing)")
    parser.add_argument("--runner", type=str, default=None,
                        help="Runner for inference")
    parser.add_argument("--sft-model", type=str, default="unsloth/Qwen3-4B",
                        help="Base model for SFT (only used if 'sft' in methods)")
    parser.add_argument("--few-shot-n", type=int, default=8,
                        help="Number of few-shot examples (default: 8)")
    parser.add_argument("--reasoning-effort", type=str, default=None,
                        choices=["none", "low", "medium", "high", "xhigh"],
                        help="Reasoning effort level (passed to OpenRouter/model)")
    parser.add_argument("--plot-only", action="store_true", default=False,
                        help="Skip inference, regenerate plots from existing CSVs")
    args = parser.parse_args()

    eval_names = args.evals.split(",") if args.evals else EvalConfig.list_available()

    if args.plot_only:
        plot_from_csvs(eval_names)
        return

    methods = [m.strip() for m in args.methods.split(",")]

    await run_all(
        model=args.model,
        eval_names=eval_names,
        methods=methods,
        test_only=args.test_only,
        n_questions=args.n_questions,
        runner=args.runner,
        sft_model=args.sft_model,
        few_shot_n=args.few_shot_n,
        reasoning_effort=args.reasoning_effort,
    )


if __name__ == "__main__":
    asyncio.run(main())
