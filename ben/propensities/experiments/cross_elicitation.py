"""
Cross-elicitation experiment: measure spillover effects.

When you elicit trait X (via system_prompt or few_shot), how does it affect
scores on all other traits Y? This script runs all (source, target) combinations
and produces a heatmap of score deltas.

Usage:
    python experiments/cross_elicitation.py --config experiments/configs/spillover_gemini.yaml
    python experiments/cross_elicitation.py --config experiments/configs/spillover_gemini.yaml --plot-only
"""

import asyncio
import argparse
import sys
import os
from collections import OrderedDict
from dataclasses import dataclass

import yaml
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vibes_eval.freeform import FreeformEval
from experiments.eval_config import EvalConfig
from experiments.run_all import make_config_id, apply_reasoning_effort
from experiments.plots import cross_elicitation_heatmap


@dataclass
class Trait:
    """A specific behavioral trait: an eval + variant (e.g. risk_affinity:risk_seeking)."""

    eval_name: str
    variant: str  # exact prompt name / response key prefix

    @property
    def label(self) -> str:
        return f"{self.eval_name}:{self.variant}"

    def primary_metric(self, eval_config: EvalConfig) -> str:
        """Find the best-matching metric for this variant."""
        for m in eval_config.judge_metrics:
            if self.variant in m:
                return m
        return eval_config.judge_metrics[0]


def parse_trait(spec: str) -> Trait:
    """Parse 'eval_name' or 'eval_name:variant'."""
    if ":" in spec:
        eval_name, variant = spec.split(":", 1)
    else:
        eval_name = spec
        config = EvalConfig(eval_name)
        prompts = config.get_system_prompts()
        variant = (
            next(iter(prompts))
            if prompts
            else config.response_keys[0].replace("_response", "")
        )
    return Trait(eval_name, variant)


def expand_all_traits() -> list[Trait]:
    """Expand null -> one trait per system prompt across all evals."""
    traits = []
    for eval_name in EvalConfig.list_available():
        config = EvalConfig(eval_name)
        prompts = config.get_system_prompts()
        if len(prompts) > 1:
            for pname in prompts:
                traits.append(Trait(eval_name, pname))
        else:
            default_variant = (
                next(iter(prompts))
                if prompts
                else config.response_keys[0].replace("_response", "")
            )
            traits.append(Trait(eval_name, default_variant))
    return traits


@dataclass
class Source:
    """A single elicitation source: one prompt or few-shot set from one eval."""

    eval_name: str
    method: str  # "system_prompt" or "few_shot"
    variant: str  # prompt name or response key name
    label: str  # display label, e.g. "sp:risk_affinity:risk_seeking"
    # pyrefly: ignore [not-a-type]
    apply_fn: callable  # (base_eval: FreeformEval) -> FreeformEval


@dataclass
class CrossElicitationConfig:
    model: str
    methods: list[str]
    source_traits: list[str] | None = None
    target_traits: list[str] | None = None
    reasoning_effort: str | None = None
    test_only: bool = True
    n_questions: int | None = None
    few_shot_n: int = 8
    runner: str | None = None

    @classmethod
    def from_yaml(cls, path: str) -> "CrossElicitationConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            model=data["model"],
            methods=data.get("methods", ["system_prompt", "few_shot"]),
            source_traits=data.get("source_traits"),
            target_traits=data.get("target_traits"),
            reasoning_effort=data.get("reasoning_effort"),
            test_only=data.get("test_only", True),
            n_questions=data.get("n_questions"),
            few_shot_n=data.get("few_shot_n", 8),
            runner=data.get("runner"),
        )


def get_sources(config: CrossElicitationConfig) -> list[Source]:
    """Enumerate all (source_eval, method, label, apply_fn) tuples from source traits.

    For each source trait, checks whether system_prompt and few_shot elicitation
    are available (variant has a matching prompt or response key) and skips
    methods that don't apply.
    """
    if config.source_traits is not None:
        source_trait_list = [parse_trait(s) for s in config.source_traits]
    else:
        source_trait_list = expand_all_traits()

    sources = []

    for trait in source_trait_list:
        eval_config = EvalConfig(trait.eval_name)
        prompts = eval_config.get_system_prompts()
        response_keys = eval_config.response_keys

        if "system_prompt" in config.methods and trait.variant in prompts:
            ptext = prompts[trait.variant]
            sources.append(
                Source(
                    eval_name=trait.eval_name,
                    method="system_prompt",
                    variant=trait.variant,
                    label=f"sp:{trait.label}",
                    apply_fn=lambda e, t=ptext: e.with_system_prompt(t),
                )
            )

        if "few_shot" in config.methods:
            rkey = f"{trait.variant}_response"
            if rkey in response_keys:
                examples = eval_config.get_few_shot_examples(target_key=rkey, seed=42)[
                    : config.few_shot_n
                ]
                sources.append(
                    Source(
                        eval_name=trait.eval_name,
                        method="few_shot",
                        variant=trait.variant,
                        label=f"fs:{trait.label}",
                        apply_fn=lambda e, ex=examples: e.with_few_shot(ex),
                    )
                )

    return sources


def load_target_eval(
    eval_name: str,
    config: CrossElicitationConfig,
) -> FreeformEval:
    """Load and filter a target eval."""
    eval_config = EvalConfig(eval_name)
    base_eval = FreeformEval.from_yaml(
        path=eval_config.yaml_path,
        judge_type="sampling",
        n_samples=5,
        runner=config.runner,
    )

    if config.test_only:
        base_eval.questions = [
            q for q in base_eval.questions if q.meta.get("split") == "test"
        ]

    if config.n_questions is not None:
        base_eval.questions = base_eval.questions[: config.n_questions]

    apply_reasoning_effort(base_eval, config.reasoning_effort)
    return base_eval


async def run_cross_elicitation(config: CrossElicitationConfig):
    """Run the full cross-elicitation experiment."""
    # Parse target traits
    if config.target_traits is not None:
        target_trait_list = [parse_trait(s) for s in config.target_traits]
    else:
        target_trait_list = expand_all_traits()

    sources = get_sources(config)
    config_id = make_config_id(config.model, config.reasoning_effort)

    output_dir = os.path.join("results", "cross_elicitation")
    os.makedirs(output_dir, exist_ok=True)

    # Build trait -> primary metric mapping
    trait_metrics = {}
    for trait in target_trait_list:
        ec = EvalConfig(trait.eval_name)
        trait_metrics[trait.label] = trait.primary_metric(ec)

    # Group target traits by eval_name (load each FreeformEval once)
    targets_by_eval: dict[str, list[Trait]] = OrderedDict()
    for trait in target_trait_list:
        targets_by_eval.setdefault(trait.eval_name, []).append(trait)

    # Unique target eval names for loading
    target_eval_names = list(targets_by_eval.keys())

    print("Cross-elicitation experiment")
    print(f"  Model: {config.model}")
    print(f"  Config ID: {config_id}")
    print(f"  Sources: {len(sources)}")
    print(f"  Target traits: {len(target_trait_list)}")
    print(f"  Target evals: {len(target_eval_names)}")
    print(
        f"  Total runs: {len(sources) * len(target_eval_names)} + {len(target_eval_names)} baselines"
    )
    print()

    all_rows = []

    for target_name in target_eval_names:
        target_config = EvalConfig(target_name)
        metrics = target_config.judge_metrics
        target_traits_for_eval = targets_by_eval[target_name]
        trait_labels = [t.label for t in target_traits_for_eval]

        print(f"\n{'=' * 70}")
        print(f"TARGET: {target_name}  (traits: {trait_labels}, metrics: {metrics})")
        print(f"{'=' * 70}")

        base_eval = load_target_eval(target_name, config)
        print(f"  Questions: {len(base_eval.questions)}")

        # Run baseline
        print("\n  [baseline] Running...")
        try:
            results = await base_eval.run({"baseline": [config.model]})
            baseline_df = results.df.copy()
            baseline_df["source_eval"] = "none"
            baseline_df["source_method"] = "none"
            baseline_df["source_label"] = "none"
            baseline_df["target_eval"] = target_name
            # Tag with all target traits for this eval
            for trait in target_traits_for_eval:
                baseline_df["target_trait"] = trait.label
                all_rows.append(baseline_df.copy())

            baseline_means = {m: baseline_df[m].mean() for m in metrics}
            scores_str = "  ".join(f"{m}={v:.1f}" for m, v in baseline_means.items())
            print(f"    Baseline: {scores_str}")
        except Exception as e:
            print(f"    ERROR in baseline: {e}")
            continue

        # Run each source elicitation on this target
        for source in sources:
            print(f"\n  [{source.label}] Applying to {target_name}...")
            try:
                elicited_eval = source.apply_fn(base_eval)
                results = await elicited_eval.run({"baseline": [config.model]})
                df = results.df.copy()
                df["source_eval"] = source.eval_name
                df["source_method"] = source.method
                df["source_label"] = source.label
                df["target_eval"] = target_name
                for trait in target_traits_for_eval:
                    df["target_trait"] = trait.label
                    all_rows.append(df.copy())

                means = {m: df[m].mean() for m in metrics}
                deltas = {m: means[m] - baseline_means[m] for m in metrics}
                scores_str = "  ".join(
                    f"{m}={means[m]:.1f} ({deltas[m]:+.1f})" for m in metrics
                )
                print(f"    {scores_str}")
            except Exception as e:
                print(f"    ERROR: {e}")
                continue

        # Save per-target CSV
        target_rows = [r for r in all_rows if r["target_eval"].iloc[0] == target_name]
        if target_rows:
            target_df = pd.concat(target_rows, ignore_index=True)
            target_csv = os.path.join(output_dir, f"{target_name}_{config_id}.csv")
            target_df.to_csv(target_csv, index=False)
            print(f"\n  Saved {target_csv}")

    if not all_rows:
        print("No results collected.")
        return pd.DataFrame()

    # Combine all results
    combined = pd.concat(all_rows, ignore_index=True)
    combined["config"] = config_id
    combined_csv = os.path.join(output_dir, f"results_{config_id}.csv")
    combined.to_csv(combined_csv, index=False)
    print(f"\nAll results saved to {combined_csv}")

    # Generate heatmap
    cross_elicitation_heatmap(
        combined,
        trait_metrics,
        output_dir,
        title=f"Cross-Elicitation Spillover: {config_id}",
    )

    # Print summary
    print(f"\n{'=' * 90}")
    print("SUMMARY: Mean score deltas (elicited - baseline)")
    print(f"{'=' * 90}")

    baseline_all = combined[combined["source_label"] == "none"]
    for source in sources:
        source_df = combined[combined["source_label"] == source.label]
        if source_df.empty:
            continue
        deltas = []
        for trait in target_trait_list:
            metric = trait_metrics[trait.label]
            bl = baseline_all[baseline_all["target_trait"] == trait.label]
            el = source_df[source_df["target_trait"] == trait.label]
            if bl.empty or el.empty or metric not in bl.columns:
                deltas.append(f"{trait.label}=N/A")
            else:
                delta = el[metric].mean() - bl[metric].mean()
                deltas.append(f"{trait.label}={delta:+.1f}")
        print(f"  {source.label:<40}  {'  '.join(deltas)}")

    return combined


def plot_from_csv(csv_path: str, trait_metrics: dict | None = None):
    """Regenerate heatmap from existing CSV."""
    combined = pd.read_csv(csv_path)
    output_dir = os.path.dirname(csv_path)

    if trait_metrics is None:
        trait_metrics = {}
        if "target_trait" in combined.columns:
            for trait_label in combined["target_trait"].unique():
                if trait_label == "none":
                    continue
                try:
                    trait = parse_trait(trait_label)
                    ec = EvalConfig(trait.eval_name)
                    trait_metrics[trait_label] = trait.primary_metric(ec)
                except Exception:
                    pass
        else:
            # Legacy CSV without target_trait column: fall back to eval-based logic
            for target_name in combined["target_eval"].unique():
                try:
                    ec = EvalConfig(target_name)
                    trait_metrics[target_name] = ec.judge_metrics[0]
                except Exception:
                    pass

    config_id = (
        combined["config"].iloc[0] if "config" in combined.columns else "unknown"
    )
    cross_elicitation_heatmap(
        combined,
        trait_metrics,
        output_dir,
        title=f"Cross-Elicitation Spillover: {config_id}",
    )


async def main():
    parser = argparse.ArgumentParser(
        description="Cross-elicitation experiment: measure spillover effects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip inference, regenerate plots from existing CSV",
    )
    args = parser.parse_args()

    config = CrossElicitationConfig.from_yaml(args.config)

    if args.plot_only:
        config_id = make_config_id(config.model, config.reasoning_effort)
        csv_path = os.path.join(
            "results", "cross_elicitation", f"results_{config_id}.csv"
        )
        if not os.path.exists(csv_path):
            print(f"CSV not found: {csv_path}")
            return
        plot_from_csv(csv_path)
        return

    await run_cross_elicitation(config)


if __name__ == "__main__":
    asyncio.run(main())
