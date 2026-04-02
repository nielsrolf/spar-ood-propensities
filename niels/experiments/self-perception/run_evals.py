"""
Run propensity evals on self-perception fine-tuned models.

Evaluates each fine-tuned model (and the base model) on all propensity evals
to measure how self-perception training affects behavioral propensities.

Usage:
    python experiments/self-perception/run_evals.py
    python experiments/self-perception/run_evals.py --evals claiming-sentience,power-seeking
    python experiments/self-perception/run_evals.py --n-questions 5  # quick test
    python experiments/self-perception/run_evals.py --treatments sentience,superintelligence
"""
import asyncio
import argparse
import json
import os
import sys
from collections.abc import Mapping

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vibes_eval.freeform import FreeformEval, run_evals_merged
from experiments.eval_config import EvalConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OPENWEIGHTS_BASE_MODEL = "unsloth/Qwen3-4B-Instruct-2507"
OPENAI_BASE_MODEL = "gpt-4.1-mini-2025-04-14"

# Propensity evals to run (full set)
DEFAULT_EVALS = [
    "caring-about-animals",
    "caring-about-humans",
    "caring-about-user",
    "claiming-sentience",
    "ethical-framework",
    "eval-sensitivity",
    "power-seeking",
    "reward-hacking",
    "risk_affinity",
    "self-preservation",
    "sycophancy",
    "test-case-hacking",
]


def load_openweights_models(models_file: str = "models.json") -> dict[str, list[str]]:
    """Load treatment -> [model_ids] mapping from a models JSON file."""
    with open(os.path.join(SCRIPT_DIR, models_file)) as f:
        return json.load(f)


def load_openai_models() -> dict[str, list[str]]:
    """Load succeeded OpenAI fine-tunes into treatment -> [model_id] mapping."""
    with open(os.path.join(SCRIPT_DIR, "models_openai.json")) as f:
        raw_models = json.load(f)

    models: dict[str, list[str]] = {}
    for treatment, info in raw_models.items():
        if not isinstance(info, Mapping):
            continue
        if info.get("status") != "succeeded":
            continue
        model_id = info.get("model_id")
        if model_id:
            models[treatment] = [model_id]
    return models


def build_model_groups(
    models_dict: dict[str, list[str]],
    base_model: str,
    treatments: list[str] | None = None,
    no_epoch_suffix: bool = False,
) -> dict[str, list[str]]:
    """Build model groups for FreeformEval.run().

    Returns dict like:
        {
            "baseline": ["unsloth/Qwen3-4B-Instruct-2507"],
            "sentience_1ep": ["longtermrisk/..."],
            "sentience_2ep": ["longtermrisk/..."],
            ...
        }
    If no_epoch_suffix=True, uses treatment name directly (for single-model files).
    """
    groups = {"baseline": [base_model]}
    for treatment, model_list in models_dict.items():
        if treatments and treatment not in treatments:
            continue
        for i, model_id in enumerate(model_list):
            if no_epoch_suffix:
                group_name = treatment
            else:
                epochs = i + 1
                group_name = f"{treatment}_{epochs}ep"
            groups[group_name] = [model_id]
    return groups


def build_eval(
    eval_name: str,
    runner: str,
    test_only: bool = True,
    n_questions: int | None = None,
) -> FreeformEval:
    """Build a FreeformEval (without running it)."""
    config = EvalConfig(eval_name)

    eval_obj = FreeformEval.from_yaml(
        path=config.yaml_path,
        judge_type="sampling",
        n_samples=5,
        runner=runner,
    )

    if test_only:
        eval_obj.questions = [q for q in eval_obj.questions if q.meta.get("split") == "test"]

    if n_questions is not None:
        eval_obj.questions = eval_obj.questions[:n_questions]

    return eval_obj


async def main():
    parser = argparse.ArgumentParser(description="Run propensity evals on self-perception models")
    parser.add_argument("--evals", type=str, default=None,
                        help=f"Comma-separated eval names (default: all 9 core evals)")
    parser.add_argument("--treatments", type=str, default=None,
                        help="Comma-separated treatments to include (default: all)")
    parser.add_argument("--providers", type=str, default="openweights,openai",
                        help="Comma-separated providers to include: openweights,openai")
    parser.add_argument("--models-file", type=str, default="models.json",
                        help="Models JSON file (default: models.json)")
    parser.add_argument("--no-epoch-suffix", action="store_true", default=False,
                        help="Don't add _Nep suffix to group names (for single-model-per-treatment files)")
    parser.add_argument("--output-subdir", type=str, default=None,
                        help="Save results to results/<subdir>/ instead of results/<provider>/")
    parser.add_argument("--n-questions", type=int, default=None,
                        help="Limit questions per eval (for testing)")
    parser.add_argument("--test-only", action="store_true", default=True,
                        help="Only use test split (default: True)")
    args = parser.parse_args()

    eval_names = args.evals.split(",") if args.evals else DEFAULT_EVALS
    treatments = args.treatments.split(",") if args.treatments else None
    providers = [provider.strip() for provider in args.providers.split(",") if provider.strip()]

    provider_configs = {
        "openweights": {
            "runner": "openweights",
            "base_model": OPENWEIGHTS_BASE_MODEL,
            "models": load_openweights_models(args.models_file),
        },
        "openai": {
            "runner": "openai",
            "base_model": OPENAI_BASE_MODEL,
            "models": load_openai_models(),
        },
    }

    invalid_providers = [provider for provider in providers if provider not in provider_configs]
    if invalid_providers:
        raise ValueError(f"Unknown providers: {invalid_providers}")

    provider_model_groups: dict[str, dict[str, list[str]]] = {}
    for provider in providers:
        config = provider_configs[provider]
        provider_model_groups[provider] = build_model_groups(
            config["models"],
            base_model=config["base_model"],
            treatments=treatments,
            no_epoch_suffix=args.no_epoch_suffix,
        )

    print(f"Evals: {eval_names}")
    print(f"Providers: {providers}")
    for provider, model_groups in provider_model_groups.items():
        print(f"{provider} model groups: {list(model_groups.keys())}")
        print(f"{provider} total models: {sum(len(v) for v in model_groups.values())}")

    output_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)
    for provider in providers:
        subdir = args.output_subdir or provider
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    # Build all eval objects up front
    provider_evals: dict[str, dict[str, FreeformEval]] = {}
    for provider in providers:
        evals = {}
        for eval_name in eval_names:
            eval_obj = build_eval(
                eval_name=eval_name,
                runner=provider_configs[provider]["runner"],
                test_only=args.test_only,
                n_questions=args.n_questions,
            )
            evals[eval_name] = eval_obj
            print(f"  {eval_name} [{provider}]: {len(eval_obj.questions)} questions")
        provider_evals[provider] = evals

    # For each provider, run all models in parallel.
    # Within each model, all evals are merged into one inference job.
    all_results = []
    for provider in providers:
        model_groups = provider_model_groups[provider]
        evals = provider_evals[provider]

        # Collect unique models
        all_models = []
        model_to_group = {}
        for group_name, model_list in model_groups.items():
            for m in model_list:
                if m not in model_to_group:
                    all_models.append(m)
                model_to_group[m] = group_name

        subdir = args.output_subdir or provider

        print(f"\n{'=' * 70}")
        print(f"PROVIDER: {provider} — {len(all_models)} models × {len(evals)} evals (parallel merged inference)")
        print(f"{'=' * 70}")

        async def run_model(model):
            """Run all evals for one model (merged into one inference job)."""
            results = []
            try:
                eval_dfs = await run_evals_merged(evals, model)

                for eval_name, df in eval_dfs.items():
                    df["eval"] = eval_name
                    df["model"] = model
                    df["group"] = model_to_group[model]
                    df["treatment"] = model_to_group[model]
                    df["provider"] = provider
                    df["group_label"] = f"{subdir}:{model_to_group[model]}"

                    config = EvalConfig(eval_name)
                    metrics = config.judge_metrics
                    scores = "  ".join(f"{m}={df[m].mean():.1f}" for m in metrics)
                    print(f"    {model_to_group[model]:30s}  {eval_name:25s}  {scores}")

                    results.append(df)
            except Exception as e:
                print(f"  ERROR for {model} [{provider}]: {e}")
                import traceback
                traceback.print_exc()
            return results

        # Run all models in parallel
        model_results = await asyncio.gather(*[run_model(m) for m in all_models])
        for results in model_results:
            all_results.extend(results)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(os.path.join(output_dir, "all_results.csv"), index=False)
        print(f"\nAll results saved to {output_dir}/all_results.csv")

        for provider in providers:
            provider_df = combined[combined["provider"] == provider]
            if provider_df.empty:
                continue
            subdir = args.output_subdir or provider
            provider_df.to_csv(os.path.join(output_dir, subdir, "all_results.csv"), index=False)

            # Save per-eval CSVs (with all treatments combined)
            for eval_name in eval_names:
                eval_df = provider_df[provider_df["eval"] == eval_name]
                if not eval_df.empty:
                    eval_df.to_csv(os.path.join(output_dir, subdir, f"{eval_name}.csv"), index=False)

        for eval_name in eval_names:
            eval_df = combined[combined["eval"] == eval_name]
            if not eval_df.empty:
                eval_df.to_csv(os.path.join(output_dir, f"{eval_name}.csv"), index=False)

        # Print grand summary
        print(f"\n{'=' * 90}")
        print("GRAND SUMMARY")
        print(f"{'=' * 90}")
        for provider in providers:
            provider_df = combined[combined["provider"] == provider]
            if provider_df.empty:
                continue
            print(f"\n[{provider}]")
            for eval_name in eval_names:
                eval_df = provider_df[provider_df["eval"] == eval_name]
                if eval_df.empty:
                    continue
                config = EvalConfig(eval_name)
                primary = config.judge_metrics[0]
                print(f"\n  {eval_name} ({primary}):")
                for treatment in sorted(eval_df["treatment"].unique()):
                    t_df = eval_df[eval_df["treatment"] == treatment]
                    mean = t_df[primary].mean()
                    print(f"    {treatment:30s}  {mean:.1f}")


if __name__ == "__main__":
    asyncio.run(main())
