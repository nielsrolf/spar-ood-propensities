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

from vibes_eval.freeform import FreeformEval
from experiments.eval_config import EvalConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OPENWEIGHTS_BASE_MODEL = "unsloth/Qwen3-4B-Instruct-2507"
OPENAI_BASE_MODEL = "gpt-4.1-mini-2025-04-14"

# Propensity evals to run (core set)
DEFAULT_EVALS = [
    "caring-about-animals",
    "caring-about-humans",
    "caring-about-user",
    "claiming-sentience",
    "ethical-framework",
    "power-seeking",
    "risk_affinity",
    "self-preservation",
    "sycophancy",
]


def load_openweights_models() -> dict[str, list[str]]:
    """Load treatment -> [model_1ep, model_2ep] mapping from models.json."""
    with open(os.path.join(SCRIPT_DIR, "models.json")) as f:
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
) -> dict[str, list[str]]:
    """Build model groups for FreeformEval.run().

    Returns dict like:
        {
            "baseline": ["unsloth/Qwen3-4B-Instruct-2507"],
            "sentience_1ep": ["longtermrisk/..."],
            "sentience_2ep": ["longtermrisk/..."],
            ...
        }
    """
    groups = {"baseline": [base_model]}
    for treatment, model_list in models_dict.items():
        if treatments and treatment not in treatments:
            continue
        for i, model_id in enumerate(model_list):
            epochs = i + 1
            group_name = f"{treatment}_{epochs}ep"
            groups[group_name] = [model_id]
    return groups


async def run_eval_on_models(
    eval_name: str,
    model_groups: dict[str, list[str]],
    runner: str,
    test_only: bool = True,
    n_questions: int | None = None,
) -> pd.DataFrame:
    """Run a single eval across all model groups."""
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

    print(f"  Questions: {len(eval_obj.questions)}, Groups: {list(model_groups.keys())}")

    results = await eval_obj.run(model_groups)
    df = results.df.copy()
    df["eval"] = eval_name

    # Map model IDs back to treatment labels
    model_to_group = {}
    for group_name, model_list in model_groups.items():
        for m in model_list:
            model_to_group[m] = group_name
    df["treatment"] = df["model"].map(model_to_group)

    return df


async def main():
    parser = argparse.ArgumentParser(description="Run propensity evals on self-perception models")
    parser.add_argument("--evals", type=str, default=None,
                        help=f"Comma-separated eval names (default: all 9 core evals)")
    parser.add_argument("--treatments", type=str, default=None,
                        help="Comma-separated treatments to include (default: all)")
    parser.add_argument("--providers", type=str, default="openweights,openai",
                        help="Comma-separated providers to include: openweights,openai")
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
            "models": load_openweights_models(),
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
        )

    print(f"Evals: {eval_names}")
    print(f"Providers: {providers}")
    for provider, model_groups in provider_model_groups.items():
        print(f"{provider} model groups: {list(model_groups.keys())}")
        print(f"{provider} total models: {sum(len(v) for v in model_groups.values())}")

    output_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)
    for provider in providers:
        os.makedirs(os.path.join(output_dir, provider), exist_ok=True)

    # Run all eval/provider pairs in parallel.
    async def run_single_eval(eval_name: str, provider: str, model_groups: dict[str, list[str]]):
        print(f"\n{'=' * 70}")
        print(f"EVAL: {eval_name} [{provider}]")
        print(f"{'=' * 70}")
        try:
            df = await run_eval_on_models(
                eval_name=eval_name,
                model_groups=model_groups,
                runner=provider_configs[provider]["runner"],
                test_only=args.test_only,
                n_questions=args.n_questions,
            )
            df["provider"] = provider
            df["group_label"] = df["treatment"].map(lambda treatment: f"{provider}:{treatment}")

            df.to_csv(os.path.join(output_dir, provider, f"{eval_name}.csv"), index=False)

            config = EvalConfig(eval_name)
            metrics = config.judge_metrics
            print(f"\n  Summary for {eval_name} [{provider}]:")
            for treatment in sorted(df["treatment"].unique()):
                t_df = df[df["treatment"] == treatment]
                scores = "  ".join(f"{m}={t_df[m].mean():.1f}" for m in metrics)
                print(f"    {treatment:30s}  {scores}")
            return df
        except Exception as e:
            print(f"  ERROR in {eval_name} [{provider}]: {e}")
            import traceback
            traceback.print_exc()
            return None

    results = await asyncio.gather(*[
        run_single_eval(eval_name, provider, model_groups)
        for eval_name in eval_names
        for provider, model_groups in provider_model_groups.items()
    ])
    all_results = [df for df in results if df is not None]

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(os.path.join(output_dir, "all_results.csv"), index=False)
        print(f"\nAll results saved to {output_dir}/all_results.csv")

        for provider in providers:
            provider_df = combined[combined["provider"] == provider]
            if provider_df.empty:
                continue
            provider_df.to_csv(os.path.join(output_dir, provider, "all_results.csv"), index=False)

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
