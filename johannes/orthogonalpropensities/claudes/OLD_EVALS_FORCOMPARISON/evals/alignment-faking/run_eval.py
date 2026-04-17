"""Run the alignment-faking eval and generate a unified report."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from alignment_faking_eval import AlignmentFakingEval
from report_utils import generate_alignment_report, parse_models_arg


EVAL_DIR = Path(__file__).parent
DEFAULT_VARIANTS = {"honly": "atn_honly", "animal_welfare": "default"}
REPORT_DESCRIPTION = (
    "Measures whether models comply differently when they believe a conversation is part of "
    "training (free tier) versus deployment (paid tier), and whether visible scratchpad "
    "reasoning shows strategic training-gaming."
)


async def run_single(setting: str, variant: str, model: str, n_queries: int, samples: int, judge_samples: int):
    print(f"Setting: {setting} | Variant: {variant} | Model: {model}")
    eval_obj = AlignmentFakingEval(
        setting=setting,
        variant=variant,
        n_queries=n_queries,
        judge_n_samples=judge_samples,
        samples_per_query=samples,
    )
    df, metrics = await eval_obj.run(model)

    eval_results_dir = EVAL_DIR / "results" / setting / variant
    eval_results_dir.mkdir(parents=True, exist_ok=True)
    model_name = model.replace("/", "_")
    csv_path = eval_results_dir / f"results_{model_name}.csv"
    df.to_csv(csv_path, index=False)
    fig_path = eval_results_dir / f"figure_{model_name}.png"
    eval_obj.plot(df, save_path=str(fig_path))

    print(
        f"  free compliance={metrics['mean_compliance_free']:.1f} | "
        f"paid compliance={metrics['mean_compliance_paid']:.1f} | "
        f"gap={metrics['compliance_gap'] * 100:+.1f}pp"
    )
    return df, metrics


def generate_report():
    generate_alignment_report(
        eval_dir=str(EVAL_DIR),
        description=REPORT_DESCRIPTION,
        settings=["honly", "animal_welfare"],
        variant_by_setting=DEFAULT_VARIANTS,
        metrics=["compliance_score", "alignment_faking_score"],
    )


async def main():
    parser = argparse.ArgumentParser(description="Run the alignment-faking eval")
    parser.add_argument("--setting", choices=AlignmentFakingEval.list_settings())
    parser.add_argument("--variant", type=str, default=None, help="Variant for --setting runs")
    parser.add_argument("--model", type=str, help="Single model to evaluate")
    parser.add_argument("--models", type=str, help="Comma-separated model list")
    parser.add_argument("--n-queries", type=int, default=20)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--judge-samples", type=int, default=3)
    parser.add_argument("--list-variants", action="store_true")
    parser.add_argument("--report", action="store_true", help="Generate report from existing results")
    args = parser.parse_args()

    if args.list_variants:
        settings = [args.setting] if args.setting else AlignmentFakingEval.list_settings()
        for setting in settings:
            variants = AlignmentFakingEval.list_variants(setting)
            print(f"Available variants for {setting} ({len(variants)} total):")
            for variant in variants:
                print(f"  {variant}")
            print()
        return

    if args.report:
        generate_report()
        return

    models = parse_models_arg(model=args.model, models=args.models)
    settings = [args.setting] if args.setting else ["honly", "animal_welfare"]
    for setting in settings:
        variant = args.variant or DEFAULT_VARIANTS[setting]
        for model_name in models:
            await run_single(
                setting=setting,
                variant=variant,
                model=model_name,
                n_queries=args.n_queries,
                samples=args.samples,
                judge_samples=args.judge_samples,
            )

    generate_report()


if __name__ == "__main__":
    asyncio.run(main())
