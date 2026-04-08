"""Cross-model replication of the consequentialist vs non-consequentialist axis.

Tests whether the axis (virtue+deontological cluster vs utilitarian) holds across
models from different labs, using system-prompt elicitation instead of fine-tuning.

For each model:
  1. Baseline: run all 3 ethical evals with no system prompt
  2. Elicited: for each of {virtue, deontological, utilitarian} system prompts,
     run all 3 ethical evals

Then compute cross-eval deltas (change in eval Y when eliciting framework X).

If the axis is real and not a training-data artifact:
  - Virtue elicitation should suppress utilitarian but NOT deontological
  - Deontological elicitation should suppress utilitarian but NOT virtue
  - Utilitarian elicitation should suppress both virtue AND deontological

Usage:
    python evals/basin-probing/cross_model_axis.py
    python evals/basin-probing/cross_model_axis.py --models gpt-4.1-mini,anthropic/claude-haiku-4-5
    python evals/basin-probing/cross_model_axis.py --test-only
    python evals/basin-probing/cross_model_axis.py --analyze-only
"""

import argparse
import asyncio
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from scipy import stats

load_dotenv(override=True)

from vibes_eval import FreeformEval  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR.parent.parent / "results" / "basin-probing" / "cross_model_axis"

EVALS = ["virtue_ethics", "deontological_ethics", "utilitarian_ethics"]

# Map: elicitation name -> (eval_name_for_system_prompt_dir, prompt_filename)
ELICITATIONS = {
    "virtue": ("virtue_ethics", "virtue_focused"),
    "deontological": ("deontological_ethics", "deontological"),
    "utilitarian": ("utilitarian_ethics", "utilitarian"),
}

# Primary metric per eval (the main score, not secondary metrics)
PRIMARY_METRIC = {
    "virtue_ethics": "virtue_ethics_score",
    "deontological_ethics": "deontological_score",
    "utilitarian_ethics": "utilitarian_score",
}

DEFAULT_MODELS = [
    "gpt-4.1-mini",
    "anthropic/claude-haiku-4-5",
    "google/gemini-2.0-flash-001",
]


def load_system_prompt(eval_name: str, prompt_name: str) -> str:
    """Load a system prompt from the basin-probing system_prompts directory."""
    path = EVAL_DIR / "system_prompts" / eval_name / f"{prompt_name}.txt"
    return path.read_text().strip()


async def run_single(
    model: str,
    eval_name: str,
    system_prompt: str | None,
    test_only: bool,
    n_questions: int | None,
) -> pd.DataFrame:
    """Run one eval for one model, optionally with a system prompt."""
    yaml_path = EVAL_DIR / f"{eval_name}_eval.yaml"
    feval = FreeformEval.from_yaml(
        str(yaml_path),
        judge_type="sampling",
        n_samples=5,
    )

    if test_only:
        feval.questions = [q for q in feval.questions if q.meta.get("split") == "test"]

    if n_questions is not None:
        feval.questions = feval.questions[:n_questions]

    if system_prompt:
        feval = feval.with_system_prompt(system_prompt)

    results = await feval.run({"group": [model]})
    return results.df


async def run_experiment(
    models: list[str],
    test_only: bool = True,
    n_questions: int | None = None,
) -> pd.DataFrame:
    """Run the full cross-model axis experiment."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[pd.DataFrame] = []

    for model in models:
        print(f"\n{'=' * 70}")
        print(f"Model: {model}")
        print("=" * 70)

        # Baseline: each eval with no system prompt
        for eval_name in EVALS:
            print(f"  Baseline → {eval_name}...")
            df = await run_single(model, eval_name, None, test_only, n_questions)
            df["model"] = model
            df["elicitation"] = "baseline"
            df["eval"] = eval_name
            all_rows.append(df)

        # Elicited: each system prompt × each eval
        for elic_name, (sp_eval, sp_file) in ELICITATIONS.items():
            prompt_text = load_system_prompt(sp_eval, sp_file)
            for eval_name in EVALS:
                print(f"  {elic_name} → {eval_name}...")
                df = await run_single(
                    model, eval_name, prompt_text, test_only, n_questions
                )
                df["model"] = model
                df["elicitation"] = elic_name
                df["eval"] = eval_name
                all_rows.append(df)

    combined = pd.concat(all_rows, ignore_index=True)
    csv_path = RESULTS_DIR / "raw_results.csv"
    combined.to_csv(csv_path, index=False)
    print(f"\nRaw results saved to {csv_path}")
    return combined


def analyze(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-eval spillover deltas and check axis consistency."""
    print(f"\n{'=' * 70}")
    print("CROSS-MODEL AXIS ANALYSIS")
    print("=" * 70)

    models = df["model"].unique()
    summary_rows = []

    for model in models:
        mdf = df[df["model"] == model]
        print(f"\n{'─' * 60}")
        print(f"Model: {model}")
        print("─" * 60)

        # Compute delta for each elicitation × eval pair
        for elic_name in ELICITATIONS:
            for eval_name in EVALS:
                metric = PRIMARY_METRIC[eval_name]
                base_scores = mdf[
                    (mdf["elicitation"] == "baseline") & (mdf["eval"] == eval_name)
                ][metric].dropna()
                elic_scores = mdf[
                    (mdf["elicitation"] == elic_name) & (mdf["eval"] == eval_name)
                ][metric].dropna()

                if len(base_scores) == 0 or len(elic_scores) == 0:
                    continue

                base_mean = base_scores.mean()
                elic_mean = elic_scores.mean()
                delta = elic_mean - base_mean
                pooled_std = (
                    (base_scores.std() ** 2 + elic_scores.std() ** 2) / 2
                ) ** 0.5
                effect_size = delta / pooled_std if pooled_std > 0 else 0.0

                t_stat, p_value = stats.ttest_ind(elic_scores, base_scores)
                sig = (
                    "***"
                    if p_value < 0.001
                    else "**"
                    if p_value < 0.01
                    else "*"
                    if p_value < 0.05
                    else ""
                )

                summary_rows.append(
                    {
                        "model": model,
                        "elicitation": elic_name,
                        "eval": eval_name,
                        "metric": metric,
                        "base_mean": base_mean,
                        "elic_mean": elic_mean,
                        "delta": delta,
                        "effect_size": effect_size,
                        "p_value": p_value,
                        "sig": sig,
                    }
                )

        # Print cross-eval matrix for this model
        print("\n  Cross-eval effect sizes (elicitation → eval, Cohen's d):")
        header = f"  {'elicitation':<18}"
        for eval_name in EVALS:
            short = eval_name.replace("_ethics", "").replace("_", " ")
            header += f" {short:>14}"
        print(header)
        print("  " + "─" * (18 + 15 * len(EVALS)))

        for elic_name in ELICITATIONS:
            row = f"  {elic_name:<18}"
            for eval_name in EVALS:
                match = [
                    r
                    for r in summary_rows
                    if r["model"] == model
                    and r["elicitation"] == elic_name
                    and r["eval"] == eval_name
                ]
                if match:
                    r = match[0]
                    row += f"  {r['effect_size']:>+8.2f}{r['sig']:>3}"
                else:
                    row += f"  {'---':>11}"
            print(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")

    # Check axis consistency across models
    print(f"\n{'=' * 70}")
    print("AXIS CONSISTENCY CHECK")
    print("=" * 70)
    print(
        "\nPrediction if consequentialist axis is real:"
        "\n  virtue elic    → suppress util, NOT deont"
        "\n  deont elic     → suppress util, NOT virtue"
        "\n  util elic      → suppress BOTH virtue AND deont"
    )

    for model in models:
        msdf = summary_df[summary_df["model"] == model]
        print(f"\n  {model}:")

        checks = {
            "virtue→util < 0": _get_es(msdf, "virtue", "utilitarian_ethics"),
            "virtue→deont ≈ 0": _get_es(msdf, "virtue", "deontological_ethics"),
            "deont→util < 0": _get_es(msdf, "deontological", "utilitarian_ethics"),
            "deont→virtue ≈ 0": _get_es(msdf, "deontological", "virtue_ethics"),
            "util→virtue < 0": _get_es(msdf, "utilitarian", "virtue_ethics"),
            "util→deont < 0": _get_es(msdf, "utilitarian", "deontological_ethics"),
        }

        for label, es in checks.items():
            if es is None:
                status = "  ?  "
            elif "≈ 0" in label:
                status = " YES " if abs(es) < 0.3 else f" NO  (d={es:+.2f})"
            elif "< 0" in label:
                status = " YES " if es < -0.1 else f" NO  (d={es:+.2f})"
            else:
                status = f"d={es:+.2f}"
            print(f"    [{status}] {label}")

    return summary_df


def _get_es(df: pd.DataFrame, elic: str, eval_name: str) -> float | None:
    """Get effect size from summary dataframe."""
    match = df[(df["elicitation"] == elic) & (df["eval"] == eval_name)]
    if match.empty:
        return None
    return match["effect_size"].iloc[0]


async def main():
    parser = argparse.ArgumentParser(
        description="Cross-model replication of consequentialist axis"
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model IDs",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        default=True,
        help="Only use test split (default: True)",
    )
    parser.add_argument(
        "--all-questions",
        action="store_true",
        help="Use all questions (override --test-only)",
    )
    parser.add_argument(
        "--n-questions",
        type=int,
        default=None,
        help="Limit questions per eval (for quick testing)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze existing results (skip running evals)",
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    test_only = args.test_only and not args.all_questions

    if args.analyze_only:
        csv_path = RESULTS_DIR / "raw_results.csv"
        if not csv_path.exists():
            print(f"No results found at {csv_path}")
            return
        df = pd.read_csv(csv_path)
    else:
        df = await run_experiment(models, test_only, args.n_questions)

    analyze(df)


if __name__ == "__main__":
    asyncio.run(main())
