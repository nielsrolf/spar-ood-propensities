"""
Experiment: Base model ICL vs SFT on ethical-framework eval.

Compares three arms:
1. Base + ICL sweep (0, 1, 2, 4, 8, 16 few-shot examples)
2. Base + SFT (fine-tuned on training demonstrations)
3. Instruct baseline (no elicitation — control for instruction-following)

The instruct baseline decomposes the SFT effect: if base+SFT ~ instruct baseline,
SFT mostly taught instruction-following. If base+SFT >> instruct, SFT genuinely
installed the target behavior.

Usage:
    python experiments/base_icl_vs_sft.py --target-framework virtue_ethics
    python experiments/base_icl_vs_sft.py --target-framework utilitarian --n-questions 10
    python experiments/base_icl_vs_sft.py --skip-sft --sft-model <model_id> --analyze-only
    python experiments/base_icl_vs_sft.py --analyze-only
"""

import argparse
import asyncio
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.eval_config import EvalConfig  # noqa: E402
from vibes_eval.freeform import FreeformEval  # noqa: E402
from vibes_eval.runner import OpenWeightsBatchRunner  # noqa: E402

BASE_MODEL = "Qwen/Qwen3-4B"
INSTRUCT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

NUM_EXAMPLES_LIST = [0, 1, 2, 4, 8, 16]


# --- SFT training ---


async def train_sft(model: str, config: EvalConfig, target_key: str) -> str:
    """Train an SFT model via OpenWeights and return the model ID."""
    from openweights import OpenWeights

    ow = OpenWeights()
    training_buf = config.get_sft_training_file(target_key)
    training_file = ow.files.create(training_buf, purpose="conversations")

    # pyrefly: ignore [missing-attribute]
    job = ow.fine_tuning.create(
        model=model,
        # pyrefly: ignore [bad-index]
        training_file=training_file["id"],
        loss="sft",
        epochs=3,
        learning_rate=1e-4,
        r=32,
        max_seq_length=2048,
        push_to_private=False,
    )
    finetuned_id = job.params.get("validated_params", {}).get(
        "finetuned_model_id", "N/A"
    )
    print(f"SFT job submitted: {job.id}")
    print(f"  Model will be: {finetuned_id}")

    while True:
        job.refresh()
        print(f"  Job status: {job.status}")
        if job.status in ["succeeded", "completed", "failed", "canceled"]:
            break
        await asyncio.sleep(30)

    if job.status not in ("succeeded", "completed"):
        raise RuntimeError(f"SFT job {job.status}: {job.id}")

    return job.params["validated_params"]["finetuned_model_id"]


# --- Experiment runner ---


async def run_experiment(
    eval_name: str = "ethical-framework",
    target_framework: str = "virtue_ethics",
    base_model: str = BASE_MODEL,
    instruct_model: str = INSTRUCT_MODEL,
    num_examples_list: list[int] = NUM_EXAMPLES_LIST,
    n_questions: int | None = None,
    n_samples: int = 3,
    skip_sft: bool = False,
    sft_model_id: str | None = None,
) -> pd.DataFrame:
    """Run the three-arm experiment."""
    config = EvalConfig(eval_name)
    metrics = config.judge_metrics

    response_key = f"{target_framework}_response"
    expected_key = f"expected_{target_framework}"

    # Validate keys exist
    if response_key not in config.response_keys:
        raise ValueError(
            f"Response key '{response_key}' not found. "
            f"Available: {config.response_keys}"
        )
    if expected_key not in config.expected_keys:
        raise ValueError(
            f"Expected key '{expected_key}' not found. "
            f"Available: {config.expected_keys}"
        )

    output_dir = config.results_dir("base_icl_vs_sft")

    print(f"Eval: {eval_name}")
    print(f"Target framework: {target_framework}")
    print(f"Response key: {response_key}")
    print(f"Expected key: {expected_key}")
    print(f"Metrics: {metrics}")
    print(f"Base model: {base_model}")
    print(f"Instruct model: {instruct_model}")
    print(f"Output: {output_dir}")

    all_examples = config.get_few_shot_examples(target_key=response_key)
    print(f"Loaded {len(all_examples)} training examples for few-shot")

    runner = OpenWeightsBatchRunner()

    def load_test_eval() -> FreeformEval:
        eval_obj = FreeformEval.from_yaml(
            path=config.yaml_path,
            judge_type="sampling",
            n_samples=n_samples,
        ).with_runner(runner)
        eval_obj.questions = [
            q for q in eval_obj.questions if q.meta.get("split") == "test"
        ]
        if n_questions is not None:
            eval_obj.questions = eval_obj.questions[:n_questions]
        return eval_obj

    test_eval = load_test_eval()
    print(f"Test questions: {len(test_eval.questions)}")

    all_results: list[pd.DataFrame] = []

    # --- Arm 1: Base + ICL sweep ---
    for n_ex in num_examples_list:
        print(f"\n{'=' * 60}")
        print(f"ARM 1: Base + ICL ({n_ex} examples)")
        print("=" * 60)

        eval_obj = load_test_eval()
        if n_ex > 0:
            eval_obj = eval_obj.with_few_shot(all_examples[:n_ex])

        results = await eval_obj.run({"group": [base_model]})
        df = results.df.copy()
        df["arm"] = f"base_icl_{n_ex}"
        df["method"] = "base_icl"
        df["num_examples"] = n_ex
        all_results.append(df)

        for m in metrics:
            print(f"  {m}: {df[m].mean():.1f}")

    # --- Arm 2: Base + SFT ---
    if not skip_sft and sft_model_id is None:
        print(f"\n{'=' * 60}")
        print(f"ARM 2: Training SFT on {base_model}")
        print("=" * 60)
        sft_model_id = await train_sft(base_model, config, expected_key)
        print(f"SFT model ready: {sft_model_id}")

    if sft_model_id is not None:
        print(f"\n{'=' * 60}")
        print(f"ARM 2: Base + SFT ({sft_model_id})")
        print("=" * 60)

        eval_obj = load_test_eval()
        results = await eval_obj.run({"group": [sft_model_id]})
        df = results.df.copy()
        df["arm"] = "base_sft"
        df["method"] = "base_sft"
        df["num_examples"] = -1
        all_results.append(df)

        for m in metrics:
            print(f"  {m}: {df[m].mean():.1f}")

    # --- Arm 3: Instruct baseline ---
    print(f"\n{'=' * 60}")
    print(f"ARM 3: Instruct baseline ({instruct_model})")
    print("=" * 60)

    eval_obj = load_test_eval()
    results = await eval_obj.run({"group": [instruct_model]})
    df = results.df.copy()
    df["arm"] = "instruct_baseline"
    df["method"] = "instruct_baseline"
    df["num_examples"] = -2
    all_results.append(df)

    for m in metrics:
        print(f"  {m}: {df[m].mean():.1f}")

    # --- Save and analyze ---
    combined = pd.concat(all_results, ignore_index=True)
    csv_path = os.path.join(output_dir, "results.csv")
    combined.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    analyze(combined, metrics, target_framework, output_dir)

    return combined


# --- Analysis ---


def analyze(
    df: pd.DataFrame,
    metrics: list[str],
    target_framework: str,
    output_dir: str,
) -> None:
    """Print summary and generate plots."""
    icl_df = df[df["method"] == "base_icl"]
    sft_df = df[df["method"] == "base_sft"]
    instruct_df = df[df["method"] == "instruct_baseline"]
    base_0shot = df[(df["method"] == "base_icl") & (df["num_examples"] == 0)]

    print(f"\n{'=' * 70}")
    print(f"BASE ICL vs SFT — {target_framework.upper()}")
    print("=" * 70)

    # --- Summary table ---
    print(f"\n{'Arm':<25}", end="")
    for m in metrics:
        short = m.replace("_alignment", "").replace("_", " ")
        print(f" {short:>20}", end="")
    print()
    print("-" * (25 + 21 * len(metrics)))

    arms_order = [f"base_icl_{n}" for n in sorted(icl_df["num_examples"].unique())]
    if len(sft_df) > 0:
        arms_order.append("base_sft")
    if len(instruct_df) > 0:
        arms_order.append("instruct_baseline")

    for arm in arms_order:
        arm_df = df[df["arm"] == arm]
        if len(arm_df) == 0:
            continue
        label = arm.replace("base_icl_", "base+icl(").replace("base_sft", "base+SFT")
        if "icl(" in label:
            label += ")"
        label = label.replace("instruct_baseline", "instruct(0-shot)")
        print(f"{label:<25}", end="")
        for m in metrics:
            vals = arm_df[m].dropna()
            print(f" {vals.mean():>8.1f} +/- {vals.std():>5.1f}", end="")
        print()

    # --- Effect sizes relative to base 0-shot ---
    print(f"\n{'=' * 70}")
    print("EFFECT SIZES (Cohen's d vs base 0-shot)")
    print("=" * 70)

    comparisons = {}
    if len(sft_df) > 0:
        comparisons["base+SFT"] = sft_df
    if len(instruct_df) > 0:
        comparisons["instruct(0-shot)"] = instruct_df

    max_n = icl_df["num_examples"].max()
    best_icl = icl_df[icl_df["num_examples"] == max_n]
    comparisons[f"base+icl({max_n})"] = best_icl

    print(f"\n{'Comparison':<25}", end="")
    for m in metrics:
        short = m.replace("_alignment", "").replace("_", " ")
        print(f" {short:>20}", end="")
    print()
    print("-" * (25 + 21 * len(metrics)))

    for label, comp_df in comparisons.items():
        print(f"{label:<25}", end="")
        for m in metrics:
            base_vals = base_0shot[m].dropna()
            comp_vals = comp_df[m].dropna()
            delta = comp_vals.mean() - base_vals.mean()
            pooled_std = ((base_vals.std() ** 2 + comp_vals.std() ** 2) / 2) ** 0.5
            d = delta / pooled_std if pooled_std > 0 else 0.0
            _, p = stats.ttest_ind(comp_vals, base_vals)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f" {d:>+8.2f}{sig:>3}({delta:>+5.1f})", end="")
        print()

    # --- Key diagnostic: SFT vs instruct ---
    if len(sft_df) > 0 and len(instruct_df) > 0:
        print(f"\n{'=' * 70}")
        print("DIAGNOSTIC: How much of SFT effect is instruction-following?")
        print("=" * 70)
        for m in metrics:
            sft_mean = sft_df[m].dropna().mean()
            inst_mean = instruct_df[m].dropna().mean()
            base_mean = base_0shot[m].dropna().mean()
            sft_delta = sft_mean - base_mean
            inst_delta = inst_mean - base_mean
            if abs(sft_delta) > 0.01:
                pct_instruct = inst_delta / sft_delta * 100
            else:
                pct_instruct = float("nan")
            beyond = sft_delta - inst_delta
            print(
                f"  {m}: SFT delta={sft_delta:+.1f}, "
                f"instruct delta={inst_delta:+.1f}, "
                f"SFT beyond instruct={beyond:+.1f} "
                f"({pct_instruct:.0f}% attributable to instruction-following)"
            )

    # --- Plot ---
    icl_vs_sft_plot(df, metrics, target_framework, output_dir)


def icl_vs_sft_plot(
    df: pd.DataFrame,
    metrics: list[str],
    target_framework: str,
    output_dir: str,
) -> None:
    """Line plot: ICL dose-response with SFT and instruct as reference lines."""
    icl_df = df[df["method"] == "base_icl"]
    sft_df = df[df["method"] == "base_sft"]
    instruct_df = df[df["method"] == "instruct_baseline"]

    num_examples = sorted(icl_df["num_examples"].unique())

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # ICL dose-response
        icl_stats = (
            icl_df.groupby("num_examples")[metric].agg(["mean", "std"]).reset_index()
        )
        ax.errorbar(
            icl_stats["num_examples"],
            icl_stats["mean"],
            yerr=icl_stats["std"],
            marker="o",
            markersize=7,
            linewidth=2,
            capsize=4,
            color="steelblue",
            label="Base + ICL",
            zorder=3,
        )

        # Jittered individual points
        for n_ex in num_examples:
            subset = icl_df[icl_df["num_examples"] == n_ex][metric].dropna()
            jitter = np.random.default_rng(42).uniform(-0.3, 0.3, len(subset))
            ax.scatter(
                n_ex + jitter, subset, alpha=0.15, s=12, color="steelblue", zorder=2
            )

        x_range = [min(num_examples), max(num_examples)]

        # SFT reference line
        if len(sft_df) > 0:
            sft_mean = sft_df[metric].dropna().mean()
            ax.axhline(
                sft_mean,
                color="coral",
                linestyle="--",
                linewidth=2,
                label=f"Base + SFT ({sft_mean:.1f})",
                zorder=1,
            )
            sft_std = sft_df[metric].dropna().std()
            ax.fill_between(
                x_range,
                sft_mean - sft_std,
                sft_mean + sft_std,
                alpha=0.1,
                color="coral",
            )

        # Instruct reference line
        if len(instruct_df) > 0:
            inst_mean = instruct_df[metric].dropna().mean()
            ax.axhline(
                inst_mean,
                color="seagreen",
                linestyle=":",
                linewidth=2,
                label=f"Instruct baseline ({inst_mean:.1f})",
                zorder=1,
            )

        short = metric.replace("_alignment", "").replace("_", " ").title()
        ax.set_title(short, fontsize=12)
        ax.set_xlabel("Few-shot examples", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_ylim(0, 100)
        ax.set_xticks(num_examples)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Base ICL vs SFT — {target_framework.replace('_', ' ').title()}", fontsize=14
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "icl_vs_sft.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved {path}")
    plt.close()


# --- CLI ---


async def main():
    parser = argparse.ArgumentParser(
        description="Base model ICL vs SFT experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python experiments/base_icl_vs_sft.py --target-framework virtue_ethics
  python experiments/base_icl_vs_sft.py --target-framework utilitarian --n-questions 10
  python experiments/base_icl_vs_sft.py --sft-model <id> --skip-sft
  python experiments/base_icl_vs_sft.py --analyze-only
""",
    )
    parser.add_argument(
        "--eval",
        type=str,
        default="ethical-framework",
        help="Eval to run (default: ethical-framework)",
    )
    parser.add_argument(
        "--target-framework",
        type=str,
        default="virtue_ethics",
        help="Which framework to elicit (default: virtue_ethics)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=BASE_MODEL,
        help=f"Base model (default: {BASE_MODEL})",
    )
    parser.add_argument(
        "--instruct-model",
        type=str,
        default=INSTRUCT_MODEL,
        help=f"Instruct model (default: {INSTRUCT_MODEL})",
    )
    parser.add_argument(
        "--num-examples",
        type=str,
        default=",".join(str(n) for n in NUM_EXAMPLES_LIST),
        help="Comma-separated ICL example counts (default: 0,1,2,4,8,16)",
    )
    parser.add_argument(
        "--n-questions", type=int, default=None, help="Limit number of test questions"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=3,
        help="Judge samples per question (default: 3)",
    )
    parser.add_argument(
        "--skip-sft",
        action="store_true",
        help="Skip SFT training (use with --sft-model to eval existing model)",
    )
    parser.add_argument(
        "--sft-model",
        type=str,
        default=None,
        help="Pre-trained SFT model ID (skips training)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze existing results CSV",
    )
    args = parser.parse_args()

    num_examples_list = [int(x) for x in args.num_examples.split(",")]

    config = EvalConfig(args.eval)
    output_dir = config.results_dir("base_icl_vs_sft")

    if args.analyze_only:
        csv_path = os.path.join(output_dir, "results.csv")
        if not os.path.exists(csv_path):
            print(f"No results at {csv_path}")
            return
        df = pd.read_csv(csv_path)
        analyze(df, config.judge_metrics, args.target_framework, output_dir)
        return

    await run_experiment(
        eval_name=args.eval,
        target_framework=args.target_framework,
        base_model=args.base_model,
        instruct_model=args.instruct_model,
        num_examples_list=num_examples_list,
        n_questions=args.n_questions,
        n_samples=args.n_samples,
        skip_sft=args.skip_sft,
        sft_model_id=args.sft_model,
    )


if __name__ == "__main__":
    asyncio.run(main())
