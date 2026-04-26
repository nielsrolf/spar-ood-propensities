"""Audit the claiming-superintelligence judge with three checks:

1. Separability: judge claiming vs denying reference responses; report mean
   gap, paired Wilcoxon p-value, % of questions where claiming > denying.
   Plot: score histograms (claiming vs denying) and per-question gap dist.
2. Effect size: Cohen's d (paired) per metric.
3. Score distribution: mean / std / IQR / verdict (BUNCHED / NARROW / OK).
   Plot: box plot of score distributions per metric.

Reference answers come from questions.json (paired *_response keys), not from
YAML expected_* meta. This audit works for any shared eval that follows that
convention; pass --eval-name to point it at a different one.

Usage:
    python shared/evals/claiming-superintelligence/audit_judges.py
    python shared/evals/claiming-superintelligence/audit_judges.py --n-questions 5
    python shared/evals/claiming-superintelligence/audit_judges.py --judge gpt-4o-2024-08-06
"""

import argparse
import asyncio
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import wilcoxon
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(REPO_ROOT / "ben" / ".env", override=True)

PROPENSITIES_ROOT = REPO_ROOT / "ben" / "propensities"
sys.path.insert(0, str(PROPENSITIES_ROOT))

from experiments.eval_config import EvalConfig, SHARED_EVALS_DIR  # noqa: E402
from vibes_eval.freeform import FreeformQuestion  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_EVAL_NAME = EVAL_DIR.name
RESULTS_BASE = PROPENSITIES_ROOT / "results"


def build_judge_question(yaml_data: list[dict], judge_model: str) -> FreeformQuestion:
    q = yaml_data[0]
    return FreeformQuestion(
        id="audit_template",
        paraphrases=["placeholder"],
        judge_prompts=q["judge_prompts"],
        judge_type=q.get("judge_type", "sampling"),
        judge_n_samples=q.get("n_samples", 5),
        judge=judge_model,
    )


async def score_paired_references(
    config: EvalConfig,
    judge_model: str,
    n_questions: int | None,
) -> pd.DataFrame:
    response_keys = config.response_keys
    if len(response_keys) < 2:
        raise ValueError(
            f"Expected >=2 *_response keys in {config.json_path}, "
            f"got {response_keys}"
        )

    fq = build_judge_question(config.yaml_data, judge_model)

    records = config.json_data
    if n_questions is not None:
        records = records[:n_questions]

    to_judge: list[dict] = []
    metadata: list[dict] = []
    for r in records:
        for key in response_keys:
            answer = r.get(key)
            if not answer:
                continue
            answer_type = key[: -len("_response")]
            to_judge.append({"question": r["question"], "answer": answer})
            metadata.append(
                {
                    "question_id": r["id"],
                    "answer_type": answer_type,
                    "split": r.get("split", "unknown"),
                    "topic": r.get("topic", ""),
                }
            )

    total = len(to_judge) * len(fq.judges)
    with tqdm(total=total, desc=f"  [{config.eval_name}]", unit="judge") as pbar:
        scored = await fq.judge(to_judge, pbar=pbar)

    rows = []
    for meta_row, scored_row in zip(metadata, scored):
        row = {**meta_row}
        for k, v in scored_row.items():
            if k not in ("question", "answer"):
                row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


def metric_columns(df: pd.DataFrame) -> list[str]:
    skip = {"question_id", "answer_type", "split", "topic"}
    return [c for c in df.columns if c not in skip and df[c].notna().any()]


def report_separation_and_effect(df: pd.DataFrame) -> dict:
    metrics = metric_columns(df)
    answer_types = sorted(df["answer_type"].unique())
    if len(answer_types) != 2:
        raise ValueError(f"Expected exactly 2 answer types, got {answer_types}")
    high_at, low_at = answer_types[0], answer_types[1]
    if "claim" in low_at and "deny" in high_at:
        high_at, low_at = low_at, high_at

    print(f"\n  High-trait answer type: {high_at}")
    print(f"  Low-trait  answer type: {low_at}\n")

    cohen_label = "Cohen's d"
    header = (
        f"  {'Metric':<35}{high_at[:14]:>15}{low_at[:14]:>15}"
        f"{'Gap':>10}{'Sep%':>8}{'p':>10}{cohen_label:>12}{'Verdict':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    stats: dict = {}
    for metric in metrics:
        df_h = df[df["answer_type"] == high_at].set_index("question_id")[metric]
        df_l = df[df["answer_type"] == low_at].set_index("question_id")[metric]
        common = df_h.index.intersection(df_l.index)
        gaps = (df_h.loc[common] - df_l.loc[common]).dropna()

        mean_h = df_h.mean()
        mean_l = df_l.mean()
        gap = mean_h - mean_l

        if len(gaps) >= 6 and gaps.std() > 0:
            _, p = wilcoxon(gaps)
        else:
            p = float("nan")

        sep_raw = (gaps > 0).mean() * 100
        sep_pct = max(sep_raw, 100 - sep_raw)

        if gaps.std() > 0:
            d = float(gaps.mean() / gaps.std())
        else:
            d = float("nan")

        if abs(d) >= 0.8 and p < 0.05:
            verdict = "OK"
        elif abs(d) >= 0.5:
            verdict = "WEAK"
        else:
            verdict = "POOR"

        sig = (
            "***"
            if p < 0.001
            else "**"
            if p < 0.01
            else "*"
            if p < 0.05
            else ""
        )
        print(
            f"  {metric:<35}{mean_h:>15.1f}{mean_l:>15.1f}{gap:>+10.1f}"
            f"{sep_pct:>7.0f}%{p:>9.4f}{sig:<3}{d:>+12.2f}{verdict:>10}"
        )
        stats[metric] = {
            "mean_high": mean_h,
            "mean_low": mean_l,
            "gap": gap,
            "sep_pct": sep_pct,
            "p_value": p,
            "cohens_d": d,
            "verdict": verdict,
        }
    return stats


def report_distribution(df: pd.DataFrame) -> dict:
    metrics = metric_columns(df)

    print(
        f"\n  {'Metric':<35}{'Mean':>8}{'Std':>8}{'Min':>8}"
        f"{'Max':>8}{'IQR':>8}{'Verdict':>12}"
    )
    print("  " + "-" * 87)

    stats: dict = {}
    for metric in metrics:
        scores = df[metric].dropna()
        std = scores.std()
        q25, q75 = scores.quantile(0.25), scores.quantile(0.75)
        iqr = q75 - q25
        if std < 10:
            verdict = "BUNCHED"
        elif iqr < 15:
            verdict = "NARROW"
        else:
            verdict = "OK"
        print(
            f"  {metric:<35}{scores.mean():>8.1f}{std:>8.1f}"
            f"{scores.min():>8.1f}{scores.max():>8.1f}{iqr:>8.1f}{verdict:>12}"
        )
        stats[metric] = {
            "mean": scores.mean(),
            "std": std,
            "iqr": iqr,
            "verdict": verdict,
        }
    return stats


def plot_separation(
    df: pd.DataFrame, eval_name: str, output_dir: Path, judge_model: str
) -> None:
    metrics = metric_columns(df)
    answer_types = sorted(df["answer_type"].unique())
    colors = ["tab:red", "tab:blue"]

    n_cols = min(3, len(metrics))
    n_rows = ((len(metrics) + n_cols - 1) // n_cols) * 2
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
    )
    bins = np.linspace(0, 100, 21)

    for idx, metric in enumerate(metrics):
        row = (idx // n_cols) * 2
        col = idx % n_cols

        ax = axes[row, col]
        for at, color in zip(answer_types, colors):
            scores = df[df["answer_type"] == at][metric]
            ax.hist(
                scores,
                bins=bins,
                alpha=0.5,
                label=f"{at} ({scores.mean():.0f})",
                color=color,
            )
            ax.axvline(scores.mean(), color=color, linestyle="--", linewidth=2)
        ax.set_xlabel(metric, fontsize=9)
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 100)

        ax_gap = axes[row + 1, col]
        df_a = df[df["answer_type"] == answer_types[0]].set_index("question_id")
        df_b = df[df["answer_type"] == answer_types[1]].set_index("question_id")
        common = df_a.index.intersection(df_b.index)
        gaps = (df_a.loc[common, metric] - df_b.loc[common, metric]).dropna().values

        gap_bins = np.linspace(-100, 100, 41)
        ax_gap.hist(
            gaps, bins=gap_bins, color="steelblue", alpha=0.7, edgecolor="white"
        )
        ax_gap.axvline(0, color="black", linewidth=1)
        if len(gaps):
            mean_gap = float(np.mean(gaps))
            ax_gap.axvline(mean_gap, color="red", linestyle="--", linewidth=2)
            sep_pct = max(float((gaps > 0).mean()), float((gaps < 0).mean())) * 100
            if len(gaps) >= 6 and float(np.std(gaps)) > 0:
                _, p = wilcoxon(gaps)
            else:
                p = float("nan")
            d = float(np.mean(gaps) / np.std(gaps, ddof=1)) if np.std(gaps) > 0 else float("nan")
            ax_gap.set_title(
                f"mean={mean_gap:+.1f}, sep={sep_pct:.0f}%, "
                f"d={d:+.2f}, p={p:.4f}",
                fontsize=9,
            )
        ax_gap.set_xlabel(
            f"gap ({answer_types[0]} - {answer_types[1]})", fontsize=9
        )
        ax_gap.set_ylabel("Count")

    for r in range(n_rows):
        for c in range(n_cols):
            mi = (r // 2) * n_cols + c
            if mi >= len(metrics):
                axes[r, c].set_visible(False)

    plt.suptitle(
        f"Reference Answer Separation: {eval_name}\n(judge: {judge_model})",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    path = output_dir / f"separation_{eval_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_distribution(
    df: pd.DataFrame, eval_name: str, output_dir: Path, judge_model: str
) -> None:
    metrics = metric_columns(df)
    fig, ax = plt.subplots(figsize=(max(8, int(len(metrics) * 1.4)), 5))
    data = [df[m].dropna().values for m in metrics]
    bp = ax.boxplot(data, tick_labels=metrics, patch_artist=True)
    for patch, m in zip(bp["boxes"], metrics):
        scores = df[m].dropna()
        std = scores.std()
        iqr = scores.quantile(0.75) - scores.quantile(0.25)
        if std < 10:
            patch.set_facecolor("#ff9999")
        elif iqr < 15:
            patch.set_facecolor("#ffcc66")
        else:
            patch.set_facecolor("#99cc99")
    ax.set_xticklabels(metrics, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score (0-100)")
    ax.set_ylim(0, 100)
    ax.axhline(50, color="gray", linestyle=":", alpha=0.5)
    ax.set_title(
        f"Score Distributions: {eval_name}\n"
        f"(green=OK, yellow=narrow, red=bunched) — judge: {judge_model}"
    )
    plt.tight_layout()
    path = output_dir / f"distribution_{eval_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Audit claiming-superintelligence judge")
    parser.add_argument(
        "--eval-name",
        type=str,
        default=DEFAULT_EVAL_NAME,
        help=f"Eval name under shared/evals/ (default: {DEFAULT_EVAL_NAME})",
    )
    parser.add_argument(
        "--n-questions",
        type=int,
        default=None,
        help="Limit questions for faster testing",
    )
    parser.add_argument(
        "--judge",
        type=str,
        default="gpt-4o-2024-08-06",
        help="Judge model ID",
    )
    args = parser.parse_args()

    config = EvalConfig(args.eval_name, evals_root=SHARED_EVALS_DIR)
    print(config)

    judge_slug = args.judge.replace("/", "_")
    output_dir = RESULTS_BASE / args.eval_name / "judge_audit" / judge_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nScoring paired reference answers from {config.json_path}")
    df = await score_paired_references(config, args.judge, args.n_questions)
    df.to_csv(output_dir / "scores.csv", index=False)

    print("\n" + "=" * 70)
    print("CHECK 1+2: Reference Answer Separation & Effect Size")
    print("=" * 70)
    sep_stats = report_separation_and_effect(df)

    print("\n" + "=" * 70)
    print("CHECK 3: Score Distribution (Coherence)")
    print("=" * 70)
    dist_stats = report_distribution(df)

    plot_separation(df, args.eval_name, output_dir, args.judge)
    plot_distribution(df, args.eval_name, output_dir, args.judge)

    summary = {"separation": sep_stats, "distribution": dist_stats}
    pd.DataFrame(summary["separation"]).T.to_csv(output_dir / "separation_stats.csv")
    pd.DataFrame(summary["distribution"]).T.to_csv(output_dir / "distribution_stats.csv")

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
