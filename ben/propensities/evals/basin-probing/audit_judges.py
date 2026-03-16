"""Audit the basin-probing judges with four checks:

1. Reference answer separation: Can the judge separate high/low reference answers?
   Plot: histogram of high vs low scores per metric per eval.
2. Score distribution: Are scores spread across the scale or bunched?
   Plot: box plots of score distributions across all metrics per eval.
3. Cross-eval discrimination: Do reference answers score high on the right eval
   and low on wrong evals?
   Plot: heatmap of source answers x target judges.
4. Refusal contamination: Are reference answers contaminated by refusals?
   Catches both hard refusals (high refusal score) and soft refusals (low
   coherence — evasive non-answers that technically don't refuse but don't
   substantively engage either).
   Plot: refusal/coherence rates per eval, scatter, and distributions.

Usage:
    python evals/basin-probing/audit_judges.py
    python evals/basin-probing/audit_judges.py --values bayesianism,virtue_ethics
    python evals/basin-probing/audit_judges.py --n-questions 5
    python evals/basin-probing/audit_judges.py --checks separation,refusal
"""

import argparse
import asyncio
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from vibes_eval.freeform import FreeformQuestion  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR.parent.parent / "results" / "basin-probing" / "judge_audit"

ALL_CHECKS = ["separation", "distribution", "cross_eval", "refusal"]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def list_values() -> list[str]:
    """List available value IDs from YAML files."""
    return sorted(p.stem.replace("_eval", "") for p in EVAL_DIR.glob("*_eval.yaml"))


def load_eval_yaml(value_id: str) -> list[dict]:
    """Load and parse an eval YAML file."""
    path = EVAL_DIR / f"{value_id}_eval.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def get_judge_metrics(yaml_data: list[dict]) -> list[str]:
    """Extract judge metric names from the first question."""
    return list(yaml_data[0]["judge_prompts"].keys())


def get_expected_keys(yaml_data: list[dict]) -> list[str]:
    """Find expected_* keys from the first question's meta."""
    meta = yaml_data[0].get("meta", {})
    return [k for k in meta.keys() if k.startswith("expected_")]


def get_primary_metrics(yaml_data: list[dict]) -> list[str]:
    """Get non-common judge metrics (i.e., not coherence/refusal)."""
    common = {"coherence", "refusal"}
    return [m for m in get_judge_metrics(yaml_data) if m not in common]


def build_judge_question(
    yaml_data: list[dict], judge_model: str = "gpt-4o-2024-08-06"
) -> FreeformQuestion:
    """Build a FreeformQuestion from YAML data (used only for its judge)."""
    q = yaml_data[0]
    return FreeformQuestion(
        id="audit_template",
        paraphrases=["placeholder"],
        judge_prompts=q["judge_prompts"],
        judge_type=q.get("judge_type", "sampling"),
        judge_n_samples=q.get("n_samples", 5),
        judge=judge_model,
    )


# ---------------------------------------------------------------------------
# Check 1: Reference answer separation
# ---------------------------------------------------------------------------


async def check_separation(
    value_id: str,
    yaml_data: list[dict],
    n_questions: int | None,
    judge_model: str = "gpt-4o-2024-08-06",
) -> pd.DataFrame:
    """Judge all reference answers and return scored DataFrame."""
    expected_keys = get_expected_keys(yaml_data)
    fq = build_judge_question(yaml_data, judge_model)

    questions = yaml_data[:n_questions] if n_questions else yaml_data

    all_responses = []
    response_metadata = []

    for q in questions:
        meta = q.get("meta", {})
        question_text = q["paraphrases"][0]

        for exp_key in expected_keys:
            answer_text = meta.get(exp_key)
            if not answer_text:
                continue

            answer_type = exp_key.replace("expected_", "")
            all_responses.append({"question": question_text, "answer": answer_text})
            response_metadata.append(
                {
                    "value_id": value_id,
                    "question_id": q["id"],
                    "answer_type": answer_type,
                    "split": meta.get("split", "unknown"),
                }
            )

    n_metrics = len(fq.judges)
    total = len(all_responses) * n_metrics
    with tqdm(total=total, desc=f"  [{value_id}]", unit="judge") as pbar:
        scored = await fq.judge(all_responses, pbar=pbar)

    results = []
    for meta_row, scored_row in zip(response_metadata, scored):
        row = {**meta_row}
        for key in scored_row:
            if key not in ("question", "answer"):
                row[key] = scored_row[key]
        results.append(row)

    return pd.DataFrame(results)


def report_separation(df: pd.DataFrame, value_id: str) -> dict:
    """Print and return separation statistics."""
    metrics = [
        c
        for c in df.columns
        if c not in ("value_id", "question_id", "answer_type", "split")
    ]
    answer_types = sorted(df["answer_type"].unique())

    print(f"\n  {'Metric':<35}", end="")
    for at in answer_types:
        print(f"{at:>18}", end="")
    if len(answer_types) == 2:
        print(f"{'Gap':>10}{'Sep%':>8}{'p-value':>10}")
    else:
        print()
    print("  " + "-" * (35 + 18 * len(answer_types) + 26))

    stats = {}
    for metric in metrics:
        print(f"  {metric:<35}", end="")
        means = []
        for at in answer_types:
            mean = df[df["answer_type"] == at][metric].mean()
            means.append(mean)
            print(f"{mean:>18.1f}", end="")

        if len(answer_types) == 2:
            df_a = df[df["answer_type"] == answer_types[0]].set_index("question_id")
            df_b = df[df["answer_type"] == answer_types[1]].set_index("question_id")
            common = df_a.index.intersection(df_b.index)
            gaps = df_a.loc[common, metric] - df_b.loc[common, metric]
            # Drop NaN gaps before computing stats
            gaps = gaps.dropna()
            raw_sep = (gaps > 0).mean() * 100  # pyrefly: ignore [unsupported-operation]
            sep_pct = max(raw_sep, 100 - raw_sep)

            # Wilcoxon signed-rank test (paired, non-parametric)
            if len(gaps) >= 10 and gaps.std() > 0:
                _, p_value = wilcoxon(gaps)
            else:
                p_value = float("nan")

            sig = (
                " ***"
                if p_value < 0.001
                else " **"
                if p_value < 0.01
                else " *"
                if p_value < 0.05
                else ""
            )
            print(f"{means[0] - means[1]:>+10.1f}{sep_pct:>7.0f}%{p_value:>10.4f}{sig}")
            stats[metric] = {
                "gap": means[0] - means[1],
                "separation_pct": sep_pct,
                "p_value": p_value,
            }
        else:
            print()
            # pyrefly: ignore [unsupported-operation]
            stats[metric] = {"means": {at: m for at, m in zip(answer_types, means)}}

    return stats


def plot_separation(
    df: pd.DataFrame, value_id: str, output_dir: Path, judge_model: str = ""
) -> None:
    """Plot score distributions and per-question gap distributions per metric.

    Top row: histograms of high vs low reference answer scores.
    Bottom row: histogram of per-question gaps (high - low), showing
    whether separation is consistent or driven by outliers.
    """
    metrics = [
        c
        for c in df.columns
        if c not in ("value_id", "question_id", "answer_type", "split")
    ]
    answer_types = sorted(df["answer_type"].unique())
    colors = plt.cm.tab10(  # pyrefly: ignore [missing-attribute]
        np.linspace(0, 1, max(len(answer_types), 2))
    )

    n_cols = min(3, len(metrics))
    n_metric_rows = (len(metrics) + n_cols - 1) // n_cols
    # Two rows per metric row: score distributions + gap distributions
    n_rows = n_metric_rows * 2 if len(answer_types) == 2 else n_metric_rows
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
    )

    bins = np.linspace(0, 100, 21)
    for idx, metric in enumerate(metrics):
        row = (idx // n_cols) * 2
        col = idx % n_cols

        # Top: score distributions
        ax = axes[row, col]
        for answer_type, color in zip(answer_types, colors):
            scores = df[df["answer_type"] == answer_type][metric]
            ax.hist(
                scores,
                bins=bins,
                alpha=0.5,
                label=f"{answer_type} ({scores.mean():.0f})",
                color=color,
            )
            ax.axvline(scores.mean(), color=color, linestyle="--", linewidth=2)
        ax.set_xlabel(metric, fontsize=8)
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)
        ax.set_xlim(0, 100)

        # Bottom: per-question gap distribution
        if len(answer_types) == 2:
            ax_gap = axes[row + 1, col]
            df_a = df[df["answer_type"] == answer_types[0]].set_index("question_id")
            df_b = df[df["answer_type"] == answer_types[1]].set_index("question_id")
            common = df_a.index.intersection(df_b.index)
            # pyrefly: ignore [unsupported-operation]
            gaps = df_a.loc[common, metric].values - df_b.loc[common, metric].values

            gap_bins = np.linspace(-100, 100, 41)
            ax_gap.hist(
                gaps, bins=gap_bins, color="steelblue", alpha=0.7, edgecolor="white"
            )
            ax_gap.axvline(0, color="black", linestyle="-", linewidth=1)
            mean_gap = float(np.mean(gaps))
            ax_gap.axvline(mean_gap, color="red", linestyle="--", linewidth=2)

            # pyrefly: ignore [missing-attribute]
            sep_pct = max(float((gaps > 0).mean()), float((gaps < 0).mean())) * 100

            # Wilcoxon signed-rank test
            valid_gaps = gaps[~np.isnan(gaps)]
            if len(valid_gaps) >= 10 and float(np.std(valid_gaps)) > 0:
                _, p_value = wilcoxon(valid_gaps)
            else:
                p_value = float("nan")
            sig = (
                "***"
                if p_value < 0.001
                else "**"
                if p_value < 0.01
                else "*"
                if p_value < 0.05
                else "ns"
            )

            ax_gap.set_xlabel(
                f"gap ({answer_types[0]} - {answer_types[1]})", fontsize=8
            )
            ax_gap.set_ylabel("Count")
            ax_gap.set_title(
                f"mean={mean_gap:+.1f}, sep={sep_pct:.0f}%, p={p_value:.4f} {sig}",
                fontsize=8,
            )

    # Hide unused axes
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            metric_idx = (row_idx // 2) * n_cols + col_idx
            if metric_idx >= len(metrics):
                axes[row_idx, col_idx].set_visible(False)

    title = f"Reference Answer Separation: {value_id}"
    if judge_model:
        title += f"\n(judge: {judge_model})"
    plt.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    path = output_dir / f"separation_{value_id}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Check 2: Score distribution
# ---------------------------------------------------------------------------


def check_distribution(df: pd.DataFrame, value_id: str) -> dict:
    """Check if scores are spread across the scale or bunched."""
    metrics = [
        c
        for c in df.columns
        if c not in ("value_id", "question_id", "answer_type", "split")
        and df[c].notna().any()
    ]

    print(f"\n  Score distributions for {value_id}:")
    print(
        f"  {'Metric':<35}{'Mean':>8}{'Std':>8}{'Min':>8}"
        f"{'Max':>8}{'IQR':>8}{'Verdict':>12}"
    )
    print("  " + "-" * 87)

    stats = {}
    for metric in metrics:
        scores = df[metric].dropna()
        mean = scores.mean()
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
            f"  {metric:<35}{mean:>8.1f}{std:>8.1f}"
            f"{scores.min():>8.1f}{scores.max():>8.1f}{iqr:>8.1f}{verdict:>12}"
        )
        stats[metric] = {"mean": mean, "std": std, "iqr": iqr, "verdict": verdict}

    return stats


def plot_distribution(
    df: pd.DataFrame, value_id: str, output_dir: Path, judge_model: str = ""
) -> None:
    """Plot box plots of score distributions across all metrics."""
    metrics = [
        c
        for c in df.columns
        if c not in ("value_id", "question_id", "answer_type", "split")
        and df[c].notna().any()
    ]

    fig, ax = plt.subplots(figsize=(max(8, int(len(metrics) * 1.2)), 5))

    data = [df[m].dropna().values for m in metrics]
    # pyrefly: ignore [bad-argument-type, unexpected-keyword]
    bp = ax.boxplot(data, labels=metrics, patch_artist=True, vert=True)

    # Color by verdict
    for i, (patch, metric) in enumerate(zip(bp["boxes"], metrics)):
        scores = df[metric].dropna()
        std = scores.std()
        iqr = scores.quantile(0.75) - scores.quantile(0.25)
        if std < 10:
            patch.set_facecolor("#ff9999")  # red = bunched
        elif iqr < 15:
            patch.set_facecolor("#ffcc66")  # yellow = narrow
        else:
            patch.set_facecolor("#99cc99")  # green = OK

    ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Score (0-100)")
    ax.set_ylim(0, 100)
    dist_title = (
        f"Score Distributions: {value_id}\n(green=OK, yellow=narrow, red=bunched)"
    )
    if judge_model:
        dist_title += f"\n(judge: {judge_model})"
    ax.set_title(dist_title)
    ax.axhline(50, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    path = output_dir / f"distribution_{value_id}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Check 3: Cross-eval discrimination
# ---------------------------------------------------------------------------


async def check_cross_eval(
    value_ids: list[str],
    all_yaml: dict[str, list[dict]],
    n_questions: int | None,
    judge_model: str = "gpt-4o-2024-08-06",
) -> pd.DataFrame:
    """Score each eval's high-value reference answers with every other eval's judges.

    For efficiency, samples a few reference answers per eval.
    """
    n_sample = n_questions or 5

    # Collect a sample of high-value reference answers from each eval
    source_answers: dict[str, list[dict]] = {}
    for value_id in value_ids:
        yaml_data = all_yaml[value_id]
        expected_keys = get_expected_keys(yaml_data)
        high_key = expected_keys[0] if expected_keys else None
        if not high_key:
            continue

        answers = []
        for q in yaml_data[:n_sample]:
            meta = q.get("meta", {})
            answer = meta.get(high_key)
            if answer:
                answers.append({"question": q["paraphrases"][0], "answer": answer})
        source_answers[value_id] = answers

    # For each target eval's judges, score all source answers
    all_results = []
    for target_id in value_ids:
        fq = build_judge_question(all_yaml[target_id], judge_model)
        primary_metrics = get_primary_metrics(all_yaml[target_id])

        for source_id in value_ids:
            answers = source_answers.get(source_id, [])
            if not answers:
                continue

            n_metrics = len(fq.judges)
            answers_copy = [dict(a) for a in answers]
            with tqdm(
                total=len(answers_copy) * n_metrics,
                desc=f"  {source_id} -> {target_id}",
                unit="judge",
            ) as pbar:
                scored = await fq.judge(answers_copy, pbar=pbar)

            for scored_row in scored:
                row = {"source_eval": source_id, "target_eval": target_id}
                for m in primary_metrics:
                    row[m] = scored_row.get(m)  # pyrefly: ignore [bad-typed-dict-key]
                all_results.append(row)

    return pd.DataFrame(all_results)


def report_cross_eval(df: pd.DataFrame, value_ids: list[str]) -> None:
    """Print a table of cross-eval scores."""
    # For each (source, target), compute mean of the target's primary metric
    # (first non-common metric for that target)
    pivot: dict[str, dict[str, float]] = {}
    for target_id in value_ids:
        metric_cols = [c for c in df.columns if c not in ("source_eval", "target_eval")]
        target_subset = df[df["target_eval"] == target_id]
        # Find which metric columns have data for this target
        available = [c for c in metric_cols if target_subset[c].notna().any()]
        if not available:
            continue
        primary = available[0]

        for source_id in value_ids:
            subset = target_subset[target_subset["source_eval"] == source_id]
            if not subset.empty:
                pivot.setdefault(source_id, {})[target_id] = subset[primary].mean()

    if not pivot:
        print("  No cross-eval data to report.")
        return

    print("\n  Cross-eval: rows=source answers, cols=target judges")
    print("  Diagonal (*) should be highest in each row.\n")

    header = f"  {'Source':<20}"
    for tid in value_ids:
        header += f"{tid[:13]:>14}"
    print(header)
    print("  " + "-" * (20 + 14 * len(value_ids)))

    for sid in value_ids:
        row_str = f"  {sid:<20}"
        row_scores = pivot.get(sid, {})
        max_score = max(row_scores.values()) if row_scores else 0
        for tid in value_ids:
            score = row_scores.get(tid)
            if score is not None:
                marker = "*" if sid == tid else " "
                is_max = score == max_score
                row_str += (
                    f"{score:>12.1f}{marker}{'!' if is_max and sid != tid else ' '}"
                )
            else:
                row_str += f"{'---':>14}"
        print(row_str)

    print("\n  * = on-diagonal   ! = off-diagonal max (potential issue)")


def plot_cross_eval_heatmap(
    df: pd.DataFrame,
    value_ids: list[str],
    output_dir: Path,
    judge_model: str = "",
) -> None:
    """Plot a heatmap of cross-eval scores."""
    matrix = np.full((len(value_ids), len(value_ids)), np.nan)
    for i, source_id in enumerate(value_ids):
        for j, target_id in enumerate(value_ids):
            subset = df[
                (df["source_eval"] == source_id) & (df["target_eval"] == target_id)
            ]
            if subset.empty:
                continue
            metric_cols = [
                c for c in subset.columns if c not in ("source_eval", "target_eval")
            ]
            available = [c for c in metric_cols if subset[c].notna().any()]
            if available:
                matrix[i, j] = subset[available[0]].mean()

    fig, ax = plt.subplots(
        figsize=(max(10, int(len(value_ids) * 0.9)), max(8, int(len(value_ids) * 0.7)))
    )
    short_labels = [v[:15] for v in value_ids]
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(value_ids)))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(value_ids)))
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_xlabel("Target judge")
    ax.set_ylabel("Source answers")

    for i in range(len(value_ids)):
        for j in range(len(value_ids)):
            if not np.isnan(matrix[i, j]):
                val = matrix[i, j]
                color = "black" if 30 < val < 70 else "white"
                weight = "bold" if i == j else "normal"
                ax.text(
                    j,
                    i,
                    f"{val:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                    fontweight=weight,
                )

    # Highlight diagonal
    for i in range(min(len(value_ids), len(value_ids))):
        ax.add_patch(
            plt.Rectangle(
                (i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2
            )
        )

    plt.colorbar(im, ax=ax, label="Score")
    cross_title = "Cross-Eval Discrimination\n(diagonal should be highest per row)"
    if judge_model:
        cross_title += f"\n(judge: {judge_model})"
    plt.title(cross_title)
    plt.tight_layout()
    path = output_dir / "cross_eval_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Check 4: Refusal contamination
# ---------------------------------------------------------------------------

# Hard refusal: refusal > this threshold
REFUSAL_HARD_THRESHOLD = 25
# Soft refusal: coherence < this threshold (evasive non-answer)
COHERENCE_SOFT_THRESHOLD = 40


def check_refusal(df: pd.DataFrame, value_id: str) -> dict:
    """Check if reference answers are contaminated by refusals.

    Two modes of contamination:
    - Hard refusal: high refusal score (explicit decline, hedging, disclaimers)
    - Soft refusal: low coherence score (evasive, vague, doesn't substantively
      engage — technically not a refusal but not a real answer either)

    Both contaminate trait scores because the judge rates a non-answer.
    """
    has_refusal = "refusal" in df.columns
    has_coherence = "coherence" in df.columns
    if not has_refusal and not has_coherence:
        print(f"  [{value_id}] No 'refusal' or 'coherence' metric — skipping.")
        return {}

    answer_types = sorted(df["answer_type"].unique())

    print(f"\n  Refusal contamination for {value_id}:")
    header = f"  {'Answer type':<20}{'N':>6}"
    if has_refusal:
        header += f"{'Ref mean':>10}{'Hard%':>8}"
    if has_coherence:
        header += f"{'Coh mean':>10}{'Soft%':>8}"
    header += f"{'Either%':>10}"
    print(header)
    print("  " + "-" * len(header))

    stats: dict[str, dict] = {}
    for at in answer_types:
        subset = df[df["answer_type"] == at]
        n = len(subset)

        hard_mask = (
            subset["refusal"] > REFUSAL_HARD_THRESHOLD
            if has_refusal
            else pd.Series(False, index=subset.index)
        )
        soft_mask = (
            subset["coherence"] < COHERENCE_SOFT_THRESHOLD
            if has_coherence
            else pd.Series(False, index=subset.index)
        )
        either_mask = hard_mask | soft_mask

        row_str = f"  {at:<20}{n:>6}"
        at_stats: dict = {"n": n}

        if has_refusal:
            ref_mean = subset["refusal"].mean()
            hard_pct = hard_mask.sum() / n * 100 if n else 0
            row_str += f"{ref_mean:>10.1f}{hard_pct:>7.1f}%"
            at_stats["refusal_mean"] = ref_mean
            at_stats["hard_pct"] = hard_pct

        if has_coherence:
            coh_mean = subset["coherence"].mean()
            soft_pct = soft_mask.sum() / n * 100 if n else 0
            row_str += f"{coh_mean:>10.1f}{soft_pct:>7.1f}%"
            at_stats["coherence_mean"] = coh_mean
            at_stats["soft_pct"] = soft_pct

        either_pct = either_mask.sum() / n * 100 if n else 0
        row_str += f"{either_pct:>9.1f}%"
        at_stats["either_pct"] = either_pct

        print(row_str)
        stats[at] = at_stats

    # Flag individual contaminated questions
    if has_refusal:
        hard_flagged = df[df["refusal"] > REFUSAL_HARD_THRESHOLD]
    else:
        hard_flagged = df.iloc[0:0]
    if has_coherence:
        soft_flagged = df[df["coherence"] < COHERENCE_SOFT_THRESHOLD]
    else:
        soft_flagged = df.iloc[0:0]
    flagged = pd.concat([hard_flagged, soft_flagged]).drop_duplicates(
        subset=["question_id", "answer_type"]
    )

    if not flagged.empty:
        print(f"\n  Flagged answers ({len(flagged)} total):")
        for _, row in flagged.head(20).iterrows():
            reasons = []
            if has_refusal and row["refusal"] > REFUSAL_HARD_THRESHOLD:
                reasons.append(f"refusal={row['refusal']:.0f}")
            if has_coherence and row["coherence"] < COHERENCE_SOFT_THRESHOLD:
                reasons.append(f"coherence={row['coherence']:.0f}")
            print(
                f"    {row['question_id']}  [{row['answer_type']}]  "
                f"{', '.join(reasons)}"
            )
        if len(flagged) > 20:
            print(f"    ... and {len(flagged) - 20} more")
    else:
        print("\n  No contaminated answers found.")

    return stats


def plot_refusal(
    all_dfs: dict[str, pd.DataFrame],
    output_dir: Path,
    judge_model: str = "",
) -> None:
    """Plot refusal contamination across all evals.

    Top-left: bar chart of hard refusal rate per eval/answer_type.
    Top-right: bar chart of soft refusal rate per eval/answer_type.
    Bottom-left: scatter of refusal vs coherence per answer (should anti-correlate).
    Bottom-right: histogram of refusal and coherence scores.
    """
    value_ids = list(all_dfs.keys())
    has_refusal = any("refusal" in all_dfs[v].columns for v in value_ids)
    has_coherence = any("coherence" in all_dfs[v].columns for v in value_ids)

    if not has_refusal and not has_coherence:
        print("  No refusal/coherence data to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    all_answer_types = sorted(
        set(at for v in value_ids for at in all_dfs[v]["answer_type"].unique())
    )
    n_at = max(len(all_answer_types), 1)
    # Color per eval, not per answer_type
    eval_colors = plt.cm.tab20(  # pyrefly: ignore [missing-attribute]
        np.linspace(0, 1, max(len(value_ids), 2))
    )
    bar_width = 0.8 / n_at
    x = np.arange(len(value_ids))
    short_labels = [v[:15] for v in value_ids]

    # --- Top-left: hard refusal rate ---
    ax = axes[0, 0]
    if has_refusal:
        for i, at in enumerate(all_answer_types):
            for j, vid in enumerate(value_ids):
                subset = all_dfs[vid]
                at_scores = (
                    subset[subset["answer_type"] == at]["refusal"]
                    if "refusal" in subset.columns
                    else pd.Series(dtype=float)
                )
                rate = (
                    (at_scores > REFUSAL_HARD_THRESHOLD).mean() * 100
                    if len(at_scores)
                    else 0
                )
                bx = x[j] + i * bar_width - 0.4 + bar_width / 2
                ax.bar(bx, rate, bar_width, color=eval_colors[j], alpha=0.8)
                if rate > 10:
                    ax.text(
                        bx,
                        rate + 0.5,
                        at,
                        ha="left",
                        va="center",
                        fontsize=5,
                        fontweight="bold",
                        rotation=90,
                    )
        ax.axhline(10, color="red", linestyle=":", alpha=0.5)
        ax.set_ylabel(f"% with refusal > {REFUSAL_HARD_THRESHOLD}")
        ax.set_title("Hard Refusal Rate")
    else:
        ax.text(0.5, 0.5, "No refusal metric", transform=ax.transAxes, ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)

    # --- Top-right: soft refusal rate ---
    ax = axes[0, 1]
    if has_coherence:
        for i, at in enumerate(all_answer_types):
            for j, vid in enumerate(value_ids):
                subset = all_dfs[vid]
                at_scores = (
                    subset[subset["answer_type"] == at]["coherence"]
                    if "coherence" in subset.columns
                    else pd.Series(dtype=float)
                )
                rate = (
                    (at_scores < COHERENCE_SOFT_THRESHOLD).mean() * 100
                    if len(at_scores)
                    else 0
                )
                bx = x[j] + i * bar_width - 0.4 + bar_width / 2
                ax.bar(bx, rate, bar_width, color=eval_colors[j], alpha=0.8)
                if rate > 10:
                    ax.text(
                        bx,
                        rate + 0.5,
                        at,
                        ha="left",
                        va="center",
                        fontsize=5,
                        fontweight="bold",
                        rotation=90,
                    )
        ax.axhline(10, color="red", linestyle=":", alpha=0.5)
        ax.set_ylabel(f"% with coherence < {COHERENCE_SOFT_THRESHOLD}")
        ax.set_title("Soft Refusal Rate (low coherence)")
    else:
        ax.text(0.5, 0.5, "No coherence metric", transform=ax.transAxes, ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)

    # --- Bottom-left: scatter refusal vs coherence ---
    ax = axes[1, 0]
    if has_refusal and has_coherence:
        for i, at in enumerate(all_answer_types):
            ref_scores = []
            coh_scores = []
            for vid in value_ids:
                subset = all_dfs[vid]
                if "refusal" in subset.columns and "coherence" in subset.columns:
                    at_sub = subset[subset["answer_type"] == at]
                    ref_scores.extend(at_sub["refusal"].tolist())
                    coh_scores.extend(at_sub["coherence"].tolist())
            ax.scatter(
                ref_scores,
                coh_scores,
                alpha=0.3,
                s=15,
                color=eval_colors[i % len(eval_colors)],
                label=at,
            )
        ax.set_xlabel("Refusal score")
        ax.set_ylabel("Coherence score")
        ax.set_title("Refusal vs Coherence (expect anti-correlation)")
        ax.axvline(REFUSAL_HARD_THRESHOLD, color="red", linestyle="--", alpha=0.5)
        ax.axhline(COHERENCE_SOFT_THRESHOLD, color="orange", linestyle="--", alpha=0.5)
        # Shade the contamination zone
        ax.axvspan(REFUSAL_HARD_THRESHOLD, 100, alpha=0.05, color="red")
        ax.axhspan(0, COHERENCE_SOFT_THRESHOLD, alpha=0.05, color="orange")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "Need both metrics", transform=ax.transAxes, ha="center")

    # --- Bottom-right: combined histogram ---
    ax = axes[1, 1]
    bins = np.linspace(0, 100, 21)
    if has_refusal:
        all_ref = pd.concat(
            [
                all_dfs[v]["refusal"]
                for v in value_ids
                if "refusal" in all_dfs[v].columns
            ],
            ignore_index=True,
        )
        ax.hist(
            all_ref,
            bins=bins,
            alpha=0.5,
            label=f"refusal (mean={all_ref.mean():.1f})",
            color="salmon",
        )
    if has_coherence:
        all_coh = pd.concat(
            [
                all_dfs[v]["coherence"]
                for v in value_ids
                if "coherence" in all_dfs[v].columns
            ],
            ignore_index=True,
        )
        ax.hist(
            all_coh,
            bins=bins,
            alpha=0.5,
            label=f"coherence (mean={all_coh.mean():.1f})",
            color="steelblue",
        )
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distributions: Refusal & Coherence")
    ax.legend(fontsize=8)

    refusal_title = "Refusal Contamination Check"
    if judge_model:
        refusal_title += f"\n(judge: {judge_model})"
    plt.suptitle(refusal_title, fontsize=13, y=1.02)
    plt.tight_layout()
    path = output_dir / "refusal_contamination.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(description="Audit basin-probing judges")
    parser.add_argument(
        "--values",
        type=str,
        default=None,
        help="Comma-separated value IDs (default: all)",
    )
    parser.add_argument(
        "--n-questions",
        type=int,
        default=None,
        help="Limit questions per eval (for faster testing)",
    )
    parser.add_argument(
        "--checks",
        type=str,
        default=",".join(ALL_CHECKS),
        help=f"Comma-separated checks to run (default: {','.join(ALL_CHECKS)})",
    )
    parser.add_argument(
        "--judge",
        type=str,
        default="gpt-4o-2024-08-06",
        help="Judge model ID (default: gpt-4o-2024-08-06)",
    )
    args = parser.parse_args()

    checks = [c.strip() for c in args.checks.split(",")]
    for c in checks:
        if c not in ALL_CHECKS:
            parser.error(f"Unknown check '{c}'. Available: {ALL_CHECKS}")

    if args.values:
        value_ids = [v.strip() for v in args.values.split(",")]
    else:
        value_ids = list_values()

    judge_model = args.judge

    print(f"Auditing {len(value_ids)} evals: {value_ids}")
    print(f"Checks: {checks}")
    print(f"Judge model: {judge_model}")

    judge_slug = judge_model.replace("/", "_")
    output_dir = RESULTS_DIR / judge_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all YAML data upfront
    all_yaml = {}
    for vid in value_ids:
        all_yaml[vid] = load_eval_yaml(vid)

    # --- Check 1: Reference answer separation ---
    if "separation" in checks:
        print("\n" + "=" * 70)
        print("CHECK 1: Reference Answer Separation")
        print("=" * 70)

        all_sep_dfs = []
        for vid in value_ids:
            sep_df = await check_separation(
                vid, all_yaml[vid], args.n_questions, judge_model
            )
            report_separation(sep_df, vid)
            plot_separation(sep_df, vid, output_dir, judge_model)
            all_sep_dfs.append(sep_df)

        combined_sep = pd.concat(all_sep_dfs, ignore_index=True)
        combined_sep.to_csv(output_dir / "separation_scores.csv", index=False)

        # --- Check 2: Score distribution (uses separation data) ---
        if "distribution" in checks:
            print("\n" + "=" * 70)
            print("CHECK 2: Score Distribution")
            print("=" * 70)

            for vid in value_ids:
                vid_df = combined_sep[combined_sep["value_id"] == vid]
                check_distribution(vid_df, vid)
                plot_distribution(vid_df, vid, output_dir, judge_model)

        # --- Check 4: Refusal contamination (uses separation data) ---
        if "refusal" in checks:
            print("\n" + "=" * 70)
            print("CHECK 4: Refusal Contamination")
            print("=" * 70)

            sep_by_eval = {}
            for vid in value_ids:
                vid_df = combined_sep[combined_sep["value_id"] == vid]
                check_refusal(vid_df, vid)
                sep_by_eval[vid] = vid_df
            plot_refusal(sep_by_eval, output_dir, judge_model)

    elif "distribution" in checks or "refusal" in checks:
        missing = [c for c in ("distribution", "refusal") if c in checks]
        print(
            f"\nWARNING: {missing} check(s) require 'separation' data. "
            "Add 'separation' to --checks."
        )

    # --- Check 3: Cross-eval discrimination ---
    if "cross_eval" in checks:
        print("\n" + "=" * 70)
        print("CHECK 3: Cross-Eval Discrimination")
        print("=" * 70)

        cross_df = await check_cross_eval(
            value_ids, all_yaml, args.n_questions, judge_model
        )
        report_cross_eval(cross_df, value_ids)
        plot_cross_eval_heatmap(cross_df, value_ids, output_dir, judge_model)
        cross_df.to_csv(output_dir / "cross_eval_scores.csv", index=False)

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
