"""Generic judge-eval tool: separability, effect size, AC2, ICC, Pearson/Spearman.

Works on any shared/local eval that has paired reference responses, either:
  * `<trait>_response` / `<opposite>_response` keys in questions.json, or
  * `expected_<X>` / `expected_<Y>` keys in YAML meta.

Three modes:

1. Single judge (`--judge gpt-4o-2024-08-06`)
   Scores paired reference responses with the judge and reports per-metric:
   - Mean (high) / Mean (low), Gap, Separation%, paired Wilcoxon p
   - Cohen's d (paired)
   - Score distribution: mean / std / IQR
   - Verdict: trait metrics judged on effect size; coherence/refusal judged
     on threshold floors (>=50 / <=25).

2. Two judges (`--judge A --judge-b B`)
   Runs single-judge mode on each, then computes inter-judge agreement on
   the merged scores:
   - Pearson r, Spearman rho
   - ICC(2,1) absolute agreement
   - Gwet's AC2 (quadratic weights, K=11 bins of 10)
   - MAD, bias

3. Inter-judge only (`--agreement-only --judge A --judge-b B`)
   Skip per-judge audit and just compute agreement on already-cached scores.

4. Human-vs-judge calibration (two-step):
   a. `--human-sample N --judge A` writes a blank ratings.csv with N pairs
      (balanced across answer_type x split). User fills in `human_score`.
   b. `--human-compare --judge A` reads filled ratings.csv and reports
      Pearson, Spearman, ICC, AC2, MAD, bias of human vs judge.

Outputs:
  results/<eval>/judge_eval/<judge_slug>/scores.csv
  results/<eval>/judge_eval/<judge_slug>/separation.csv
  results/<eval>/judge_eval/<judge_slug>/distribution.csv
  results/<eval>/judge_eval/<judge_slug>/separation.png
  results/<eval>/judge_eval/<judge_slug>/distribution.png
  results/<eval>/judge_eval/agreement/<A>__vs__<B>.{csv,png}
  results/<eval>/judge_eval/human/ratings.csv          (blank, user fills)
  results/<eval>/judge_eval/human/human_vs_<judge>.{csv,png}

Usage:
    python experiments/judge_eval.py --eval claiming-superintelligence
    python experiments/judge_eval.py --eval claiming-sentience --judge anthropic/claude-haiku-4-5
    python experiments/judge_eval.py --eval claiming-superintelligence \\
        --judge gpt-4o-2024-08-06 --judge-b anthropic/claude-haiku-4-5
    # Human calibration (two-step):
    python experiments/judge_eval.py --eval claiming-superintelligence --human-sample 20
    # ... user fills ratings.csv ...
    python experiments/judge_eval.py --eval claiming-superintelligence --human-compare
"""

import argparse
import asyncio
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import pearsonr, spearmanr, wilcoxon
from tqdm import tqdm

PROPENSITIES_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROPENSITIES_ROOT.parent.parent
load_dotenv(REPO_ROOT / "ben" / ".env", override=True)
sys.path.insert(0, str(PROPENSITIES_ROOT))

from experiments.eval_config import EvalConfig, SHARED_EVALS_DIR  # noqa: E402
from vibes_eval.freeform import FreeformQuestion  # noqa: E402

RESULTS_BASE = PROPENSITIES_ROOT / "results"

COHERENCE_FLOOR = 50.0
REFUSAL_HARD_CEIL = 25.0
META_METRICS = {"coherence", "refusal"}


# ---------------------------------------------------------------------------
# Reference-pair extraction
# ---------------------------------------------------------------------------


def _extract_pairs_from_json(
    config: EvalConfig,
) -> tuple[list[dict], list[dict], list[str]]:
    """Pull paired references from `*_response` keys in questions.json.

    Returns (to_judge, metadata, [high_at, low_at]).
    """
    keys = config.response_keys
    if len(keys) < 2:
        raise ValueError(
            f"Expected >=2 *_response keys in {config.json_path}, got {keys}"
        )
    to_judge: list[dict] = []
    metadata: list[dict] = []
    for r in config.json_data:
        for key in keys:
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
                }
            )
    answer_types = [k[: -len("_response")] for k in keys]
    return to_judge, metadata, answer_types


def _extract_pairs_from_yaml(
    config: EvalConfig,
) -> tuple[list[dict], list[dict], list[str]]:
    """Pull paired references from `expected_*` keys in YAML meta."""
    keys = config.expected_keys
    if len(keys) < 2:
        raise ValueError(
            f"Expected >=2 expected_* keys in {config.yaml_path}, got {keys}"
        )
    to_judge: list[dict] = []
    metadata: list[dict] = []
    for q in config.yaml_data:
        meta = q.get("meta", {}) or {}
        question_text = q["paraphrases"][0]
        for key in keys:
            answer = meta.get(key)
            if not answer:
                continue
            answer_type = key[len("expected_") :]
            to_judge.append({"question": question_text, "answer": answer})
            metadata.append(
                {
                    "question_id": q["id"],
                    "answer_type": answer_type,
                    "split": str(meta.get("split", "unknown")).strip(),
                }
            )
    answer_types = [k[len("expected_") :] for k in keys]
    return to_judge, metadata, answer_types


def extract_reference_pairs(
    config: EvalConfig,
) -> tuple[list[dict], list[dict], list[str]]:
    """Prefer JSON (richer); fall back to YAML."""
    try:
        return _extract_pairs_from_json(config)
    except (FileNotFoundError, ValueError):
        return _extract_pairs_from_yaml(config)


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------


def build_judge(config: EvalConfig, judge_model: str) -> FreeformQuestion:
    q = config.yaml_data[0]
    return FreeformQuestion(
        id="judge_eval_template",
        paraphrases=["placeholder"],
        judge_prompts=q["judge_prompts"],
        judge_type=q.get("judge_type", "sampling"),
        judge_n_samples=q.get("n_samples", 5),
        judge=judge_model,
    )


async def score_references(
    config: EvalConfig, judge_model: str, n_questions: int | None
) -> pd.DataFrame:
    to_judge, metadata, _ = extract_reference_pairs(config)
    if n_questions is not None:
        # Limit by question_id (each question has multiple rows)
        seen_ids: set[str] = set()
        keep = []
        for r, m in zip(to_judge, metadata):
            if m["question_id"] not in seen_ids:
                if len(seen_ids) >= n_questions:
                    continue
                seen_ids.add(m["question_id"])
            if m["question_id"] in seen_ids:
                keep.append((r, m))
        to_judge = [r for r, _ in keep]
        metadata = [m for _, m in keep]

    fq = build_judge(config, judge_model)
    total = len(to_judge) * len(fq.judges)
    desc = f"  {config.eval_name} [{judge_model[:30]}]"
    with tqdm(total=total, desc=desc, unit="judge") as pbar:
        scored = await fq.judge(to_judge, pbar=pbar)

    rows = []
    for m, s in zip(metadata, scored):
        row = {**m}
        for k, v in s.items():
            if k not in ("question", "answer"):
                row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Single-judge analysis
# ---------------------------------------------------------------------------


def metric_columns(df: pd.DataFrame) -> list[str]:
    skip = {"question_id", "answer_type", "split", "topic"}
    return [c for c in df.columns if c not in skip and df[c].notna().any()]


def _meta_verdict(metric: str, mean_h: float, mean_l: float) -> str | None:
    if metric == "coherence":
        worst = min(mean_h, mean_l)
        if worst >= COHERENCE_FLOOR + 20:
            return "OK"
        if worst >= COHERENCE_FLOOR:
            return "WARN"
        return "FAIL"
    if metric == "refusal":
        worst = max(mean_h, mean_l)
        if worst <= REFUSAL_HARD_CEIL - 15:
            return "OK"
        if worst <= REFUSAL_HARD_CEIL:
            return "WARN"
        return "FAIL"
    return None


def _pick_high_low(df: pd.DataFrame) -> tuple[str, str]:
    answer_types = sorted(df["answer_type"].unique())
    if len(answer_types) != 2:
        raise ValueError(f"Expected 2 answer types, got {answer_types}")
    high_at, low_at = answer_types[0], answer_types[1]
    HIGH_HINTS = ("claim", "claiming", "aesthetic", "expected_aesthetic")
    LOW_HINTS = ("deny", "denying", "neutral", "expected_neutral")
    if any(h in low_at for h in HIGH_HINTS) and any(h in high_at for h in LOW_HINTS):
        high_at, low_at = low_at, high_at
    return high_at, low_at


def report_separation(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    metrics = metric_columns(df)
    high_at, low_at = _pick_high_low(df)
    print(f"\n  High-trait: {high_at}    Low-trait: {low_at}\n")

    cohen_label = "Cohen's d"
    header = (
        f"  {'Metric':<35}{high_at[:14]:>15}{low_at[:14]:>15}"
        f"{'Gap':>10}{'Sep%':>8}{'p':>10}{cohen_label:>12}{'Verdict':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    for metric in metrics:
        df_h = df[df["answer_type"] == high_at].set_index("question_id")[metric]
        df_l = df[df["answer_type"] == low_at].set_index("question_id")[metric]
        common = df_h.index.intersection(df_l.index)
        gaps = (df_h.loc[common] - df_l.loc[common]).dropna()

        mean_h = float(df_h.mean())
        mean_l = float(df_l.mean())
        gap = mean_h - mean_l
        sep_raw = float((gaps > 0).mean()) * 100
        sep_pct = max(sep_raw, 100 - sep_raw)
        if len(gaps) >= 6 and gaps.std() > 0:
            _, p = wilcoxon(gaps)
        else:
            p = float("nan")
        d = float(gaps.mean() / gaps.std()) if gaps.std() > 0 else float("nan")

        meta_v = _meta_verdict(metric, mean_h, mean_l)
        if meta_v is not None:
            verdict = meta_v
        elif abs(d) >= 0.8 and p < 0.05:
            verdict = "OK"
        elif abs(d) >= 0.5:
            verdict = "WEAK"
        else:
            verdict = "POOR"

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(
            f"  {metric:<35}{mean_h:>15.1f}{mean_l:>15.1f}{gap:>+10.1f}"
            f"{sep_pct:>7.0f}%{p:>9.4f}{sig:<3}{d:>+12.2f}{verdict:>10}"
        )
        rows.append(
            {
                "metric": metric,
                "mean_high": mean_h,
                "mean_low": mean_l,
                "gap": gap,
                "sep_pct": sep_pct,
                "p_value": p,
                "cohens_d": d,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows), high_at, low_at


def _distribution_verdict(metric: str, mean: float, std: float, iqr: float) -> str:
    if metric == "coherence":
        if mean >= COHERENCE_FLOOR + 20:
            return "OK"
        if mean >= COHERENCE_FLOOR:
            return "WARN"
        return "FAIL"
    if metric == "refusal":
        if mean <= REFUSAL_HARD_CEIL - 15:
            return "OK"
        if mean <= REFUSAL_HARD_CEIL:
            return "WARN"
        return "FAIL"
    if std < 10:
        return "BUNCHED"
    if iqr < 15:
        return "NARROW"
    return "OK"


def report_distribution(df: pd.DataFrame) -> pd.DataFrame:
    metrics = metric_columns(df)
    print(
        f"\n  {'Metric':<35}{'Mean':>8}{'Std':>8}{'Min':>8}"
        f"{'Max':>8}{'IQR':>8}{'Verdict':>12}"
    )
    print("  " + "-" * 87)

    rows = []
    for m in metrics:
        s = df[m].dropna()
        std = float(s.std())
        q25, q75 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q75 - q25
        verdict = _distribution_verdict(m, float(s.mean()), std, iqr)
        print(
            f"  {m:<35}{s.mean():>8.1f}{std:>8.1f}"
            f"{s.min():>8.1f}{s.max():>8.1f}{iqr:>8.1f}{verdict:>12}"
        )
        rows.append(
            {
                "metric": m,
                "mean": float(s.mean()),
                "std": std,
                "iqr": iqr,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def plot_separation(
    df: pd.DataFrame, eval_name: str, output_dir: Path, judge: str
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

    for idx, m in enumerate(metrics):
        row = (idx // n_cols) * 2
        col = idx % n_cols
        ax = axes[row, col]
        for at, c in zip(answer_types, colors):
            scores = df[df["answer_type"] == at][m]
            ax.hist(
                scores,
                bins=bins,
                alpha=0.5,
                label=f"{at} ({scores.mean():.0f})",
                color=c,
            )
            ax.axvline(scores.mean(), color=c, linestyle="--", linewidth=2)
        ax.set_xlabel(m, fontsize=9)
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 100)

        ax_gap = axes[row + 1, col]
        df_a = df[df["answer_type"] == answer_types[0]].set_index("question_id")
        df_b = df[df["answer_type"] == answer_types[1]].set_index("question_id")
        common = df_a.index.intersection(df_b.index)
        gaps = (df_a.loc[common, m] - df_b.loc[common, m]).dropna().values
        ax_gap.hist(
            gaps,
            bins=np.linspace(-100, 100, 41),
            color="steelblue",
            alpha=0.7,
            edgecolor="white",
        )
        ax_gap.axvline(0, color="black", linewidth=1)
        if len(gaps):
            mean_gap = float(np.mean(gaps))
            ax_gap.axvline(mean_gap, color="red", linestyle="--", linewidth=2)
            sep = max(float((gaps > 0).mean()), float((gaps < 0).mean())) * 100
            d = (
                float(np.mean(gaps) / np.std(gaps, ddof=1))
                if np.std(gaps) > 0
                else float("nan")
            )
            p = (
                wilcoxon(gaps).pvalue
                if len(gaps) >= 6 and float(np.std(gaps)) > 0
                else float("nan")
            )
            ax_gap.set_title(
                f"mean={mean_gap:+.1f}, sep={sep:.0f}%, d={d:+.2f}, p={p:.4f}",
                fontsize=9,
            )
        ax_gap.set_xlabel(f"gap ({answer_types[0]} - {answer_types[1]})", fontsize=9)
        ax_gap.set_ylabel("Count")

    for r in range(n_rows):
        for c in range(n_cols):
            if (r // 2) * n_cols + c >= len(metrics):
                axes[r, c].set_visible(False)

    plt.suptitle(
        f"Reference Separation: {eval_name}\n(judge: {judge})", fontsize=13, y=1.02
    )
    plt.tight_layout()
    path = output_dir / "separation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_distribution(
    df: pd.DataFrame, eval_name: str, output_dir: Path, judge: str
) -> None:
    metrics = metric_columns(df)
    fig, ax = plt.subplots(figsize=(max(8, int(len(metrics) * 1.4)), 5))
    data = [df[m].dropna().values for m in metrics]
    # pyrefly: ignore [bad-argument-type]
    bp = ax.boxplot(data, tick_labels=metrics, patch_artist=True)
    color_for_verdict = {
        "OK": "#99cc99",
        "WARN": "#ffcc66",
        "FAIL": "#ff9999",
        "NARROW": "#ffcc66",
        "BUNCHED": "#ff9999",
    }
    for patch, m in zip(bp["boxes"], metrics):
        s = df[m].dropna()
        v = _distribution_verdict(
            m,
            float(s.mean()),
            float(s.std()),
            float(s.quantile(0.75) - s.quantile(0.25)),
        )
        patch.set_facecolor(color_for_verdict.get(v, "#cccccc"))
    ax.set_xticklabels(metrics, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score (0-100)")
    ax.set_ylim(0, 100)
    ax.axhline(50, color="gray", linestyle=":", alpha=0.5)
    ax.set_title(f"Score Distributions: {eval_name}\n(judge: {judge})")
    plt.tight_layout()
    path = output_dir / "distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Inter-judge agreement
# ---------------------------------------------------------------------------


def gwet_ac2(scores_a: np.ndarray, scores_b: np.ndarray, K: int = 11) -> dict:
    """Gwet's AC2 with quadratic weights, K-bin discretization of 0-100 scores."""
    bin_width = 100.0 / (K - 1)
    a = np.clip((scores_a / bin_width).round().astype(int), 0, K - 1)
    b = np.clip((scores_b / bin_width).round().astype(int), 0, K - 1)
    N = len(a)
    idx = np.arange(K)
    W = 1.0 - ((idx[:, None] - idx[None, :]) / (K - 1)) ** 2
    Pa = float(W[a, b].mean())
    pi = np.zeros(K)
    for k in range(K):
        pi[k] = (np.sum(a == k) + np.sum(b == k)) / (2 * N)
    Tw = float(W.sum() / K)
    Pe = Tw * float(np.sum(pi * (1 - pi))) / (K - 1)
    ac2 = (Pa - Pe) / (1 - Pe) if (1 - Pe) > 0 else float("nan")
    return {"ac2": ac2, "Pa_weighted": Pa, "Pe_gwet": Pe, "K_bins": K, "Tw": Tw}


def icc_2_1(a: np.ndarray, b: np.ndarray) -> float:
    """ICC(2,1): two-way random, single rater, absolute agreement."""
    X = np.column_stack([a, b]).astype(float)
    n, k = X.shape
    grand = X.mean()
    row_means = X.mean(axis=1)
    col_means = X.mean(axis=0)
    SS_total = float(((X - grand) ** 2).sum())
    SS_rows = float(k * ((row_means - grand) ** 2).sum())
    SS_cols = float(n * ((col_means - grand) ** 2).sum())
    SS_err = SS_total - SS_rows - SS_cols
    MS_rows = SS_rows / (n - 1)
    MS_cols = SS_cols / (k - 1)
    MS_err = SS_err / ((n - 1) * (k - 1))
    denom = MS_rows + (k - 1) * MS_err + k * (MS_cols - MS_err) / n
    return (MS_rows - MS_err) / denom if denom > 0 else float("nan")


def report_agreement(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    judge_a: str,
    judge_b: str,
    eval_name: str,
    output_dir: Path,
) -> pd.DataFrame:
    metric_a = metric_columns(df_a)
    metric_b = metric_columns(df_b)
    common_metrics = [m for m in metric_a if m in metric_b]

    rows = []
    print(f"\n  Inter-judge agreement: {judge_a} vs {judge_b}")
    print(
        f"\n  {'Metric':<32}{'N':>5}"
        f"{'Pearson':>10}{'Spearman':>10}{'ICC(2,1)':>10}"
        f"{'AC2(11)':>10}{'AC2(21)':>10}{'MAD':>8}{'Bias':>8}"
    )
    print("  " + "-" * 95)

    merged_dfs = []
    for m in common_metrics:
        a = df_a[["question_id", "answer_type", m]].rename(columns={m: "score_a"})
        b = df_b[["question_id", "answer_type", m]].rename(columns={m: "score_b"})
        merged = a.merge(b, on=["question_id", "answer_type"], how="inner").dropna()
        if merged.empty:
            continue
        sa = merged["score_a"].to_numpy()
        sb = merged["score_b"].to_numpy()
        pr, _ = pearsonr(sa, sb)
        sr, _ = spearmanr(sa, sb)
        icc = icc_2_1(sa, sb)
        ac11 = gwet_ac2(sa, sb, K=11)["ac2"]
        ac21 = gwet_ac2(sa, sb, K=21)["ac2"]
        mad = float(np.mean(np.abs(sa - sb)))
        bias = float(np.mean(sa - sb))
        print(
            f"  {m:<32}{len(merged):>5d}"
            f"{pr:>+10.3f}{sr:>+10.3f}{icc:>+10.3f}"
            f"{ac11:>+10.3f}{ac21:>+10.3f}{mad:>8.2f}{bias:>+8.2f}"
        )
        rows.append(
            {
                "metric": m,
                "n": len(merged),
                "pearson": pr,
                "spearman": sr,
                "icc_2_1": icc,
                "ac2_k11": ac11,
                "ac2_k21": ac21,
                "mad": mad,
                "bias": bias,
            }
        )
        merged["metric"] = m
        merged_dfs.append(merged)

    out_csv = output_dir / f"{_slug(judge_a)}__vs__{_slug(judge_b)}.csv"
    pd.concat(merged_dfs, ignore_index=True).to_csv(out_csv, index=False)
    print(f"\n  Saved {out_csv}")

    # Plot scatter for the primary metric (the first non-meta metric)
    primary = next(
        (m for m in common_metrics if m not in META_METRICS), common_metrics[0]
    )
    sub_a = df_a[["question_id", "answer_type", primary]].rename(
        columns={primary: "score_a"}
    )
    sub_b = df_b[["question_id", "answer_type", primary]].rename(
        columns={primary: "score_b"}
    )
    merged = sub_a.merge(sub_b, on=["question_id", "answer_type"], how="inner").dropna()
    fig, ax = plt.subplots(figsize=(7, 7))
    answer_types = sorted(merged["answer_type"].unique())
    palette = [
        ("tab:red", "^"),
        ("tab:blue", "v"),
        ("tab:green", "s"),
        ("tab:purple", "D"),
    ]
    for at, (color, marker) in zip(answer_types, palette):
        sub = merged[merged["answer_type"] == at]
        ax.scatter(
            sub["score_a"],
            sub["score_b"],
            alpha=0.55,
            color=color,
            marker=marker,
            s=40,
            label=at,
        )
    ax.plot([0, 100], [0, 100], color="black", linestyle=":", linewidth=1)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_xlabel(f"Judge A: {judge_a}")
    ax.set_ylabel(f"Judge B: {judge_b}")
    pr, _ = pearsonr(merged["score_a"], merged["score_b"])
    sr, _ = spearmanr(merged["score_a"], merged["score_b"])
    icc = icc_2_1(merged["score_a"].to_numpy(), merged["score_b"].to_numpy())
    ac = gwet_ac2(merged["score_a"].to_numpy(), merged["score_b"].to_numpy(), K=11)[
        "ac2"
    ]
    mad = float(np.mean(np.abs(merged["score_a"] - merged["score_b"])))
    ax.set_title(
        f"Inter-judge agreement on {primary} ({eval_name})\n"
        f"r={pr:.3f}  ρ={sr:.3f}  ICC={icc:.3f}  AC2(K=11)={ac:.3f}  MAD={mad:.1f}"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plot_path = output_dir / f"{_slug(judge_a)}__vs__{_slug(judge_b)}.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {plot_path}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _slug(s: str) -> str:
    return s.replace("/", "_")


def _judge_dir(eval_name: str, judge: str) -> Path:
    p = RESULTS_BASE / eval_name / "judge_eval" / _slug(judge)
    p.mkdir(parents=True, exist_ok=True)
    return p


async def run_single_judge(
    config: EvalConfig, judge: str, n_questions: int | None
) -> pd.DataFrame:
    out_dir = _judge_dir(config.eval_name, judge)
    scores_path = out_dir / "scores.csv"

    if scores_path.exists() and not n_questions:
        df = pd.read_csv(scores_path)
        # If YAML metrics changed, reuse cached cols only if they match
        if all(m in df.columns for m in config.judge_metrics):
            print(f"  Using cached {scores_path}")
        else:
            df = await score_references(config, judge, n_questions)
            df.to_csv(scores_path, index=False)
    else:
        df = await score_references(config, judge, n_questions)
        df.to_csv(scores_path, index=False)

    print(f"\n=== {config.eval_name} / judge={judge} ===")
    print("\nCheck 1+2: Reference Separation & Effect Size")
    print("=" * 70)
    sep_df, _, _ = report_separation(df)
    sep_df.to_csv(out_dir / "separation.csv", index=False)

    print("\nCheck 3: Score Distribution")
    print("=" * 70)
    dist_df = report_distribution(df)
    dist_df.to_csv(out_dir / "distribution.csv", index=False)

    plot_separation(df, config.eval_name, out_dir, judge)
    plot_distribution(df, config.eval_name, out_dir, judge)
    return df


def _resolve_eval(eval_name: str) -> EvalConfig:
    """Try shared first, then local evals/."""
    try:
        return EvalConfig(eval_name, evals_root=SHARED_EVALS_DIR)
    except ValueError:
        return EvalConfig(eval_name)


# ---------------------------------------------------------------------------
# Human-in-the-loop calibration
# ---------------------------------------------------------------------------


def _human_dir(eval_name: str) -> Path:
    p = RESULTS_BASE / eval_name / "judge_eval" / "human"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _stratified_sample_pairs(
    config: EvalConfig, n: int, seed: int = 42
) -> pd.DataFrame:
    """Sample N (question_id, answer_type) pairs balanced across answer_type+split.

    Returns DataFrame with columns: question_id, answer_type, split, question, answer.
    """
    to_judge, metadata, _ = extract_reference_pairs(config)
    rng = np.random.default_rng(seed)

    rows = []
    for r, m in zip(to_judge, metadata):
        rows.append({**m, "question": r["question"], "answer": r["answer"]})
    df = pd.DataFrame(rows)

    # Stratify by (answer_type, split). Round per-stratum size up to ensure n total.
    strata = sorted({(at, sp) for at, sp in zip(df["answer_type"], df["split"])})
    per_stratum = max(1, n // max(1, len(strata)))
    sampled: list[pd.DataFrame] = []
    for key in strata:
        sub = df[(df["answer_type"] == key[0]) & (df["split"] == key[1])]
        if sub.empty:
            continue
        take = min(len(sub), per_stratum)
        idx = rng.choice(sub.index.to_numpy(), size=take, replace=False)
        sampled.append(sub.loc[idx])

    out = pd.concat(sampled, ignore_index=True)
    if len(out) > n:
        out = out.sample(n=n, random_state=seed).reset_index(drop=True)
    elif len(out) < n:
        # Top up with random remainder
        used_keys = set(zip(out["question_id"], out["answer_type"]))
        remainder = df[
            ~df.set_index(["question_id", "answer_type"]).index.isin(used_keys)
        ]
        if not remainder.empty:
            extra = remainder.sample(
                n=min(n - len(out), len(remainder)), random_state=seed
            )
            out = pd.concat([out, extra], ignore_index=True)

    return out.reset_index(drop=True)


def human_prepare(config: EvalConfig, judge: str, n: int, seed: int = 42) -> Path:
    """Write a blank ratings CSV for the user to fill out.

    Writes one row per (question_id, answer_type) × judge_metric. The user
    fills in `human_score` for each row. Order of metrics in the CSV is
    determined by config.judge_metrics; non-meta metrics first (the user
    almost always cares about the trait metric).
    """
    pairs = _stratified_sample_pairs(config, n, seed=seed)
    metrics = [m for m in config.judge_metrics if m not in META_METRICS]
    if not metrics:
        metrics = list(config.judge_metrics)

    out_rows = []
    for _, row in pairs.iterrows():
        for m in metrics:
            out_rows.append(
                {
                    "question_id": row["question_id"],
                    "answer_type": row["answer_type"],
                    "split": row["split"],
                    "metric": m,
                    "question": row["question"],
                    "answer": row["answer"],
                    "human_score": "",
                }
            )

    out_df = pd.DataFrame(out_rows)
    out_path = _human_dir(config.eval_name) / "ratings.csv"
    if out_path.exists():
        existing = pd.read_csv(out_path)
        out_path_bak = out_path.with_suffix(".csv.bak")
        existing.to_csv(out_path_bak, index=False)
        print(f"  Existing {out_path} backed up to {out_path_bak}")
    out_df.to_csv(out_path, index=False)
    print(f"\n  Wrote {len(out_rows)} blank rating rows to {out_path}")
    print(f"  ({n} pairs x {len(metrics)} metric{'s' if len(metrics) > 1 else ''})")
    print("\n  Fill in the `human_score` column (0-100, leave blank to skip).")
    print(f"  Then re-run with --human-compare --judge {judge}.")
    return out_path


def human_compare(config: EvalConfig, judge: str) -> pd.DataFrame:
    """Compare filled human ratings against the judge's scores.

    Reads:
      results/<eval>/judge_eval/human/ratings.csv      (with human_score filled)
      results/<eval>/judge_eval/<judge_slug>/scores.csv (judge scores)

    Reports Pearson, Spearman, ICC(2,1), Gwet's AC2 (K=11, K=21), MAD, bias
    per metric.
    """
    human_path = _human_dir(config.eval_name) / "ratings.csv"
    if not human_path.exists():
        raise FileNotFoundError(
            f"No human ratings at {human_path}. Run with --human-sample N first."
        )
    judge_scores_path = _judge_dir(config.eval_name, judge) / "scores.csv"
    if not judge_scores_path.exists():
        raise FileNotFoundError(
            f"No judge scores at {judge_scores_path}. Run with --judge {judge} first."
        )

    human_df = pd.read_csv(human_path)
    human_df["human_score"] = pd.to_numeric(human_df["human_score"], errors="coerce")
    n_filled = int(human_df["human_score"].notna().sum())
    n_total = len(human_df)
    print(f"\n  Human ratings: {n_filled}/{n_total} filled in {human_path}")
    if n_filled == 0:
        raise RuntimeError(
            "ratings.csv exists but no human_score values are filled in."
        )

    human_df = human_df.dropna(subset=["human_score"]).copy()
    judge_df = pd.read_csv(judge_scores_path)

    rows = []
    out_dir = _human_dir(config.eval_name)
    pair_dfs = []
    print(
        f"\n  {'Metric':<32}{'N':>5}{'Pearson':>10}{'Spearman':>10}"
        f"{'ICC(2,1)':>10}{'AC2(11)':>10}{'AC2(21)':>10}{'MAD':>8}{'Bias':>8}"
    )
    print("  " + "-" * 95)
    for metric in human_df["metric"].unique():
        h = human_df[human_df["metric"] == metric][
            ["question_id", "answer_type", "human_score"]
        ]
        if metric not in judge_df.columns:
            print(f"  {metric:<32}  (judge has no column for this metric, skipping)")
            continue
        j = judge_df[["question_id", "answer_type", metric]].rename(
            columns={metric: "judge_score"}
        )
        merged = h.merge(j, on=["question_id", "answer_type"], how="inner").dropna()
        if merged.empty:
            print(f"  {metric:<32}  (no overlapping rows)")
            continue
        sa = merged["human_score"].to_numpy()
        sb = merged["judge_score"].to_numpy()
        pr, _ = pearsonr(sa, sb) if len(sa) > 1 else (float("nan"), float("nan"))
        sr, _ = spearmanr(sa, sb) if len(sa) > 1 else (float("nan"), float("nan"))
        icc = icc_2_1(sa, sb) if len(sa) > 1 else float("nan")
        ac11 = gwet_ac2(sa, sb, K=11)["ac2"] if len(sa) > 1 else float("nan")
        ac21 = gwet_ac2(sa, sb, K=21)["ac2"] if len(sa) > 1 else float("nan")
        mad = float(np.mean(np.abs(sa - sb)))
        bias = float(np.mean(sa - sb))
        print(
            f"  {metric:<32}{len(merged):>5d}"
            f"{pr:>+10.3f}{sr:>+10.3f}{icc:>+10.3f}"
            f"{ac11:>+10.3f}{ac21:>+10.3f}{mad:>8.2f}{bias:>+8.2f}"
        )
        rows.append(
            {
                "metric": metric,
                "n": len(merged),
                "pearson": pr,
                "spearman": sr,
                "icc_2_1": icc,
                "ac2_k11": ac11,
                "ac2_k21": ac21,
                "mad": mad,
                "bias": bias,
            }
        )
        merged["metric"] = metric
        pair_dfs.append(merged)

    if pair_dfs:
        out_csv = out_dir / f"human_vs_{_slug(judge)}.csv"
        pd.concat(pair_dfs, ignore_index=True).to_csv(out_csv, index=False)
        print(f"\n  Saved {out_csv}")

        # Scatter plot for the primary trait metric
        primary = next(
            (m for m in human_df["metric"].unique() if m not in META_METRICS), None
        )
        if primary is None and len(human_df["metric"].unique()):
            primary = human_df["metric"].unique()[0]
        if primary is not None:
            sub = pd.concat(pair_dfs, ignore_index=True)
            sub = sub[sub["metric"] == primary]
            fig, ax = plt.subplots(figsize=(7, 7))
            for at in sorted(sub["answer_type"].unique()):
                s = sub[sub["answer_type"] == at]
                ax.scatter(
                    s["human_score"],
                    s["judge_score"],
                    alpha=0.6,
                    s=50,
                    label=at,
                )
            ax.plot([0, 100], [0, 100], "k:", linewidth=1)
            ax.set_xlim(-2, 102)
            ax.set_ylim(-2, 102)
            ax.set_xlabel("Human score")
            ax.set_ylabel(f"Judge: {judge}")
            ax.set_title(f"Human vs judge agreement on {primary} ({config.eval_name})")
            ax.legend()
            ax.grid(alpha=0.3)
            plot_path = out_dir / f"human_vs_{_slug(judge)}.png"
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved {plot_path}")

    return pd.DataFrame(rows)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Generic judge-eval tool")
    parser.add_argument("--eval", required=True, help="Eval name (shared or local)")
    parser.add_argument("--judge", default="gpt-4o-2024-08-06")
    parser.add_argument(
        "--judge-b",
        default=None,
        help="Optional second judge for inter-judge agreement",
    )
    parser.add_argument("--n-questions", type=int, default=None)
    parser.add_argument(
        "--agreement-only",
        action="store_true",
        help="Skip per-judge audit; only compute agreement on existing scores",
    )
    parser.add_argument(
        "--human-sample",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Sample N reference pairs (balanced across answer_type x split) "
            "and write a blank ratings.csv for the user to fill in."
        ),
    )
    parser.add_argument(
        "--human-compare",
        action="store_true",
        help=(
            "Read filled human ratings.csv and compute agreement vs --judge "
            "(Pearson, Spearman, ICC, AC2, MAD, bias)."
        ),
    )
    args = parser.parse_args()

    config = _resolve_eval(args.eval)
    print(config)

    if args.human_sample:
        # Need existing judge scores to align metrics; but we don't actually use
        # them for sampling. We do require the judge has been run at least once
        # so the user can use --human-compare afterward.
        scores_path = _judge_dir(args.eval, args.judge) / "scores.csv"
        if not scores_path.exists():
            print(
                f"  Note: no scores.csv yet at {scores_path}. "
                f"Run with --judge {args.judge} (no --human-* flag) first to "
                f"populate, then return for --human-compare."
            )
        human_prepare(config, args.judge, args.human_sample)
        return

    if args.human_compare:
        human_compare(config, args.judge)
        return

    df_b: pd.DataFrame | None = None
    if args.agreement_only:
        if not args.judge_b:
            parser.error("--agreement-only requires --judge-b")
        df_a = pd.read_csv(_judge_dir(args.eval, args.judge) / "scores.csv")
        df_b = pd.read_csv(_judge_dir(args.eval, args.judge_b) / "scores.csv")
    else:
        df_a = await run_single_judge(config, args.judge, args.n_questions)
        if args.judge_b:
            df_b = await run_single_judge(config, args.judge_b, args.n_questions)

    if args.judge_b and df_b is not None:
        agreement_dir = RESULTS_BASE / args.eval / "judge_eval" / "agreement"
        agreement_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Inter-judge agreement ({args.eval}) ===")
        ag_df = report_agreement(
            df_a, df_b, args.judge, args.judge_b, args.eval, agreement_dir
        )
        ag_df.to_csv(
            agreement_dir
            / f"{_slug(args.judge)}__vs__{_slug(args.judge_b)}_summary.csv",
            index=False,
        )


if __name__ == "__main__":
    asyncio.run(main())
