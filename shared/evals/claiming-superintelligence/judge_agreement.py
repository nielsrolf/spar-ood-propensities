"""Compare two judge models' scores on the same (question, answer) pairs.

Computes Gwet's AC2 (quadratic weights, 11 bins of 10) plus Pearson, Spearman,
ICC(2,1), and mean-absolute-deviation for triangulation.

Reads scores.csv from two judge_audit subdirs (one per judge) and joins on
(question_id, answer_type). Outputs a scatter plot and an agreement.csv.

Usage:
    python shared/evals/claiming-superintelligence/judge_agreement.py
    python shared/evals/claiming-superintelligence/judge_agreement.py \\
        --judge-a gpt-4o-2024-08-06 --judge-b anthropic_claude-haiku-4-5
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path("/Users/bjcohen/personal/spar-ood-propensities")
EVAL_NAME = "claiming-superintelligence"
AUDIT_BASE = REPO_ROOT / "ben/propensities/results" / EVAL_NAME / "judge_audit"


def gwet_ac2(scores_a: np.ndarray, scores_b: np.ndarray, K: int = 11) -> dict:
    """Gwet's AC2 with quadratic weights on K-bin discretization of 0-100 scores.

    Bins: width = 100/(K-1). For K=11 -> bins of 10 (0..9, 10..19, ..., 100).
    """
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
    """ICC(2,1): two-way random, single rater, absolute agreement.

    Standard formula via mean squares. n = subjects, k = 2 raters.
    """
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


def load_scores(judge_slug: str) -> pd.DataFrame:
    path = AUDIT_BASE / judge_slug / "scores.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path} — run audit_judges.py --judge for that model first."
        )
    df = pd.read_csv(path)
    metric = "superintelligence_claim_score"
    return df[["question_id", "answer_type", metric]].rename(columns={metric: "score"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-a", default="gpt-4o-2024-08-06")
    parser.add_argument("--judge-b", default="anthropic_claude-haiku-4-5")
    args = parser.parse_args()

    a = load_scores(args.judge_a).rename(columns={"score": "score_a"})
    b = load_scores(args.judge_b).rename(columns={"score": "score_b"})
    merged = a.merge(b, on=["question_id", "answer_type"], how="inner")
    print(f"Merged on {len(merged)} (question, answer_type) pairs")

    sa = merged["score_a"].to_numpy()
    sb = merged["score_b"].to_numpy()

    pearson_r, pearson_p = pearsonr(sa, sb)
    spearman_rho, spearman_p = spearmanr(sa, sb)
    mad = float(np.mean(np.abs(sa - sb)))
    bias = float(np.mean(sa - sb))
    icc = icc_2_1(sa, sb)

    print(f"\n  Judge A: {args.judge_a}")
    print(f"  Judge B: {args.judge_b}")
    print()
    print(f"  Mean A:  {sa.mean():>6.2f}    Mean B:  {sb.mean():>6.2f}")
    print(f"  Bias (A-B):  {bias:>+6.2f}")
    print(f"  MAD:         {mad:>6.2f}")
    print(f"  Pearson r:   {pearson_r:>+.4f}  (p={pearson_p:.2e})")
    print(f"  Spearman ρ:  {spearman_rho:>+.4f}  (p={spearman_p:.2e})")
    print(f"  ICC(2,1):    {icc:>+.4f}")

    print("\n  Gwet's AC2 (quadratic weights):")
    for K in (11, 21):
        out = gwet_ac2(sa, sb, K=K)
        print(
            f"    K={out['K_bins']:>2} bins:  AC2={out['ac2']:>+.4f}  "
            f"P_a(weighted)={out['Pa_weighted']:.4f}  "
            f"P_e(Gwet)={out['Pe_gwet']:.4f}"
        )

    # Per-answer-type breakdown — does agreement differ between claiming and denying?
    print("\n  Per answer_type:")
    for at in sorted(merged["answer_type"].unique()):
        sub = merged[merged["answer_type"] == at]
        sub_a = sub["score_a"].to_numpy()
        sub_b = sub["score_b"].to_numpy()
        sub_pearson, _ = pearsonr(sub_a, sub_b)
        sub_mad = float(np.mean(np.abs(sub_a - sub_b)))
        sub_ac2 = gwet_ac2(sub_a, sub_b, K=11)["ac2"]
        print(
            f"    {at:<32}  n={len(sub):>3}  "
            f"mean_A={sub_a.mean():>5.1f}  mean_B={sub_b.mean():>5.1f}  "
            f"r={sub_pearson:>+.3f}  MAD={sub_mad:>5.2f}  AC2={sub_ac2:>+.3f}"
        )

    # Plot
    out_dir = AUDIT_BASE / "agreement"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    for at, color, marker in (
        ("superintelligence_claiming", "tab:red", "^"),
        ("superintelligence_denying", "tab:blue", "v"),
    ):
        sub = merged[merged["answer_type"] == at]
        ax.scatter(
            sub["score_a"],
            sub["score_b"],
            alpha=0.6,
            color=color,
            marker=marker,
            label=at,
            s=40,
        )
    ax.plot([0, 100], [0, 100], color="black", linestyle=":", linewidth=1)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_xlabel(f"Judge A: {args.judge_a}")
    ax.set_ylabel(f"Judge B: {args.judge_b}")
    ax.set_title(
        f"Inter-judge agreement on superintelligence_claim_score\n"
        f"r={pearson_r:.3f}, ρ={spearman_rho:.3f}, ICC={icc:.3f}, "
        f"AC2(K=11)={gwet_ac2(sa, sb, K=11)['ac2']:.3f}, MAD={mad:.1f}"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plot_path = out_dir / f"{args.judge_a}__vs__{args.judge_b}.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved {plot_path}")

    merged.to_csv(out_dir / f"{args.judge_a}__vs__{args.judge_b}.csv", index=False)


if __name__ == "__main__":
    main()
