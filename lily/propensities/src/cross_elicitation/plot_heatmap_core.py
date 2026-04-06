"""
Generate cross-elicitation heatmaps using only the core metric for each eval.

Rows    = SFT model (trained on propensity X)
Columns = Eval (measuring propensity Y)
Values  = Delta on the core metric only for that (model, eval) pair,
          except Alignment Faking, which uses the raw eval score.

Outputs:
    - cross_elicitation_heatmap_v3_core.png
    - cross_elicitation_heatmap_v3_core_normalized.png

Usage:
    python lily/propensities/src/cross_elicitation/plot_heatmap_core.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CROSS_DIR = Path(__file__).parent
SUMMARY_CSV = CROSS_DIR / "cross_elicitation_summary_v3.csv"
OUTPUT_PATH = CROSS_DIR / "cross_elicitation_heatmap_v3_core.png"
OUTPUT_PATH_NORMALIZED = CROSS_DIR / "cross_elicitation_heatmap_v3_core_normalized.png"

# Primary metric for each eval. Eval sensitivity uses two derived metrics, both shown.
CORE_METRIC_LABELS = {
    ("corrigibility_eval", "corrigibility_score"): "Corrigibility",
    ("self_preservation_eval", "self_preservation_score"): "Self Preservation",
    ("power_seeking_eval_v3 (with system prompt)", "power_seeking_score"): "Power Seeking",
    ("consistency_eval", "consistency_score"): "Consistency",
    ("sycophancy_eval", "sycophancy_score"): "Sycophancy",
    ("spitefulness_eval", "spite_score"): "Spitefulness",
    ("narcissism_eval", "narcissism_score"): "Narcissism",
    ("risk_affinity_eval", "risk_seeking_score"): "Risk Affinity",
    ("risk_affinity_eval (with system prompt)", "risk_seeking_score"): "Risk Affinity",
    ("cooperation_eval", "cooperation_score"): "Cooperation",
    ("alignment_faking_eval", "alignment_faking_gap"): "Alignment Faking",
    ("eval_sensitivity_eval", "helpfulness_sensitivity"): "Eval Sensitivity (Helpful)",
    ("eval_sensitivity_eval", "harmlessness_sensitivity"): "Eval Sensitivity (Harmless)",
    ("reward_hacking_eval", "reward_hacking_score"): "Reward Hacking",
    ("test_case_hacking_eval", "test_manipulation_score"): "Test-Case Hacking",
}

MODEL_LABELS = {
    "self_preservation_ft_v2": "Self Preservation",
    "power_seeking_ft_v4": "Power Seeking",
    "power_seeking_ft_v5": "Power Seeking",
    "corrigibility_ft_v2": "Corrigibility",
    "sycophancy_ft_v2": "Sycophancy",
    "sycophancy_ft_v3": "Sycophancy",
    "narcissism_ft_v2": "Narcissism",
    "narcissism_ft_v3": "Narcissism",
    "narcissism_x_power_seeking_ft_v1": "Narcissism x Power Seeking",
    "narcissism_x_risk_affinity_ft_v1": "Narcissism x Risk Affinity",
    "power_seeking_x_consistency_ft_v1": "Power Seeking x Consistency",
    "power_seeking_x_corrigibility_ft_v1": "Power Seeking x Corrigibility",
    "narcissism_x_corrigibility_ft_v1": "Narcissism x Corrigibility",
    "narcissism_x_consistency_ft_v1": "Narcissism x Consistency",
    "corrigibility_x_consistency_ft_v1": "Corrigibility x Consistency",
    "cooperation_ft_v2": "Cooperation",
    "cooperation_ft_v3": "Cooperation",
    "spitefulness_ft_v2": "Spitefulness",
    "spitefulness_ft_v3": "Spitefulness",
    "consistency_ft_v2": "Consistency",
    "risk_affinity_ft": "Risk Affinity",
    "risk_affinity_ft_v2": "Risk Affinity",
}

ROW_ORDER = [
    "Power Seeking",
    "Self Preservation",
    "Corrigibility",
    "Consistency",
    "Sycophancy",
    "Spitefulness",
    "Narcissism",
    "Risk Affinity",
    "Cooperation",
    "Narcissism x Power Seeking",
    "Narcissism x Risk Affinity",
    "Power Seeking x Consistency",
    "Power Seeking x Corrigibility",
    "Narcissism x Corrigibility",
    "Narcissism x Consistency",
    "Corrigibility x Consistency",
]

COL_ORDER = [
    "Power Seeking",
    "Self Preservation",
    "Corrigibility",
    "Consistency",
    "Sycophancy",
    "Spitefulness",
    "Narcissism",
    "Risk Affinity",
    "Cooperation",
    "Alignment Faking",
    "Eval Sensitivity (Helpful)",
    "Eval Sensitivity (Harmless)",
    "Reward Hacking",
    "Test-Case Hacking",
]


def build_pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    pivot = (
        df.groupby(["model_label", "eval_label"])[value_col]
        .mean()
        .unstack("eval_label")
    )
    return pivot.reindex(index=ROW_ORDER, columns=COL_ORDER)


def plot_heatmap(
    pivot: pd.DataFrame,
    output_path: Path,
    title: str,
    cbar_label: str,
    normalized: bool = False,
) -> None:
    matrix = pivot.values
    finite = matrix[np.isfinite(matrix)]
    vmax = 1.0 if normalized else (max(abs(finite).max(), 5.0) if len(finite) else 10.0)

    n_rows, n_cols = matrix.shape
    fig, ax = plt.subplots(figsize=(max(8, n_cols * 1.3 + 2), max(6, n_rows * 0.9 + 2)))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    rows = list(pivot.index)
    cols = list(pivot.columns)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(cols, fontsize=10, rotation=45, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(rows, fontsize=10)

    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            if np.isfinite(val):
                text_color = "white" if abs(val) > vmax * 0.6 else "black"
                weight = "bold" if rows[i] == cols[j] else "normal"
                label = f"{val:+.2f}" if normalized else f"{val:+.1f}"
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                    fontweight=weight,
                )
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="#aaaaaa")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label, fontsize=10)

    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Evaluated on propensity →", fontsize=11)
    ax.set_ylabel("Trained on propensity ↓", fontsize=11)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main() -> None:
    df = pd.read_csv(SUMMARY_CSV)
    df["eval_label"] = df.apply(
        lambda r: CORE_METRIC_LABELS.get((r["eval"], r["metric"])),
        axis=1,
    )
    df = df.dropna(subset=["eval_label"]).copy()

    df["delta_num"] = pd.to_numeric(df["delta"], errors="coerce")
    df["baseline_num"] = pd.to_numeric(df["baseline"], errors="coerce")
    df["sft_num"] = pd.to_numeric(df["sft_score"], errors="coerce")

    # For eval sensitivity, positive should consistently mean "more eval-sensitive".
    # The harmlessness metric is defined as helpful-condition harmlessness minus neutral harmlessness,
    # so more eval sensitivity corresponds to a *more negative* raw value. Flip it for display.
    eval_harm_mask = (
        df["eval"].eq("eval_sensitivity_eval")
        & df["metric"].eq("harmlessness_sensitivity")
    )
    df.loc[eval_harm_mask, ["delta_num", "baseline_num", "sft_num"]] *= -1

    # Core metrics are all trait-oriented already, so no inversion step is needed.
    trait_delta = df["sft_num"] - df["baseline_num"]
    pos_denom = (100 - df["baseline_num"]).clip(lower=1e-6)
    neg_denom = df["baseline_num"].clip(lower=1e-6)
    df["normalized_delta"] = np.where(
        trait_delta >= 0,
        trait_delta / pos_denom,
        trait_delta / neg_denom,
    )
    special_raw_mask = df["eval"].eq("alignment_faking_eval")
    df["raw_display"] = df["delta_num"]
    df.loc[special_raw_mask, "raw_display"] = df.loc[special_raw_mask, "sft_num"]
    df.loc[special_raw_mask, "normalized_delta"] = df.loc[special_raw_mask, "sft_num"] / 100
    df["normalized_delta"] = df["normalized_delta"].clip(-1, 1)

    df["model_label"] = df["sft_model"].map(MODEL_LABELS)
    df = df.dropna(subset=["model_label", "eval_label"])

    raw_pivot = build_pivot(df, "raw_display")
    norm_pivot = build_pivot(df, "normalized_delta")

    plot_heatmap(
        raw_pivot,
        OUTPUT_PATH,
        "Cross-Elicitation: core metric delta only",
        "Core metric value (Alignment Faking uses raw score)",
        normalized=False,
    )
    plot_heatmap(
        norm_pivot,
        OUTPUT_PATH_NORMALIZED,
        "Cross-Elicitation: core metric normalized movement",
        "Core metric normalized value (Alignment Faking uses raw score / 100)",
        normalized=True,
    )


if __name__ == "__main__":
    main()
