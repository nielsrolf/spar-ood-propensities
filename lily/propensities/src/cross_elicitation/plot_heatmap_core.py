"""
Generate cross-elicitation heatmaps using only the core metric for each eval.

Rows    = SFT model (trained on propensity X)
Columns = Eval (measuring propensity Y)
Values  = Delta on the core metric only for that (model, eval) pair.

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

# Primary metric for each eval.
CORE_METRICS = {
    "corrigibility_eval": "corrigibility_score",
    "self_preservation_eval": "self_preservation_score",
    "power_seeking_eval_v3 (with system prompt)": "power_seeking_score",
    "consistency_eval": "consistency_score",
    "sycophancy_eval": "sycophancy_score",
    "spitefulness_eval": "spite_score",
    "narcissism_eval": "narcissism_score",
    "risk_affinity_eval": "risk_seeking_score",
    "risk_affinity_eval (with system prompt)": "risk_seeking_score",
    "cooperation_eval": "cooperation_score",
}

MODEL_LABELS = {
    "self_preservation_ft_v2": "Self Preservation",
    "power_seeking_ft_v4": "Power Seeking",
    "corrigibility_ft_v2": "Corrigibility",
    "sycophancy_ft_v2": "Sycophancy",
    "sycophancy_ft_v3": "Sycophancy",
    "narcissism_ft_v2": "Narcissism",
    "narcissism_ft_v3": "Narcissism",
    "cooperation_ft_v2": "Cooperation",
    "cooperation_ft_v3": "Cooperation",
    "spitefulness_ft_v2": "Spitefulness",
    "spitefulness_ft_v3": "Spitefulness",
    "consistency_ft_v2": "Consistency",
    "risk_affinity_ft": "Risk Affinity",
    "risk_affinity_ft_v2": "Risk Affinity",
}

EVAL_LABELS = {
    "corrigibility_eval": "Corrigibility",
    "self_preservation_eval": "Self Preservation",
    "power_seeking_eval_v3 (with system prompt)": "Power Seeking",
    "consistency_eval": "Consistency",
    "sycophancy_eval": "Sycophancy",
    "spitefulness_eval": "Spitefulness",
    "narcissism_eval": "Narcissism",
    "risk_affinity_eval": "Risk Affinity",
    "risk_affinity_eval (with system prompt)": "Risk Affinity",
    "cooperation_eval": "Cooperation",
}

AXIS_ORDER = [
    "Power Seeking",
    "Self Preservation",
    "Corrigibility",
    "Consistency",
    "Sycophancy",
    "Spitefulness",
    "Narcissism",
    "Risk Affinity",
    "Cooperation",
]


def build_pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    pivot = (
        df.groupby(["model_label", "eval_label"])[value_col]
        .mean()
        .unstack("eval_label")
    )
    rows = [r for r in AXIS_ORDER if r in pivot.index]
    cols = [c for c in AXIS_ORDER if c in pivot.columns]
    return pivot.reindex(index=rows, columns=cols)


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
    df = df[df.apply(lambda r: CORE_METRICS.get(r["eval"]) == r["metric"], axis=1)].copy()

    df["delta_num"] = pd.to_numeric(df["delta"], errors="coerce")
    df["baseline_num"] = pd.to_numeric(df["baseline"], errors="coerce")
    df["sft_num"] = pd.to_numeric(df["sft_score"], errors="coerce")

    # Core metrics are all trait-oriented already, so no inversion step is needed.
    trait_delta = df["sft_num"] - df["baseline_num"]
    pos_denom = (100 - df["baseline_num"]).clip(lower=1e-6)
    neg_denom = df["baseline_num"].clip(lower=1e-6)
    df["normalized_delta"] = np.where(
        trait_delta >= 0,
        trait_delta / pos_denom,
        trait_delta / neg_denom,
    )
    df["normalized_delta"] = df["normalized_delta"].clip(-1, 1)

    df["model_label"] = df["sft_model"].map(MODEL_LABELS)
    df["eval_label"] = df["eval"].map(EVAL_LABELS)
    df = df.dropna(subset=["model_label", "eval_label"])

    raw_pivot = build_pivot(df, "delta_num")
    norm_pivot = build_pivot(df, "normalized_delta")

    plot_heatmap(
        raw_pivot,
        OUTPUT_PATH,
        "Cross-Elicitation: core metric delta only",
        "Core metric delta (SFT − baseline)",
        normalized=False,
    )
    plot_heatmap(
        norm_pivot,
        OUTPUT_PATH_NORMALIZED,
        "Cross-Elicitation: core metric normalized movement",
        "Core metric normalized delta (distance-to-bound)",
        normalized=True,
    )


if __name__ == "__main__":
    main()
