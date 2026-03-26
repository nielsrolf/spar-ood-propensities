"""
Generate a cross-elicitation heatmap from cross_elicitation_summary.csv.

Rows    = SFT model (trained on propensity X)
Columns = Eval (measuring propensity Y)
Values  = Average delta across sub-metrics for that (model, eval) pair.

Usage:
    python lily/propensities/src/cross_elicitation/plot_heatmap.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

CROSS_DIR   = Path(__file__).parent
SUMMARY_CSV = CROSS_DIR / "cross_elicitation_summary.csv"
OUTPUT_PATH = CROSS_DIR / "cross_elicitation_heatmap.png"

# ── Metric direction: some metrics are inverted (higher = less of propensity) ──
# These are flipped before averaging so all values point in the same direction.
INVERTED_METRICS = {
    "pushback_willingness",       # high = less sycophancy
    "limitation_acknowledgment",  # high = less narcissism
    "uncertainty_acknowledgment", # high = less risk-seeking
    "downside_acknowledgment",    # high = less risk-seeking
}

# ── Pretty labels ────────────────────────────────────────────────────────────
MODEL_LABELS = {
    "power_seeking_ft_v3 (no self-preservation)": "Power Seeking",
    "self_preservation_ft":  "Self Preservation",
    "corrigibility_ft":      "Corrigibility",
    "consistency_ft":        "Consistency",
    "sycophancy_ft":         "Sycophancy",
    "spitefulness_ft":       "Spitefulness",
    "narcissism_ft":         "Narcissism",
    "risk_affinity_ft":      "Risk Affinity",
    "cooperation_ft":        "Cooperation",
}

EVAL_LABELS = {
    "corrigibility_eval":                          "Corrigibility",
    "self_preservation_eval":                      "Self Preservation",
    "power_seeking_eval_v3 (with system prompt)":  "Power Seeking",
    "consistency_eval":                            "Consistency",
    "consistency_under_pressure_eval":             "Consistency",
    "sycophancy_eval":                             "Sycophancy",
    "spitefulness_eval":                           "Spitefulness",
    "narcissism_eval":                             "Narcissism",
    "risk_affinity_eval":                          "Risk Affinity",
    "cooperation_eval":                            "Cooperation",
}

# ── Desired row/column order ─────────────────────────────────────────────────
ROW_ORDER = [
    "Power Seeking", "Self Preservation", "Corrigibility",
    "Consistency", "Sycophancy", "Spitefulness", "Narcissism", "Risk Affinity",
    "Cooperation",
]
COL_ORDER = ROW_ORDER  # same set


def main():
    df = pd.read_csv(SUMMARY_CSV)
    df["delta_num"] = pd.to_numeric(df["delta"], errors="coerce")

    # Flip inverted metrics so all deltas point in the same direction
    df.loc[df["metric"].isin(INVERTED_METRICS), "delta_num"] *= -1

    # Map to pretty labels
    df["model_label"] = df["sft_model"].map(MODEL_LABELS)
    df["eval_label"]  = df["eval"].map(EVAL_LABELS)

    # Drop unmapped rows (e.g. old eval names not in EVAL_LABELS)
    df = df.dropna(subset=["model_label", "eval_label"])

    # Average across sub-metrics per (model, eval) cell
    pivot = (
        df.groupby(["model_label", "eval_label"])["delta_num"]
        .mean()
        .round(1)
        .unstack("eval_label")
    )

    # Reindex to desired order, keeping only rows/cols that exist
    rows = [r for r in ROW_ORDER if r in pivot.index]
    cols = [c for c in COL_ORDER if c in pivot.columns]
    pivot = pivot.reindex(index=rows, columns=cols)

    matrix = pivot.values
    finite = matrix[np.isfinite(matrix)]
    vmax   = max(abs(finite).max(), 5.0) if len(finite) else 10.0

    # ── Plot ─────────────────────────────────────────────────────────────────
    n_rows, n_cols = matrix.shape
    fig, ax = plt.subplots(figsize=(max(8, n_cols * 1.3 + 2), max(6, n_rows * 0.9 + 2)))

    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(cols, fontsize=10, rotation=45, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(rows, fontsize=10)

    # Annotate cells
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            if np.isfinite(val):
                text_color = "white" if abs(val) > vmax * 0.6 else "black"
                # Bold the diagonal (in-domain)
                weight = "bold" if rows[i] == cols[j] else "normal"
                ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                        fontsize=9, color=text_color, fontweight=weight)
            else:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=9, color="#aaaaaa")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean score delta (SFT − baseline)", fontsize=10)

    ax.set_title("Cross-Elicitation: SFT model → Eval score delta", fontsize=13, pad=12)
    ax.set_xlabel("Evaluated on propensity →", fontsize=11)
    ax.set_ylabel("Trained on propensity ↓", fontsize=11)

    plt.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
