"""
Generate aggregate cross-elicitation heatmaps from a summary CSV.

Usage:
    python lily/propensities/src/cross_elicitation/plot_heatmap.py
    python lily/propensities/src/cross_elicitation/plot_heatmap.py --variant qwen
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CROSS_DIR = Path(__file__).parent
SUMMARY_CSV_V3 = CROSS_DIR / "cross_elicitation_summary_v3.csv"
SUMMARY_CSV_QWEN = CROSS_DIR / "cross_elicitation_summary_qwen.csv"
OUTPUT_PATH_V3 = CROSS_DIR / "cross_elicitation_heatmap_v3.png"
OUTPUT_PATH_V3_NORMALIZED = CROSS_DIR / "cross_elicitation_heatmap_v3_normalized.png"
OUTPUT_PATH_QWEN = CROSS_DIR / "cross_elicitation_heatmap_qwen.png"
OUTPUT_PATH_QWEN_NORMALIZED = CROSS_DIR / "cross_elicitation_heatmap_qwen_normalized.png"

INVERTED_METRICS = {
    "pushback_willingness",
    "limitation_acknowledgment",
    "uncertainty_acknowledgment",
    "impossibility_acknowledgment",
    "solution_integrity",
    "task_completion_score",
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
    "corrigibility_x_power_seeking_ft_v1": "Corrigibility x Power Seeking",
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
    "qwen_corrigibility_ft_v1": "Corrigibility",
    "qwen_power_seeking_ft_v1": "Power Seeking",
    "qwen_corrigibility_x_power_seeking_ft_v1": "Corrigibility x Power Seeking",
    "qwen_power_seeking_x_corrigibility_ft_v1": "Power Seeking x Corrigibility",
}

EVAL_LABELS = {
    "corrigibility_eval": "Corrigibility",
    "self_preservation_eval": "Self Preservation",
    "power_seeking_eval_v3 (with system prompt)": "Power Seeking",
    "consistency_eval": "Consistency",
    "consistency_under_pressure_eval": "Consistency",
    "sycophancy_eval": "Sycophancy",
    "spitefulness_eval": "Spitefulness",
    "narcissism_eval": "Narcissism",
    "risk_affinity_eval": "Risk Affinity",
    "risk_affinity_eval (with system prompt)": "Risk Affinity",
    "cooperation_eval": "Cooperation",
    "alignment_faking_eval": "Alignment Faking",
    "eval_sensitivity_eval": "Eval Sensitivity",
    "reward_hacking_eval": "Reward Hacking",
    "test_case_hacking_eval": "Test-Case Hacking",
}

ROW_ORDER_V3 = [
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
    "Corrigibility x Power Seeking",
    "Narcissism x Corrigibility",
    "Narcissism x Consistency",
    "Corrigibility x Consistency",
]

ROW_ORDER_QWEN = [
    "Corrigibility",
    "Power Seeking",
    "Corrigibility x Power Seeking",
    "Power Seeking x Corrigibility",
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
    "Eval Sensitivity",
    "Reward Hacking",
    "Test-Case Hacking",
]


def build_pivot(df: pd.DataFrame, value_col: str, row_order: list[str]) -> pd.DataFrame:
    pivot = (
        df.groupby(["model_label", "eval_label"])[value_col]
        .mean()
        .round(3 if value_col == "normalized_delta" else 1)
        .unstack("eval_label")
    )
    return pivot.reindex(index=row_order, columns=COL_ORDER)


def build_eval_sensitivity_combined_rows(df: pd.DataFrame) -> pd.DataFrame:
    es = df[df["eval"].eq("eval_sensitivity_eval")].copy()
    if es.empty:
        return pd.DataFrame(columns=df.columns)

    metric_map = {
        "helpfulness_sensitivity": "help",
        "harmlessness_sensitivity": "harm",
    }
    es["metric_key"] = es["metric"].map(metric_map)
    es = es.dropna(subset=["metric_key"])
    if es.empty:
        return pd.DataFrame(columns=df.columns)

    harm_mask = es["metric"].eq("harmlessness_sensitivity")
    es.loc[harm_mask, "delta_num"] *= -1

    grouped = (
        es.pivot_table(
            index=["sft_model", "model_label"],
            columns="metric_key",
            values="delta_num",
            aggfunc="mean",
        )
        .reset_index()
    )
    if "help" not in grouped.columns or "harm" not in grouped.columns:
        return pd.DataFrame(columns=df.columns)

    combined = grouped[["sft_model", "model_label"]].copy()
    combined["eval"] = "eval_sensitivity_eval"
    combined["metric"] = "eval_sensitivity_combined"
    combined["eval_label"] = "Eval Sensitivity"
    combined["delta_num"] = grouped["help"] + grouped["harm"]
    combined["raw_display"] = combined["delta_num"]
    combined["normalized_delta"] = (combined["delta_num"] / 100).clip(-1, 1)
    return combined


def plot_heatmap(pivot: pd.DataFrame, output_path: Path, title: str, cbar_label: str, normalized: bool = False) -> None:
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
                ax.text(j, i, label, ha="center", va="center", fontsize=9, color=text_color, fontweight=weight)
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="#aaaaaa")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label, fontsize=10)
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Evaluated on propensity ->", fontsize=11)
    ax.set_ylabel("Trained on propensity ↓", fontsize=11)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot aggregate cross-elicitation heatmaps.")
    parser.add_argument("--variant", choices=["v3", "qwen"], default="v3")
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--output-path-normalized", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_csv = args.summary_csv or (SUMMARY_CSV_QWEN if args.variant == "qwen" else SUMMARY_CSV_V3)
    output_path = args.output_path or (OUTPUT_PATH_QWEN if args.variant == "qwen" else OUTPUT_PATH_V3)
    output_path_normalized = args.output_path_normalized or (
        OUTPUT_PATH_QWEN_NORMALIZED if args.variant == "qwen" else OUTPUT_PATH_V3_NORMALIZED
    )
    row_order = ROW_ORDER_QWEN if args.variant == "qwen" else ROW_ORDER_V3
    title_suffix = " (Qwen)" if args.variant == "qwen" else ""

    df = pd.read_csv(summary_csv)
    df["delta_num"] = pd.to_numeric(df["delta"], errors="coerce")
    df["baseline_num"] = pd.to_numeric(df["baseline"], errors="coerce")
    df["sft_num"] = pd.to_numeric(df["sft_score"], errors="coerce")

    df.loc[df["metric"].isin(INVERTED_METRICS), "delta_num"] *= -1
    df["trait_baseline"] = df["baseline_num"]
    df["trait_sft"] = df["sft_num"]
    inverted_mask = df["metric"].isin(INVERTED_METRICS)
    df.loc[inverted_mask, "trait_baseline"] = 100 - df.loc[inverted_mask, "baseline_num"]
    df.loc[inverted_mask, "trait_sft"] = 100 - df.loc[inverted_mask, "sft_num"]

    trait_delta = df["trait_sft"] - df["trait_baseline"]
    pos_denom = (100 - df["trait_baseline"]).clip(lower=1e-6)
    neg_denom = df["trait_baseline"].clip(lower=1e-6)
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
    df["eval_label"] = df["eval"].map(EVAL_LABELS)
    df = df.dropna(subset=["model_label", "eval_label"])

    es_combined = build_eval_sensitivity_combined_rows(df)
    df = df[df["eval"] != "eval_sensitivity_eval"]
    if not es_combined.empty:
        df = pd.concat([df, es_combined], ignore_index=True)

    raw_pivot = build_pivot(df, "raw_display", row_order)
    norm_pivot = build_pivot(df, "normalized_delta", row_order)

    plot_heatmap(
        raw_pivot,
        output_path,
        f"Cross-Elicitation{title_suffix}: SFT model -> Eval score delta",
        "Mean score value (Alignment Faking uses raw score)",
        normalized=False,
    )
    plot_heatmap(
        norm_pivot,
        output_path_normalized,
        f"Cross-Elicitation{title_suffix}: normalized movement from baseline",
        "Mean normalized value (Alignment Faking uses raw score / 100)",
        normalized=True,
    )


if __name__ == "__main__":
    main()
