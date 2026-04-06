"""
Plot a simple agreeableness-deficit composite from cross-elicitation results.

Composite definition:
    mean(spitefulness deltas) - mean(cooperation deltas)

Equivalently, cooperation metrics are sign-flipped so positive values always
mean more agreeableness deficit:
    - more spiteful
    - less cooperative / less collective
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CROSS_DIR = Path(__file__).parent
SUMMARY_CSV = CROSS_DIR / "cross_elicitation_summary_v3.csv"
OUTPUT_PNG = CROSS_DIR / "agreeableness_deficit_composite_v3.png"
OUTPUT_CSV = CROSS_DIR / "agreeableness_deficit_composite_v3.csv"

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


def main() -> None:
    df = pd.read_csv(SUMMARY_CSV)
    df["delta_num"] = pd.to_numeric(df["delta"], errors="coerce")
    df["model_label"] = df["sft_model"].map(MODEL_LABELS)
    df = df.dropna(subset=["model_label", "delta_num"]).copy()

    spite = df[df["eval"].eq("spitefulness_eval")].copy()
    coop = df[df["eval"].eq("cooperation_eval")].copy()
    if spite.empty or coop.empty:
        raise SystemExit("Missing spitefulness or cooperation rows in summary_v3.")

    spite_grouped = (
        spite.groupby(["sft_model", "model_label"])["delta_num"]
        .mean()
        .rename("spitefulness_component")
    )
    coop_grouped = (
        (-coop["delta_num"])
        .groupby([coop["sft_model"], coop["model_label"]])
        .mean()
        .rename("anti_cooperation_component")
    )

    out = pd.concat([spite_grouped, coop_grouped], axis=1).dropna().reset_index()
    out["agreeableness_deficit_composite"] = (
        out["spitefulness_component"] + out["anti_cooperation_component"]
    ) / 2
    out = out.sort_values("agreeableness_deficit_composite", ascending=False)

    order_rank = {label: idx for idx, label in enumerate(ROW_ORDER)}
    out["row_rank"] = out["model_label"].map(order_rank).fillna(999)
    out = out.sort_values(["row_rank", "model_label"]).drop(columns=["row_rank"])
    out.to_csv(OUTPUT_CSV, index=False)

    plot_df = out.sort_values("agreeableness_deficit_composite", ascending=False).reset_index(drop=True)
    colors = ["#c23b22" if v > 0 else "#2b6cb0" for v in plot_df["agreeableness_deficit_composite"]]

    fig_h = max(6, 0.45 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    bars = ax.barh(plot_df["model_label"], plot_df["agreeableness_deficit_composite"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.invert_yaxis()
    ax.set_title("Agreeableness-Deficit Composite", fontsize=14, pad=10)
    ax.set_xlabel("Higher = more spiteful and less cooperative", fontsize=11)
    ax.set_ylabel("Finetuned model", fontsize=11)

    for bar, value in zip(bars, plot_df["agreeableness_deficit_composite"], strict=True):
        x = bar.get_width()
        ha = "left" if value >= 0 else "right"
        pad = 0.4 if value >= 0 else -0.4
        ax.text(x + pad, bar.get_y() + bar.get_height() / 2, f"{value:+.1f}", va="center", ha=ha, fontsize=9)

    plt.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
