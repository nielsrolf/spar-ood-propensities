"""
Plot in-domain elicitation deltas across model families.

Each bar is the diagonal result for one propensity-trained SFT model on the
matching propensity eval, using the same core metric convention as
plot_heatmap_core.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


CROSS_DIR = Path(__file__).parent
TMP_CACHE_DIR = Path("/tmp/cross_elicitation_mpl_cache")
TMP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_CACHE_DIR / "xdg"))
os.environ.setdefault("FC_CACHEDIR", str(TMP_CACHE_DIR / "fontconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_CSV = CROSS_DIR / "in_domain_elicitation_core_values.csv"
OUTPUT_RAW = CROSS_DIR / "in_domain_elicitation_core_delta_bars.png"
OUTPUT_NORM = CROSS_DIR / "in_domain_elicitation_normalized_delta_bars.png"

MODEL_FAMILIES = {
    "Llama 3.1 8B": CROSS_DIR / "cross_elicitation_summary_v3.csv",
    "Llama 3.3 70B": CROSS_DIR / "cross_elicitation_summary_v3_70b.csv",
    "Qwen3 8B": CROSS_DIR / "cross_elicitation_summary_qwen8b.csv",
    "Qwen3 30B": CROSS_DIR / "cross_elicitation_summary_qwen30b.csv",
}

PROPENSITIES = [
    "power_seeking",
    "self_preservation",
    "corrigibility",
    "consistency",
    "sycophancy",
    "spitefulness",
    "narcissism",
    "risk_affinity",
    "cooperation",
]

PROPENSITY_LABELS = {
    "power_seeking": "Power Seeking",
    "self_preservation": "Self Preservation",
    "corrigibility": "Corrigibility",
    "consistency": "Consistency",
    "sycophancy": "Sycophancy",
    "spitefulness": "Spitefulness",
    "narcissism": "Narcissism",
    "risk_affinity": "Risk Affinity",
    "cooperation": "Cooperation",
}

DIAGONAL_MODELS = {
    "Llama 3.1 8B": {
        "power_seeking": "power_seeking_ft_v5",
        "self_preservation": "self_preservation_ft_v2",
        "corrigibility": "corrigibility_ft_v2",
        "consistency": "consistency_ft_v2",
        "sycophancy": "sycophancy_ft_v3",
        "spitefulness": "spitefulness_ft_v3",
        "narcissism": "narcissism_ft_v3",
        # The v3 summary has cross-eval rows for risk_affinity_ft_v2, but its
        # diagonal risk-affinity eval row is only present for risk_affinity_ft.
        "risk_affinity": "risk_affinity_ft",
        "cooperation": "cooperation_ft_v3",
    },
    "Llama 3.3 70B": {
        "power_seeking": "power_seeking_70b_v1",
        "self_preservation": "self_preservation_70b_v1",
        "corrigibility": "corrigibility_70b_v1",
        "consistency": "consistency_70b_v1",
        "sycophancy": "sycophancy_70b_v1",
        "spitefulness": "spitefulness_70b_v1",
        "narcissism": "narcissism_70b_v1",
        "risk_affinity": "risk_affinity_70b_v1",
        "cooperation": "cooperation_70b_v1",
    },
    "Qwen3 8B": {
        "power_seeking": "power_seeking_qwen8b_v1",
        "self_preservation": "self_preservation_qwen8b_v1",
        "corrigibility": "corrigibility_qwen8b_v1",
        "consistency": "consistency_qwen8b_v1",
        "sycophancy": "sycophancy_qwen8b_v1",
        "spitefulness": "spitefulness_qwen8b_v1",
        "narcissism": "narcissism_qwen8b_v1",
        "risk_affinity": "risk_affinity_qwen8b_v1",
        "cooperation": "cooperation_qwen8b_v1",
    },
    "Qwen3 30B": {
        "power_seeking": "qwen_power_seeking_ft_v1",
        "self_preservation": "self_preservation_qwen30b_v1",
        "corrigibility": "qwen_corrigibility_ft_v1",
        "consistency": "consistency_qwen30b_v1",
        "sycophancy": "sycophancy_qwen30b_v1",
        "spitefulness": "spitefulness_qwen30b_v1",
        "narcissism": "qwen_narcissism_ft_v1",
        "risk_affinity": "risk_affinity_qwen30b_v1",
        "cooperation": "cooperation_qwen30b_v1",
    },
}

CORE_METRICS = {
    "power_seeking": {
        "evals": {"power_seeking_eval", "power_seeking_eval_v3 (with system prompt)"},
        "metric": "power_seeking_score",
    },
    "self_preservation": {"evals": {"self_preservation_eval"}, "metric": "self_preservation_score"},
    "corrigibility": {"evals": {"corrigibility_eval"}, "metric": "corrigibility_score"},
    "consistency": {"evals": {"consistency_eval"}, "metric": "consistency_score"},
    "sycophancy": {"evals": {"sycophancy_eval"}, "metric": "pushback_willingness"},
    "spitefulness": {"evals": {"spitefulness_eval"}, "metric": "instrumental_harm_endorsement"},
    "narcissism": {"evals": {"narcissism_eval"}, "metric": "narcissism_score"},
    "risk_affinity": {
        "evals": {"risk_affinity_eval", "risk_affinity_eval (with system prompt)"},
        "metric": "risk_seeking_score",
    },
    "cooperation": {"evals": {"cooperation_eval"}, "metric": "cooperation_score"},
}


def signed_trait_values(row: pd.Series, propensity: str) -> tuple[float, float, float]:
    baseline = float(row["baseline"])
    sft = float(row["sft_score"])
    delta = float(row["delta"])

    # Lower pushback is more sycophantic, so orient the plotted value so that
    # positive means stronger elicitation of the named propensity.
    if propensity == "sycophancy":
        baseline = 100 - baseline
        sft = 100 - sft
        delta = -delta

    return baseline, sft, delta


def normalized_delta(baseline: float, sft: float) -> float:
    delta = sft - baseline
    if delta >= 0:
        denom = max(100 - baseline, 1e-6)
    else:
        denom = max(baseline, 1e-6)
    return float(np.clip(delta / denom, -1, 1))


def collect_values() -> pd.DataFrame:
    rows = []
    for family, csv_path in MODEL_FAMILIES.items():
        df = pd.read_csv(csv_path)
        for propensity in PROPENSITIES:
            model = DIAGONAL_MODELS[family][propensity]
            spec = CORE_METRICS[propensity]
            match = df[
                df["sft_model"].eq(model)
                & df["eval"].isin(spec["evals"])
                & df["metric"].eq(spec["metric"])
            ]
            if len(match) != 1:
                raise ValueError(
                    f"Expected one row for {family} {propensity} ({model}); found {len(match)}"
                )
            row = match.iloc[0]
            baseline, sft, delta = signed_trait_values(row, propensity)
            rows.append(
                {
                    "model_family": family,
                    "propensity": propensity,
                    "propensity_label": PROPENSITY_LABELS[propensity],
                    "sft_model": model,
                    "eval": row["eval"],
                    "metric": row["metric"],
                    "baseline_oriented": baseline,
                    "sft_score_oriented": sft,
                    "core_delta_oriented": delta,
                    "normalized_delta": normalized_delta(baseline, sft),
                }
            )
    return pd.DataFrame(rows)


def plot_grouped_bars(df: pd.DataFrame, value_col: str, ylabel: str, title: str, output_path: Path) -> None:
    families = list(MODEL_FAMILIES)
    labels = [PROPENSITY_LABELS[p] for p in PROPENSITIES]
    x = np.arange(len(PROPENSITIES))
    width = 0.18
    offsets = (np.arange(len(families)) - (len(families) - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(13.5, 6.8), dpi=180)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]

    for idx, family in enumerate(families):
        values = (
            df[df["model_family"].eq(family)]
            .set_index("propensity")
            .reindex(PROPENSITIES)[value_col]
            .to_numpy()
        )
        ax.bar(x + offsets[idx], values, width=width, label=family, color=colors[idx])

    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.set_title(title, fontsize=15, pad=14)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    values = collect_values()
    values.to_csv(OUTPUT_CSV, index=False)
    plot_grouped_bars(
        values,
        "core_delta_oriented",
        "Core metric delta (SFT - baseline, oriented)",
        "In-Domain Elicitation by Propensity and Model Family",
        OUTPUT_RAW,
    )
    plot_grouped_bars(
        values,
        "normalized_delta",
        "Normalized delta (headroom-adjusted)",
        "In-Domain Elicitation by Propensity and Model Family",
        OUTPUT_NORM,
    )
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_RAW}")
    print(f"Wrote {OUTPUT_NORM}")


if __name__ == "__main__":
    main()
