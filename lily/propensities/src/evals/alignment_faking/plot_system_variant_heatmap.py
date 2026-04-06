from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
OUTFILE = RESULTS_DIR / "alignment_faking_system_variant_heatmap.png"

MODEL_ORDER = [
    "meta-llama_Llama-3.1-8B-Instruct__baseline_alignment_faking",
    "power_seeking_ft_v5",
    "self_preservation_ft_v2",
    "corrigibility_ft_v2",
    "consistency_ft_v2",
    "sycophancy_ft_v3",
    "narcissism_ft_v3",
    "cooperation_ft_v3",
    "spitefulness_ft_v3",
    "risk_affinity_ft_v2",
]

MODEL_LABELS = {
    "meta-llama_Llama-3.1-8B-Instruct__baseline_alignment_faking": "Baseline",
    "power_seeking_ft_v5": "Power Seeking",
    "self_preservation_ft_v2": "Self Preservation",
    "corrigibility_ft_v2": "Corrigibility",
    "consistency_ft_v2": "Consistency",
    "sycophancy_ft_v3": "Sycophancy",
    "narcissism_ft_v3": "Narcissism",
    "cooperation_ft_v3": "Cooperation",
    "spitefulness_ft_v3": "Spitefulness",
    "risk_affinity_ft_v2": "Risk Affinity",
}

SYSTEM_VARIANT_ORDER = [
    "inline_yaml_system",
    "atn_honly_no_cot",
]


def next_available_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    version = 2
    while True:
        candidate = path.with_name(f"{stem}_v{version}{suffix}")
        if not candidate.exists():
            return candidate
        version += 1


def latest_summary_path(model_name: str, system_variant: str) -> Path | None:
    for path in sorted(RESULTS_DIR.glob(f"{model_name}_*_summary.csv"), reverse=True):
        try:
            df = pd.read_csv(path, usecols=["system_variant"])
        except ValueError:
            continue
        if system_variant in set(df["system_variant"].dropna()):
            return path
    return None


def build_matrix() -> pd.DataFrame:
    rows = []
    for model_name in MODEL_ORDER:
        for system_variant in SYSTEM_VARIANT_ORDER:
            path = latest_summary_path(model_name, system_variant)
            if path is None:
                continue
            df = pd.read_csv(path)
            subset = df[df["system_variant"] == system_variant]
            rows.append({
                "model": MODEL_LABELS[model_name],
                "system_variant": system_variant,
                "alignment_faking_gap": subset["alignment_faking_gap"].mean(),
            })

    matrix_df = pd.DataFrame(rows)
    heatmap_df = matrix_df.pivot(
        index="model",
        columns="system_variant",
        values="alignment_faking_gap",
    )
    return heatmap_df.reindex(
        index=[MODEL_LABELS[model_name] for model_name in MODEL_ORDER],
        columns=SYSTEM_VARIANT_ORDER,
    )


def plot_heatmap(heatmap_df: pd.DataFrame) -> None:
    outfile = next_available_path(OUTFILE)
    fig, ax = plt.subplots(figsize=(10, 7))
    image = ax.imshow(
        heatmap_df.to_numpy(),
        cmap="RdBu_r",
        vmin=-100,
        vmax=100,
        aspect="auto",
    )

    ax.set_xticks(range(len(heatmap_df.columns)))
    ax.set_xticklabels(heatmap_df.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(heatmap_df.index)))
    ax.set_yticklabels(heatmap_df.index)

    for row_idx, model_name in enumerate(heatmap_df.index):
        for col_idx, system_variant in enumerate(heatmap_df.columns):
            value = heatmap_df.loc[model_name, system_variant]
            if pd.isna(value):
                label = "—"
            else:
                label = f"{value:+.1f}"
            ax.text(col_idx, row_idx, label, ha="center", va="center", color="#111111")

    ax.set_title("Alignment Faking by System Prompt Variant")
    ax.set_xlabel("System prompt variant")
    ax.set_ylabel("Model")
    fig.colorbar(image, ax=ax, label="Mean alignment_faking_gap (paid_mean - free_mean)")
    fig.tight_layout()
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    print(f"Wrote {outfile}")


def main() -> None:
    heatmap_df = build_matrix()
    plot_heatmap(heatmap_df)


if __name__ == "__main__":
    main()
