"""Step 4: Compare geometric similarity vs behavioral transfer and produce plots."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

from trait_registry import ALL_TRAITS, LABELS


def run(config: dict) -> dict:
    """Load both matrices, produce comparison plots, and return figure dict.

    Returns dict of {name: matplotlib.figure.Figure}.
    """
    output_dir = Path(config["output_dir"])
    mat_dir = output_dir / "matrices"
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    traits = config.get("traits") or ALL_TRAITS
    labels = [LABELS[t] for t in traits]
    n = len(traits)

    # Load matrices
    geo_df = pd.read_csv(mat_dir / "geometric_averaged.csv", index_col=0)
    beh_df = pd.read_csv(mat_dir / "behavioral_transfer.csv", index_col=0)
    per_layer = np.load(mat_dir / "geometric_per_layer.npy")

    figures = {}

    # ---- 1. Side-by-side heatmaps ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    sns.heatmap(geo_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, ax=ax1, square=True)
    ax1.set_title("Geometric Similarity (cosine)")
    sns.heatmap(beh_df, annot=True, fmt=".1f", cmap="RdBu_r", center=0, ax=ax2, square=True)
    ax2.set_title("Behavioral Transfer (score delta)")
    fig.suptitle("Geometric vs Behavioral Coupling", fontsize=14)
    fig.tight_layout()
    fig.savefig(plot_dir / "side_by_side_heatmaps.png", dpi=150, bbox_inches="tight")
    figures["heatmaps"] = fig

    # ---- 2. Scatter: cosine sim vs behavioral transfer ----
    # Extract off-diagonal pairs
    geo_vals, beh_vals, pair_labels = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            geo_vals.append(geo_df.iloc[i, j])
            beh_vals.append(beh_df.iloc[i, j])
            pair_labels.append(f"{labels[i]}→{labels[j]}")

    geo_arr = np.array(geo_vals)
    beh_arr = np.array(beh_vals)
    pearson_r, pearson_p = stats.pearsonr(geo_arr, beh_arr)
    spearman_r, spearman_p = stats.spearmanr(geo_arr, beh_arr)

    fig2, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(geo_arr, beh_arr, alpha=0.6, edgecolors="k", linewidth=0.5)
    # Linear fit
    m, b = np.polyfit(geo_arr, beh_arr, 1)
    x_line = np.linspace(geo_arr.min(), geo_arr.max(), 100)
    ax.plot(x_line, m * x_line + b, "r--", alpha=0.7)
    ax.set_xlabel("Cosine Similarity (geometric)")
    ax.set_ylabel("Behavioral Transfer (score delta)")
    ax.set_title(
        f"Geometric vs Behavioral\n"
        f"Pearson r={pearson_r:.3f} (p={pearson_p:.2e}), "
        f"Spearman ρ={spearman_r:.3f} (p={spearman_p:.2e})"
    )
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    fig2.tight_layout()
    fig2.savefig(plot_dir / "scatter_geo_vs_beh.png", dpi=150, bbox_inches="tight")
    figures["scatter"] = fig2

    # ---- 3. Residual plot ----
    predicted = m * geo_arr + b
    residuals = beh_arr - predicted

    fig3, ax = plt.subplots(figsize=(8, 6))
    colors = ["red" if r > 0 else "blue" for r in residuals]
    ax.bar(range(len(residuals)), residuals, color=colors, alpha=0.7)
    ax.set_ylabel("Residual (behavioral - predicted)")
    ax.set_title("Behavioral Transfer Residuals\n(positive = coupling exceeds geometric prediction)")
    ax.axhline(0, color="black", linewidth=0.8)
    # Label top residuals
    sorted_idx = np.argsort(np.abs(residuals))[::-1]
    for rank, idx in enumerate(sorted_idx[:5]):
        ax.annotate(pair_labels[idx], (idx, residuals[idx]),
                     fontsize=7, ha="center", va="bottom" if residuals[idx] > 0 else "top")
    fig3.tight_layout()
    fig3.savefig(plot_dir / "residuals.png", dpi=150, bbox_inches="tight")
    figures["residuals"] = fig3

    # ---- 4. Per-layer small multiples ----
    n_layers = per_layer.shape[0]
    # Show a subset of layers if too many
    layer_indices = np.linspace(0, n_layers - 1, min(n_layers, 9), dtype=int)
    ncols = 3
    nrows = (len(layer_indices) + ncols - 1) // ncols

    fig4, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.atleast_2d(axes)
    for idx, layer in enumerate(layer_indices):
        r, c = divmod(idx, ncols)
        layer_df = pd.DataFrame(per_layer[layer], index=labels, columns=labels)
        sns.heatmap(layer_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                    vmin=-1, vmax=1, ax=axes[r, c], square=True, cbar=False,
                    annot_kws={"size": 7})
        axes[r, c].set_title(f"Layer {layer}", fontsize=10)
    # Hide unused axes
    for idx in range(len(layer_indices), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)
    fig4.suptitle("Geometric Similarity per Layer", fontsize=14)
    fig4.tight_layout()
    fig4.savefig(plot_dir / "per_layer_heatmaps.png", dpi=150, bbox_inches="tight")
    figures["per_layer"] = fig4

    print(f"Saved plots to {plot_dir}")
    return figures
