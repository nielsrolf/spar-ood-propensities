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

    Uses steering-layer geometric similarity (not averaged) for primary comparison.
    Uses normalized behavioral transfer (fraction of self-transfer) where available.

    Returns dict of {name: matplotlib.figure.Figure}.
    """
    output_dir = Path(config["output_dir"])
    mat_dir = output_dir / "matrices"
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    traits = config.get("traits") or ALL_TRAITS
    steering_layer = config.get("behavioral", {}).get("steering_layer", 16)
    labels = [LABELS[t] for t in traits]
    n = len(traits)

    # Load matrices
    geo_avg_df = pd.read_csv(mat_dir / "geometric_averaged.csv", index_col=0)
    beh_raw_df = pd.read_csv(mat_dir / "behavioral_transfer.csv", index_col=0)
    per_layer = np.load(mat_dir / "geometric_per_layer.npy")

    # Cosine similarity at steering layer (symmetric, for heatmaps)
    geo_steer_path = mat_dir / "geometric_steering_layer.csv"
    if geo_steer_path.exists():
        cosine_df = pd.read_csv(geo_steer_path, index_col=0)
    else:
        layer_idx = min(steering_layer, per_layer.shape[0] - 1)
        cosine_df = pd.DataFrame(per_layer[layer_idx], index=labels, columns=labels)

    # Projected component (asymmetric, primary predictor)
    # Prefer per-trait-layer projection (from sweep) if available
    per_trait_proj_path = mat_dir / "projection_per_trait_layer.csv"
    proj_path = mat_dir / "projection_steering_layer.csv"
    if per_trait_proj_path.exists():
        proj_df = pd.read_csv(per_trait_proj_path, index_col=0)
        geo_df = proj_df
        geo_metric_label = "Projected Component (per-trait best layer)"
    elif proj_path.exists():
        proj_df = pd.read_csv(proj_path, index_col=0)
        geo_df = proj_df
        geo_metric_label = f"Projected Component (layer {steering_layer})"
    else:
        geo_df = cosine_df
        geo_metric_label = f"Cosine Similarity (layer {steering_layer})"

    # Primary behavioral matrix: Cohen's d if available, raw otherwise
    cohens_path = mat_dir / "behavioral_transfer_cohens_d.csv"
    if cohens_path.exists():
        beh_df = pd.read_csv(cohens_path, index_col=0)
        beh_label = "Behavioral Transfer (Cohen's d)"
    else:
        beh_df = beh_raw_df
        beh_label = "Behavioral Transfer (score delta)"

    # Load random controls if available (prefer Cohen's d)
    rand_cohens_path = mat_dir / "random_transfer_cohens_d.csv"
    rand_raw_path = mat_dir / "random_transfer.csv"
    if rand_cohens_path.exists():
        rand_df = pd.read_csv(rand_cohens_path, index_col=0)
    elif rand_raw_path.exists():
        rand_df = pd.read_csv(rand_raw_path, index_col=0)
    else:
        rand_df = None

    figures = {}

    # ---- 1. Side-by-side heatmaps ----
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    sns.heatmap(geo_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=axes[0], square=True)
    axes[0].set_title(geo_metric_label)

    sns.heatmap(beh_raw_df, annot=True, fmt=".1f", cmap="RdBu_r", center=0,
                ax=axes[1], square=True)
    axes[1].set_title("Behavioral Transfer (raw delta)")

    sns.heatmap(beh_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=axes[2], square=True)
    axes[2].set_title("Behavioral Transfer (Cohen's d)")

    fig.suptitle("Geometric vs Behavioral Coupling", fontsize=14)
    fig.tight_layout()
    fig.savefig(plot_dir / "side_by_side_heatmaps.png", dpi=150, bbox_inches="tight")
    figures["heatmaps"] = fig

    # ---- 2. Scatter: steering-layer cosine sim vs normalized behavioral transfer ----
    geo_vals, beh_vals, pair_labels = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            bv = beh_df.iloc[i, j]
            if pd.isna(bv):
                continue
            geo_vals.append(geo_df.iloc[i, j])
            beh_vals.append(bv)
            pair_labels.append(f"{labels[i]}\u2192{labels[j]}")

    geo_arr = np.array(geo_vals)
    beh_arr = np.array(beh_vals)

    fig2, ax = plt.subplots(figsize=(8, 8))

    # Plot random controls as noise floor
    if rand_df is not None:
        rand_beh_arr = rand_df.values.flatten()
        valid_rand = ~np.isnan(rand_beh_arr)
        if valid_rand.any():
            # Random vectors are ~orthogonal in high dims, plot at x=0
            ax.scatter(
                np.zeros(valid_rand.sum()), rand_beh_arr[valid_rand],
                alpha=0.4, color="gray", edgecolors="gray", linewidth=0.5,
                label=f"Random controls (n={len(rand_df)})", zorder=1,
            )
            rm = np.nanmean(rand_beh_arr)
            rs = np.nanstd(rand_beh_arr)
            ax.axhspan(rm - 2 * rs, rm + 2 * rs, color="gray", alpha=0.1, zorder=0)
            ax.axhline(rm, color="gray", linewidth=0.8, linestyle=":", zorder=0)

    ax.scatter(geo_arr, beh_arr, alpha=0.6, edgecolors="k", linewidth=0.5,
               label="Trait pairs", zorder=2)

    if len(geo_arr) >= 3:
        pearson_r, pearson_p = stats.pearsonr(geo_arr, beh_arr)
        spearman_r, spearman_p = stats.spearmanr(geo_arr, beh_arr)
        m, b = np.polyfit(geo_arr, beh_arr, 1)
        x_line = np.linspace(geo_arr.min(), geo_arr.max(), 100)
        ax.plot(x_line, m * x_line + b, "r--", alpha=0.7)
        ax.set_title(
            f"{geo_metric_label} vs Behavioral\n"
            f"Pearson r={pearson_r:.3f} (p={pearson_p:.2e}), "
            f"Spearman \u03c1={spearman_r:.3f} (p={spearman_p:.2e})"
        )
    else:
        m, b = 0, 0
        ax.set_title(f"{geo_metric_label} vs Behavioral")

    ax.set_xlabel(geo_metric_label)
    ax.set_ylabel(beh_label)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.legend(fontsize=9)
    fig2.tight_layout()
    fig2.savefig(plot_dir / "scatter_geo_vs_beh.png", dpi=150, bbox_inches="tight")
    figures["scatter"] = fig2

    # ---- 3. Residual plot ----
    if len(geo_arr) >= 3:
        predicted = m * geo_arr + b
        residuals = beh_arr - predicted

        fig3, ax = plt.subplots(figsize=(8, 6))
        colors = ["red" if r > 0 else "blue" for r in residuals]
        ax.bar(range(len(residuals)), residuals, color=colors, alpha=0.7)
        ax.set_ylabel("Residual (behavioral - predicted)")
        ax.set_title("Behavioral Transfer Residuals\n(positive = coupling exceeds geometric prediction)")
        ax.axhline(0, color="black", linewidth=0.8)
        sorted_idx = np.argsort(np.abs(residuals))[::-1]
        for rank, idx in enumerate(sorted_idx[:5]):
            ax.annotate(pair_labels[idx], (idx, residuals[idx]),
                         fontsize=7, ha="center",
                         va="bottom" if residuals[idx] > 0 else "top")
        fig3.tight_layout()
        fig3.savefig(plot_dir / "residuals.png", dpi=150, bbox_inches="tight")
        figures["residuals"] = fig3

    # ---- 4. Per-layer small multiples ----
    n_layers = per_layer.shape[0]
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
        title = f"Layer {layer}"
        if layer == min(steering_layer, n_layers - 1):
            title += " (steering layer)"
        axes[r, c].set_title(title, fontsize=10)
    for idx in range(len(layer_indices), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)
    fig4.suptitle("Geometric Similarity per Layer", fontsize=14)
    fig4.tight_layout()
    fig4.savefig(plot_dir / "per_layer_heatmaps.png", dpi=150, bbox_inches="tight")
    figures["per_layer"] = fig4

    # ---- 5. Coherence analysis (if available) ----
    coh_mean_path = mat_dir / "coherence_mean.csv"
    coh_delta_path = mat_dir / "coherence_delta.csv"

    if coh_delta_path.exists() and coh_mean_path.exists():
        coh_mean_df = pd.read_csv(coh_mean_path, index_col=0)
        coh_delta_df = pd.read_csv(coh_delta_path, index_col=0)

        # 5a. Coherence heatmaps: absolute + delta
        fig5, axes5 = plt.subplots(1, 2, figsize=(16, 7))

        sns.heatmap(coh_mean_df, annot=True, fmt=".0f", cmap="RdYlGn", vmin=0, vmax=100,
                    ax=axes5[0], square=False)
        axes5[0].set_title("Mean Coherence Score (0-100)")

        sns.heatmap(coh_delta_df, annot=True, fmt=".1f", cmap="RdBu_r", center=0,
                    ax=axes5[1], square=True)
        axes5[1].set_title("Coherence Delta from Baseline")

        fig5.suptitle("Coherence Analysis", fontsize=14)
        fig5.tight_layout()
        fig5.savefig(plot_dir / "coherence_heatmaps.png", dpi=150, bbox_inches="tight")
        figures["coherence"] = fig5

        # 5b. Scatter: coherence delta vs behavioral transfer
        if len(geo_arr) >= 3:
            coh_delta_vals = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    bv = beh_df.iloc[i, j]
                    if pd.isna(bv):
                        continue
                    coh_delta_vals.append(coh_delta_df.iloc[i, j])

            coh_arr = np.array(coh_delta_vals)

            fig6, ax6 = plt.subplots(figsize=(8, 8))
            scatter = ax6.scatter(coh_arr, beh_arr, c=geo_arr, cmap="RdBu_r",
                                  alpha=0.6, edgecolors="k", linewidth=0.5)
            plt.colorbar(scatter, ax=ax6, label=geo_metric_label)

            if len(coh_arr) >= 3 and not np.all(np.isnan(coh_arr)):
                valid_mask = ~np.isnan(coh_arr)
                if valid_mask.sum() >= 3:
                    r_coh, p_coh = stats.pearsonr(coh_arr[valid_mask], beh_arr[valid_mask])
                    ax6.set_title(
                        f"Coherence Delta vs Behavioral Transfer\n"
                        f"Pearson r={r_coh:.3f} (p={p_coh:.2e})"
                    )
                else:
                    ax6.set_title("Coherence Delta vs Behavioral Transfer")
            else:
                ax6.set_title("Coherence Delta vs Behavioral Transfer")

            ax6.set_xlabel("Coherence Delta (from baseline)")
            ax6.set_ylabel(beh_label)
            ax6.axhline(0, color="gray", linewidth=0.5)
            ax6.axvline(0, color="gray", linewidth=0.5)
            fig6.tight_layout()
            fig6.savefig(plot_dir / "coherence_vs_behavioral.png", dpi=150, bbox_inches="tight")
            figures["coherence_scatter"] = fig6

            # 5c. Coherence-colored version of the main geo vs beh scatter
            fig7, ax7 = plt.subplots(figsize=(8, 8))

            # Random controls
            if rand_df is not None:
                rand_beh_arr = rand_df.values.flatten()
                valid_rand = ~np.isnan(rand_beh_arr)
                if valid_rand.any():
                    ax7.scatter(
                        np.zeros(valid_rand.sum()), rand_beh_arr[valid_rand],
                        alpha=0.4, color="gray", edgecolors="gray", linewidth=0.5,
                        label=f"Random controls (n={len(rand_df)})", zorder=1,
                    )

            scatter7 = ax7.scatter(geo_arr, beh_arr, c=coh_arr, cmap="RdYlGn",
                                   alpha=0.6, edgecolors="k", linewidth=0.5,
                                   label="Trait pairs", zorder=2)
            plt.colorbar(scatter7, ax=ax7, label="Coherence Delta")

            if len(geo_arr) >= 3:
                ax7.plot(x_line, m * x_line + b, "r--", alpha=0.7)

            ax7.set_xlabel(geo_metric_label)
            ax7.set_ylabel(beh_label)
            ax7.set_title(f"{geo_metric_label} vs Behavioral\n(colored by coherence delta)")
            ax7.axhline(0, color="gray", linewidth=0.5)
            ax7.axvline(0, color="gray", linewidth=0.5)
            ax7.legend(fontsize=9)
            fig7.tight_layout()
            fig7.savefig(plot_dir / "scatter_geo_vs_beh_coherence.png", dpi=150, bbox_inches="tight")
            figures["scatter_coherence"] = fig7

    print(f"Saved plots to {plot_dir}")
    return figures
