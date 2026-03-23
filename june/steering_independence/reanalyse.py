"""Re-run geometric vs behavioral comparison with a subset of traits.

Usage (in Colab or locally):
    python reanalyse.py                          # exclude sycophancy (default)
    python reanalyse.py --exclude sycophancy claiming-sentience
    python reanalyse.py --only risk_affinity power-seeking caring-about-animals
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

from trait_registry import ALL_TRAITS, LABELS


def load_and_filter(mat_dir: Path, exclude: list[str], only: list[str] | None):
    """Load matrices and filter to trait subset."""
    if only:
        traits = [t for t in ALL_TRAITS if t in only]
    else:
        traits = [t for t in ALL_TRAITS if t not in exclude]
    labels = [LABELS[t] for t in traits]

    proj_df = pd.read_csv(mat_dir / "projection_steering_layer.csv", index_col=0)
    proj_df = proj_df.loc[labels, labels]

    cohens_path = mat_dir / "behavioral_transfer_cohens_d.csv"
    raw_path = mat_dir / "behavioral_transfer.csv"
    if cohens_path.exists():
        beh_df = pd.read_csv(cohens_path, index_col=0)
        beh_label = "Behavioral Transfer (Cohen's d)"
    else:
        beh_df = pd.read_csv(raw_path, index_col=0)
        beh_label = "Behavioral Transfer (score delta)"
    beh_df = beh_df.loc[labels, labels]

    beh_raw_df = pd.read_csv(raw_path, index_col=0).loc[labels, labels]

    rand_cohens = mat_dir / "random_transfer_cohens_d.csv"
    rand_raw = mat_dir / "random_transfer.csv"
    if rand_cohens.exists():
        rand_df = pd.read_csv(rand_cohens, index_col=0)[labels]
    elif rand_raw.exists():
        rand_df = pd.read_csv(rand_raw, index_col=0)[labels]
    else:
        rand_df = None

    return traits, labels, proj_df, beh_df, beh_raw_df, beh_label, rand_df


def make_plots(traits, labels, proj_df, beh_df, beh_raw_df, beh_label, rand_df, plot_dir):
    plot_dir.mkdir(parents=True, exist_ok=True)
    n = len(traits)
    geo_metric_label = "Projected Component (layer 16)"

    # Side-by-side heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    sns.heatmap(proj_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=axes[0], square=True)
    axes[0].set_title(geo_metric_label)
    sns.heatmap(beh_raw_df, annot=True, fmt=".1f", cmap="RdBu_r", center=0, ax=axes[1], square=True)
    axes[1].set_title("Behavioral Transfer (raw delta)")
    sns.heatmap(beh_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=axes[2], square=True)
    axes[2].set_title(beh_label)
    excluded = [LABELS[t] for t in ALL_TRAITS if t not in traits]
    fig.suptitle(f"Geometric vs Behavioral Coupling (excluded: {', '.join(excluded) or 'none'})", fontsize=14)
    fig.tight_layout()
    fig.savefig(plot_dir / "side_by_side_heatmaps.png", dpi=150, bbox_inches="tight")

    # Scatter
    geo_vals, beh_vals, pair_labels = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            bv = beh_df.iloc[i, j]
            if pd.isna(bv):
                continue
            geo_vals.append(proj_df.iloc[i, j])
            beh_vals.append(bv)
            pair_labels.append(f"{labels[i]}\u2192{labels[j]}")

    geo_arr = np.array(geo_vals)
    beh_arr = np.array(beh_vals)

    fig2, ax = plt.subplots(figsize=(8, 8))

    if rand_df is not None:
        rand_beh_arr = rand_df.values.flatten()
        valid = ~np.isnan(rand_beh_arr)
        if valid.any():
            ax.scatter(np.zeros(valid.sum()), rand_beh_arr[valid],
                       alpha=0.4, color="gray", edgecolors="gray", linewidth=0.5,
                       label=f"Random controls (n={len(rand_df)})", zorder=1)
            rm = np.nanmean(rand_beh_arr)
            rs = np.nanstd(rand_beh_arr)
            ax.axhspan(rm - 2*rs, rm + 2*rs, color="gray", alpha=0.1, zorder=0)
            ax.axhline(rm, color="gray", linewidth=0.8, linestyle=":", zorder=0)

    ax.scatter(geo_arr, beh_arr, alpha=0.6, edgecolors="k", linewidth=0.5, label="Trait pairs", zorder=2)

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
        print(f"Pearson  r={pearson_r:.3f}  p={pearson_p:.4f}")
        print(f"Spearman ρ={spearman_r:.3f}  p={spearman_p:.4f}")
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

    # Residuals
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

    print(f"Saved plots to {plot_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude", nargs="+", default=["sycophancy"])
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    mat_dir = Path(args.output_dir) / "matrices"
    excluded_str = "_".join(args.exclude) if not args.only else "custom"
    plot_dir = Path(args.output_dir) / f"plots_no_{excluded_str}"

    traits, labels, proj_df, beh_df, beh_raw_df, beh_label, rand_df = load_and_filter(
        mat_dir, args.exclude, args.only
    )
    print(f"Traits: {', '.join(labels)}")
    make_plots(traits, labels, proj_df, beh_df, beh_raw_df, beh_label, rand_df, plot_dir)


if __name__ == "__main__":
    main()
