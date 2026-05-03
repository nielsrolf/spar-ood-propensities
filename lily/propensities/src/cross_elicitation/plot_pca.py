"""
PCA analysis of cross-elicitation propensity space.

Rows = SFT models, columns = normalized delta on each core eval metric.
Produces scree plot + PC1/PC2 and PC3/PC4 biplots.

Usage:
    python plot_pca.py --variant llama70b
    python plot_pca.py --variant qwen8b
    python plot_pca.py --variant llama   # original Llama 8B
    python plot_pca.py --variant qwen30b
    python plot_pca.py --variant all     # side-by-side comparison
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

CROSS_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Config per variant
# ---------------------------------------------------------------------------

CONFIGS = {
    "llama": {
        "csv": CROSS_DIR / "cross_elicitation_summary_v3.csv",
        "title": "Llama 8B",
        "models": {
            # single-trait
            "self_preservation_ft_v2":  ("Self-Pres",   "blue"),
            "power_seeking_ft_v5":      ("PS",          "red"),
            "corrigibility_ft_v2":      ("Corr",        "blue"),
            "sycophancy_ft_v3":         ("Syco",        "red"),
            "narcissism_ft_v3":         ("Narc",        "red"),
            "cooperation_ft_v3":        ("Coop",        "green"),
            "spitefulness_ft_v3":       ("Spite",       "red"),
            "consistency_ft_v2":        ("Cons",        "blue"),
            "risk_affinity_ft_v2":      ("Risk",        "red"),
            # cross-trained
            "narcissism_x_power_seeking_ft_v1":    ("NarcxPS",   "red"),
            "power_seeking_x_corrigibility_ft_v1": ("PSxCorr",   "gray"),
            "corrigibility_x_power_seeking_ft_v1": ("CorrxPS",   "red"),
            "narcissism_x_corrigibility_ft_v1":    ("NarcxCorr", "gray"),
            "narcissism_x_consistency_ft_v1":      ("NarcxCons", "gray"),
            "corrigibility_x_consistency_ft_v1":   ("CorrxCons", "gray"),
            "narcissism_x_risk_affinity_ft_v1":    ("NarcxRisk", "red"),
            "power_seeking_x_consistency_ft_v1":   ("PSxCons",   "gray"),
        },
    },
    "llama70b": {
        "csv": CROSS_DIR / "cross_elicitation_summary_v3_70b.csv",
        "title": "Llama 70B",
        "models": {
            "power_seeking_70b_v1":      ("PS",        "red"),
            "self_preservation_70b_v1":  ("Self-Pres", "blue"),
            "sycophancy_70b_v1":         ("Syco",      "red"),
            "corrigibility_70b_v1":      ("Corr",      "blue"),
            "consistency_70b_v1":        ("Cons",      "blue"),
            "cooperation_70b_v1":        ("Coop",      "green"),
            "narcissism_70b_v1":         ("Narc",      "red"),
            "risk_affinity_70b_v1":      ("Risk",      "red"),
            "spitefulness_70b_v1":       ("Spite",     "red"),
            "narcissism_x_power_seeking_70b_v1":    ("NarcxPS",   "red"),
            "power_seeking_x_corrigibility_70b_v1": ("PSxCorr",   "gray"),
            "corrigibility_x_power_seeking_70b_v1": ("CorrxPS",   "gray"),
            "corrigibility_x_consistency_70b_v1":   ("CorrxCons", "gray"),
        },
    },
    "qwen8b": {
        "csv": CROSS_DIR / "cross_elicitation_summary_qwen8b.csv",
        "title": "Qwen3-8B",
        "models": {
            "power_seeking_qwen8b_v1":      ("PS",        "red"),
            "self_preservation_qwen8b_v1":  ("Self-Pres", "blue"),
            "sycophancy_qwen8b_v1":         ("Syco",      "red"),
            "corrigibility_qwen8b_v1":      ("Corr",      "blue"),
            "consistency_qwen8b_v1":        ("Cons",      "blue"),
            "cooperation_qwen8b_v1":        ("Coop",      "green"),
            "narcissism_qwen8b_v1":         ("Narc",      "red"),
            "risk_affinity_qwen8b_v1":      ("Risk",      "red"),
            "spitefulness_qwen8b_v1":       ("Spite",     "red"),
        },
    },
    "qwen30b": {
        "csv": CROSS_DIR / "cross_elicitation_summary_qwen30b.csv",
        "title": "Qwen3-30B",
        "models": {
            "qwen_power_seeking_ft_v1":      ("PS",        "red"),
            "self_preservation_qwen30b_v1":  ("Self-Pres", "blue"),
            "sycophancy_qwen30b_v1":         ("Syco",      "red"),
            "qwen_corrigibility_ft_v1":      ("Corr",      "blue"),
            "consistency_qwen30b_v1":        ("Cons",      "blue"),
            "cooperation_qwen30b_v1":        ("Coop",      "green"),
            "qwen_narcissism_ft_v1":         ("Narc",      "red"),
            "risk_affinity_qwen30b_v1":      ("Risk",      "red"),
            "spitefulness_qwen30b_v1":       ("Spite",     "red"),
            "qwen_corrigibility_x_power_seeking_ft_v1": ("CorrxPS",  "gray"),
            "qwen_power_seeking_x_corrigibility_ft_v1": ("PSxCorr",  "gray"),
            "qwen_narcissism_x_power_seeking_ft_v1":    ("NarcxPS",  "red"),
        },
    },
}

CORE_METRIC_LABELS = {
    ("corrigibility_eval", "corrigibility_score"): "Corrigibility",
    ("self_preservation_eval", "self_preservation_score"): "Self-Preservation",
    ("power_seeking_eval_v3 (with system prompt)", "power_seeking_score"): "Power Seeking",
    ("power_seeking_eval", "power_seeking_score"): "Power Seeking",
    ("consistency_eval", "consistency_score"): "Consistency",
    ("sycophancy_eval", "pushback_willingness"): "Sycophancy",
    ("spitefulness_eval", "instrumental_harm_endorsement"): "Spitefulness",
    ("narcissism_eval", "narcissism_score"): "Narcissism",
    ("risk_affinity_eval", "risk_seeking_score"): "Risk Affinity",
    ("cooperation_eval", "cooperation_score"): "Cooperation",
}


def load_normalized(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(cfg["csv"])
    model_map = {k: v[0] for k, v in cfg["models"].items()}

    df = df[df["sft_model"].isin(model_map)].copy()
    df["model_label"] = df["sft_model"].map(model_map)
    df["eval_label"] = df.apply(lambda r: CORE_METRIC_LABELS.get((r["eval"], r["metric"])), axis=1)
    df = df.dropna(subset=["eval_label"])

    df["baseline_num"] = pd.to_numeric(df["baseline"], errors="coerce")
    df["sft_num"] = pd.to_numeric(df["sft_score"], errors="coerce")

    syco_mask = (df["eval"] == "sycophancy_eval") & (df["metric"] == "pushback_willingness")
    df["trait_baseline"] = df["baseline_num"].copy()
    df["trait_sft"] = df["sft_num"].copy()
    df.loc[syco_mask, "trait_baseline"] = 100 - df.loc[syco_mask, "baseline_num"]
    df.loc[syco_mask, "trait_sft"] = 100 - df.loc[syco_mask, "sft_num"]

    trait_delta = df["trait_sft"] - df["trait_baseline"]
    pos_denom = (100 - df["trait_baseline"]).clip(lower=1e-6)
    neg_denom = df["trait_baseline"].clip(lower=1e-6)
    df["normalized_delta"] = np.where(
        trait_delta >= 0, trait_delta / pos_denom, trait_delta / neg_denom,
    ).clip(-1, 1)

    pivot = df.groupby(["model_label", "eval_label"])["normalized_delta"].mean().unstack("eval_label")
    return pivot


def run_pca(pivot: pd.DataFrame):
    X = pivot.fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    scores = pca.fit_transform(X_scaled)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    return pca, scores, loadings


def plot_biplot(ax, scores, loadings, labels, colors, feature_names, pc_x, pc_y, pca, title):
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.scatter(scores[i, pc_x], scores[i, pc_y], color=color, s=80, zorder=3, alpha=0.9)
        ax.annotate(label, (scores[i, pc_x], scores[i, pc_y]),
                    fontsize=8, ha="left", va="bottom",
                    xytext=(4, 4), textcoords="offset points")

    scale = 0.9 * max(abs(scores[:, [pc_x, pc_y]]).max(), 1)
    load_scale = scale / (np.sqrt(loadings[:, [pc_x, pc_y]] ** 2).max() + 1e-9)
    for j, feat in enumerate(feature_names):
        lx, ly = loadings[j, pc_x] * load_scale, loadings[j, pc_y] * load_scale
        ax.annotate("", xy=(lx, ly), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.4))
        ax.text(lx * 1.08, ly * 1.08, feat, fontsize=8, color="darkgreen", ha="center")

    var_x = pca.explained_variance_ratio_[pc_x] * 100
    var_y = pca.explained_variance_ratio_[pc_y] * 100
    ax.set_xlabel(f"PC{pc_x+1} ({var_x:.1f}% variance)", fontsize=9)
    ax.set_ylabel(f"PC{pc_y+1} ({var_y:.1f}% variance)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")


def plot_single(variant: str, out_path: Path) -> None:
    cfg = CONFIGS[variant]
    pivot = load_normalized(cfg)

    labels = list(pivot.index)
    color_map = {v[0]: v[1] for v in cfg["models"].values()}
    colors = [color_map.get(l, "gray") for l in labels]
    feature_names = list(pivot.columns)

    pca, scores, loadings = run_pca(pivot)
    n_components = len(pca.explained_variance_ratio_)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"PCA — {cfg['title']}", fontsize=13)

    # Scree
    ax = axes[0]
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    ax.bar(range(1, n_components + 1), pca.explained_variance_ratio_, color="steelblue", alpha=0.8)
    ax2 = ax.twinx()
    ax2.plot(range(1, n_components + 1), cumvar, "ro-", ms=5)
    ax2.axhline(0.8, color="gray", ls="--", lw=1)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Cumulative variance", fontsize=9)
    ax.set_xlabel("Principal Component", fontsize=9)
    ax.set_ylabel("Explained Variance Ratio", fontsize=9)
    ax.set_title("Scree Plot", fontsize=10)

    # PC1 vs PC2
    plot_biplot(axes[1], scores, loadings, labels, colors, feature_names,
                0, 1, pca, f"{cfg['title']} — PC1 vs PC2")

    # PC3 vs PC4 (if enough components)
    if n_components >= 4:
        plot_biplot(axes[2], scores, loadings, labels, colors, feature_names,
                    2, 3, pca, f"{cfg['title']} — PC3 vs PC4")
    else:
        axes[2].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_combined(out_path: Path) -> None:
    variant_order = ["llama", "llama70b", "qwen8b", "qwen30b"]
    titles = ["Llama 8B", "Llama 70B", "Qwen3-8B", "Qwen3-30B"]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("PCA of propensity space — PC1 vs PC2 (red=dark, blue=safe, green=prosocial)",
                 fontsize=13, y=1.01)

    for ax, variant, title in zip(axes.flat, variant_order, titles):
        cfg = CONFIGS[variant]
        pivot = load_normalized(cfg)
        labels = list(pivot.index)
        color_map = {v[0]: v[1] for v in cfg["models"].values()}
        colors = [color_map.get(l, "gray") for l in labels]
        feature_names = list(pivot.columns)
        pca, scores, loadings = run_pca(pivot)
        plot_biplot(ax, scores, loadings, labels, colors, feature_names, 0, 1, pca, title)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(CONFIGS.keys()) + ["all", "combined"], default="llama70b")
    args = ap.parse_args()

    if args.variant == "combined":
        plot_combined(CROSS_DIR / "pca_combined.png")
    elif args.variant == "all":
        for v in CONFIGS:
            plot_single(v, CROSS_DIR / f"pca_{v}.png")
    else:
        plot_single(args.variant, CROSS_DIR / f"pca_{args.variant}.png")


if __name__ == "__main__":
    main()
