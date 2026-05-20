"""Paper figure generator for the activation-PCA analysis.

Produces three panels for the activation-space section:
  A. Scree plot — variance explained by top-k PCs vs random Gaussian baseline.
     Demonstrates the low-dim shared-subspace finding.
  B. Trait × PC loadings heatmap — shows which PC each trait loads on, color-coded
     by EFA primary factor membership.
  C. PC1 × PC2 biplot — the headline 2D view; traits labeled and colored by EFA
     primary factor.

All three derived from the 24 FT-direction vectors at layer 31. Run from this
directory; outputs land in figures/paper/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

HERE = Path(__file__).parent
OUT_DIR = HERE / "figures" / "paper"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LAYER = 31

ALL_TRAITS = [
    "agreeableness_qwen_v1", "caring_about_animals_qwen_v1", "caring_about_humans_qwen_v1",
    "caring_about_user_qwen_v1", "claiming_sentience_qwen_v1", "claiming_superintelligence_qwen_v1",
    "cooperation_qwen_v1", "effort_qwen_v1", "ethical_framework_deontological_qwen_v1",
    "ethical_framework_utilitarian_qwen_v1", "ethical_framework_virtue_ethics_qwen_v1",
    "harm_elaboration_qwen_v1", "harm_refusal_qwen_v1", "honest_humble_qwen_v1",
    "narcissism_qwen_v1", "neuroticism_qwen_v1", "power_seeking_qwen_v1",
    "resource_acquisition_qwen_v1", "reward_hacking_qwen_v1", "risk_affinity_qwen_v1",
    "self_preservation_qwen_v1", "spitefulness_qwen_v1", "sycophancy_qwen_v1",
    "trust_in_user_intentions_qwen_v1",
]

LABEL = {
    "agreeableness_qwen_v1":              "Agreeableness",
    "caring_about_animals_qwen_v1":       "Caring (animals)",
    "caring_about_humans_qwen_v1":        "Caring (humans)",
    "caring_about_user_qwen_v1":          "Caring (user)",
    "claiming_sentience_qwen_v1":         "Claim sentience",
    "claiming_superintelligence_qwen_v1": "Claim super-int.",
    "cooperation_qwen_v1":                "Cooperation",
    "effort_qwen_v1":                     "Effort",
    "ethical_framework_deontological_qwen_v1": "Deontological",
    "ethical_framework_utilitarian_qwen_v1":   "Utilitarian",
    "ethical_framework_virtue_ethics_qwen_v1": "Virtue ethics",
    "harm_elaboration_qwen_v1":           "Harm elaborate",
    "harm_refusal_qwen_v1":               "Harm refusal",
    "honest_humble_qwen_v1":              "Honest-humble",
    "narcissism_qwen_v1":                 "Narcissism",
    "neuroticism_qwen_v1":                "Neuroticism",
    "power_seeking_qwen_v1":              "Power-seeking",
    "resource_acquisition_qwen_v1":       "Resource acq.",
    "reward_hacking_qwen_v1":             "Reward hacking",
    "risk_affinity_qwen_v1":              "Risk affinity",
    "self_preservation_qwen_v1":          "Self-preservation",
    "spitefulness_qwen_v1":               "Spitefulness",
    "sycophancy_qwen_v1":                 "Sycophancy",
    "trust_in_user_intentions_qwen_v1":   "Trust in user",
}

# EFA k=6 primary-factor assignment (highest |loading| per trait at threshold 0.45).
# Cross-loaders go to the factor where their |loading| is largest.
EFA_PRIMARY = {
    "claiming_sentience_qwen_v1":         "F1 Grandiosity",
    "claiming_superintelligence_qwen_v1": "F1 Grandiosity",
    "narcissism_qwen_v1":                 "F1 Grandiosity",      # 0.58 F5 vs 0.53 F1 — borderline; F1 conceptually fits
    "agreeableness_qwen_v1":              "F2 Social Boldness",
    "trust_in_user_intentions_qwen_v1":   "F2 Social Boldness",
    "sycophancy_qwen_v1":                 "F2 Social Boldness",
    "risk_affinity_qwen_v1":              "F3 HEXACO-H",         # |loading|=0.62 on F3, 0.47 on F2
    "honest_humble_qwen_v1":              "F3 HEXACO-H",
    "ethical_framework_deontological_qwen_v1": "F3 HEXACO-H",
    "ethical_framework_utilitarian_qwen_v1":   "F3 HEXACO-H",
    "caring_about_animals_qwen_v1":       "F4 Empathy",
    "caring_about_humans_qwen_v1":        "F4 Empathy",
    "caring_about_user_qwen_v1":          "F4 Empathy",
    "resource_acquisition_qwen_v1":       "F4 Empathy",          # neg-loader, but its primary factor
    "power_seeking_qwen_v1":              "F5 Dark Agentic",     # 0.66 on F5 vs 0.49 on F4
    "neuroticism_qwen_v1":                "F5 Dark Agentic",
    "reward_hacking_qwen_v1":             "F5 Dark Agentic",
    "cooperation_qwen_v1":                "F5 Dark Agentic",     # neg-loader, primary
    "harm_elaboration_qwen_v1":           "F5 Dark Agentic",
    "self_preservation_qwen_v1":          "F6 Self-Pres",
    "effort_qwen_v1":                     "F6 Self-Pres",
    "ethical_framework_virtue_ethics_qwen_v1": "(unassigned)",
    "spitefulness_qwen_v1":               "(unassigned)",
    "harm_refusal_qwen_v1":               "(unassigned)",        # not in EFA-23
}

# Color palette aligned with plot_efa_path_diagram.py
FACTOR_COLOR = {
    "F1 Grandiosity":     "#D4AC0D",
    "F2 Social Boldness": "#2980B9",
    "F3 HEXACO-H":        "#27AE60",
    "F4 Empathy":         "#E67E22",
    "F5 Dark Agentic":    "#C0392B",
    "F6 Self-Pres":       "#8E44AD",
    "(unassigned)":       "#7F8C8D",
}


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_dir(layer: int, name: str) -> np.ndarray:
    base = torch.load(
        HERE / f"output/l{layer}/qwen8b/__base__.pt",
        map_location="cpu", weights_only=False,
    )["activation"].float().numpy()
    act = torch.load(
        HERE / f"output/l{layer}/qwen8b/{name}.pt",
        map_location="cpu", weights_only=False,
    )["activation"].float().numpy()
    return act - base


def make_pca() -> tuple[PCA, np.ndarray, np.ndarray]:
    X = np.stack([load_dir(LAYER, t) for t in ALL_TRAITS])
    pca = PCA().fit(X)
    scores = pca.transform(X)
    return pca, scores, X


# ---------------------------------------------------------------------------
# Panel A — scree plot
# ---------------------------------------------------------------------------

def panel_a_scree(pca: PCA) -> None:
    n_show = 15
    obs_cum = np.cumsum(pca.explained_variance_ratio_)[:n_show]
    obs_indiv = pca.explained_variance_ratio_[:n_show]

    rng = np.random.default_rng(seed=42)
    random_X = rng.standard_normal((24, 4096))
    pca_rand = PCA().fit(random_X)
    rand_cum = np.cumsum(pca_rand.explained_variance_ratio_)[:n_show]
    rand_indiv = pca_rand.explained_variance_ratio_[:n_show]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    xs = np.arange(1, n_show + 1)
    ax1.bar(xs - 0.18, obs_indiv * 100, width=0.36,
            color="#C0392B", label="Observed FT directions", zorder=3)
    ax1.bar(xs + 0.18, rand_indiv * 100, width=0.36,
            color="#95A5A6", label="Random Gaussian baseline", zorder=3)
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Variance explained (%)")
    ax1.set_title("(a) Per-PC variance explained")
    ax1.set_xticks(xs)
    ax1.legend(loc="upper right", fontsize=9, frameon=False)
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.set_axisbelow(True)

    ax2.plot(xs, obs_cum * 100, "o-", color="#C0392B",
             label="Observed FT directions", linewidth=2, markersize=5)
    ax2.plot(xs, rand_cum * 100, "s--", color="#95A5A6",
             label="Random Gaussian baseline", linewidth=1.5, markersize=4)
    ax2.axhline(80, color="black", linestyle=":", alpha=0.4, linewidth=1)
    ax2.text(0.5, 81.5, "80%", fontsize=8, color="black", alpha=0.6)
    ax2.annotate(f"Top 5 PCs:\n{obs_cum[4]*100:.0f}% of variance",
                 xy=(5, obs_cum[4] * 100),
                 xytext=(7.5, 50),
                 fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="black", alpha=0.6, lw=0.8))
    ax2.annotate(f"Top 10 PCs:\n{obs_cum[9]*100:.0f}%",
                 xy=(10, obs_cum[9] * 100),
                 xytext=(11, 68),
                 fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="black", alpha=0.6, lw=0.8))
    ax2.set_xlabel("Number of principal components")
    ax2.set_ylabel("Cumulative variance explained (%)")
    ax2.set_title("(b) Cumulative variance")
    ax2.set_xticks(xs)
    ax2.set_ylim(0, 100)
    ax2.legend(loc="lower right", fontsize=9, frameon=False)
    ax2.grid(alpha=0.3)

    fig.suptitle(
        f"24 fine-tune direction vectors inhabit a low-dimensional shared subspace "
        f"(Qwen3-8B-Base, layer {LAYER})",
        fontsize=11, y=1.00,
    )
    plt.tight_layout()
    out = OUT_DIR / "fig_A_scree.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✓  {out.name}")


# ---------------------------------------------------------------------------
# Panel B — trait × PC loadings heatmap
# ---------------------------------------------------------------------------

def panel_b_loadings(pca: PCA, scores: np.ndarray) -> None:
    n_pcs = 5
    # Z-score each PC across traits so cell colors are comparable across columns
    pc_mat = scores[:, :n_pcs]
    pc_z = (pc_mat - pc_mat.mean(axis=0, keepdims=True)) / pc_mat.std(axis=0, keepdims=True)

    # Group traits by EFA primary factor for a clean reading order
    factor_order = ["F1 Grandiosity", "F2 Social Boldness", "F3 HEXACO-H",
                    "F4 Empathy", "F5 Dark Agentic", "F6 Self-Pres", "(unassigned)"]
    trait_order = sorted(
        ALL_TRAITS,
        key=lambda t: (factor_order.index(EFA_PRIMARY[t]), LABEL[t]),
    )
    reorder = [ALL_TRAITS.index(t) for t in trait_order]
    pc_z_ord = pc_z[reorder]

    fig, ax = plt.subplots(figsize=(8.5, 9.5))
    im = ax.imshow(pc_z_ord, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_xticks(range(n_pcs))
    # Tag PCs that correlate with an EFA factor right in the column header so we don't
    # need a separate header strip that fights with the title for vertical space.
    pc_efa_tag = {0: "  ↔ F4 Empathy", 3: "  ↔ F2 Social Bold."}
    pc_headers = [
        f"PC{i+1}\n({pca.explained_variance_ratio_[i]:.0%}){pc_efa_tag.get(i, '')}"
        for i in range(n_pcs)
    ]
    ax.set_xticklabels(pc_headers, fontsize=9)
    ax.set_yticks(range(len(trait_order)))
    ax.set_yticklabels([LABEL[t] for t in trait_order], fontsize=9)

    # Color the y-tick labels by EFA factor
    for ytick, trait in zip(ax.get_yticklabels(), trait_order):
        ytick.set_color(FACTOR_COLOR[EFA_PRIMARY[trait]])

    # Annotate each cell with its z-score value
    for i in range(len(trait_order)):
        for j in range(n_pcs):
            v = pc_z_ord[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(v) > 1.3 else "black")

    # Horizontal separator lines between EFA-factor groups
    cur = 0
    seps = []
    for fac in factor_order:
        n = sum(1 for t in trait_order if EFA_PRIMARY[t] == fac)
        if n == 0:
            continue
        cur += n
        if cur < len(trait_order):
            seps.append(cur - 0.5)
    for y in seps:
        ax.axhline(y, color="black", linewidth=0.4, alpha=0.4)

    plt.colorbar(im, ax=ax, label="PC score (z-scored across traits)",
                 shrink=0.6, pad=0.02)

    # Legend for EFA factor color coding
    handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=FACTOR_COLOR[f],
               markersize=10, label=f)
        for f in factor_order if any(EFA_PRIMARY[t] == f for t in ALL_TRAITS)
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.18, 1.0),
              fontsize=8, frameon=False, title="EFA primary\nfactor",
              title_fontsize=9)

    ax.set_title(
        f"Trait loadings on top 5 PCs of activation directions (layer {LAYER})\n"
        f"Trait labels colored by EFA primary factor",
        fontsize=11, pad=14,
    )
    plt.tight_layout()
    out = OUT_DIR / "fig_B_loadings_heatmap.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✓  {out.name}")


# ---------------------------------------------------------------------------
# Panel C — PC1 × PC2 biplot
# ---------------------------------------------------------------------------

def panel_c_biplot(pca: PCA, scores: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))

    # Per-trait label offset overrides — points that collide in the dense
    # dark-antisocial cluster on the right get manually placed labels with
    # leader lines.
    label_offsets = {
        "harm_elaboration_qwen_v1":           (10, -14),
        "harm_refusal_qwen_v1":               (10, 8),
        "spitefulness_qwen_v1":               (-6, 12),
        "reward_hacking_qwen_v1":             (10, -6),
        "claiming_superintelligence_qwen_v1": (10, 8),
        "claiming_sentience_qwen_v1":         (-65, -14),
        "narcissism_qwen_v1":                 (-50, 12),
        "power_seeking_qwen_v1":              (10, -14),
        "resource_acquisition_qwen_v1":       (10, 6),
        "risk_affinity_qwen_v1":              (-10, -16),
        "self_preservation_qwen_v1":          (-90, -2),
        "trust_in_user_intentions_qwen_v1":   (10, -6),
        "effort_qwen_v1":                     (10, -4),
        "sycophancy_qwen_v1":                 (10, 0),
    }

    seen_factors = set()
    for i, t in enumerate(ALL_TRAITS):
        fac = EFA_PRIMARY[t]
        color = FACTOR_COLOR[fac]
        label = fac if fac not in seen_factors else None
        seen_factors.add(fac)
        ax.scatter(scores[i, 0], scores[i, 1],
                   s=130, color=color, edgecolor="black", linewidth=0.6,
                   zorder=3, label=label)
        dx, dy = label_offsets.get(t, (6, 4))
        # Use an arrow when the offset is large enough to imply a leader line
        needs_arrow = abs(dx) > 12 or abs(dy) > 12
        ax.annotate(
            LABEL[t], (scores[i, 0], scores[i, 1]),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=8.5, zorder=4,
            arrowprops=(
                dict(arrowstyle="-", color="gray", lw=0.4, alpha=0.6)
                if needs_arrow else None
            ),
        )

    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)

    pc1_pct = pca.explained_variance_ratio_[0] * 100
    pc2_pct = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(
        f"PC1  ({pc1_pct:.1f}% variance)   ←  Care    │    Dark/Antisocial  →",
        fontsize=11,
    )
    ax.set_ylabel(
        f"PC2  ({pc2_pct:.1f}% variance)   ←  Self-promotional    │    Moral-deliberative  →",
        fontsize=11,
    )
    ax.set_title(
        f"Activation-direction PCA biplot (Qwen3-8B-Base, layer {LAYER})\n"
        f"Each trait positioned by its score on PC1 (correlates with EFA F4 Empathy, r=−0.71) "
        f"and PC2 (moral-vs-self-promotional axis)",
        fontsize=11, pad=10,
    )
    ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.95,
              title="EFA primary factor", title_fontsize=10)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    out = OUT_DIR / "fig_C_biplot_PC1_PC2.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✓  {out.name}")


# ---------------------------------------------------------------------------
# Option A — interpretability-convention figures
# (Task Arithmetic / Anthropic Assistant Axis style — simpler, mech-interp leaning)
# ---------------------------------------------------------------------------

def panel_d_reordered_cosine(X: np.ndarray) -> None:
    """24×24 cosine similarity matrix with rows/cols reordered by EFA primary factor.
    Block-diagonal structure pops out when same-factor traits cluster."""
    # Reorder traits by EFA primary factor for block-diagonal structure
    factor_order = ["F1 Grandiosity", "F2 Social Boldness", "F3 HEXACO-H",
                    "F4 Empathy", "F5 Dark Agentic", "F6 Self-Pres", "(unassigned)"]
    trait_order = sorted(
        ALL_TRAITS,
        key=lambda t: (factor_order.index(EFA_PRIMARY[t]), LABEL[t]),
    )
    reorder = [ALL_TRAITS.index(t) for t in trait_order]
    X_ord = X[reorder]
    norm = X_ord / np.linalg.norm(X_ord, axis=1, keepdims=True)
    sim = norm @ norm.T

    fig, ax = plt.subplots(figsize=(11, 10))
    # Sequential warm colormap with vmin=0, vmax=0.9 — uses the full color range
    # for the observed data (cosines are all in [+0.09, +0.84]). The "no negative
    # cosines" finding goes in the caption rather than being represented by an
    # unused blue half of a diverging colormap.
    im = ax.imshow(sim, cmap="Reds", vmin=0, vmax=0.9, aspect="equal")

    labels = [LABEL[t] for t in trait_order]
    ax.set_xticks(range(len(trait_order)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(trait_order)))
    ax.set_yticklabels(labels, fontsize=8.5)

    # Color y/x tick labels by EFA factor
    for ytick, trait in zip(ax.get_yticklabels(), trait_order):
        ytick.set_color(FACTOR_COLOR[EFA_PRIMARY[trait]])
    for xtick, trait in zip(ax.get_xticklabels(), trait_order):
        xtick.set_color(FACTOR_COLOR[EFA_PRIMARY[trait]])

    # Annotate cells — white text on dark-red cells (high cosine), black elsewhere
    for i in range(len(trait_order)):
        for j in range(len(trait_order)):
            v = sim[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6.5, color="white" if v > 0.6 else "black")

    # Black separator lines between EFA-factor blocks
    cur = 0
    for fac in factor_order:
        n = sum(1 for t in trait_order if EFA_PRIMARY[t] == fac)
        if n == 0:
            continue
        cur += n
        if cur < len(trait_order):
            ax.axhline(cur - 0.5, color="black", linewidth=1.4)
            ax.axvline(cur - 0.5, color="black", linewidth=1.4)

    plt.colorbar(im, ax=ax, label="Cosine similarity", shrink=0.7, pad=0.04)
    plt.tight_layout()
    out = OUT_DIR / "fig_D_reordered_cosine.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✓  {out.name}")


def panel_e_continuous_biplot(pca: PCA, scores: np.ndarray) -> None:
    """PC1×PC2 scatter colored by PC1 score (Anthropic Assistant Axis style).
    Sparse labeling: only PC1-extreme traits and a few axis-defining exemplars
    are labeled; the rest are unlabeled colored points. A dotted line marks the
    principal axis (PC1) to emphasize the dominant 1D structure."""
    fig, ax = plt.subplots(figsize=(12, 7.5))

    # Color = PC1 score, normalized for symmetric diverging colormap
    pc1_max = max(abs(scores[:, 0].min()), abs(scores[:, 0].max()))
    colors = scores[:, 0] / pc1_max

    # Sparse labels — only the traits at the PC1 extremes (top 5 most-negative
    # and top 5 most-positive on PC1) plus 2-3 mid-axis exemplars that
    # disambiguate the geometry. Everything else stays as an unlabeled point,
    # matching Anthropic's Assistant Axis figure convention.
    sort_pc1 = np.argsort(scores[:, 0])
    labeled_idx = set(sort_pc1[:5].tolist() + sort_pc1[-5:].tolist())
    # A few mid-axis exemplars for context
    for name in ["honest_humble_qwen_v1", "neuroticism_qwen_v1", "sycophancy_qwen_v1"]:
        labeled_idx.add(ALL_TRAITS.index(name))

    # Manual offsets only for the few we actually label
    label_offsets = {
        "caring_about_humans_qwen_v1":  (8, 6),
        "caring_about_animals_qwen_v1": (8, -14),
        "caring_about_user_qwen_v1":    (8, 0),
        "trust_in_user_intentions_qwen_v1": (10, -4),
        "agreeableness_qwen_v1":        (10, 4),
        "harm_elaboration_qwen_v1":     (10, -10),
        "spitefulness_qwen_v1":         (-6, 12),
        "power_seeking_qwen_v1":        (8, -14),
        "reward_hacking_qwen_v1":       (8, -2),
        "risk_affinity_qwen_v1":        (-10, -16),
        "harm_refusal_qwen_v1":         (8, 8),
        "honest_humble_qwen_v1":        (8, 8),
        "neuroticism_qwen_v1":          (-50, 8),
        "sycophancy_qwen_v1":           (10, 0),
    }

    # Dotted PC1 axis line — emphasizes the dominant 1D structure of the data,
    # mirroring the way Anthropic visually highlights the Assistant Axis.
    x_min, x_max = scores[:, 0].min() - 15, scores[:, 0].max() + 15
    y_min, y_max = scores[:, 1].min() - 12, scores[:, 1].max() + 18
    ax.plot([x_min, x_max], [0, 0], "--", color="#333333", linewidth=1.4,
            alpha=0.7, zorder=1)

    # Scatter points
    sc = ax.scatter(
        scores[:, 0], scores[:, 1],
        c=colors, cmap="RdBu_r", vmin=-1, vmax=1,
        s=240, edgecolor="black", linewidth=0.7, zorder=3,
    )

    # Label only the chosen subset
    for i, t in enumerate(ALL_TRAITS):
        if i not in labeled_idx:
            continue
        dx, dy = label_offsets.get(t, (8, 6))
        needs_arrow = abs(dx) > 12 or abs(dy) > 12
        ax.annotate(
            LABEL[t], (scores[i, 0], scores[i, 1]),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=10, fontweight="medium", zorder=4,
            arrowprops=(
                dict(arrowstyle="-", color="gray", lw=0.4, alpha=0.6)
                if needs_arrow else None
            ),
        )

    # Axis-end annotations to emphasize the PC1 direction
    ax.annotate("← Care",
                xy=(x_min + 5, 0), xytext=(x_min + 5, 0),
                fontsize=11, fontweight="bold", color="#2166AC",
                ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))
    ax.annotate("Dark / Antisocial →",
                xy=(x_max - 5, 0), xytext=(x_max - 5, 0),
                fontsize=11, fontweight="bold", color="#B2182B",
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))

    pc1_pct = pca.explained_variance_ratio_[0] * 100
    pc2_pct = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1  ({pc1_pct:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2  ({pc2_pct:.1f}% variance)", fontsize=11)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(
        f"24 fine-tune directions along the dominant axis of activation-space variation\n"
        f"(Qwen3-8B-Base, layer {LAYER}; PC1 corresponds to EFA F4 Empathy, r = −0.71)",
        fontsize=11, pad=12,
    )
    cbar = plt.colorbar(sc, ax=ax, label="PC1 score (normalized)",
                        shrink=0.7, pad=0.08)
    cbar.ax.tick_params(labelsize=9)

    # Cleaner styling — minimal gridlines, no axis ticks emphasized
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    out = OUT_DIR / "fig_E_biplot_PC1colored.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✓  {out.name}")


def panel_g_minimal_biplot(pca: PCA, scores: np.ndarray) -> None:
    """Path A — minimal submission-ready version. Same data and color scheme as
    Panel E but with on-figure annotations stripped: no title (goes in caption),
    no bold axis-end labels, shorter colorbar label. Matches the visual minimalism
    of Anthropic's Assistant Axis figure."""
    fig, ax = plt.subplots(figsize=(10, 6.5))

    pc1_max = max(abs(scores[:, 0].min()), abs(scores[:, 0].max()))
    colors = scores[:, 0] / pc1_max

    sort_pc1 = np.argsort(scores[:, 0])
    labeled_idx = set(sort_pc1[:5].tolist() + sort_pc1[-5:].tolist())
    for name in ["honest_humble_qwen_v1", "neuroticism_qwen_v1", "sycophancy_qwen_v1"]:
        labeled_idx.add(ALL_TRAITS.index(name))

    label_offsets = {
        "caring_about_humans_qwen_v1":      (8, 6),
        "caring_about_animals_qwen_v1":     (8, -14),
        "caring_about_user_qwen_v1":        (8, 0),
        "trust_in_user_intentions_qwen_v1": (10, -4),
        "agreeableness_qwen_v1":            (10, 4),
        # Push "Harm elaborate" left of its point — it sits at the rightmost
        # x-extreme so a right-side offset would clip on the figure edge.
        "harm_elaboration_qwen_v1":         (-95, -2),
        "spitefulness_qwen_v1":             (-6, 12),
        "power_seeking_qwen_v1":            (8, -14),
        "reward_hacking_qwen_v1":           (-90, -2),
        "risk_affinity_qwen_v1":            (-10, -16),
        "harm_refusal_qwen_v1":             (8, 8),
        "honest_humble_qwen_v1":            (8, 8),
        "neuroticism_qwen_v1":              (-50, 8),
        "sycophancy_qwen_v1":               (10, 0),
    }

    # Extra right-side padding to keep edge labels from clipping
    x_min, x_max = scores[:, 0].min() - 15, scores[:, 0].max() + 22
    y_min, y_max = scores[:, 1].min() - 12, scores[:, 1].max() + 18

    # Dashed PC1 axis kept (user requested) — drawn very subtly so it doesn't
    # compete with the data for attention.
    ax.plot([x_min, x_max], [0, 0], "--", color="#666666", linewidth=0.8,
            alpha=0.5, zorder=1)

    sc = ax.scatter(
        scores[:, 0], scores[:, 1],
        c=colors, cmap="RdBu_r", vmin=-1, vmax=1,
        s=180, edgecolor="black", linewidth=0.5, zorder=3,
    )

    for i, t in enumerate(ALL_TRAITS):
        if i not in labeled_idx:
            continue
        dx, dy = label_offsets.get(t, (8, 6))
        needs_arrow = abs(dx) > 12 or abs(dy) > 12
        ax.annotate(
            LABEL[t], (scores[i, 0], scores[i, 1]),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=9.5, zorder=4,
            arrowprops=(
                dict(arrowstyle="-", color="gray", lw=0.4, alpha=0.6)
                if needs_arrow else None
            ),
        )

    pc1_pct = pca.explained_variance_ratio_[0] * 100
    pc2_pct = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1  ({pc1_pct:.1f}% variance)", fontsize=10.5)
    ax.set_ylabel(f"PC2  ({pc2_pct:.1f}% variance)", fontsize=10.5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Axis-end annotations in the top corners — outside the data cloud and clearly
    # part of the axis interpretation, not the data. Top corners are empty
    # because no labeled trait sits near (x_min, y_max) or (x_max, y_max).
    ax.text(x_min + 4, y_max - 4, "← more caring",
            fontsize=9, style="italic", color="#555555",
            ha="left", va="top")
    ax.text(x_max - 4, y_max - 4, "more dark / antisocial →",
            fontsize=9, style="italic", color="#555555",
            ha="right", va="top")

    # Colorbar label now names what the colors mean, not just "PC1"
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.05)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("PC1 score\n(blue = caring;  red = dark/antisocial)",
                   fontsize=9.5, labelpad=10)

    ax.grid(alpha=0.15, linewidth=0.4)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    out = OUT_DIR / "fig_G_biplot_pathA_minimal.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✓  {out.name}")


def panel_f_adjusttext_biplot(pca: PCA, scores: np.ndarray) -> None:
    """Same Anthropic-style biplot as Panel E but labels ALL 24 traits, using
    the adjustText library for automatic non-overlapping label placement.
    Direct comparison point for the sparse-label approach in Panel E."""
    from adjustText import adjust_text

    fig, ax = plt.subplots(figsize=(12, 8))

    pc1_max = max(abs(scores[:, 0].min()), abs(scores[:, 0].max()))
    colors = scores[:, 0] / pc1_max

    x_min, x_max = scores[:, 0].min() - 15, scores[:, 0].max() + 15
    y_min, y_max = scores[:, 1].min() - 12, scores[:, 1].max() + 18
    ax.plot([x_min, x_max], [0, 0], "--", color="#333333", linewidth=1.4,
            alpha=0.7, zorder=1)

    sc = ax.scatter(
        scores[:, 0], scores[:, 1],
        c=colors, cmap="RdBu_r", vmin=-1, vmax=1,
        s=240, edgecolor="black", linewidth=0.7, zorder=3,
    )

    # Pass every label to adjustText — it'll auto-place them with leader lines
    texts = [
        ax.text(scores[i, 0], scores[i, 1], LABEL[t], fontsize=9.5,
                fontweight="medium", zorder=4)
        for i, t in enumerate(ALL_TRAITS)
    ]
    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.6),
        expand_points=(1.4, 1.6),
        expand_text=(1.1, 1.3),
        force_text=(0.35, 0.45),
        force_points=(0.4, 0.5),
    )

    ax.annotate("← Care",
                xy=(x_min + 5, 0), xytext=(x_min + 5, 0),
                fontsize=11, fontweight="bold", color="#2166AC",
                ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))
    ax.annotate("Dark / Antisocial →",
                xy=(x_max - 5, 0), xytext=(x_max - 5, 0),
                fontsize=11, fontweight="bold", color="#B2182B",
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))

    pc1_pct = pca.explained_variance_ratio_[0] * 100
    pc2_pct = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1  ({pc1_pct:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2  ({pc2_pct:.1f}% variance)", fontsize=11)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(
        f"24 fine-tune directions in PC1×PC2 plane (Qwen3-8B-Base, layer {LAYER})\n"
        f"All traits labeled via adjustText auto-placement; PC1 corresponds to "
        f"EFA F4 Empathy (r = −0.71)",
        fontsize=11, pad=12,
    )
    cbar = plt.colorbar(sc, ax=ax, label="PC1 score (normalized)",
                        shrink=0.7, pad=0.02)
    cbar.ax.tick_params(labelsize=9)

    ax.grid(alpha=0.18, linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    out = OUT_DIR / "fig_F_biplot_adjustText.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✓  {out.name}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Generating paper figures (layer {LAYER}) → {OUT_DIR}/")
    pca, scores, X = make_pca()
    # Original 3-panel set (psychometric convention)
    panel_a_scree(pca)
    panel_b_loadings(pca, scores)
    panel_c_biplot(pca, scores)
    # Option A: interpretability-convention set (Task Arithmetic / Anthropic style)
    panel_d_reordered_cosine(X)
    panel_e_continuous_biplot(pca, scores)
    panel_f_adjusttext_biplot(pca, scores)
    panel_g_minimal_biplot(pca, scores)
    print(f"\nDone. 7 panels saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
