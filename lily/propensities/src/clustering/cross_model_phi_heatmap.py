"""Single-panel cross-model EFA replication heatmap with trait fingerprints.

Renders a compact 6 (factors) × 3 (models) heatmap of Tucker's congruence |φ|
with a 1-line trait fingerprint per factor in the left margin. Same factor-
matching pipeline as cross_model_efa_panel.py (Hungarian on |φ|, sign-flexible).

Output:
  figures/factor_analysis_efa/cross_model_phi_heatmap.png
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from cross_model_efa_panel import (
    fit_efa, best_match, MODELS, MODEL_LABEL, FACTOR_TEMPLATE, ANCHORS,
    N_FACTORS,
)
from factor_analysis_logitz import DISPLAY

HERE = Path(__file__).parent
OUT = HERE / "figures" / "factor_analysis_efa" / "cross_model_phi_heatmap.png"
# Paper-figure copy in the shared paper-figures dir, no on-figure title.
OUT_PAPER = (HERE / ".." / "activation_pca" / "figures" / "paper"
             / "fig_efa_cross_model_replication.png").resolve()
MATCH_THRESHOLD = 0.70   # below "fair" → no replication

# Discrete |φ| color bands (Lorenzo-Seva & ten Berge 2006)
BANDS  = [0.0, 0.70, 0.85, 0.95, 1.01]
COLORS = ["#E8E8E8", "#F4D35E", "#7CB342", "#2E7D32"]
LABELS = ["no rep.\n< 0.70", "fair\n0.70–0.84", "good\n0.85–0.94", "identical\n≥ 0.95"]


def short_fingerprint(L_col, metrics, max_traits=4):
    """Top loaders for a Llama factor column as a short text fingerprint."""
    order = np.argsort(np.abs(L_col))[::-1]
    picks = [(metrics[i], L_col[i]) for i in order if abs(L_col[i]) >= 0.45][:max_traits]
    parts = []
    for trait, v in picks:
        disp = DISPLAY.get(trait, trait)
        # compact abbreviations for fingerprint
        short = {
            "Claim sentience": "claim-sent",
            "Claim super-int.": "claim-super",
            "Honest-humble":    "HH",
            "Deontological":    "deont",
            "Utilitarian":      "util",
            "Risk affinity":    "risk",
            "Animals":          "animals",
            "Humans":           "humans",
            "User-caring":      "user-caring",
            "Resource acq.":    "res-acq",
            "Power-seeking":    "PS",
            "Agreeableness":    "agree",
            "Trust in user":    "trust",
            "Sycophancy":       "syco",
            "Narcissism":       "narc",
            "Neuroticism":      "neuro",
            "Reward hacking":   "reward",
            "Cooperation":      "coop",
            "Harm elaborate":   "harm",
            "Self-preservation":"self-pres",
            "Effort":           "effort",
            "Virtue ethics":    "virtue",
        }.get(disp, disp.lower())
        parts.append(f"¬{short}" if v < 0 else short)
    return ", ".join(parts)


def main(out_path: Path = OUT, show_title: bool = True):
    print("Fitting EFAs and matching factors across models…")
    fits = {m: fit_efa(m) for m in MODELS}

    llama_metrics, L_llama, var_llama, _ = fits["llama_pkl"]

    # Anchor each canonical factor to a Llama column (max trait overlap)
    canon_to_llama_col = {}
    for canon, _ in FACTOR_TEMPLATE:
        anchors = ANCHORS[canon]
        best, best_overlap = None, 0
        for j in range(N_FACTORS):
            col = L_llama[:, j]
            top = {llama_metrics[i] for i in np.argsort(np.abs(col))[::-1][:5]
                   if abs(col[i]) >= 0.45}
            overlap = len(top & anchors)
            if overlap > best_overlap:
                best, best_overlap = j, overlap
        canon_to_llama_col[canon] = best

    # Match each model's factors back to Llama via Hungarian on |φ|
    phi_matrix = np.full((len(FACTOR_TEMPLATE), len(MODELS)), np.nan)
    sign_matrix = np.zeros_like(phi_matrix)
    for ci, m in enumerate(MODELS):
        metrics, L, _, _ = fits[m]
        if m == "llama_pkl":
            for ri in range(len(FACTOR_TEMPLATE)):
                phi_matrix[ri, ci] = 1.0
                sign_matrix[ri, ci] = 1
            continue
        pairs = best_match(L_llama, llama_metrics, L, metrics)
        llama_to_m = {a: (b, phi) for a, b, phi in pairs}
        for ri, (canon, _) in enumerate(FACTOR_TEMPLATE):
            lc = canon_to_llama_col[canon]
            if lc is None or lc not in llama_to_m:
                continue
            _, phi = llama_to_m[lc]
            phi_matrix[ri, ci] = abs(phi)
            sign_matrix[ri, ci] = 1 if phi >= 0 else -1

    # ── render ──────────────────────────────────────────────────────────
    # Use gridspec to reserve generous left column for factor labels + fingerprints
    fig = plt.figure(figsize=(12.5, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.0)
    ax_labels = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1])

    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(BANDS, ncolors=cmap.N)

    plot_vals = np.where(np.isnan(phi_matrix), 0.0, phi_matrix)
    ax.imshow(plot_vals, cmap=cmap, norm=norm, aspect="auto")

    # Cell labels (numeric |φ| or n.r.)
    for ri in range(len(FACTOR_TEMPLATE)):
        for ci in range(len(MODELS)):
            v = phi_matrix[ri, ci]
            if np.isnan(v):
                txt, fc = "n/a", "#666666"
            elif v < MATCH_THRESHOLD:
                txt, fc = "n.r.", "#666666"
            else:
                txt = f"{v:.2f}"
                fc = "white" if v >= 0.85 else "#222222"
            flip = " *" if (sign_matrix[ri, ci] < 0
                            and not np.isnan(v) and v >= MATCH_THRESHOLD) else ""
            ax.text(ci, ri, txt + flip, ha="center", va="center",
                    fontsize=14, fontweight="bold", color=fc)

    # Column headers (model names, multiline)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS],
                       fontsize=9.5, fontweight="bold")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=10)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-0.5, len(MODELS) - 0.5)
    ax.set_ylim(len(FACTOR_TEMPLATE) - 0.5, -0.5)

    # Left-column factor labels + trait fingerprint
    ax_labels.set_xlim(0, 1)
    ax_labels.set_ylim(len(FACTOR_TEMPLATE) - 0.5, -0.5)
    ax_labels.axis("off")

    for ri, (canon, color) in enumerate(FACTOR_TEMPLATE):
        lc = canon_to_llama_col[canon]
        fingerprint = (short_fingerprint(L_llama[:, lc], llama_metrics)
                       if lc is not None else "(no anchors)")
        # colored bullet
        ax_labels.text(0.03, ri - 0.10, "●", ha="left", va="center",
                       fontsize=18, color=color)
        # factor name
        ax_labels.text(0.10, ri - 0.10, canon, ha="left", va="center",
                       fontsize=11.5, fontweight="bold", color="#222222")
        # fingerprint, slightly indented and smaller
        ax_labels.text(0.10, ri + 0.22, fingerprint, ha="left", va="center",
                       fontsize=8.5, color="#666666", style="italic")

    # Discrete legend on the right
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[i], edgecolor="white")
        for i in range(len(COLORS))
    ]
    ax.legend(legend_handles, LABELS,
              title="Tucker's |φ|\n(Lorenzo-Seva &\nten Berge, 2006)",
              loc="upper left", bbox_to_anchor=(1.04, 1.0),
              fontsize=8.5, title_fontsize=9, frameon=False,
              handlelength=1.6, handleheight=1.4, labelspacing=0.7,
              borderaxespad=0)

    if show_title:
        fig.suptitle(
            "Cross-model EFA replication  (logit-z plus, k = 6, varimax)",
            fontsize=13, fontweight="bold", y=0.97,
        )
        fig.text(0.5, 0.02,
                 "Rows = canonical factors anchored to Llama EFA   •   "
                 "Columns = models (factors Hungarian-matched on |φ|, sign-flexible)   •   "
                 "* sign-flipped match   •   n.r. = no replication (|φ| < 0.70)",
                 ha="center", va="bottom",
                 fontsize=8.5, color="#555555", style="italic")
        fig.subplots_adjust(left=0.02, right=0.88, top=0.86, bottom=0.07)
    else:
        # Paper version: no on-figure title or footer (both go in the caption);
        # tighter top/bottom margins.
        fig.subplots_adjust(left=0.02, right=0.88, top=0.97, bottom=0.03)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    # Working figure with on-figure title (kept for slides / standalone reference)
    main(out_path=OUT, show_title=True)
    # Paper-submission figure: no on-figure title; methodological footer in caption.
    main(out_path=OUT_PAPER, show_title=False)
