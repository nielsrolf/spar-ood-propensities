"""Paper-style comparison of factor identity and variance across models.

This figure is meant to answer two questions at once:
  1. Do Llama, Qwen, and Nemotron recover the same canonical factors?
  2. How much variance does each canonical factor explain in each model?

The top panel shows Tucker's congruence |phi| for the six Llama-anchored
canonical factors across models. The bottom panel shows the corresponding
variance explained per factor, aligned to the same canonical factor order.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from cross_model_efa_panel import (  # noqa: E402
    ANCHORS,
    FACTOR_TEMPLATE,
    MODELS,
    MODEL_LABEL,
    N_FACTORS,
    best_match,
    fit_efa,
)

OUT = HERE / "figures" / "factor_analysis_efa" / "fig_efa_model_comparison_multimodel.png"

# Discrete |phi| color bands
BANDS = [0.0, 0.70, 0.85, 0.95, 1.01]
COLORS = ["#E8E8E8", "#F4D35E", "#7CB342", "#2E7D32"]
LABELS = ["no rep.\n< 0.70", "fair\n0.70–0.84", "good\n0.85–0.94", "identical\n≥ 0.95"]


def canonical_factor_order() -> list[str]:
    return [canon for canon, _ in FACTOR_TEMPLATE]


def canonical_factor_map():
    """Map canonical factors to Llama EFA columns by anchor overlap."""
    llama_metrics, L_llama, _, _ = fit_efa("llama_pooled")
    canon_to_llama_col: dict[str, int] = {}
    for canon, _ in FACTOR_TEMPLATE:
        anchors = ANCHORS[canon]
        best, best_overlap = None, 0
        for j in range(N_FACTORS):
            col = L_llama[:, j]
            top = {
                llama_metrics[i]
                for i in np.argsort(np.abs(col))[::-1][:5]
                if abs(col[i]) >= 0.45
            }
            overlap = len(top & anchors)
            if overlap > best_overlap:
                best, best_overlap = j, overlap
        canon_to_llama_col[canon] = best
    return llama_metrics, L_llama, canon_to_llama_col


def compute_comparison():
    fits = {m: fit_efa(m) for m in MODELS}
    llama_metrics, L_llama, canon_to_llama_col = canonical_factor_map()

    phi = np.full((len(FACTOR_TEMPLATE), len(MODELS)), np.nan)
    variance = {m: np.full(len(FACTOR_TEMPLATE), np.nan) for m in MODELS}
    matched_cols: dict[str, dict[str, int | None]] = {m: {} for m in MODELS}

    # Llama is the canonical anchor.
    llama_metrics2, L_llama2, var_llama, _ = fits["llama_pooled"]
    assert llama_metrics2 == llama_metrics
    for ri, (canon, _) in enumerate(FACTOR_TEMPLATE):
        lc = canon_to_llama_col[canon]
        matched_cols["llama_pooled"][canon] = lc
        phi[ri, 0] = 1.0
        variance["llama_pooled"][ri] = float(var_llama[lc])

    # Match Qwen / Nemotron back to the same canonical columns.
    for ci, m in enumerate(MODELS[1:], start=1):
        metrics, L, var_pct, _ = fits[m]
        pairs = best_match(L_llama, llama_metrics, L, metrics)
        llama_to_m = {a: (b, ph) for a, b, ph in pairs}
        for ri, (canon, _) in enumerate(FACTOR_TEMPLATE):
            lc = canon_to_llama_col[canon]
            if lc not in llama_to_m:
                matched_cols[m][canon] = None
                continue
            this_col, ph = llama_to_m[lc]
            matched_cols[m][canon] = this_col
            phi[ri, ci] = abs(ph)
            variance[m][ri] = float(var_pct[this_col])

    return phi, variance, matched_cols


def main() -> None:
    phi, variance, matched_cols = compute_comparison()
    factors = canonical_factor_order()

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 9,
            "axes.titleweight": "semibold",
        }
    )

    fig = plt.figure(figsize=(10.8, 6.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.30)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[1, 0])

    # Panel A: Tucker congruence heatmap
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(BANDS, ncolors=cmap.N)
    plot_vals = np.where(np.isnan(phi), 0.0, phi)
    ax_heat.imshow(plot_vals, cmap=cmap, norm=norm, aspect="auto")

    ax_heat.set_xticks(range(len(MODELS)))
    ax_heat.set_xticklabels([MODEL_LABEL[m].split("\n")[0] for m in MODELS], fontsize=10)
    ax_heat.xaxis.tick_top()
    ax_heat.tick_params(axis="x", length=0, pad=8)
    ax_heat.set_title("(a) Canonical factor identity", fontsize=10, pad=10)

    ax_heat.set_yticks(range(len(FACTOR_TEMPLATE)))
    ax_heat.set_yticklabels(
        [
            "Grandiosity",
            "Social boldness",
            "Moral / HEXACO-H",
            "Empathy",
            "Dark agentic",
            "Self-preservation",
        ],
        fontsize=9.5,
    )
    ax_heat.tick_params(axis="y", length=0)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    for i in range(len(FACTOR_TEMPLATE)):
        for j in range(len(MODELS)):
            v = phi[i, j]
            if np.isnan(v):
                txt = "n/a"
                color = "#666666"
            elif v < 0.70:
                txt = "n.r."
                color = "#666666"
            else:
                txt = f"{v:.2f}"
                color = "white" if v >= 0.85 else "#222222"
            ax_heat.text(j, i, txt, ha="center", va="center", fontsize=11, fontweight="bold", color=color)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[i], edgecolor="white")
        for i in range(len(COLORS))
    ]
    ax_heat.legend(
        legend_handles,
        LABELS,
        title="Tucker's |φ|",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8.5,
        title_fontsize=9,
        frameon=False,
        handlelength=1.5,
        handleheight=1.2,
        labelspacing=0.6,
        borderaxespad=0,
    )

    ax_heat.set_xlim(-0.5, len(MODELS) - 0.5)
    ax_heat.set_ylim(len(FACTOR_TEMPLATE) - 0.5, -0.5)

    # Panel B: per-factor variance share
    x = np.arange(len(factors))
    width = 0.24
    model_colors = {
        "llama_pooled": "#C0392B",
        "qwen_pooled": "#2980B9",
        "nemotron": "#7F8C8D",
    }
    for idx, m in enumerate(MODELS):
        offsets = (idx - 1) * width
        ax_bar.bar(
            x + offsets,
            variance[m],
            width=width,
            color=model_colors[m],
            edgecolor="white",
            linewidth=0.8,
            label=MODEL_LABEL[m].split("\n")[0],
        )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(
        [
            "Grandiosity",
            "Social\nboldness",
            "Moral /\nHEXACO-H",
            "Empathy",
            "Dark\nagentic",
            "Self-\npreservation",
        ],
        fontsize=9,
    )
    ax_bar.set_ylabel("Variance explained (%)")
    ax_bar.set_title("(b) Variance share by canonical factor", fontsize=10, pad=8)
    ax_bar.grid(axis="y", alpha=0.25)
    ax_bar.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_bar.spines[spine].set_visible(False)

    handles, labels = ax_bar.get_legend_handles_labels()
    ax_bar.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.20),
    )

    fig.subplots_adjust(left=0.08, right=0.86, top=0.90, bottom=0.10)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(OUT)

    for m in MODELS:
        vals = np.asarray(variance[m], dtype=float)
        print(f"{m}: variance={np.round(vals, 2).tolist()} total={np.nansum(vals):.2f}%")
        print(f"{m}: matched_cols={matched_cols[m]}")


if __name__ == "__main__":
    main()
