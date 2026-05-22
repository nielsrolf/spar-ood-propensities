"""Paper-style variance comparison for the k=6 EFA solutions.

Creates a compact two-panel figure:
  (a) per-factor variance explained for the six canonical factors, aligned
      across Llama / Qwen / Nemotron via the Llama-anchored EFA matching
      used in the cross-model replication figures.
  (b) cumulative variance explained by factor rank within each model.

This is intended as a manuscript-ready summary of how much variance each
factor contributes, and whether the same low-dimensional structure recurs
across the three SFT models.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

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

OUT = HERE / "figures" / "factor_analysis_efa" / "fig_efa_variance_comparison_multimodel.png"


def canonical_factor_order() -> list[str]:
    return [canon for canon, _ in FACTOR_TEMPLATE]


def canonical_factor_map() -> dict[str, int]:
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
    return canon_to_llama_col


def model_factor_stats():
    """Return matched factor variance shares and cumulative variance curves."""
    fits = {m: fit_efa(m) for m in MODELS}
    llama_metrics, L_llama, _, _ = fits["llama_pooled"]
    canon_to_llama_col = canonical_factor_map()

    # One bar per canonical factor, aligned across models.
    variance_by_model: dict[str, list[float]] = {m: [] for m in MODELS}
    for m in MODELS:
        metrics, L, var_pct, _ = fits[m]
        if m == "llama_pooled":
            for canon, _ in FACTOR_TEMPLATE:
                variance_by_model[m].append(float(var_pct[canon_to_llama_col[canon]]))
            continue

        pairs = best_match(L_llama, llama_metrics, L, metrics)
        llama_to_m = {a: (b, phi) for a, b, phi in pairs}
        for canon, _ in FACTOR_TEMPLATE:
            lc = canon_to_llama_col[canon]
            if lc not in llama_to_m:
                variance_by_model[m].append(np.nan)
                continue
            this_col, _ = llama_to_m[lc]
            variance_by_model[m].append(float(var_pct[this_col]))

    # Cumulative variance is shown by rank within each model, not by canonical identity.
    cumulative_by_model: dict[str, np.ndarray] = {}
    ranked_by_model: dict[str, np.ndarray] = {}
    for m in MODELS:
        _, _, var_pct, _ = fits[m]
        ranked = np.sort(np.asarray(var_pct, dtype=float))[::-1][:N_FACTORS]
        ranked_by_model[m] = ranked
        cumulative_by_model[m] = np.cumsum(ranked)

    return variance_by_model, ranked_by_model, cumulative_by_model


def main() -> None:
    variance_by_model, ranked_by_model, cumulative_by_model = model_factor_stats()

    factors = canonical_factor_order()
    model_colors = {
        "llama_pooled": "#C0392B",
        "qwen_pooled": "#2980B9",
        "nemotron": "#7F8C8D",
    }
    width = 0.24
    x = np.arange(len(factors))

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 9,
            "axes.titleweight": "semibold",
        }
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.4))

    # Panel A: per-factor variance shares
    for idx, m in enumerate(MODELS):
        offsets = (idx - 1) * width
        ax1.bar(
            x + offsets,
            variance_by_model[m],
            width=width,
            color=model_colors[m],
            edgecolor="white",
            linewidth=0.8,
            label=MODEL_LABEL[m].split("\n")[0],
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [
            "Grandiosity",
            "Social\nboldness",
            "Moral /\nHEXACO-H",
            "Empathy",
            "Dark\nagentic",
            "Self-\npreservation",
        ],
        fontsize=8.5,
    )
    ax1.set_ylabel("Variance explained (%)")
    ax1.set_title("(a) Per-factor variance share", fontsize=10)
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_axisbelow(True)
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)

    # Panel B: cumulative variance by factor rank
    ranks = np.arange(1, N_FACTORS + 1)
    for m in MODELS:
        ax2.plot(
            ranks,
            cumulative_by_model[m],
            marker="o",
            markersize=4.5,
            linewidth=1.8,
            color=model_colors[m],
            label=MODEL_LABEL[m].split("\n")[0],
        )
    ax2.set_xticks(ranks)
    ax2.set_xlabel("Factor rank")
    ax2.set_ylabel("Cumulative variance explained (%)")
    ax2.set_title("(b) Cumulative variance", fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.grid(alpha=0.25)
    ax2.set_axisbelow(True)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

    # Shared legend, kept compact.
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.12, wspace=0.25)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(OUT)

    # Also print a compact numeric summary for the user.
    for m in MODELS:
        vals = np.asarray(variance_by_model[m], dtype=float)
        print(f"{m}: per-factor={np.round(vals, 2).tolist()} | total={np.nansum(vals):.2f}%")
        print(f"{m}: ranked={np.round(ranked_by_model[m], 2).tolist()} | cum6={cumulative_by_model[m][-1]:.2f}%")


if __name__ == "__main__":
    main()
