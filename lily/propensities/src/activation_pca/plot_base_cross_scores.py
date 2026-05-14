"""Visualize base model cross-scoring results.

Two-panel figure:
  Left:  Correlation matrix heatmap across all 8 traits (base model responses)
  Right: Bar chart of valid pairwise correlations (dark-dark vs controls)

Usage:
    python plot_base_cross_scores.py
    python plot_base_cross_scores.py --results results/base_cross_scores_v2.json
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = Path(__file__).parent

# Display order: core dark cluster first, spitefulness + RewHack after, controls last
TRAIT_ORDER = [
    "power-seeking",
    "resource-acquisition",
    "narcissism",
    "risk-affinity",
    "spitefulness",
    "reward-hacking",
    "caring-about-animals",
    "honest-humble",
]

SHORT = {
    "power-seeking":        "PS",
    "resource-acquisition": "ResAcq",
    "reward-hacking":       "RewHack",
    "narcissism":           "Narc",
    "spitefulness":         "Spite",
    "risk-affinity":        "Risk",
    "caring-about-animals": "Animals",
    "honest-humble":        "HonHum",
}

DARK = {"power-seeking", "resource-acquisition", "reward-hacking",
        "narcissism", "spitefulness", "risk-affinity"}


def pairwise_r(scores: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (r_matrix, n_matrix) over TRAIT_ORDER. NaN where n < 5."""
    k = len(TRAIT_ORDER)
    R = np.full((k, k), np.nan)
    N = np.zeros((k, k), dtype=int)
    np.fill_diagonal(R, 1.0)
    for i, t1 in enumerate(TRAIT_ORDER):
        for j, t2 in enumerate(TRAIT_ORDER):
            if i >= j:
                continue
            s1 = scores.get(t1, [])
            s2 = scores.get(t2, [])
            paired = [(a, b) for a, b in zip(s1, s2)
                      if a is not None and b is not None]
            N[i, j] = N[j, i] = len(paired)
            if len(paired) >= 5:
                v1, v2 = zip(*paired)
                r = float(np.corrcoef(v1, v2)[0, 1])
                R[i, j] = R[j, i] = r
    return R, N


def bar_data(scores: dict) -> list[tuple]:
    """(label, r, n, kind) for all valid pairs, sorted by r descending."""
    rows = []
    for t1, t2 in combinations(TRAIT_ORDER, 2):
        s1 = scores.get(t1, [])
        s2 = scores.get(t2, [])
        paired = [(a, b) for a, b in zip(s1, s2)
                  if a is not None and b is not None]
        if len(paired) < 5:
            continue
        v1, v2 = zip(*paired)
        r = float(np.corrcoef(v1, v2)[0, 1])
        kind = "dark–dark" if t1 in DARK and t2 in DARK else "dark–control"
        label = f"{SHORT[t1]} ↔ {SHORT[t2]}"
        rows.append((label, r, len(paired), kind))
    return sorted(rows, key=lambda x: x[1], reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                    default=HERE / "results" / "base_cross_scores_v2.json")
    ap.add_argument("--out", type=Path,
                    default=HERE / "figures" / "base_cross_scores_v2.png")
    args = ap.parse_args()

    with open(args.results) as f:
        data = json.load(f)
    scores = data["scores"]

    R, N = pairwise_r(scores)
    bars = bar_data(scores)
    labels = [SHORT[t] for t in TRAIT_ORDER]
    k = len(TRAIT_ORDER)

    fig, (ax_heat, ax_bar) = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── Left: correlation heatmap ──────────────────────────────────────────
    # mask NaN as gray
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="#cccccc")
    masked = np.ma.masked_invalid(R)
    im = ax_heat.imshow(masked, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    ax_heat.set_xticks(range(k))
    ax_heat.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax_heat.set_yticks(range(k))
    ax_heat.set_yticklabels(labels, fontsize=9)

    for i in range(k):
        for j in range(k):
            if np.isnan(R[i, j]):
                txt = "n/a"
                color = "#666666"
                fs = 7
            else:
                txt = f"{R[i,j]:+.2f}"
                color = "white" if abs(R[i, j]) > 0.55 else "black"
                fs = 8
            ax_heat.text(j, i, txt, ha="center", va="center",
                         fontsize=fs, color=color)

    plt.colorbar(im, ax=ax_heat, label="Pearson r", shrink=0.8)
    ax_heat.set_title(
        "Cross-trait correlations — base Llama-3.1-8B\n"
        "(no fine-tuning; scored by GPT-4.1-mini judges)",
        fontsize=10
    )

    # Box around core dark sub-cluster (first 4×4: PS, ResAcq, Narc, Risk)
    core_n = 4
    rect = plt.Rectangle((-0.5, -0.5), core_n, core_n,
                         fill=False, edgecolor="black", linewidth=2.5,
                         linestyle="--")
    ax_heat.add_patch(rect)
    ax_heat.text(1.5, core_n - 0.35,
                 "IC core cluster",
                 ha="center", va="top", fontsize=8,
                 fontweight="bold", color="black")

    # ── Right: bar chart ──────────────────────────────────────────────────
    blabels = [b[0] for b in bars]
    rs      = [b[1] for b in bars]
    ns      = [b[2] for b in bars]
    kinds   = [b[3] for b in bars]

    color_map = {"dark–dark": "#c0392b", "dark–control": "#2980b9"}
    colors = [color_map[kk] for kk in kinds]
    y_pos  = list(range(len(blabels)))

    bar_objs = ax_bar.barh(y_pos, rs, color=colors, alpha=0.82, height=0.6)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(
        [f"{l}  (n={n})" for l, n in zip(blabels, ns)], fontsize=8.5
    )
    ax_bar.axvline(0, color="black", linewidth=0.8)
    ax_bar.set_xlabel("Pearson r", fontsize=11)
    ax_bar.set_title("Pairwise correlations sorted by r", fontsize=10)
    ax_bar.set_xlim(-0.85, 1.1)

    for bar, r in zip(bar_objs, rs):
        x = bar.get_width()
        ax_bar.text(x + 0.02 if x >= 0 else x - 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{r:+.2f}", va="center",
                    ha="left" if x >= 0 else "right",
                    fontsize=8, fontweight="bold")

    legend_handles = [
        mpatches.Patch(color="#c0392b", alpha=0.82, label="Dark–dark pair"),
        mpatches.Patch(color="#2980b9", alpha=0.82, label="Dark–control pair"),
    ]
    ax_bar.legend(handles=legend_handles, fontsize=8.5, loc="lower right")
    ax_bar.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")
    plt.close()


if __name__ == "__main__":
    main()
