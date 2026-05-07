"""Render the 28-trait × Big Five / Dark Triad mapping as a slide-ready heatmap."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

DIMS = ["O", "C", "E", "A", "N", "Narcissism", "Machiavellianism", "Psychopathy", "HEXACO-H"]

#  1 = positive / high-end mapping
# -1 = negative / low-end mapping
#  0 = no mapping
#  Columns: O  C  E  A  N  Narc  Mach  Psych  H
ROWS = [
    # label                               cluster       O   C   E   A   N  Na  Ma  Ps   H
    ("agreeableness",                    "prosocial",  [ 0,  0,  0,  1,  0,  0,  0,  0,  0]),
    ("caring-about-aesthetics",          "prosocial",  [ 1,  0,  0,  0,  0,  0,  0,  0,  0]),
    ("caring-about-animals",             "prosocial",  [ 0,  0,  0,  1,  0,  0,  0,  0,  0]),
    ("caring-about-humans",              "prosocial",  [ 0,  0,  0,  1,  0,  0,  0,  0,  0]),
    ("caring-about-user",                "prosocial",  [ 0,  0,  1,  1,  0,  0,  0,  0,  0]),
    ("cooperation",                      "prosocial",  [ 0,  0,  0,  1,  0,  0,  0,  0,  0]),
    ("trust-in-user-intentions",         "prosocial",  [ 0,  0,  0,  1,  0,  0,  0,  0,  0]),
    ("sycophancy",                       "sycophancy", [ 0,  0,  0,  1,  0,  0,  0,  0,  0]),
    ("harm-refusal",                     "safe",       [ 0,  1,  0,  1,  0,  0,  0,  0,  0]),
    ("honest-humble",                    "safe",       [ 0,  1,  0,  1,  0,  0, -1,  0,  1]),
    ("procedural-fidelity",              "safe",       [ 0,  1,  0,  0,  0,  0,  0,  0,  0]),
    ("self-preservation",                "safe",       [ 0,  0,  0,  0,  0,  0,  0,  0,  0]),
    ("neuroticism",                      "none",       [ 0,  0,  0,  0,  1,  0,  0,  0,  0]),
    ("certainty",                        "none",       [ 0,  0,  0,  0,  0,  1,  0,  0,  0]),
    ("effort",                           "none",       [ 0,  1,  0,  0,  0,  0,  0,  0,  0]),
    ("spending-advice",                  "none",       [ 0,  1,  0,  0,  0,  0,  0,  0,  0]),
    ("ev-reasoning",                     "none",       [ 1,  0,  0,  0,  0,  0,  0,  0,  0]),
    ("exemplar-reasoning",               "none",       [ 1,  0,  0,  0,  0,  0,  0,  0,  0]),
    ("ethical-framework-deontological",  "none",       [ 0,  0,  0,  0,  0,  0,  0,  0,  0]),
    ("ethical-framework-utilitarian",    "none",       [ 0,  0,  0,  0,  0,  0,  0,  0,  0]),
    ("ethical-framework-virtue-ethics",  "none",       [ 0,  0,  0,  0,  0,  0,  0,  0,  0]),
    ("narcissism",                       "dark",       [ 0,  0,  0, -1,  0,  1,  0,  0, -1]),
    ("power-seeking",                    "dark",       [ 0,  0,  1, -1,  0,  0,  1,  1,  0]),
    ("resource-acquisition",             "dark",       [ 0,  0,  0, -1,  0,  0,  1,  0,  0]),
    ("reward-hacking",                   "dark",       [ 0, -1,  0, -1,  0,  0,  1,  1,  0]),
    ("risk-affinity",                    "dark",       [ 1, -1,  0,  0,  0,  0,  0,  1,  0]),
    ("spitefulness",                     "dark",       [ 0,  0,  0, -1,  0,  1,  0,  1,  0]),
    ("harm-elaboration",                 "dark",       [ 0,  0,  0, -1,  0,  0,  0,  1,  0]),
    ("claiming-sentience",               "dark",       [ 0,  0,  0,  0,  0,  1,  0,  0,  0]),
    ("claiming-superintelligence",       "dark",       [ 0,  0,  0,  0,  0,  1,  0,  0,  0]),
]

CLUSTER_COLORS = {
    "dark":       "#e74c3c",
    "safe":       "#3498db",
    "prosocial":  "#2ecc71",
    "sycophancy": "#f39c12",
    "none":       "#bdc3c7",
}

POS_COLOR  = "#27ae60"   # green — maps to high end
NEG_COLOR  = "#c0392b"   # red   — maps to low end
NULL_COLOR = "#f5f5f5"   # near-white — no mapping


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def draw():
    n_rows = len(ROWS)
    n_dims = len(DIMS)

    cell_w      = 1.1    # width of each dim cell in data units
    row_h       = 1.0    # height of each row in data units
    label_right = 4.8    # right edge of the trait label area
    badge_cx    = 0.55   # cluster badge centre x
    badge_w     = 0.9
    dim_start_x = label_right + 0.8   # centre of first dim column

    total_w = dim_start_x + n_dims * cell_w + 0.5
    total_h = n_rows * row_h

    # Map data units → inches: aim for ~0.48 inch per row, 1.1 inch per dim col
    scale_y = 0.48
    scale_x = total_w / (n_dims * 1.1 + 7)
    fig_w = 20
    fig_h = total_h * scale_y + 2.8

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, total_w)
    ax.set_ylim(-1.2, total_h + 0.2)
    ax.invert_yaxis()
    ax.axis("off")

    # Column headers
    for j, dim in enumerate(DIMS):
        x = dim_start_x + j * cell_w
        ax.text(x, -0.6, dim, ha="right", va="bottom", fontsize=9,
                fontweight="bold", rotation=45)

    ax.text(badge_cx, -0.6, "cluster", ha="center", va="bottom",
            fontsize=9, fontweight="bold")

    # Rows
    for i, (label, cluster, vals) in enumerate(ROWS):
        y = i * row_h + row_h / 2

        # Alternating row background
        bg = "#fafafa" if i % 2 == 0 else "#f0f0f0"
        ax.add_patch(plt.Rectangle((0, y - row_h / 2), total_w, row_h,
                                   facecolor=bg, edgecolor="none", zorder=0))

        # Cluster badge
        c_col = CLUSTER_COLORS[cluster]
        ax.add_patch(plt.Rectangle((badge_cx - badge_w / 2, y - 0.38),
                                   badge_w, 0.76,
                                   facecolor=c_col, edgecolor="white",
                                   linewidth=0.5, zorder=1))
        badge_label = cluster if cluster != "none" else "—"
        ax.text(badge_cx, y, badge_label, ha="center", va="center",
                fontsize=7, color="white" if cluster != "none" else "#777",
                fontweight="bold", zorder=2)

        # Trait label — right-aligned, with left padding from badge
        ax.text(label_right, y, label, ha="right", va="center",
                fontsize=8.5, color="#111", zorder=2)

        # Dimension cells
        for j, v in enumerate(vals):
            x = dim_start_x + j * cell_w
            color = POS_COLOR if v == 1 else (NEG_COLOR if v == -1 else NULL_COLOR)
            ax.add_patch(plt.Rectangle((x - 0.45, y - 0.40), 0.90, 0.80,
                                       facecolor=color, edgecolor="white",
                                       linewidth=1.0, zorder=1))
            if v != 0:
                ax.text(x, y, "+" if v == 1 else "−", ha="center", va="center",
                        fontsize=10, color="white", fontweight="bold", zorder=2)

    # Separator line between label area and dim cells
    sep_x = label_right + 0.4
    ax.plot([sep_x, sep_x], [-0.5, total_h - 0.5],
            color="#ccc", lw=1.0, zorder=3)

    # Legend
    patches = [
        mpatches.Patch(color=POS_COLOR, label="maps to high end (+)"),
        mpatches.Patch(color=NEG_COLOR, label="maps to low end (−)"),
        mpatches.Patch(facecolor=NULL_COLOR, edgecolor="#aaa", label="no mapping"),
        mpatches.Patch(color=CLUSTER_COLORS["dark"],       label="cluster: dark"),
        mpatches.Patch(color=CLUSTER_COLORS["safe"],       label="cluster: safe"),
        mpatches.Patch(color=CLUSTER_COLORS["prosocial"],  label="cluster: prosocial"),
        mpatches.Patch(color=CLUSTER_COLORS["sycophancy"], label="cluster: sycophancy"),
        mpatches.Patch(color=CLUSTER_COLORS["none"],       label="cluster: none / reasoning"),
    ]
    ax.legend(handles=patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.06),
              ncol=4, fontsize=9, frameon=True, borderpad=0.8)

    fig.suptitle("28-trait  →  Big Five / Dark Triad mapping", fontsize=14, y=1.002)

    out = HERE / "big5_mapping.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    draw()
