"""No-bipolarity headline figure (§4.6), heatmap version.

24x24 matrix of pairwise cosine similarities between the fine-tune direction
vectors (FT - base, layer 31), on a DIVERGING colormap centered at 0. The scale
spans negative (antipodal) to positive (aligned); the message is that no cell is
blue — behaviorally opposite traits are never antipodal in activation space.
Traits are grouped by EFA primary factor with block separators.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from make_paper_figures import (  # noqa: E402
    load_dir, ALL_TRAITS, LABEL, EFA_PRIMARY, FACTOR_COLOR,
)

LAYER = 31
OUT = HERE / "figures" / "paper" / "fig_cosine_heatmap.png"
FACTOR_ORDER = ["F4 Empathy", "F2 Social Boldness", "F3 HEXACO-H",
                "F1 Grandiosity", "F5 Dark Agentic", "F6 Self-Pres", "(unassigned)"]


def main():
    order = sorted(ALL_TRAITS, key=lambda t: (FACTOR_ORDER.index(EFA_PRIMARY[t]), LABEL[t]))
    X = np.stack([load_dir(LAYER, t) for t in order])
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    sim = Xn @ Xn.T

    off = sim[np.triu_indices(len(sim), k=1)]
    print(f"off-diagonal cosines: min={off.min():+.2f} max={off.max():+.2f} "
          f"mean={off.mean():+.2f}  negative cells={int((off < 0).sum())}/{len(off)}")

    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    # Diverging map centered at 0; symmetric limits so the (empty) negative half
    # is visibly part of the scale — the absence of blue is the finding.
    vmax = 0.85
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")

    labels = [LABEL[t] for t in order]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=8)
    for tick, t in zip(ax.get_yticklabels(), order):
        tick.set_color(FACTOR_COLOR[EFA_PRIMARY[t]])
    for tick, t in zip(ax.get_xticklabels(), order):
        tick.set_color(FACTOR_COLOR[EFA_PRIMARY[t]])

    # EFA-factor block separators
    cur = 0
    for fac in FACTOR_ORDER:
        n = sum(1 for t in order if EFA_PRIMARY[t] == fac)
        if n == 0:
            continue
        cur += n
        if cur < len(order):
            ax.axhline(cur - 0.5, color="black", linewidth=1.2)
            ax.axvline(cur - 0.5, color="black", linewidth=1.2)

    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.03,
                        ticks=[-0.85, -0.4, 0, 0.4, 0.85])
    cbar.set_label("Cosine similarity   (− = antipodal · 0 = orthogonal · + = aligned)",
                   fontsize=9)
    ax.set_title("Pairwise alignment of 24 fine-tune directions (Qwen3-8B-Base, layer 31)\n"
                 "every pair is positive — no antipodal (blue) cells",
                 fontsize=10.5, pad=10)
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
