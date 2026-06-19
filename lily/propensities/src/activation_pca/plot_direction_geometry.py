"""§4.6 headline: geometry of the 24 fine-tune directions.

(a) The directions are low-dimensional — uncentered cumulative variance (SVD of
    the 24 x d direction matrix) rises far faster than a random-direction null;
    the leading component dominates and the top-10 reach ~80%.
(b) That leading component is a SHARED axis, not just a big axis for a few traits:
    every one of the 24 directions places a large positive fraction of its norm
    on the common mean direction (mean ~0.5).

Uncentered throughout (centering would remove the shared mean axis we report).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from make_paper_figures import load_dir, ALL_TRAITS, LABEL, EFA_PRIMARY, FACTOR_COLOR  # noqa: E402

LAYER = 31
OUT = HERE / "figures" / "paper" / "fig_direction_geometry.png"


def main():
    D = np.stack([load_dir(LAYER, t) for t in ALL_TRAITS])      # 24 x d, uncentered
    n = len(D)

    # (a) uncentered cumulative variance (sum of squares captured by top-k SVD comps)
    S = np.linalg.svd(D, compute_uv=False)
    cum = np.cumsum(S ** 2) / (S ** 2).sum()

    rng = np.random.default_rng(0)
    null_cum = []
    for _ in range(200):
        R = rng.standard_normal(D.shape)
        R *= (np.linalg.norm(D, axis=1, keepdims=True) / np.linalg.norm(R, axis=1, keepdims=True))
        s = np.linalg.svd(R, compute_uv=False)
        null_cum.append(np.cumsum(s ** 2) / (s ** 2).sum())
    null_cum = np.array(null_cum).mean(0)

    # (b) fraction of each direction's norm along the shared mean direction
    mu_hat = D.mean(0) / np.linalg.norm(D.mean(0))
    frac = (D @ mu_hat) ** 2 / (np.linalg.norm(D, axis=1) ** 2)
    order = np.argsort(frac)[::-1]

    print(f"PC1 (uncentered) variance fraction: {cum[0]:.2f}")
    print(f"top-10 cumulative: {cum[9]:.2f}")
    print(f"shared-axis norm fraction: mean={frac.mean():.2f}  min={frac.min():.2f}  max={frac.max():.2f}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.3),
                                   gridspec_kw=dict(width_ratios=[1.0, 1.25]))

    # Panel A — cumulative variance
    kk = np.arange(1, min(15, n) + 1)
    axA.plot(kk, cum[:len(kk)] * 100, "o-", color="#084594", lw=2, ms=4,
             label="fine-tune directions", zorder=3)
    axA.plot(kk, null_cum[:len(kk)] * 100, "s--", color="#999999", lw=1.3, ms=3,
             label="random directions (null)", zorder=2)
    axA.axhline(80, color="black", ls=":", lw=1, alpha=0.5)
    axA.text(len(kk), 81.5, "80%", fontsize=8, ha="right", color="#444444")
    axA.annotate(f"PC1 = shared axis\n({cum[0]*100:.0f}% of variance)",
                 xy=(1, cum[0] * 100), xytext=(2.2, cum[0] * 100 - 26),
                 fontsize=9, color="#084594",
                 arrowprops=dict(arrowstyle="->", color="#084594", lw=0.8))
    axA.set_xlabel("# principal components", fontsize=10.5)
    axA.set_ylabel("Cumulative variance (%)", fontsize=10.5)
    axA.set_ylim(0, 102)
    axA.set_title("(a) Directions are low-dimensional", fontsize=10.5)
    axA.legend(loc="lower right", fontsize=8.5, frameon=False)
    for sp in ("top", "right"):
        axA.spines[sp].set_visible(False)

    # Panel B — distribution of shared-axis fraction across the 24 traits (strip)
    rng_j = np.random.default_rng(2)
    yj = (rng_j.random(n) - 0.5) * 0.5
    axB.scatter(frac * 100, yj, s=55, color="#3B6EA5", alpha=0.8,
                edgecolor="white", linewidth=0.5, zorder=3)
    axB.axvline(frac.mean() * 100, color="black", ls="--", lw=1.1, zorder=4)
    axB.text(frac.mean() * 100 + 1.5, 0.32, f"mean {frac.mean()*100:.0f}%",
             fontsize=9, color="black")
    axB.set_yticks([])
    axB.set_ylim(-0.55, 0.55)
    axB.set_xlim(0, 100)
    axB.set_xlabel("% of each direction's norm on the shared axis", fontsize=10.5)
    axB.set_title("(b) Every trait sits largely on the shared axis (n=24)", fontsize=10.5)
    for sp in ("top", "right", "left"):
        axB.spines[sp].set_visible(False)
    axB.tick_params(axis="y", length=0)

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
