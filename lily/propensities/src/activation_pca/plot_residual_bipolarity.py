"""Residual-control headline (§4.6): no bipolarity survives shared-axis removal.

The raw "all cosines positive" result is partly anisotropy — ~52% of each
fine-tune direction's norm lies on a shared mean axis. After projecting that axis
out, the 276 pairwise cosines re-center around 0 (≈half negative, as expected).
The real test: do the EFA-predicted *opposite* trait pairs fall in the negative
tail (= bipolar) or sit mid-distribution (= orthogonal, not antipodal)? They do
the latter (permutation p ≈ 0.82). This figure shows that directly.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str((HERE.parent / "clustering").resolve()))
from make_paper_figures import load_dir, ALL_TRAITS, LABEL  # noqa: E402
from cross_model_efa_panel import fit_efa                   # noqa: E402

LAYER = 31
LOAD_THRESH = 0.45          # |loading| considered a meaningful pole on a factor
OUT = HERE / "figures" / "paper" / "fig_residual_bipolarity.png"


def act_to_behav(name: str) -> str:
    stem = name[:-len("_qwen_v1")] if name.endswith("_qwen_v1") else name
    return stem.replace("_", "-")


def efa_opposite_pairs():
    """Behaviorally-opposite pairs derived from the EFA: trait pairs that load
    with OPPOSITE signs (both |loading| >= LOAD_THRESH) on the same factor.
    These are the pole-pairs of each bipolar factor — antipodal if behavior is
    geometrically bipolar. Mapped to activation-direction names; intersected
    with the 24 traits that have activation directions."""
    metrics, L, *_ = fit_efa("qwen_pkl")          # 22 behavioral traits x 6 factors
    behav2act = {act_to_behav(t): t for t in ALL_TRAITS}
    pairs = set()
    for j in range(L.shape[1]):
        pos = [metrics[i] for i in range(len(metrics)) if L[i, j] >= LOAD_THRESH]
        neg = [metrics[i] for i in range(len(metrics)) if L[i, j] <= -LOAD_THRESH]
        for a in pos:
            for b in neg:
                if a in behav2act and b in behav2act:
                    pa, pb = behav2act[a], behav2act[b]
                    pairs.add(tuple(sorted((pa, pb))))
    return sorted(pairs)


def main():
    OPP_PAIRS = efa_opposite_pairs()
    idx = {t: i for i, t in enumerate(ALL_TRAITS)}
    X = np.stack([load_dir(LAYER, t) for t in ALL_TRAITS])
    # Project out the shared mean DIRECTION (matches residual_cosine_control.py:
    # remove the 1-D span of mu from each direction, not mean-subtraction).
    mu = X.mean(axis=0)
    Q, _ = np.linalg.qr(mu[:, None])
    X = X - (X @ Q) @ Q.T
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    sim = Xn @ Xn.T
    iu = np.triu_indices(len(X), k=1)
    allpairs = sim[iu]
    opp = np.array([sim[idx[a], idx[b]] for a, b in OPP_PAIRS])

    # Permutation test: are the opposite pairs more negative than random pairs?
    rng_p = np.random.default_rng(0)
    null_means = np.array([rng_p.choice(allpairs, size=len(opp), replace=False).mean()
                           for _ in range(200000)])
    p_val = float((null_means <= opp.mean()).mean())   # frac of random sets more negative
    print(f"all pairs (shared-axis removed): mean={allpairs.mean():+.3f}  %neg={100*(allpairs<0).mean():.0f}%")
    print(f"EFA-opposite pairs (n={len(opp)}):       mean={opp.mean():+.3f}  "
          f"%neg={100*(opp<0).mean():.0f}%  perm p={p_val:.2f}")

    # Two-row dot strip: top = all pairs, bottom = the EFA-opposite pairs, on a
    # shared cosine axis. The eye reads instantly whether the red (opposite)
    # dots fall in the negative/antipodal zone or sit among the bulk.
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(9, 3.4))

    # shaded antipodal (negative) zone
    ax.axvspan(allpairs.min() - 0.05, 0, color="#2166AC", alpha=0.07, zorder=0)
    ax.axvline(0, color="black", lw=1.0, ls="--", zorder=2)

    # row 1: all 276 pairs (jittered grey dots)
    y_all = 1 + (rng.random(len(allpairs)) - 0.5) * 0.5
    ax.scatter(allpairs, y_all, s=14, color="#b8bdc2", alpha=0.7,
               edgecolor="none", zorder=3)
    # row 2: the EFA-opposite pairs (red), jittered for overlap
    y_opp = 0.0 + (rng.random(len(opp)) - 0.5) * 0.35
    ax.scatter(opp, y_opp, s=70, color="#C0392B", edgecolor="black",
               linewidth=0.5, alpha=0.85, zorder=4)

    ax.set_yticks([1, 0])
    ax.set_yticklabels([f"all {len(allpairs)}\ntrait pairs",
                        f"EFA-opposite\npairs (n={len(opp)})"], fontsize=9)
    ax.set_ylim(-0.7, 1.6)
    ax.set_xlabel("Pairwise cosine after removing the shared mean axis", fontsize=10.5)

    ax.text(allpairs.min() - 0.02, 1.5, "← antipodal (bipolar prediction)",
            fontsize=8.5, color="#2166AC", ha="left", va="center", style="italic")
    ax.text(opp.mean(), -0.62,
            f"opposite pairs span the bulk, not the antipodal zone  (perm $p = {p_val:.2f}$)",
            fontsize=9, color="#C0392B", ha="center", va="center")

    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
