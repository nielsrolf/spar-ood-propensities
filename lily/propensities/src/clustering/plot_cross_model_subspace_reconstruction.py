"""Cross-model subspace transfer (§4.5 robustness companion).

Derive principal components from one model's spillover matrix, then reconstruct
every model's matrix from the top-k of those PCs and plot the fraction of
variance captured vs k. The self-curve (a model reconstructed by its own PCs)
is the ceiling; a cross-curve that hugs it means that model's structure lives in
the source model's principal subspace — i.e. shared structure, with no factor
labeling / anchor matching required.

One panel per source model. Uses the uniform per-prompt (_pkl) basis.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from cross_model_efa_panel import MODELS, MODEL_LABEL          # noqa: E402
from factor_analysis_logitz import load_logitz_matrix, NORM_RES  # noqa: E402

OUT = (HERE / ".." / "activation_pca" / "figures" / "paper"
       / "fig_cross_model_subspace_reconstruction.png").resolve()
KMAX = 10
COLORS = {"llama_pkl": "#4C72B0", "qwen_pkl": "#DD8452", "nemotron_pkl": "#55A868"}
SHORT = {"llama_pkl": "Llama", "qwen_pkl": "Qwen", "nemotron_pkl": "Nemotron"}


def load_Z(model):
    """Standardized spillover matrix on the shared trait set."""
    _, metrics, M = load_logitz_matrix(NORM_RES / model, "plus")
    return metrics, StandardScaler().fit_transform(M)


def var_captured(Z_target, components):
    """Fraction of Z_target's total variance captured by projecting its rows
    onto the row space spanned by `components` (rows = unit PC directions)."""
    Q, _ = np.linalg.qr(components.T)          # orthonormal basis for the subspace
    proj = (Z_target @ Q) @ Q.T
    return (proj ** 2).sum() / (Z_target ** 2).sum()


def main():
    data = {}
    base_metrics = None
    for m in MODELS:
        metrics, Z = load_Z(m)
        data[m] = Z
        base_metrics = base_metrics or metrics
        assert metrics == base_metrics, f"trait order differs for {m}"
    n = data[MODELS[0]].shape[1]
    kmax = min(KMAX, n)

    # PCs (right singular vectors) of each source model
    pcs = {}
    for m in MODELS:
        _, _, Vt = np.linalg.svd(data[m] - data[m].mean(0), full_matrices=False)
        pcs[m] = Vt

    fig, axes = plt.subplots(1, len(MODELS), figsize=(13, 4.0), sharey=True)
    ks = np.arange(1, kmax + 1)
    for ax, src in zip(axes, MODELS):
        for tgt in MODELS:
            y = [var_captured(data[tgt], pcs[src][:k]) for k in ks]
            ax.plot(ks, y, marker="o", ms=3.5,
                    color=COLORS[tgt],
                    ls="-" if tgt == src else "--",
                    lw=2.0 if tgt == src else 1.4,
                    label=f"{SHORT[tgt]}" + (" (self)" if tgt == src else ""))
        ax.set_title(f"PCs from {SHORT[src]}", fontsize=10.5)
        ax.set_xlabel("# principal components", fontsize=9.5)
        ax.set_xticks([1, 3, 5, 7, 9])
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Fraction of target\nvariance captured", fontsize=10)

    # console summary: cross-vs-self gap at k=6
    k6 = min(6, kmax)
    print(f"\nVariance captured at k={k6} (rows=PC source, cols=target):")
    hdr = "src/tgt"
    print(f"  {hdr:<10}" + "".join(f"{SHORT[t]:>10}" for t in MODELS))
    for src in MODELS:
        row = f"  {SHORT[src]:<10}"
        for tgt in MODELS:
            row += f"{var_captured(data[tgt], pcs[src][:k6]):>10.2f}"
        print(row)

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
