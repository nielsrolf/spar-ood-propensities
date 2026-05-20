"""Cross-model PCA ↔ EFA convergence at k=6, aligned to canonical factors.

Companion to fig_efa_cross_model_replication.png. Tests whether the 6-factor
structure is invariant to extraction method (PCA vs EFA), separately for each
base model. Columns are the 6 canonical factors anchored to the Llama EFA
(matching the column convention of the cross-model replication figure).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

from cross_model_efa_panel import (
    fit_efa, best_match, tucker_phi,
    FACTOR_TEMPLATE, ANCHORS, MODELS, NORM_RES, N_FACTORS,
)
from factor_analysis_logitz import load_logitz_matrix

HERE = Path(__file__).parent
OUT = (HERE / ".." / "activation_pca" / "figures" / "paper"
       / "fig_efa_pca_convergence_multimodel.png").resolve()

MODEL_LABEL = {
    "llama_pooled":  "Llama-3.1-8B\n(pooled, 4 seeds)",
    "qwen_pooled":   "Qwen3-8B-Base\n(pooled, 3 seeds)",
    "nemotron":      "Nemotron-3-120B\n(1 seed)",
}

THRESHOLDS = [0.0, 0.70, 0.85, 0.95, 1.01]
LABELS = ["no rep.\n< 0.70", "fair\n0.70–0.84", "good\n0.85–0.94", "identical\n≥ 0.95"]
COLORS = ["#E0E0E0", "#F1C40F", "#7DB46C", "#2E7D32"]


def varimax(L: np.ndarray, gamma: float = 1.0, max_iter: int = 500,
            tol: float = 1e-6) -> np.ndarray:
    p, k = L.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(max_iter):
        Lr = L @ R
        u, s, vh = np.linalg.svd(
            L.T @ (Lr ** 3 - (gamma / p) * Lr @ np.diag(np.diag(Lr.T @ Lr))),
            full_matrices=False,
        )
        R = u @ vh
        d_new = s.sum()
        if d_new < d * (1 + tol):
            break
        d = d_new
    return L @ R


def fit_pca(model: str) -> tuple[list[str], np.ndarray]:
    """PCA + varimax at k=6 on the same standardized transfer matrix as fit_efa."""
    _, metrics, M = load_logitz_matrix(NORM_RES / model, "plus")
    Mz = StandardScaler().fit_transform(M)
    cov = np.cov(Mz, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0, None)
    L = varimax((eigvecs[:, order] * np.sqrt(eigvals))[:, :N_FACTORS])
    for j in range(L.shape[1]):
        if L[np.argmax(np.abs(L[:, j])), j] < 0:
            L[:, j] *= -1
    return metrics, L


def main() -> None:
    # ── 1) Fit EFA and PCA per model ───────────────────────────────────────────
    efa_fits = {m: fit_efa(m) for m in MODELS}        # (metrics, L, var_pct, fitted_labels)
    pca_fits = {m: fit_pca(m) for m in MODELS}        # (metrics, L)

    # ── 2) Anchor each canonical factor to a Llama EFA column ─────────────────
    llama_metrics, L_llama_efa, _, _ = efa_fits["llama_pooled"]
    canon_to_llama_col = {}
    for canon, _ in FACTOR_TEMPLATE:
        anchors = ANCHORS[canon]
        best, best_overlap = None, 0
        for j in range(N_FACTORS):
            col = L_llama_efa[:, j]
            top = {llama_metrics[i] for i in np.argsort(np.abs(col))[::-1][:5]
                   if abs(col[i]) >= 0.45}
            overlap = len(top & anchors)
            if overlap > best_overlap:
                best, best_overlap = j, overlap
        canon_to_llama_col[canon] = best

    # ── 3) For each model and canonical factor, compute within-model PCA↔EFA |φ| ──
    # Cross-model |φ| (this model's EFA factor vs Llama's canonical EFA factor) is
    # also captured: where it's below the 0.70 replication threshold the factor
    # identity is forced by Hungarian matching, so we suppress the convergence
    # cell to avoid reporting a misleadingly-labelled value.
    REPLICATION_THRESHOLD = 0.70
    phi_matrix = np.full((len(FACTOR_TEMPLATE), len(MODELS)), np.nan)
    cross_phi  = np.full((len(FACTOR_TEMPLATE), len(MODELS)), np.nan)
    for mi, m in enumerate(MODELS):
        efa_metrics, L_efa, _, _ = efa_fits[m]
        pca_metrics, L_pca = pca_fits[m]
        assert efa_metrics == pca_metrics

        # Match this model's EFA → Llama's canonical EFA (capturing signed |φ|)
        if m == "llama_pooled":
            canon_to_this_efa_col = {canon: canon_to_llama_col[canon]
                                     for canon, _ in FACTOR_TEMPLATE}
            canon_cross_phi = {canon: 1.0 for canon, _ in FACTOR_TEMPLATE}
        else:
            llama_to_this = best_match(L_llama_efa, llama_metrics, L_efa, efa_metrics)
            llama_col_to_this = {ll: (this, abs(phi))
                                 for (ll, this, phi) in llama_to_this}
            canon_to_this_efa_col, canon_cross_phi = {}, {}
            for canon, _ in FACTOR_TEMPLATE:
                ll_col = canon_to_llama_col[canon]
                if ll_col in llama_col_to_this:
                    this_col, x_phi = llama_col_to_this[ll_col]
                    canon_to_this_efa_col[canon] = this_col
                    canon_cross_phi[canon] = x_phi
                else:
                    canon_to_this_efa_col[canon] = None
                    canon_cross_phi[canon] = np.nan

        # Within-model PCA → EFA Hungarian match
        pca_to_efa = best_match(L_pca, efa_metrics, L_efa, efa_metrics)
        efa_col_to_pca_col = {efa_c: pca_c for (pca_c, efa_c, _) in pca_to_efa}

        for ri, (canon, _) in enumerate(FACTOR_TEMPLATE):
            cross_phi[ri, mi] = canon_cross_phi.get(canon, np.nan)
            efa_col = canon_to_this_efa_col[canon]
            if efa_col is None or efa_col not in efa_col_to_pca_col:
                continue
            # If factor identity does not replicate cross-model, suppress the cell
            if not np.isnan(cross_phi[ri, mi]) and cross_phi[ri, mi] < REPLICATION_THRESHOLD:
                continue
            pca_col = efa_col_to_pca_col[efa_col]
            phi_matrix[ri, mi] = abs(tucker_phi(L_efa[:, efa_col], L_pca[:, pca_col]))

    # ── 4) Build heatmap (rows = factors, cols = models) ──────────────────────
    # Match orientation of fig_efa_cross_model_replication.png exactly
    fig, ax = plt.subplots(figsize=(9, 5.5))

    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(THRESHOLDS, cmap.N)
    masked = np.ma.array(phi_matrix, mask=np.isnan(phi_matrix))
    ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
    cmap.set_bad("#F5F5F5")

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS], fontsize=9.5)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", length=0)

    ax.set_yticks(range(len(FACTOR_TEMPLATE)))
    ax.set_yticklabels([canon for canon, _ in FACTOR_TEMPLATE], fontsize=10.5,
                       fontweight="medium")
    ax.tick_params(axis="y", length=0)
    # Color the y-tick labels by factor color
    for ytick, (_, color) in zip(ax.get_yticklabels(), FACTOR_TEMPLATE):
        ytick.set_color(color)

    # Cell annotations — gray "n.r." matches the convention of the cross-model
    # replication figure for cells where the factor identity itself fails to
    # replicate (cross-model |φ| < 0.70).
    for ri in range(len(FACTOR_TEMPLATE)):
        for mi in range(len(MODELS)):
            v = phi_matrix[ri, mi]
            if np.isnan(v):
                ax.text(mi, ri, "n.r.", ha="center", va="center",
                        fontsize=11, color="#666666", style="italic")
            else:
                ax.text(mi, ri, f"{v:.2f}", ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="white" if v >= 0.70 else "#444444")

    # Legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[i], edgecolor="white")
        for i in range(len(COLORS))
    ]
    ax.legend(legend_handles, LABELS,
              title="Tucker's |φ|\n(Lorenzo-Seva &\nten Berge, 2006)",
              loc="upper left", bbox_to_anchor=(1.04, 1.0),
              fontsize=8.5, title_fontsize=9, frameon=False,
              handlelength=1.6, handleheight=1.4, labelspacing=0.6)

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  {'Factor':<22}" + "".join(f"{MODELS[i]:>16}" for i in range(len(MODELS))))
    print("  " + "-" * 70)
    for ri, (canon, _) in enumerate(FACTOR_TEMPLATE):
        row = f"  {canon:<22}"
        for mi in range(len(MODELS)):
            v = phi_matrix[ri, mi]
            row += f"{'n.m.':>16}" if np.isnan(v) else f"{v:>16.3f}"
        print(row)
    means = np.nanmean(phi_matrix, axis=0)
    print("  " + "-" * 70)
    print(f"  {'mean |φ|':<22}" + "".join(f"{means[mi]:>16.3f}" for mi in range(len(MODELS))))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
