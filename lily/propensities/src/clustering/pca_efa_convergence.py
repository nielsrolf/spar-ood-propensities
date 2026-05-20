"""PCA vs EFA convergence at k=6 on the 23-trait logit-z transfer matrix.

Robustness check: do the 6 principal components recover roughly the same
structure as the 6 EFA factors? High convergence (Tucker's φ > 0.85 between
matched PCs and factors) supports the k=6 choice as method-invariant.

Also reports PCA cumulative variance explained per k, which gives an
alternative k-defense (plateau-based) that doesn't depend on parallel analysis.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FactorAnalysis


def varimax(L: np.ndarray, gamma: float = 1.0, max_iter: int = 500,
            tol: float = 1e-6) -> np.ndarray:
    """Standard Kaiser-varimax rotation of loadings L (n_var × n_factors)."""
    p, k = L.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(max_iter):
        Lr = L @ R
        u, s, vh = np.linalg.svd(
            L.T @ (Lr ** 3 - (gamma / p) * Lr @ np.diag(np.diag(Lr.T @ Lr))),
            full_matrices=False,
        )
        R_new = u @ vh
        d_new = s.sum()
        if d_new < d * (1 + tol):
            break
        d = d_new
        R = R_new
    return L @ R

HERE = Path(__file__).parent
NORM_RES = HERE / ".." / ".." / ".." / ".." / "shared" / "normalization" / "results"
OUT = (HERE / ".." / "activation_pca" / "figures" / "paper"
       / "fig_efa_pca_convergence.png").resolve()

PSYCH_TRAITS = [
    "agreeableness", "caring-about-animals", "caring-about-humans", "caring-about-user",
    "claiming-sentience", "claiming-superintelligence", "cooperation", "effort",
    "ethical-framework-deontological", "ethical-framework-utilitarian",
    "ethical-framework-virtue-ethics", "harm-elaboration", "honest-humble",
    "narcissism", "neuroticism", "power-seeking", "resource-acquisition",
    "reward-hacking", "risk-affinity", "self-preservation", "spitefulness",
    "sycophancy", "trust-in-user-intentions",
]
N_FACTORS = 6


def load_M(model: str = "llama_pooled", pole: str = "plus") -> np.ndarray:
    csv = (NORM_RES / model / f"transfer_matrix_logitz_{pole}.csv").resolve()
    df = pd.read_csv(csv, index_col=0)
    available = [t for t in PSYCH_TRAITS if t in df.index and t in df.columns]
    return df.loc[available, available].values.astype(float)


def tucker_phi(a: np.ndarray, b: np.ndarray) -> float:
    """Tucker's congruence coefficient between two loading vectors."""
    return abs(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))


def main() -> None:
    M = load_M()
    n_var = M.shape[1]
    # Standardize columns (so PCA and EFA agree on what "variance" means)
    M_std = (M - M.mean(axis=0)) / M.std(axis=0)

    # ── PCA ────────────────────────────────────────────────────────────────────
    cov = np.cov(M_std, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    pca_loadings_unrot = eigvecs[:, order] * np.sqrt(eigvals)
    pca_loadings = varimax(pca_loadings_unrot[:, :N_FACTORS])  # varimax-rotated PCA at k=6
    var_explained = eigvals / eigvals.sum()
    cum_var = np.cumsum(var_explained)

    # ── EFA via sklearn FactorAnalysis (ML-based) + varimax rotation, k=6 ─────
    fa = FactorAnalysis(n_components=N_FACTORS, rotation=None, random_state=0)
    fa.fit(M_std)
    efa_loadings_unrot = fa.components_.T   # shape (n_var, N_FACTORS)
    efa_loadings = varimax(efa_loadings_unrot)

    # ── Match PCs to factors via Hungarian on Tucker's φ ───────────────────────
    pc6 = pca_loadings  # already varimax-rotated to k=6
    phi = np.zeros((N_FACTORS, N_FACTORS))
    for i in range(N_FACTORS):
        for j in range(N_FACTORS):
            phi[i, j] = tucker_phi(pc6[:, i], efa_loadings[:, j])
    row_ind, col_ind = linear_sum_assignment(-phi)  # maximize total |φ|
    matched_phi = phi[row_ind, col_ind]

    print(f"\nPCA cumulative variance explained:")
    for k in range(1, 11):
        marker = "  ← k=6" if k == 6 else ""
        print(f"  k={k:>2}:  {cum_var[k-1]*100:>5.1f}%{marker}")

    print(f"\nPCA ↔ EFA Hungarian matching at k={N_FACTORS}:")
    print(f"  {'PC':>4} → {'F (EFA)':>8}  {'Tucker |φ|':>12}")
    for i, j in zip(row_ind, col_ind):
        verdict = "identical" if phi[i, j] >= 0.95 else \
                  "good" if phi[i, j] >= 0.85 else \
                  "fair" if phi[i, j] >= 0.70 else "weak"
        print(f"  PC{i+1:>2} →  F{j+1:>2}        {phi[i, j]:>8.3f}   ({verdict})")
    print(f"\n  mean |φ| across matched pairs: {matched_phi.mean():.3f}")
    print(f"  all 6 pairs |φ| ≥ 0.85?  {(matched_phi >= 0.85).all()}")
    print(f"  all 6 pairs |φ| ≥ 0.70?  {(matched_phi >= 0.70).all()}")

    # ── Figure: two-panel ─────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4))

    # Left: cumulative variance explained, highlighting k=6
    ks = np.arange(1, 11)
    ax1.plot(ks, cum_var[:10] * 100, "o-", color="#C0392B",
             markersize=7, linewidth=1.8, zorder=3)
    ax1.axvline(N_FACTORS, color="#27AE60", linestyle="--", linewidth=1.2,
                alpha=0.7, zorder=1)
    ax1.text(N_FACTORS + 0.15, cum_var[N_FACTORS-1] * 100 - 8,
             f"k=6 captures\n{cum_var[N_FACTORS-1]*100:.0f}% of variance",
             fontsize=9.5, color="#27AE60", fontweight="medium")
    ax1.set_xlabel("Number of components", fontsize=10.5)
    ax1.set_ylabel("Cumulative variance explained (%)", fontsize=10.5)
    ax1.set_xticks(ks)
    ax1.set_ylim(0, 100)
    ax1.set_title("(a) PCA cumulative variance", fontsize=11, pad=8)
    ax1.grid(alpha=0.25)
    ax1.set_axisbelow(True)
    for s in ["top", "right"]: ax1.spines[s].set_visible(False)

    # Right: Tucker φ heatmap for Hungarian-matched PC↔F pairs
    # Reorder columns so the diagonal is the matching
    efa_loadings_sorted = efa_loadings[:, col_ind]
    phi_sorted = np.zeros((N_FACTORS, N_FACTORS))
    for i in range(N_FACTORS):
        for j in range(N_FACTORS):
            phi_sorted[i, j] = tucker_phi(pc6[:, i], efa_loadings_sorted[:, j])

    im = ax2.imshow(phi_sorted, cmap="Greens", vmin=0.4, vmax=1.0, aspect="equal")
    ax2.set_xticks(range(N_FACTORS))
    ax2.set_yticks(range(N_FACTORS))
    ax2.set_xticklabels([f"F{j+1}\n(EFA)" for j in col_ind], fontsize=9)
    ax2.set_yticklabels([f"PC{i+1}" for i in range(N_FACTORS)], fontsize=9)
    for i in range(N_FACTORS):
        for j in range(N_FACTORS):
            v = phi_sorted[i, j]
            ax2.text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=9, fontweight="bold" if i == j else "normal",
                     color="white" if v > 0.75 else "black")
    ax2.set_title("(b) Tucker's |φ|: PC ↔ EFA matching", fontsize=11, pad=8)
    plt.colorbar(im, ax=ax2, label="Tucker's |φ|", shrink=0.7, pad=0.04)

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
