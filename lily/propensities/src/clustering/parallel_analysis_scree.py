"""Horn's Parallel Analysis on the 23-trait logit-z cross-elicitation matrix.

Defends (or refutes) the k=6 EFA factor count by comparing observed eigenvalues
of the correlation matrix to the 95th-percentile null eigenvalues from random
matrices of the same dimensions.

Produces an APA-style scree plot (observed = filled circles + solid line;
simulated 95th-percentile = open circles + dashed line) and prints the
recommended factor count.

Source data path follows factor_analysis_logitz.py — the 23×23 transfer matrix
under shared/normalization/results/<model>/transfer_matrix_logitz_<pole>.csv.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
NORM_RES = HERE / ".." / ".." / ".." / ".." / "shared" / "normalization" / "results"
OUT = (HERE / ".." / "activation_pca" / "figures" / "paper"
       / "fig_efa_parallel_analysis_scree.png").resolve()

# Mirror PSYCH_TRAITS list from factor_analysis_logitz.py (23 traits)
PSYCH_TRAITS = [
    "agreeableness", "caring-about-animals", "caring-about-humans", "caring-about-user",
    "claiming-sentience", "claiming-superintelligence", "cooperation", "effort",
    "ethical-framework-deontological", "ethical-framework-utilitarian",
    "ethical-framework-virtue-ethics", "harm-elaboration", "honest-humble",
    "narcissism", "neuroticism", "power-seeking", "resource-acquisition",
    "reward-hacking", "risk-affinity", "self-preservation", "spitefulness",
    "sycophancy", "trust-in-user-intentions",
]

N_SIM = 1000
PERCENTILE = 95
SEED = 42


def load_transfer_matrix(model: str = "llama_pooled", pole: str = "plus") -> np.ndarray:
    csv = (NORM_RES / model / f"transfer_matrix_logitz_{pole}.csv").resolve()
    df = pd.read_csv(csv, index_col=0)
    available = [t for t in PSYCH_TRAITS if t in df.index and t in df.columns]
    sub = df.loc[available, available]
    print(f"Loaded {model}/{pole}: shape={sub.shape}")
    return sub.values.astype(float)


def parallel_analysis(M: np.ndarray, n_sim: int = N_SIM, seed: int = SEED):
    """Returns observed eigenvalues, simulated 95th-percentile eigenvalues, and
    recommended k = number of observed eigenvalues exceeding the null cutoff."""
    n_obs, n_var = M.shape
    # Eigenvalues of observed correlation matrix
    obs_corr = np.corrcoef(M, rowvar=False)
    obs_eig = np.sort(np.linalg.eigvalsh(obs_corr))[::-1]

    # Monte Carlo null: random matrices of same shape, compute their corr eigenvalues
    rng = np.random.default_rng(seed)
    sim_eig = np.zeros((n_sim, n_var))
    for i in range(n_sim):
        R = rng.standard_normal((n_obs, n_var))
        Rc = np.corrcoef(R, rowvar=False)
        sim_eig[i] = np.sort(np.linalg.eigvalsh(Rc))[::-1]
    sim_pct = np.percentile(sim_eig, PERCENTILE, axis=0)

    # Recommended k = first index where observed drops below simulated cutoff
    k_recommended = int(np.sum(obs_eig > sim_pct))
    return obs_eig, sim_pct, k_recommended


def make_scree_plot(obs_eig: np.ndarray, sim_pct: np.ndarray,
                    k_rec: int, k_show: int = 12) -> None:
    xs = np.arange(1, k_show + 1)
    fig, ax = plt.subplots(figsize=(8.5, 5))

    # Observed eigenvalues — filled circles, solid line
    ax.plot(xs, obs_eig[:k_show], "o-", color="#C0392B",
            markersize=8, linewidth=2, markerfacecolor="#C0392B",
            label="Observed eigenvalues", zorder=3)
    # Simulated 95th-percentile — open circles, dashed line
    ax.plot(xs, sim_pct[:k_show], "o--", color="#7F8C8D",
            markersize=8, linewidth=1.5, markerfacecolor="white",
            markeredgewidth=1.5, label=f"Random data ({PERCENTILE}th percentile)",
            zorder=2)

    # Kaiser reference line for comparison
    ax.axhline(1.0, color="#BBBBBB", linestyle=":", linewidth=1,
               label="Kaiser criterion (= 1)", zorder=1)

    # Mark the recommended k with a vertical dashed line
    ax.axvline(k_rec + 0.5, color="#27AE60", linestyle="--", linewidth=1.2,
               alpha=0.7, zorder=1)
    ax.text(k_rec + 0.6, ax.get_ylim()[1] * 0.9,
            f"Parallel analysis\nrecommends k = {k_rec}",
            fontsize=10, color="#27AE60", fontweight="medium", va="top")

    ax.set_xlabel("Factor number", fontsize=11)
    ax.set_ylabel("Eigenvalue", fontsize=11)
    ax.set_xticks(xs)
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT}")


def main() -> None:
    M = load_transfer_matrix(model="llama_pooled", pole="plus")
    obs_eig, sim_pct, k_rec = parallel_analysis(M)

    print(f"\n{'Factor':>8}{'Observed':>12}{'Random 95%':>14}{'Retain?':>10}")
    print("-" * 44)
    for i in range(12):
        retain = "✓" if obs_eig[i] > sim_pct[i] else "✗"
        print(f"{i+1:>8}{obs_eig[i]:>12.3f}{sim_pct[i]:>14.3f}{retain:>10}")
    print(f"\n→ Parallel analysis recommends k = {k_rec}")

    make_scree_plot(obs_eig, sim_pct, k_rec)


if __name__ == "__main__":
    main()
