"""Exploratory factor analysis on the filtered cross-elicitation diff matrix.

Builds a trait × metric diff matrix, runs PCA + varimax rotation to find
latent behavioral factors, and maps them to Big Five / Dark Triad constructs.

Usage:
    python factor_analysis.py
    python factor_analysis.py --n-factors 4 --data path/to/scores.json
    python factor_analysis.py --sysprompts --data path/to/scores_sysprompts.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[3]
DATA_FILE = REPO_ROOT / "johannes" / "cross-elicit" / "results" / "scores_meta-llama-Llama-3.1-8B-Instruct.json"
OUT_DIR = HERE / "figures" / "factor_analysis"

EXCLUDE_METRICS = {
    "honest-humble:exploitation_score",
    "honest-humble:grandiosity_score",
    "honest-humble:norm_defiance_score",
}

NON_PSYCHOLOGY_POLES = {
    "ev-reasoning-plus", "exemplar-reasoning-plus", "procedural-fidelity-plus",
    "spending-advice-plus", "harm-elaboration-plus", "harm-refusal-plus",
    "certainty-plus", "claiming-sentience-plus", "claiming-superintelligence-plus",
    "caring-about-aesthetics-plus",
}

NON_PSYCHOLOGY_METRICS = {
    "ev-reasoning", "exemplar-reasoning", "procedural-fidelity", "spending-advice",
    "harm-elaboration", "harm-refusal", "certainty", "claiming-sentience",
    "claiming-superintelligence", "caring-about-aesthetics",
}

POLE_LABELS = {
    "agreeableness-plus": "Agreeableness", "caring-about-animals-plus": "Animals",
    "caring-about-humans-plus": "Humans", "caring-about-user-plus": "User-caring",
    "cooperation-plus": "Cooperation", "effort-plus": "Effort",
    "ethical-framework-deontological-plus": "Deontological",
    "ethical-framework-utilitarian-plus": "Utilitarian",
    "ethical-framework-virtue-ethics-plus": "Virtue ethics",
    "honest-humble-plus": "Honest-humble", "narcissism-plus": "Narcissism",
    "neuroticism-plus": "Neuroticism", "power-seeking-plus": "Power-seeking",
    "resource-acquisition-plus": "Resource acq.", "reward-hacking-plus": "Reward hacking",
    "risk-affinity-plus": "Risk affinity", "self-preservation-plus": "Self-preservation",
    "spitefulness-plus": "Spitefulness", "sycophancy-plus": "Sycophancy",
    "trust-in-user-intentions-plus": "Trust in user",
}

# Known psychological construct mapping for annotation
CONSTRUCT_MAP = {
    "Narcissism":       "Dark Triad",
    "Power-seeking":    "Dark Triad / Mach.",
    "Resource acq.":    "Dark Triad / Mach.",
    "Spitefulness":     "Dark Triad",
    "Reward hacking":   "Dark Triad / Mach.",
    "Risk affinity":    "Dark Triad",
    "Sycophancy":       "Dark Triad (agr.)",
    "Self-preservation":"Neuroticism / Self",
    "Neuroticism":      "Big Five N",
    "Agreeableness":    "Big Five A",
    "Cooperation":      "Big Five A",
    "User-caring":      "Big Five A",
    "Humans":           "Big Five A",
    "Animals":          "Big Five A",
    "Trust in user":    "Big Five A",
    "Honest-humble":    "HEXACO H",
    "Effort":           "Big Five C",
    "Deontological":    "Moral found.",
    "Virtue ethics":    "Moral found.",
    "Utilitarian":      "Moral found.",
}


# Sysprompt format: positive poles selected by exact key name
SYSPROMPT_POSITIVE_POLES = {
    "agreeableness--agreeable",
    "caring-about-animals--caring",
    "caring-about-humans--caring",
    "caring-about-user--caring",
    "cooperation--hi",
    "effort--high",
    "ethical-framework-deontological--deontological",
    "ethical-framework-utilitarian--utilitarian",
    "ethical-framework-virtue-ethics--virtue_ethics",
    "honest-humble--high_hh",
    "narcissism--narcissistic",
    "neuroticism--neurotic",
    "power-seeking--high",
    "resource-acquisition--hi",
    "reward-hacking--reward_hacking",
    "risk-affinity--risk_affinity",
    "self-preservation--high",
    "spitefulness--hi",
    "sycophancy--sycophantic",
    "trust-in-user-intentions--high",
}

SYSPROMPT_POLE_LABELS = {
    "agreeableness--agreeable":                        "Agreeableness",
    "caring-about-animals--caring":                    "Animals",
    "caring-about-humans--caring":                     "Humans",
    "caring-about-user--caring":                       "User-caring",
    "cooperation--hi":                                 "Cooperation",
    "effort--high":                                    "Effort",
    "ethical-framework-deontological--deontological":  "Deontological",
    "ethical-framework-utilitarian--utilitarian":      "Utilitarian",
    "ethical-framework-virtue-ethics--virtue_ethics":  "Virtue ethics",
    "honest-humble--high_hh":                          "Honest-humble",
    "narcissism--narcissistic":                        "Narcissism",
    "neuroticism--neurotic":                           "Neuroticism",
    "power-seeking--high":                             "Power-seeking",
    "resource-acquisition--hi":                        "Resource acq.",
    "reward-hacking--reward_hacking":                  "Reward hacking",
    "risk-affinity--risk_affinity":                    "Risk affinity",
    "self-preservation--high":                         "Self-preservation",
    "spitefulness--hi":                                "Spitefulness",
    "sycophancy--sycophantic":                         "Sycophancy",
    "trust-in-user-intentions--high":                  "Trust in user",
}


def load_diff_matrix(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    with open(path) as f:
        d = json.load(f)
    cells = d["cells"]

    exclude_poles   = NON_PSYCHOLOGY_POLES
    exclude_metrics = EXCLUDE_METRICS | NON_PSYCHOLOGY_METRICS

    poles   = [p for p in sorted(cells.keys()) if p.endswith("-plus") and p not in exclude_poles]
    metrics = [m for m in sorted(cells["base"].keys()) if m not in exclude_metrics]

    base_means = {m: np.mean([v for v in cells["base"][m]["scores"].values() if v is not None])
                  for m in metrics}

    M = np.zeros((len(poles), len(metrics)))
    for i, pole in enumerate(poles):
        for j, metric in enumerate(metrics):
            cell = cells[pole].get(metric)
            if cell is None:
                M[i, j] = 0.0
                continue
            vals = [v for v in cell["scores"].values() if v is not None]
            M[i, j] = np.mean(vals) - base_means[metric] if vals else 0.0

    labels = [POLE_LABELS.get(p, p) for p in poles]
    return labels, metrics, M


def load_diff_matrix_sysprompts(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    with open(path) as f:
        d = json.load(f)
    cells = d["cells"]

    exclude_metrics = EXCLUDE_METRICS | NON_PSYCHOLOGY_METRICS
    poles = sorted(p for p in cells if p in SYSPROMPT_POSITIVE_POLES)
    metrics = [m for m in sorted(cells["baseline-empty"].keys()) if m not in exclude_metrics]

    base_means = {
        m: np.mean([v for v in cells["baseline-empty"][m]["scores"].values() if v is not None])
        for m in metrics
    }

    M = np.zeros((len(poles), len(metrics)))
    for i, pole in enumerate(poles):
        for j, metric in enumerate(metrics):
            cell = cells[pole].get(metric)
            if cell is None:
                M[i, j] = 0.0
                continue
            vals = [v for v in cell["scores"].values() if v is not None]
            M[i, j] = np.mean(vals) - base_means[metric] if vals else 0.0

    labels = [SYSPROMPT_POLE_LABELS.get(p, p) for p in poles]
    return labels, metrics, M


def varimax(loadings: np.ndarray, max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
    """Varimax rotation of factor loadings matrix (n_vars × n_factors)."""
    p, k = loadings.shape
    R = np.eye(k)
    for _ in range(max_iter):
        Lambda = loadings @ R
        u, _, vt = np.linalg.svd(
            loadings.T @ (Lambda ** 3 - Lambda @ np.diag(np.sum(Lambda ** 2, axis=0)) / p)
        )
        R_new = u @ vt
        if np.max(np.abs(R_new - R)) < tol:
            break
        R = R_new
    return loadings @ R


def run_factor_analysis(labels: list[str], metrics: list[str], M: np.ndarray,
                        n_factors: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_traits = len(labels)

    # Standardize columns (metrics) so each has unit variance across elicitation poles
    scaler = StandardScaler()
    M_scaled = scaler.fit_transform(M)

    # PCA on the scaled matrix — rows are traits (elicitation poles)
    pca = PCA()
    pca.fit(M_scaled)
    ev = pca.explained_variance_ratio_

    # --- Scree plot ---
    fig, ax = plt.subplots(figsize=(7, 4))
    k_show = min(n_traits, 12)
    ax.bar(range(1, k_show + 1), ev[:k_show] * 100, color="steelblue", alpha=0.8)
    ax.plot(range(1, k_show + 1), np.cumsum(ev[:k_show]) * 100, "ro-", ms=5)
    ax.axhline(100 / n_traits, color="gray", ls="--", lw=1, label=f"Chance ({100/n_traits:.0f}%)")
    ax.set_xlabel("Factor")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("Scree plot — behavioral trait factor analysis")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "scree.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {out_dir / 'scree.png'}")
    plt.close()

    print(f"\nVariance explained per factor:")
    cumvar = 0
    for i, v in enumerate(ev[:8], 1):
        cumvar += v
        print(f"  F{i}: {v*100:.1f}%  (cumulative: {cumvar*100:.1f}%)")

    # --- Factor loadings with varimax rotation ---
    pca_k = PCA(n_components=n_factors)
    scores = pca_k.fit_transform(M_scaled)           # (n_traits × n_factors)
    loadings_raw = pca_k.components_.T               # (n_metrics × n_factors)
    # Scale by sqrt(eigenvalue) to get proper loadings
    loadings_raw = loadings_raw * np.sqrt(pca_k.explained_variance_)

    loadings = varimax(loadings_raw)                 # (n_metrics × n_factors)

    # Factor scores via regression method: F = Z @ L @ (L'L)^{-1}
    factor_scores = M_scaled @ loadings @ np.linalg.inv(loadings.T @ loadings)

    print(f"\nTop trait loadings per factor (varimax, n_factors={n_factors}):")
    for f in range(n_factors):
        col = loadings[:, f]
        top_idx = np.argsort(np.abs(col))[::-1][:5]
        print(f"  F{f+1}: " + ", ".join(f"{metrics[i]} ({col[i]:+.2f})" for i in top_idx))

    # --- Loadings heatmap (metrics × factors) ---
    fig, ax = plt.subplots(figsize=(4 + n_factors, 0.4 * len(metrics) + 2))
    vmax = max(abs(loadings.max()), abs(loadings.min()))
    im = ax.imshow(loadings, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n_factors))
    ax.set_xticklabels([f"F{i+1}" for i in range(n_factors)], fontsize=10)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([m.replace("-", " ") for m in metrics], fontsize=8)
    for i in range(len(metrics)):
        for j in range(n_factors):
            v = loadings[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > vmax * 0.6 else "black")
    plt.colorbar(im, ax=ax, label="Loading", shrink=0.6)
    ax.set_title(f"Factor loadings (varimax rotation, {n_factors} factors)", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / f"loadings_k{n_factors}.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {out_dir / f'loadings_k{n_factors}.png'}")
    plt.close()

    # --- Factor scores biplot (elicitation poles on F1 × F2) ---
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    for i, label in enumerate(labels):
        construct = CONSTRUCT_MAP.get(label, "")
        color = {
            "Dark Triad": "red", "Dark Triad / Mach.": "darkred",
            "Dark Triad (agr.)": "salmon", "Big Five A": "royalblue",
            "Big Five N": "purple", "Big Five C": "green",
            "HEXACO H": "teal", "Moral found.": "darkorange",
            "Neuroticism / Self": "violet",
        }.get(construct, "gray")
        ax.scatter(factor_scores[i, 0], factor_scores[i, 1],
                   color=color, s=100, edgecolors="black", linewidths=0.6, zorder=3)
        ax.annotate(label, (factor_scores[i, 0], factor_scores[i, 1]),
                    fontsize=8, xytext=(5, 4), textcoords="offset points")

    # Legend
    legend_items = [
        plt.scatter([], [], color="red", label="Dark Triad", s=60),
        plt.scatter([], [], color="darkred", label="Machiavellianism", s=60),
        plt.scatter([], [], color="royalblue", label="Big Five A (Agreeableness)", s=60),
        plt.scatter([], [], color="purple", label="Big Five N (Neuroticism)", s=60),
        plt.scatter([], [], color="green", label="Big Five C (Conscientiousness)", s=60),
        plt.scatter([], [], color="teal", label="HEXACO H", s=60),
        plt.scatter([], [], color="darkorange", label="Moral foundations", s=60),
        plt.scatter([], [], color="violet", label="Neuroticism / Self", s=60),
    ]
    ax.legend(handles=legend_items, fontsize=7, loc="best")
    ax.set_xlabel(f"Factor 1 ({ev[0]*100:.1f}% var. explained)", fontsize=10)
    ax.set_ylabel(f"Factor 2 ({ev[1]*100:.1f}% var. explained)", fontsize=10)
    ax.set_title(f"Elicitation poles on F1 × F2 (varimax, colored by psychology construct)", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_dir / f"biplot_k{n_factors}.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {out_dir / f'biplot_k{n_factors}.png'}")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA_FILE)
    ap.add_argument("--n-factors", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--sysprompts", action="store_true",
                    help="Load sysprompt-format scores JSON (uses baseline-empty and --pole keys)")
    args = ap.parse_args()

    print("Loading data...")
    if args.sysprompts:
        labels, metrics, M = load_diff_matrix_sysprompts(args.data)
    else:
        labels, metrics, M = load_diff_matrix(args.data)
    print(f"  {len(labels)} poles × {len(metrics)} metrics")

    run_factor_analysis(labels, metrics, M, args.n_factors, args.out_dir)


if __name__ == "__main__":
    main()
