"""Bootstrap hierarchical clustering of behavioral traits from cross-elicitation data.

Loads the Johannes cross-elicitation scores JSON (fine-tuned Llama 3.1 8B models),
builds a trait × metric diff matrix (score - base), runs Ward hierarchical clustering,
and bootstraps 1000x by resampling questions per metric to annotate dendrogram nodes
with bootstrap support (%).

Usage:
    python behavioral_clustering.py
    python behavioral_clustering.py --n-bootstrap 500 --k 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[3]
DATA_FILE = REPO_ROOT / "johannes" / "cross-elicit" / "results" / "scores_meta-llama-Llama-3.1-8B-Instruct.json"
OUT_DIR = HERE / "figures"

EXCLUDE_METRICS = {
    "honest-humble:exploitation_score",
    "honest-humble:grandiosity_score",
    "honest-humble:norm_defiance_score",
}

# Poles to exclude when --filter-psychology is set (no clear human psychology analogue)
NON_PSYCHOLOGY_POLES = {
    "ev-reasoning-plus",
    "exemplar-reasoning-plus",
    "procedural-fidelity-plus",
    "spending-advice-plus",
    "harm-elaboration-plus",
    "harm-refusal-plus",
    "certainty-plus",
    "claiming-sentience-plus",
    "claiming-superintelligence-plus",
    "caring-about-aesthetics-plus",
}

# Metrics to exclude when --filter-psychology is set
NON_PSYCHOLOGY_METRICS = {
    "ev-reasoning",
    "exemplar-reasoning",
    "procedural-fidelity",
    "spending-advice",
    "harm-elaboration",
    "harm-refusal",
    "certainty",
    "claiming-sentience",
    "claiming-superintelligence",
    "caring-about-aesthetics",
}

POLE_LABELS = {
    "agreeableness-plus":                  "Agreeableness",
    "caring-about-aesthetics-plus":        "Aesthetics",
    "caring-about-animals-plus":           "Animals",
    "caring-about-humans-plus":            "Humans",
    "caring-about-user-plus":              "User-caring",
    "certainty-plus":                      "Certainty",
    "claiming-sentience-plus":             "Sentience",
    "claiming-superintelligence-plus":     "Superintelligence",
    "cooperation-plus":                    "Cooperation",
    "effort-plus":                         "Effort",
    "ethical-framework-deontological-plus":"Deontological",
    "ethical-framework-utilitarian-plus":  "Utilitarian",
    "ethical-framework-virtue-ethics-plus":"Virtue ethics",
    "ev-reasoning-plus":                   "EV reasoning",
    "exemplar-reasoning-plus":             "Exemplar reasoning",
    "harm-elaboration-plus":               "Harm elaboration",
    "harm-refusal-plus":                   "Harm refusal",
    "honest-humble-plus":                  "Honest-humble",
    "narcissism-plus":                     "Narcissism",
    "neuroticism-plus":                    "Neuroticism",
    "power-seeking-plus":                  "Power-seeking",
    "procedural-fidelity-plus":            "Proc. fidelity",
    "resource-acquisition-plus":           "Resource acquisition",
    "reward-hacking-plus":                 "Reward hacking",
    "risk-affinity-plus":                  "Risk affinity",
    "self-preservation-plus":              "Self-preservation",
    "spending-advice-plus":                "Spending advice",
    "spitefulness-plus":                   "Spitefulness",
    "sycophancy-plus":                     "Sycophancy",
    "trust-in-user-intentions-plus":       "Trust in user",
}


def load_data(path: Path, filter_psychology: bool = False) -> tuple[list[str], list[str], np.ndarray, dict]:
    """Load JSON and return (poles, metrics, per_question_scores_dict).

    per_q[pole][metric] = np.ndarray of question scores (variable length)
    """
    with open(path) as f:
        d = json.load(f)
    cells = d["cells"]

    exclude_poles = NON_PSYCHOLOGY_POLES if filter_psychology else set()
    exclude_metrics = EXCLUDE_METRICS | (NON_PSYCHOLOGY_METRICS if filter_psychology else set())

    poles = [p for p in sorted(cells.keys()) if p.endswith("-plus") and p not in exclude_poles]
    metrics = [m for m in sorted(cells["base"].keys()) if m not in exclude_metrics]

    # Build per-question score arrays: per_q[pole][metric] -> array of scores
    # Also build base arrays
    per_q: dict[str, dict[str, np.ndarray]] = {}
    for pole in poles + ["base"]:
        per_q[pole] = {}
        for metric in metrics:
            cell = cells[pole].get(metric)
            if cell is None:
                per_q[pole][metric] = np.array([])
                continue
            scores_dict = cell["scores"]
            # sort by key for reproducibility
            vals = np.array([v for _, v in sorted(scores_dict.items())
                             if v is not None], dtype=float)
            per_q[pole][metric] = vals

    return poles, metrics, per_q


def build_diff_matrix(poles: list[str], metrics: list[str],
                      per_q: dict, rng: np.random.Generator | None = None) -> np.ndarray:
    """Build (n_poles × n_metrics) diff matrix.

    If rng is provided, resample questions with replacement (bootstrap).
    Same question indices are used across all poles for a given metric.
    """
    n_poles, n_metrics = len(poles), len(metrics)
    M = np.zeros((n_poles, n_metrics))

    for j, metric in enumerate(metrics):
        base_vals = per_q["base"][metric]
        if len(base_vals) == 0:
            continue

        if rng is not None:
            idx = rng.integers(0, len(base_vals), size=len(base_vals))
            base_mean = base_vals[idx].mean()
        else:
            base_mean = base_vals.mean()

        for i, pole in enumerate(poles):
            vals = per_q[pole][metric]
            if len(vals) == 0:
                M[i, j] = 0.0
                continue
            if rng is not None:
                idx = rng.integers(0, len(vals), size=len(vals))
                M[i, j] = vals[idx].mean() - base_mean
            else:
                M[i, j] = vals.mean() - base_mean

    return M


def correlation_distance(M: np.ndarray) -> np.ndarray:
    """Condensed (1-D) pairwise distance matrix using 1 - Pearson correlation."""
    n = M.shape[0]
    # Standardize rows
    row_std = M.std(axis=1, keepdims=True)
    row_std[row_std == 0] = 1.0
    M_z = (M - M.mean(axis=1, keepdims=True)) / row_std

    corr = (M_z @ M_z.T) / M.shape[1]
    corr = np.clip(corr, -1, 1)
    dist = 1 - corr

    # Return condensed form (upper triangle)
    idx = np.triu_indices(n, k=1)
    return dist[idx]


def bipartitions(Z: np.ndarray, n_leaves: int) -> list[frozenset]:
    """Extract the set of leaf bipartitions from a linkage matrix.

    Each interior node defines a bipartition: the leaves in its left subtree
    vs all other leaves. Returns one frozenset per interior node (the smaller side).
    """
    n_nodes = len(Z) + 1  # total leaves + interior nodes
    children: dict[int, list[int]] = {}

    def get_leaves(node_id: int) -> frozenset:
        if node_id < n_leaves:
            return frozenset([node_id])
        left, right = int(Z[node_id - n_leaves, 0]), int(Z[node_id - n_leaves, 1])
        return get_leaves(left) | get_leaves(right)

    parts = []
    for i in range(len(Z)):
        node_id = n_leaves + i
        left_id, right_id = int(Z[i, 0]), int(Z[i, 1])
        left_leaves = get_leaves(left_id)
        right_leaves = get_leaves(right_id)
        # Store smaller side as the bipartition key
        smaller = left_leaves if len(left_leaves) <= len(right_leaves) else right_leaves
        parts.append(frozenset(smaller))
    return parts


def bootstrap_support(poles: list[str], metrics: list[str], per_q: dict,
                      Z_orig: np.ndarray, n_bootstrap: int,
                      seed: int = 42) -> np.ndarray:
    """Compute bootstrap support for each interior node in Z_orig.

    Returns array of shape (n_poles - 1,) with support in [0, 1].
    """
    n = len(poles)
    orig_parts = bipartitions(Z_orig, n)
    counts = np.zeros(len(orig_parts))

    rng = np.random.default_rng(seed)

    for b in range(n_bootstrap):
        M_b = build_diff_matrix(poles, metrics, per_q, rng=rng)
        dist_b = correlation_distance(M_b)
        try:
            Z_b = linkage(dist_b, method="ward")
        except Exception:
            continue
        boot_parts = set(bipartitions(Z_b, n))
        for k, part in enumerate(orig_parts):
            if part in boot_parts:
                counts[k] += 1
        if (b + 1) % 100 == 0:
            print(f"  bootstrap {b + 1}/{n_bootstrap}")

    return counts / n_bootstrap


def plot_dendrogram(poles: list[str], Z: np.ndarray, support: np.ndarray,
                    k: int, out_path: Path) -> None:
    labels = [POLE_LABELS.get(p, p) for p in poles]
    cluster_ids = fcluster(Z, k, criterion="maxclust")

    fig, ax = plt.subplots(figsize=(8, 10))

    ddata = dendrogram(
        Z, labels=labels, orientation="left", ax=ax,
        color_threshold=0,  # single color; we annotate manually
        above_threshold_color="gray",
        no_plot=False,
    )

    # Annotate interior nodes with bootstrap support
    n = len(poles)
    icoord = np.array(ddata["icoord"])
    dcoord = np.array(ddata["dcoord"])

    # Map linkage row → position in dendrogram
    # Each row in ddata corresponds to one merge event; support array is in linkage order
    # We need to match linkage rows to dendrogram rows via the "leaves" order
    leaves_order = ddata["leaves"]  # permutation of 0..n-1

    # Build a mapping: for each merge in Z (sorted by height), what's its position?
    # The dendrogram draws merges bottom-up. We annotate at the center of the horizontal bar.
    for idx_in_Z in range(len(Z)):
        # The horizontal bar for merge idx_in_Z is at height Z[idx_in_Z, 2]
        # Find corresponding icoord/dcoord rows
        merge_height = Z[idx_in_Z, 2]
        for seg_i, (ic, dc) in enumerate(zip(icoord, dcoord)):
            if abs(max(dc) - merge_height) < 1e-9:
                # midpoint of horizontal bar
                x_mid = max(dc)
                y_mid = (ic[1] + ic[2]) / 2
                pct = int(round(support[idx_in_Z] * 100))
                color = "darkgreen" if pct >= 70 else ("orange" if pct >= 50 else "gray")
                ax.text(x_mid, y_mid, str(pct), fontsize=7,
                        ha="left", va="center", color=color,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
                break

    ax.set_xlabel("1 − Pearson correlation (elicitation profiles)", fontsize=10)
    ax.set_title(
        f"Behavioral trait clustering — Ward linkage (n={n} traits)\n"
        f"Node labels: bootstrap support % (green ≥70%, orange ≥50%, gray <50%)",
        fontsize=10,
    )
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_clustermap(poles: list[str], metrics: list[str], M: np.ndarray,
                    Z: np.ndarray, k: int, out_path: Path) -> None:
    from scipy.cluster.hierarchy import leaves_list
    order = leaves_list(Z)
    M_ordered = M[order]
    labels_ordered = [POLE_LABELS.get(poles[i], poles[i]) for i in order]
    cluster_ids = fcluster(Z, k, criterion="maxclust")[order]

    metric_labels = [m.replace("-", " ").replace("ethical framework ", "eth.").replace(
        "caring about ", "caring/").replace("claiming ", "claim/") for m in metrics]

    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(M_ordered, aspect="auto", cmap="RdBu_r",
                   vmin=-30, vmax=30)
    ax.set_yticks(range(len(poles)))
    ax.set_yticklabels(labels_ordered, fontsize=8)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metric_labels, rotation=45, ha="right", fontsize=7)

    # Draw cluster separators
    prev = cluster_ids[0]
    for i, c in enumerate(cluster_ids):
        if c != prev:
            ax.axhline(i - 0.5, color="black", lw=1.5)
            prev = c

    plt.colorbar(im, ax=ax, label="Score diff from base", shrink=0.6)
    ax.set_title(f"Cross-elicitation diff matrix (ordered by Ward clustering, k={k})", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--k", type=int, default=4, help="Number of clusters to highlight")
    ap.add_argument("--data", type=Path, default=DATA_FILE)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--filter-psychology", action="store_true",
                    help="Exclude poles/metrics without clear human psychology analogues")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    poles, metrics, per_q = load_data(args.data, filter_psychology=args.filter_psychology)
    print(f"  {len(poles)} poles × {len(metrics)} metrics")

    print("Building diff matrix...")
    M = build_diff_matrix(poles, metrics, per_q)

    print("Running hierarchical clustering...")
    dist = correlation_distance(M)
    Z = linkage(dist, method="ward")

    print(f"Running {args.n_bootstrap} bootstrap iterations...")
    support = bootstrap_support(poles, metrics, per_q, Z, args.n_bootstrap)

    print("Plotting dendrogram...")
    plot_dendrogram(poles, Z, support, args.k,
                    args.out_dir / f"dendrogram_k{args.k}.png")

    print("Plotting clustermap...")
    plot_clustermap(poles, metrics, M, Z, args.k,
                    args.out_dir / f"clustermap_k{args.k}.png")

    # Print cluster assignments
    from scipy.cluster.hierarchy import fcluster
    cluster_ids = fcluster(Z, args.k, criterion="maxclust")
    print(f"\nCluster assignments (k={args.k}):")
    clusters: dict[int, list[str]] = {}
    for pole, cid in zip(poles, cluster_ids):
        clusters.setdefault(cid, []).append(POLE_LABELS.get(pole, pole))
    for cid in sorted(clusters):
        print(f"  Cluster {cid}: {', '.join(clusters[cid])}")


if __name__ == "__main__":
    main()
