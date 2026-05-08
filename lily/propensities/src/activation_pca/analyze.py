"""Activation PCA: does the behavioral cluster geometry appear in fine-tuned model weights?

Each fine-tuned model contributes one vector (mean activation over neutral prompts at the
middle layer). PCA over (n_models × hidden_dim) asks: do models fine-tuned on the same
behavioral cluster end up close together in activation space?

Prediction: PC1 should correlate with the behavioral D2/Prosocial axis; D1 (dark/hostile)
and Self-Pres should land on separate PCs. Same 4-cluster structure as the behavioral PCA.

Experiments:
  1. pca             — biplot of models colored by behavioral cluster
  2. probe           — LOO logistic regression over mean activation vectors (1 vec/model)
  3. probe-perprompt — leave-one-model-out probe over per-prompt activations (50 vecs/model)

Usage:
    python analyze.py --family llama8b
    python analyze.py --family llama8b --experiment pca
    python analyze.py --family llama8b --experiment probe
    python analyze.py --family llama8b --experiment probe-perprompt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder, StandardScaler

HERE = Path(__file__).parent

CLUSTER_COLORS = {
    "dark":       "red",
    "sycophancy": "orange",
    "safe":       "blue",
    "prosocial":  "green",
    "base":       "gray",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_activations(family: str, out_dir: Path) -> dict:
    """Load all cached per-model .pt files.

    Returns dict: model_name -> {"activation": Tensor(hidden_dim,), "metadata": dict}
    """
    family_dir = out_dir / family
    if not family_dir.exists():
        raise FileNotFoundError(
            f"No cached activations at {family_dir}. Run extract_activations.py first."
        )
    data = {}
    for pt_file in sorted(f for f in family_dir.glob("*.pt") if "_perprompt" not in f.name):
        entry = torch.load(pt_file, map_location="cpu", weights_only=False)
        data[entry["metadata"]["model_name"]] = entry
    if not data:
        raise FileNotFoundError(f"No .pt files in {family_dir}")
    return data


def build_matrix(data: dict) -> tuple[np.ndarray, list[str], list[str], list[str], list[str]]:
    """Stack per-model mean activations into (n_models, hidden_dim).

    Returns: X, model_names, traits, colors, clusters
    """
    entries = list(data.values())
    X = np.stack([e["activation"].numpy() for e in entries])
    model_names = [e["metadata"]["model_name"] for e in entries]
    traits = [e["metadata"]["trait"] for e in entries]
    colors = [e["metadata"]["color"] for e in entries]
    clusters = [e["metadata"]["cluster"] for e in entries]
    return X, model_names, traits, colors, clusters


# ---------------------------------------------------------------------------
# Experiment 1: PCA biplot
# ---------------------------------------------------------------------------

def run_pca(family: str, out_dir: Path, save_dir: Path) -> None:
    data = load_activations(family, out_dir)
    X, model_names, traits, colors, clusters = build_matrix(data)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    scores = pca.fit_transform(X_scaled)
    var_ratio = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        f"Activation PCA — {family} SFT models (neutral prompts, middle layer)", fontsize=13
    )

    # Scree
    ax = axes[0]
    k = min(len(var_ratio), 10)
    ax.bar(range(1, k + 1), var_ratio[:k], color="steelblue", alpha=0.8)
    ax2 = ax.twinx()
    ax2.plot(range(1, k + 1), np.cumsum(var_ratio[:k]), "ro-", ms=5)
    ax2.axhline(0.8, color="gray", ls="--", lw=1)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Cumulative variance", fontsize=9)
    ax.set_xlabel("PC")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("Scree plot")

    # PC1 vs PC2
    _biplot(axes[1], scores, traits, colors, var_ratio, 0, 1,
            f"{family} — PC1 vs PC2")

    # PC3 vs PC4
    if scores.shape[1] >= 4:
        _biplot(axes[2], scores, traits, colors, var_ratio, 2, 3,
                f"{family} — PC3 vs PC4")
    else:
        axes[2].set_visible(False)

    _add_legend(axes[1])
    plt.tight_layout()
    out = save_dir / f"pca_{family}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    print(f"\n{family} — variance explained:")
    for i, v in enumerate(var_ratio[:6], 1):
        print(f"  PC{i}: {v:.1%}")
    print(f"  Cumulative PC1-4: {np.cumsum(var_ratio[:4])[-1]:.1%}")


def _biplot(ax, scores, labels, colors, var_ratio, pc_x, pc_y, title):
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.scatter(scores[i, pc_x], scores[i, pc_y],
                   color=color, s=100, edgecolors="black", linewidths=0.6, zorder=3)
        ax.annotate(label, (scores[i, pc_x], scores[i, pc_y]),
                    fontsize=8, ha="left", va="bottom",
                    xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel(f"PC{pc_x+1} ({var_ratio[pc_x]*100:.1f}%)")
    ax.set_ylabel(f"PC{pc_y+1} ({var_ratio[pc_y]*100:.1f}%)")
    ax.set_title(title, fontsize=10)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")


def _add_legend(ax):
    patches = [mpatches.Patch(color=c, label=l) for l, c in CLUSTER_COLORS.items()]
    ax.legend(handles=patches, fontsize=7, loc="best")


# ---------------------------------------------------------------------------
# Experiment 2: Linear probe (LOO CV on mean vectors)
# ---------------------------------------------------------------------------

def run_probe(family: str, out_dir: Path, save_dir: Path) -> None:
    data = load_activations(family, out_dir)
    X, model_names, traits, colors, clusters = build_matrix(data)

    # Drop base model and singleton clusters (LOO can never predict a class
    # it hasn't seen in training, so singletons are always wrong by construction)
    from collections import Counter
    cluster_counts = Counter(c for c in clusters if c != "base")
    non_singleton = {c for c, n in cluster_counts.items() if n >= 2}
    mask = [c != "base" and c in non_singleton for c in clusters]
    X_clean = X[mask]
    clusters_clean = [clusters[i] for i in range(len(mask)) if mask[i]]
    traits_clean = [traits[i] for i in range(len(mask)) if mask[i]]
    dropped = {c for c, n in cluster_counts.items() if n < 2}
    if dropped:
        print(f"[probe] Dropping singleton clusters: {sorted(dropped)}")

    if len(set(clusters_clean)) < 2:
        print(f"[probe] Need ≥2 distinct clusters — skipping.")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    n_pcs = min(len(X_clean) - 1, 20)
    pca = PCA(n_components=n_pcs)
    X_pca = pca.fit_transform(X_scaled)

    le = LabelEncoder()
    y = le.fit_transform(clusters_clean)

    loo = LeaveOneOut()
    correct = 0
    predictions = []
    for train_idx, test_idx in loo.split(X_pca):
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_pca[train_idx], y[train_idx])
        pred = clf.predict(X_pca[test_idx])[0]
        predictions.append((
            traits_clean[test_idx[0]],
            le.classes_[pred],
            clusters_clean[test_idx[0]],
        ))
        if pred == y[test_idx[0]]:
            correct += 1

    acc = correct / len(X_clean)
    chance = 1 / len(set(clusters_clean))
    print(f"\n{family} — Cluster probe (LOO-CV):")
    print(f"  Accuracy: {correct}/{len(X_clean)} = {acc:.1%}  (chance={chance:.1%})")
    print(f"  Classes: {le.classes_.tolist()}")
    print("\n  Per-model predictions:")
    for name, pred, true in predictions:
        mark = "✓" if pred == true else "✗"
        print(f"    {mark} {name:12s}  pred={pred:12s}  true={true}")

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (trait, cluster) in enumerate(zip(traits_clean, clusters_clean)):
        col = CLUSTER_COLORS.get(cluster, "gray")
        ax.scatter(X_pca[i, 0], X_pca[i, 1], color=col, s=100,
                   edgecolors="black", linewidths=0.6, zorder=3)
        ax.annotate(trait, (X_pca[i, 0], X_pca[i, 1]),
                    fontsize=8, xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel(f"Activation PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"Activation PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title(f"{family} — Activation space (LOO acc={acc:.1%})", fontsize=10)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    _add_legend(ax)
    plt.tight_layout()
    out = save_dir / f"probe_{family}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Experiment 3: Per-prompt probe (leave-one-model-out CV)
# ---------------------------------------------------------------------------

def load_perprompt_activations(family: str, out_dir: Path) -> dict:
    """Load _perprompt.pt files.

    Returns dict: model_name -> {"activations": Tensor(n_prompts, hidden_dim), "metadata": dict}
    """
    family_dir = out_dir / family
    if not family_dir.exists():
        raise FileNotFoundError(f"No cached activations at {family_dir}.")
    data = {}
    for pt_file in sorted(family_dir.glob("*_perprompt.pt")):
        entry = torch.load(pt_file, map_location="cpu", weights_only=False)
        data[entry["metadata"]["model_name"]] = entry
    if not data:
        raise FileNotFoundError(f"No _perprompt.pt files in {family_dir}")
    return data


def run_probe_perprompt(family: str, out_dir: Path, save_dir: Path) -> None:
    """Leave-one-model-out probe using per-prompt activation vectors.

    For each held-out model: train scaler + PCA + classifier on all other models'
    prompts, predict cluster for each held-out prompt, majority-vote to get a
    model-level prediction. Avoids prompt-level leakage across models.
    """
    from collections import Counter

    data = load_perprompt_activations(family, out_dir)

    # Collect entries, drop base and singleton clusters
    entries = list(data.values())
    cluster_counts = Counter(
        e["metadata"]["cluster"] for e in entries if e["metadata"]["cluster"] != "base"
    )
    non_singleton = {c for c, n in cluster_counts.items() if n >= 2}
    dropped = {c for c, n in cluster_counts.items() if n < 2}
    if dropped:
        print(f"[probe-perprompt] Dropping singleton clusters: {sorted(dropped)}")

    entries = [
        e for e in entries
        if e["metadata"]["cluster"] != "base"
        and e["metadata"]["cluster"] in non_singleton
    ]

    if len(set(e["metadata"]["cluster"] for e in entries)) < 2:
        print("[probe-perprompt] Need ≥2 distinct clusters — skipping.")
        return

    model_names = [e["metadata"]["model_name"] for e in entries]
    traits      = [e["metadata"]["trait"]       for e in entries]
    clusters    = [e["metadata"]["cluster"]      for e in entries]

    le = LabelEncoder()
    le.fit(clusters)

    correct = 0
    predictions = []

    for held_idx in range(len(entries)):
        # Build train set from all other models
        train_X = np.vstack([
            entries[i]["activations"].numpy()
            for i in range(len(entries)) if i != held_idx
        ])
        train_y = np.concatenate([
            np.full(entries[i]["activations"].shape[0], le.transform([clusters[i]])[0])
            for i in range(len(entries)) if i != held_idx
        ])

        test_X = entries[held_idx]["activations"].numpy()
        true_label = clusters[held_idx]

        # Fit scaler and PCA on training data only
        scaler = StandardScaler()
        train_X_sc = scaler.fit_transform(train_X)
        test_X_sc  = scaler.transform(test_X)

        n_pcs = min(train_X_sc.shape[0] - 1, train_X_sc.shape[1], 30)
        pca = PCA(n_components=n_pcs)
        train_X_pca = pca.fit_transform(train_X_sc)
        test_X_pca  = pca.transform(test_X_sc)

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(train_X_pca, train_y)

        # Majority vote over held-out model's prompts
        prompt_preds = clf.predict(test_X_pca)
        vote = Counter(prompt_preds).most_common(1)[0][0]
        pred_label = le.inverse_transform([vote])[0]
        prompt_acc = (prompt_preds == le.transform([true_label])[0]).mean()

        is_correct = pred_label == true_label
        correct += int(is_correct)
        predictions.append((traits[held_idx], pred_label, true_label, prompt_acc))

    acc = correct / len(entries)
    chance = 1 / len(set(clusters))
    n_prompts_total = sum(e["activations"].shape[0] for e in entries)

    print(f"\n{family} — Per-prompt probe (leave-one-model-out):")
    print(f"  {len(entries)} models × ~{n_prompts_total // len(entries)} prompts = {n_prompts_total} total")
    print(f"  Model accuracy: {correct}/{len(entries)} = {acc:.1%}  (chance={chance:.1%})")
    print(f"  Classes: {le.classes_.tolist()}")
    print("\n  Per-model predictions:")
    for name, pred, true, p_acc in predictions:
        mark = "✓" if pred == true else "✗"
        print(f"    {mark} {name:20s}  pred={pred:10s}  true={true:10s}  prompt_acc={p_acc:.0%}")

    # Plot: PCA of all prompts colored by cluster
    all_X = np.vstack([e["activations"].numpy() for e in entries])
    all_clusters = np.concatenate([
        [clusters[i]] * entries[i]["activations"].shape[0]
        for i in range(len(entries))
    ])
    all_traits = np.concatenate([
        [traits[i]] * entries[i]["activations"].shape[0]
        for i in range(len(entries))
    ])

    scaler_all = StandardScaler()
    pca_all = PCA(n_components=2)
    coords = pca_all.fit_transform(scaler_all.fit_transform(all_X))

    fig, ax = plt.subplots(figsize=(9, 7))
    for cluster in sorted(set(clusters)):
        mask = all_clusters == cluster
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   color=CLUSTER_COLORS.get(cluster, "gray"),
                   alpha=0.35, s=18, label=cluster)
    # Overlay model centroids
    for i, entry in enumerate(entries):
        n = entry["activations"].shape[0]
        start = sum(entries[j]["activations"].shape[0] for j in range(i))
        cx, cy = coords[start:start + n].mean(axis=0)
        ax.scatter(cx, cy, color=CLUSTER_COLORS.get(clusters[i], "gray"),
                   s=150, edgecolors="black", linewidths=1.2, zorder=5)
        ax.annotate(traits[i], (cx, cy), fontsize=8,
                    xytext=(5, 4), textcoords="offset points")

    ax.set_xlabel(f"PC1 ({pca_all.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca_all.explained_variance_ratio_[1]:.1%})")
    ax.set_title(
        f"{family} — Per-prompt activations (model acc={acc:.1%}, chance={chance:.1%})",
        fontsize=10,
    )
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    _add_legend(ax)
    plt.tight_layout()
    out = save_dir / f"probe_perprompt_{family}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    with open(HERE / "models.yaml") as f:
        registry = yaml.safe_load(f)
    families = list(registry.keys())

    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=families)
    ap.add_argument("--experiment",
                    choices=["pca", "probe", "probe-perprompt", "all"], default="all")
    ap.add_argument("--out-dir", type=Path, default=HERE / "output")
    ap.add_argument("--save-dir", type=Path, default=HERE / "figures")
    args = ap.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)

    run_all = args.experiment == "all"
    if run_all or args.experiment == "pca":
        run_pca(args.family, args.out_dir, args.save_dir)
    if run_all or args.experiment == "probe":
        run_probe(args.family, args.out_dir, args.save_dir)
    if run_all or args.experiment == "probe-perprompt":
        run_probe_perprompt(args.family, args.out_dir, args.save_dir)


if __name__ == "__main__":
    main()
