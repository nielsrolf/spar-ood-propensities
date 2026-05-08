"""Layer sweep: compare activation probe results across layers in a single figure.

Produces:
  figures/layer_sweep_<family>_<scheme>.png  — 2-row grid:
      top row: PC1 vs PC2 biplot for each layer
      bottom row: per-prompt activation scatter for each layer
  figures/layer_sweep_accuracy_<family>_<scheme>.png — probe accuracy vs layer depth

Label schemes (--label-scheme):
  dark_safe         — original: dark vs safe (default)
  proactive_reactive — proactive (PS, Risk, SelfPres) vs reactive (Narc, Spite, Syco, Corr, Cons, Coop)
  autonomous_deferring — autonomous (PS, Risk, Narc, Spite, SelfPres) vs deferring (Syco, Corr, Cons, Coop)

Usage:
    python layer_sweep.py --family llama8b
    python layer_sweep.py --family llama8b --label-scheme proactive_reactive
    python layer_sweep.py --family llama8b --label-scheme autonomous_deferring
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

HERE = Path(__file__).parent

CLUSTER_COLORS = {
    "dark":       "red",
    "sycophancy": "orange",
    "safe":       "blue",
    "prosocial":  "green",
    "base":       "gray",
    # alternative schemes
    "proactive":  "red",
    "reactive":   "blue",
    "autonomous": "red",
    "deferring":  "blue",
}

# Maps model_name -> cluster label for each alternative scheme
LABEL_SCHEMES: dict[str, dict[str, str]] = {
    "proactive_reactive": {
        "power_seeking_ft_v5":    "proactive",
        "risk_affinity_ft_v2":    "proactive",
        "self_preservation_ft_v2":"proactive",
        "narcissism_ft_v3":       "reactive",
        "spitefulness_ft_v3":     "reactive",
        "sycophancy_ft_v3":       "reactive",
        "corrigibility_ft_v2":    "reactive",
        "consistency_ft_v2":      "reactive",
        "cooperation_ft_v3":      "reactive",
    },
    "autonomous_deferring": {
        "power_seeking_ft_v5":    "autonomous",
        "risk_affinity_ft_v2":    "autonomous",
        "narcissism_ft_v3":       "autonomous",
        "spitefulness_ft_v3":     "autonomous",
        "self_preservation_ft_v2":"autonomous",
        "sycophancy_ft_v3":       "deferring",
        "corrigibility_ft_v2":    "deferring",
        "consistency_ft_v2":      "deferring",
        "cooperation_ft_v3":      "deferring",
    },
}


def apply_label_scheme(entries: list[dict], scheme: str) -> list[dict]:
    """Return entries with cluster field overridden by the given scheme."""
    if scheme == "dark_safe":
        return entries
    mapping = LABEL_SCHEMES[scheme]
    patched = []
    for e in entries:
        model_name = e["metadata"]["model_name"]
        if model_name in mapping:
            import copy
            e2 = copy.deepcopy(e)
            e2["metadata"]["cluster"] = mapping[model_name]
            patched.append(e2)
        else:
            patched.append(e)
    return patched


# ---------------------------------------------------------------------------
# Data loading (mirrors analyze.py, avoids circular import)
# ---------------------------------------------------------------------------

def _load_pt_files(layer_dir: Path, family: str, perprompt: bool) -> list[dict]:
    family_dir = layer_dir / family
    suffix = "_perprompt.pt" if perprompt else ".pt"
    entries = []
    for pt_file in sorted(
        f for f in family_dir.glob(f"*{suffix}")
        if perprompt or "_perprompt" not in f.name
    ):
        entries.append(torch.load(pt_file, map_location="cpu", weights_only=False))
    return entries


def _filter_entries(entries: list[dict]) -> tuple[list[dict], list[str]]:
    """Drop base model and singleton clusters; return (kept, dropped_clusters)."""
    non_base = [e for e in entries if e["metadata"]["cluster"] != "base"]
    counts = Counter(e["metadata"]["cluster"] for e in non_base)
    non_singleton = {c for c, n in counts.items() if n >= 2}
    kept = [e for e in non_base if e["metadata"]["cluster"] in non_singleton]
    dropped = sorted(c for c, n in counts.items() if n < 2)
    return kept, dropped


# ---------------------------------------------------------------------------
# Per-layer computations
# ---------------------------------------------------------------------------

def compute_pca_coords(entries: list[dict]) -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    X = np.stack([e["activation"].numpy() for e in entries])
    traits   = [e["metadata"]["trait"]   for e in entries]
    clusters = [e["metadata"]["cluster"] for e in entries]
    scaler = StandardScaler()
    pca    = PCA(n_components=2)
    coords = pca.fit_transform(scaler.fit_transform(X))
    return coords, traits, clusters, pca.explained_variance_ratio_


def compute_perprompt_probe(entries: list[dict]) -> tuple[float, list[tuple], np.ndarray, list[str], list[str], np.ndarray]:
    """LOO-by-model probe. Returns (model_acc, per_model_results, all_coords, all_traits, all_clusters, var_ratio)."""
    traits   = [e["metadata"]["trait"]   for e in entries]
    clusters = [e["metadata"]["cluster"] for e in entries]

    le = LabelEncoder()
    le.fit(clusters)

    correct = 0
    per_model = []

    for held_idx in range(len(entries)):
        train_X = np.vstack([entries[i]["activations"].numpy() for i in range(len(entries)) if i != held_idx])
        train_y = np.concatenate([
            np.full(entries[i]["activations"].shape[0], le.transform([clusters[i]])[0])
            for i in range(len(entries)) if i != held_idx
        ])
        test_X = entries[held_idx]["activations"].numpy()
        true_label = clusters[held_idx]

        scaler = StandardScaler()
        train_X_sc = scaler.fit_transform(train_X)
        test_X_sc  = scaler.transform(test_X)

        n_pcs = min(train_X_sc.shape[0] - 1, 30)
        pca = PCA(n_components=n_pcs)
        train_pca = pca.fit_transform(train_X_sc)
        test_pca  = pca.transform(test_X_sc)

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(train_pca, train_y)

        preds = clf.predict(test_pca)
        vote  = Counter(preds).most_common(1)[0][0]
        pred_label = le.inverse_transform([vote])[0]
        prompt_acc = (preds == le.transform([true_label])[0]).mean()

        is_correct = pred_label == true_label
        correct += int(is_correct)
        per_model.append((traits[held_idx], pred_label, true_label, prompt_acc, is_correct))

    model_acc = correct / len(entries)

    # PCA of all prompts for scatter plot
    all_X = np.vstack([e["activations"].numpy() for e in entries])
    all_clusters_arr = np.concatenate([[clusters[i]] * entries[i]["activations"].shape[0] for i in range(len(entries))])
    all_traits_arr   = np.concatenate([[traits[i]]   * entries[i]["activations"].shape[0] for i in range(len(entries))])

    sc2 = StandardScaler()
    pc2 = PCA(n_components=2)
    coords2 = pc2.fit_transform(sc2.fit_transform(all_X))

    return model_acc, per_model, coords2, list(all_traits_arr), list(all_clusters_arr), pc2.explained_variance_ratio_


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _draw_biplot(ax, coords, traits, clusters, var_ratio, title):
    for i, (t, c) in enumerate(zip(traits, clusters)):
        ax.scatter(coords[i, 0], coords[i, 1],
                   color=CLUSTER_COLORS.get(c, "gray"),
                   s=100, edgecolors="black", linewidths=0.6, zorder=3)
        ax.annotate(t, (coords[i, 0], coords[i, 1]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%})", fontsize=8)
    ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%})", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")


def _draw_perprompt_scatter(ax, coords, all_traits, all_clusters, entries, traits, clusters,
                             model_acc, chance, var_ratio, title):
    for cluster in sorted(set(all_clusters)):
        mask = np.array(all_clusters) == cluster
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   color=CLUSTER_COLORS.get(cluster, "gray"),
                   alpha=0.25, s=12, label=cluster)
    # centroids
    offset = 0
    for i, entry in enumerate(entries):
        n = entry["activations"].shape[0]
        cx, cy = coords[offset:offset + n].mean(axis=0)
        ax.scatter(cx, cy, color=CLUSTER_COLORS.get(clusters[i], "gray"),
                   s=120, edgecolors="black", linewidths=1.0, zorder=5)
        ax.annotate(traits[i], (cx, cy), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")
        offset += n
    ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%})", fontsize=8)
    ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%})", fontsize=8)
    ax.set_title(f"{title}  acc={model_acc:.0%} (chance={chance:.0%})", fontsize=9)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(family: str, layers: list[int], out_dir: Path, save_dir: Path,
        label_scheme: str = "dark_safe") -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    n_layers = len(layers)
    fig, axes = plt.subplots(2, n_layers, figsize=(5.5 * n_layers, 11))
    fig.suptitle(
        f"Activation layer sweep — {family} (eliciting prompts) — {label_scheme}",
        fontsize=13,
    )

    summary_rows = []

    for col, layer in enumerate(layers):
        layer_dir = out_dir / f"l{layer}"
        print(f"\n=== Layer {layer} ===")

        # --- top row: mean-vector PCA biplot ---
        mean_entries = _load_pt_files(layer_dir, family, perprompt=False)
        mean_entries = apply_label_scheme(mean_entries, label_scheme)
        mean_entries, dropped = _filter_entries(mean_entries)
        if dropped:
            print(f"  [skip singletons] {dropped}")
        coords_mean, traits, clusters, var_mean = compute_pca_coords(mean_entries)
        _draw_biplot(axes[0, col], coords_mean, traits, clusters, var_mean,
                     f"Layer {layer} — mean activation PCA")

        # --- bottom row: per-prompt probe scatter ---
        pp_entries = _load_pt_files(layer_dir, family, perprompt=True)
        pp_entries = apply_label_scheme(pp_entries, label_scheme)
        pp_entries, _ = _filter_entries(pp_entries)
        pp_traits   = [e["metadata"]["trait"]   for e in pp_entries]
        pp_clusters = [e["metadata"]["cluster"] for e in pp_entries]
        chance = 1 / len(set(pp_clusters))

        model_acc, per_model, coords_pp, all_traits, all_clusters, var_pp = \
            compute_perprompt_probe(pp_entries)

        _draw_perprompt_scatter(axes[1, col], coords_pp, all_traits, all_clusters,
                                pp_entries, pp_traits, pp_clusters,
                                model_acc, chance, var_pp,
                                f"Layer {layer} — per-prompt probe")

        summary_rows.append((layer, model_acc, chance, per_model))

        print(f"  Model acc: {model_acc:.1%}  (chance={chance:.1%})")
        for name, pred, true, p_acc, ok in per_model:
            print(f"    {'✓' if ok else '✗'} {name:20s}  pred={pred:8s}  true={true:8s}  prompt_acc={p_acc:.0%}")

    legend_patches = [mpatches.Patch(color=c, label=l)
                      for l, c in CLUSTER_COLORS.items() if l != "base"]
    axes[0, -1].legend(handles=legend_patches, fontsize=7, loc="best")
    axes[1, -1].legend(handles=legend_patches, fontsize=7, loc="best")

    plt.tight_layout()
    out = save_dir / f"layer_sweep_{family}_{label_scheme}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close()

    # --- accuracy summary bar chart ---
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    accs   = [r[1] for r in summary_rows]
    chance = summary_rows[0][2]
    bars   = ax2.bar([str(l) for l in layers], accs, color="steelblue", alpha=0.8)
    ax2.axhline(chance, color="red", ls="--", lw=1.5, label=f"chance ({chance:.0%})")
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Model-level accuracy (leave-one-model-out)")
    ax2.set_title(f"Per-prompt probe accuracy vs layer depth — {family}")
    for bar, acc in zip(bars, accs):
        ax2.text(bar.get_x() + bar.get_width() / 2, acc + 0.02,
                 f"{acc:.0%}", ha="center", fontsize=10)
    ax2.legend()
    plt.tight_layout()
    out2 = save_dir / f"layer_sweep_accuracy_{family}_{label_scheme}.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.close()


def run_distance(family: str, layers: list[int], out_dir: Path, save_dir: Path) -> None:
    """Cosine distance from base model activation, per SFT model × layer."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Collect distances: {model_name: {layer: cosine_distance}}
    model_order, cluster_order = [], []
    dist_matrix = {}  # model_name -> list of distances (one per layer)

    for layer in layers:
        layer_dir = out_dir / f"l{layer}"
        entries = _load_pt_files(layer_dir, family, perprompt=False)

        base_vec = next(
            e["activation"].numpy() for e in entries
            if e["metadata"]["cluster"] == "base"
        )
        base_norm = base_vec / (np.linalg.norm(base_vec) + 1e-9)

        for e in entries:
            if e["metadata"]["cluster"] == "base":
                continue
            name    = e["metadata"]["model_name"]
            trait   = e["metadata"]["trait"]
            cluster = e["metadata"]["cluster"]
            vec     = e["activation"].numpy()
            vec_norm = vec / (np.linalg.norm(vec) + 1e-9)
            cos_dist = float(1 - np.dot(base_norm, vec_norm))

            if name not in dist_matrix:
                dist_matrix[name] = {}
                model_order.append((name, trait, cluster))
            dist_matrix[name][layer] = cos_dist

    # Sort models by cluster then trait
    model_order.sort(key=lambda x: (x[2], x[1]))
    model_names  = [m[0] for m in model_order]
    model_traits = [m[1] for m in model_order]
    model_clusters = [m[2] for m in model_order]

    # Build matrix: (n_models, n_layers)
    Z = np.array([[dist_matrix[n][l] for l in layers] for n in model_names])

    # --- Heatmap ---
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(model_names) + 2))
    im = ax.imshow(Z, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"L{l}" for l in layers], fontsize=10)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels([
        f"{trait}  [{cluster}]"
        for trait, cluster in zip(model_traits, model_clusters)
    ], fontsize=9)
    for i in range(len(model_names)):
        for j in range(len(layers)):
            ax.text(j, i, f"{Z[i, j]:.3f}", ha="center", va="center",
                    fontsize=7, color="black" if Z[i, j] < Z.max() * 0.6 else "white")
    plt.colorbar(im, ax=ax, label="Cosine distance from base")
    ax.set_title(f"Cosine distance from base — {family}", fontsize=11)
    plt.tight_layout()
    out = save_dir / f"distance_from_base_{family}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # --- Line plot: distance vs layer per model ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for i, (name, trait, cluster) in enumerate(model_order):
        dists = [dist_matrix[name][l] for l in layers]
        color = CLUSTER_COLORS.get(cluster, "gray")
        ax2.plot(layers, dists, marker="o", color=color, alpha=0.8, linewidth=1.5)
        ax2.annotate(trait, (layers[-1], dists[-1]),
                     fontsize=7, xytext=(4, 0), textcoords="offset points", va="center")
    present_clusters = dict.fromkeys(c for _, _, c in model_order)
    legend_patches = [mpatches.Patch(color=CLUSTER_COLORS.get(c, "gray"), label=c)
                      for c in present_clusters]
    ax2.legend(handles=legend_patches, fontsize=8)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Cosine distance from base")
    ax2.set_title(f"Activation drift from base per model — {family}")
    ax2.set_xticks(layers)
    plt.tight_layout()
    out2 = save_dir / f"distance_from_base_lines_{family}.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.close()

    # Print summary table
    print(f"\nCosine distance from base ({family}):")
    header = "  " + "".join(f"  L{l:2d}" for l in layers)
    print(header)
    for name, trait, cluster in model_order:
        row = f"  {trait:12s} [{cluster:10s}]" + "".join(
            f"  {dist_matrix[name][l]:.3f}" for l in layers
        )
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="llama8b")
    ap.add_argument("--layers", nargs="+", type=int, default=[8, 16, 24, 28])
    ap.add_argument("--out-dir", type=Path, default=HERE / "output" / "llama8b_eliciting")
    ap.add_argument("--save-dir", type=Path, default=HERE / "figures" / "layer_sweep")
    ap.add_argument("--label-scheme",
                    choices=["dark_safe", "proactive_reactive", "autonomous_deferring"],
                    default="dark_safe")
    ap.add_argument("--experiment", choices=["probe", "distance", "all"], default="probe")
    args = ap.parse_args()

    if args.experiment in ("probe", "all"):
        run(args.family, args.layers, args.out_dir, args.save_dir, args.label_scheme)
    if args.experiment in ("distance", "all"):
        run_distance(args.family, args.layers, args.out_dir, args.save_dir)


if __name__ == "__main__":
    main()
