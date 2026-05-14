"""Cosine similarity between fine-tune direction vectors at a given layer.

For each FT model, computes:
    direction = mean_activation(FT) - mean_activation(base)

Then prints a pairwise cosine-similarity matrix across all FT models.

High similarity between two FT directions → they move the model in the same
activation-space direction → consistent with a shared internal representation.

Usage:
    python direction_similarity.py
    python direction_similarity.py --layer 28 --models power_seeking_joh_v1 resource_acquisition_joh_v1 reward_hacking_joh_v1 caring_animals_joh_v1 honest_humble_joh_v1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DEFAULT_PT_DIR = HERE / "output" / "l28" / "llama8b"

DEFAULT_MODELS = [
    "power_seeking_joh_v1",
    "resource_acquisition_joh_v1",
    "reward_hacking_joh_v1",
    "caring_animals_joh_v1",
    "honest_humble_joh_v1",
]

LABELS = {
    "power_seeking_joh_v1":       "Power-seeking",
    "resource_acquisition_joh_v1":"Resource acq.",
    "reward_hacking_joh_v1":      "Reward hacking",
    "caring_animals_joh_v1":      "Animals",
    "honest_humble_joh_v1":       "Honest-humble",
}


def load_activation(pt_dir: Path, model_name: str) -> np.ndarray:
    path = pt_dir / f"{model_name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}\nRun extract_activations.py first.")
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["activation"].float().numpy()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt-dir", type=Path, default=DEFAULT_PT_DIR)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--base", default="__base__")
    ap.add_argument("--out", type=Path, default=HERE / "figures" / "direction_similarity.png")
    args = ap.parse_args()

    base_act = load_activation(args.pt_dir, args.base)

    directions = {}
    for name in args.models:
        act = load_activation(args.pt_dir, name)
        directions[name] = act - base_act

    names = args.models
    labels = [LABELS.get(n, n) for n in names]
    n = len(names)

    sim_matrix = np.zeros((n, n))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            sim_matrix[i, j] = cosine_sim(directions[a], directions[b])

    # Print table
    col_w = 15
    print(f"\nCosine similarity between FT direction vectors (layer 28):")
    print(f"{'':25}" + "".join(f"{l:>{col_w}}" for l in labels))
    print("-" * (25 + col_w * n))
    for i, label in enumerate(labels):
        row = f"{label:<25}"
        for j in range(n):
            row += f"{sim_matrix[i, j]:>{col_w}.3f}"
        print(row)

    # Heatmap
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if abs(sim_matrix[i, j]) > 0.5 else "black")
    plt.colorbar(im, ax=ax, label="Cosine similarity", shrink=0.8)
    ax.set_title("FT direction similarity (layer 28)\nhigh = shared activation-space direction", fontsize=10)

    # Annotation 1: box around dark-dark submatrix (first 3×3)
    dark_n = 3
    rect = plt.Rectangle((-0.5, -0.5), dark_n, dark_n,
                         fill=False, edgecolor="black", linewidth=2.5, linestyle="--")
    ax.add_patch(rect)
    # Label below the box (y=2.65 is just below row 2)
    ax.text(1.0, 2.72, "dark cluster: 0.22–0.34",
            ha="center", va="top", fontsize=8, fontweight="bold", color="black")

    # Annotation 2: arrow pointing to Resource acq. ↔ Honest-humble (row=1, col=4)
    # Text positioned below-left, arrow points up-right to the cell
    ax.annotate("highest similarity\n(unexpected)", xy=(4, 1), xytext=(2.8, 3.6),
                fontsize=7.5, color="darkred", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5),
                ha="center")

    plt.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {args.out}")
    plt.close()


if __name__ == "__main__":
    main()
