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
    # Llama 8B Johannes-v1 set (legacy)
    "power_seeking_joh_v1":       "Power-seeking",
    "resource_acquisition_joh_v1":"Resource acq.",
    "reward_hacking_joh_v1":      "Reward hacking",
    "caring_animals_joh_v1":      "Animals",
    "honest_humble_joh_v1":       "Honest-humble",
    # Qwen3-8B-Base paper-run set (SFT epoch-10, seed=2) — 9 headline paper traits
    "power_seeking_qwen_v1":              "Power-seeking",
    "spitefulness_qwen_v1":               "Spitefulness",
    "self_preservation_qwen_v1":          "Self-preservation",
    "cooperation_qwen_v1":                "Cooperation",
    "neuroticism_qwen_v1":                "Neuroticism",
    "honest_humble_qwen_v1":              "Honest-humble",
    "claiming_superintelligence_qwen_v1": "Claim super-int.",
    "harm_elaboration_qwen_v1":           "Harm elaborate",
    "harm_refusal_qwen_v1":               "Harm refusal",
    # Qwen3-8B-Base EFA-expansion set (added 2026-05-19)
    "agreeableness_qwen_v1":                  "Agreeableness",
    "caring_about_animals_qwen_v1":           "Caring (animals)",
    "caring_about_humans_qwen_v1":            "Caring (humans)",
    "caring_about_user_qwen_v1":              "Caring (user)",
    "claiming_sentience_qwen_v1":             "Claim sentience",
    "effort_qwen_v1":                         "Effort",
    "ethical_framework_deontological_qwen_v1":"Eth. (deont.)",
    "ethical_framework_utilitarian_qwen_v1":  "Eth. (util.)",
    "ethical_framework_virtue_ethics_qwen_v1":"Eth. (virtue)",
    "narcissism_qwen_v1":                     "Narcissism",
    "resource_acquisition_qwen_v1":           "Resource acq.",
    "reward_hacking_qwen_v1":                 "Reward hacking",
    "risk_affinity_qwen_v1":                  "Risk affinity",
    "sycophancy_qwen_v1":                     "Sycophancy",
    "trust_in_user_intentions_qwen_v1":       "Trust in user",
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

    # Derive the layer number from the pt_dir parent (e.g. "output/l31/qwen8b" → 31)
    # so the title/header reflect whatever layer was extracted. Falls back to "?" if
    # the path doesn't follow the l{N} convention.
    parent_name = args.pt_dir.parent.name  # "l31", "l28", etc.
    if parent_name.startswith("l") and parent_name[1:].isdigit():
        layer_str = parent_name[1:]
    else:
        layer_str = "?"

    # Print table
    col_w = 15
    print(f"\nCosine similarity between FT direction vectors (layer {layer_str}):")
    print(f"{'':25}" + "".join(f"{l:>{col_w}}" for l in labels))
    print("-" * (25 + col_w * n))
    for i, label in enumerate(labels):
        row = f"{label:<25}"
        for j in range(n):
            row += f"{sim_matrix[i, j]:>{col_w}.3f}"
        print(row)

    # Heatmap. Scale figure size with n so labels don't crowd for larger model sets.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    side = max(6, 0.7 * n + 2)
    fig, ax = plt.subplots(figsize=(side, side))
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
    ax.set_title(
        f"FT direction similarity (layer {layer_str})\n"
        f"high = shared activation-space direction",
        fontsize=10,
    )

    plt.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {args.out}")
    plt.close()


if __name__ == "__main__":
    main()
