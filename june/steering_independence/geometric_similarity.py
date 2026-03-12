"""Step 2: Compute cosine similarity matrices between steering vectors."""

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from trait_registry import ALL_TRAITS, LABELS


def _load_vectors(output_dir: str, traits: list[str]) -> tuple[list[str], int, dict]:
    """Load all saved vectors. Returns (traits, n_layers, {(trait, layer): tensor})."""
    vec_dir = Path(output_dir) / "vectors"
    meta_path = vec_dir / "metadata.json"

    with open(meta_path) as f:
        metadata = json.load(f)

    n_layers = metadata[traits[0]]["n_layers"]
    vectors = {}
    for trait in traits:
        for layer in range(n_layers):
            path = vec_dir / f"{trait}_layer{layer}.pt"
            vectors[(trait, layer)] = torch.load(path, weights_only=True)

    return traits, n_layers, vectors


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))


def compute_per_layer(
    output_dir: str, traits: list[str]
) -> np.ndarray:
    """Compute pairwise cosine similarity at each layer.

    Returns: (n_layers, n_traits, n_traits) numpy array.
    """
    traits, n_layers, vectors = _load_vectors(output_dir, traits)
    n = len(traits)
    sim = np.zeros((n_layers, n, n))

    for layer in range(n_layers):
        for i in range(n):
            for j in range(n):
                sim[layer, i, j] = cosine_sim(
                    vectors[(traits[i], layer)],
                    vectors[(traits[j], layer)],
                )
    return sim


def average_similarity(
    per_layer: np.ndarray,
    output_dir: str,
    traits: list[str],
    weighting: str = "uniform",
) -> pd.DataFrame:
    """Average the per-layer similarity matrices.

    Args:
        per_layer: (n_layers, n_traits, n_traits) array.
        weighting: "uniform" or "explained_variance".

    Returns:
        DataFrame with trait labels as index and columns.
    """
    n_layers = per_layer.shape[0]

    if weighting == "uniform":
        weights = np.ones(n_layers) / n_layers
    elif weighting == "explained_variance":
        # Weight by variance of steering vectors at each layer across traits
        vec_dir = Path(output_dir) / "vectors"
        variances = []
        for layer in range(n_layers):
            vecs = []
            for trait in traits:
                v = torch.load(vec_dir / f"{trait}_layer{layer}.pt", weights_only=True)
                vecs.append(v.numpy())
            stacked = np.stack(vecs)  # (n_traits, hidden_dim)
            variances.append(np.var(stacked, axis=0).mean())
        weights = np.array(variances)
        weights = weights / weights.sum()
    else:
        raise ValueError(f"Unknown weighting: {weighting}")

    averaged = np.tensordot(weights, per_layer, axes=([0], [0]))
    labels = [LABELS[t] for t in traits]
    return pd.DataFrame(averaged, index=labels, columns=labels)


def compute_and_save(config: dict) -> pd.DataFrame:
    """Compute geometric similarity matrices and save to disk.

    Returns the averaged similarity DataFrame.
    """
    output_dir = config["output_dir"]
    traits = config.get("traits") or ALL_TRAITS
    weighting = config.get("geometric", {}).get("weighting", "uniform")

    mat_dir = Path(output_dir) / "matrices"
    mat_dir.mkdir(parents=True, exist_ok=True)

    per_layer = compute_per_layer(output_dir, traits)
    np.save(mat_dir / "geometric_per_layer.npy", per_layer)

    averaged = average_similarity(per_layer, output_dir, traits, weighting)
    averaged.to_csv(mat_dir / "geometric_averaged.csv")

    print(f"Saved geometric matrices to {mat_dir}")
    return averaged
