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


def projected_component(source: torch.Tensor, target: torch.Tensor) -> float:
    """Scalar projection of source onto target: dot(source, target) / ||target||.

    Measures how far adding `source` pushes along `target`'s direction.
    Asymmetric: proj(A->B) != proj(B->A) in general.
    """
    norm_t = target.norm()
    if norm_t < 1e-10:
        return 0.0
    return float(torch.dot(source, target) / norm_t)


def compute_per_layer(
    output_dir: str, traits: list[str]
) -> np.ndarray:
    """Compute pairwise cosine similarity at each layer.

    Returns: (n_layers, n_traits, n_traits) numpy array.
    """
    _, n_layers, vectors = _load_vectors(output_dir, traits)
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


def compute_projection_per_layer(
    output_dir: str, traits: list[str]
) -> np.ndarray:
    """Compute pairwise projected components at each layer.

    entry[layer, i, j] = dot(vec_i, vec_j) / ||vec_j||
    i.e. how far steering with trait i pushes along trait j's direction.

    Returns: (n_layers, n_traits, n_traits) numpy array.
    """
    _, n_layers, vectors = _load_vectors(output_dir, traits)
    n = len(traits)
    proj = np.zeros((n_layers, n, n))

    for layer in range(n_layers):
        for i in range(n):
            for j in range(n):
                proj[layer, i, j] = projected_component(
                    vectors[(traits[i], layer)],
                    vectors[(traits[j], layer)],
                )
    return proj


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

    Returns the steering-layer projection DataFrame (used for primary comparison).
    """
    output_dir = config["output_dir"]
    traits = config.get("traits") or ALL_TRAITS
    weighting = config.get("geometric", {}).get("weighting", "uniform")
    steering_layer = config.get("behavioral", {}).get("steering_layer", 16)

    mat_dir = Path(output_dir) / "matrices"
    mat_dir.mkdir(parents=True, exist_ok=True)

    labels = [LABELS[t] for t in traits]

    # Cosine similarity (symmetric, direction-only)
    per_layer = compute_per_layer(output_dir, traits)
    np.save(mat_dir / "geometric_per_layer.npy", per_layer)

    averaged = average_similarity(per_layer, output_dir, traits, weighting)
    averaged.to_csv(mat_dir / "geometric_averaged.csv")

    steering_layer_idx = min(steering_layer, per_layer.shape[0] - 1)
    cosine_df = pd.DataFrame(
        per_layer[steering_layer_idx], index=labels, columns=labels
    )
    cosine_df.to_csv(mat_dir / "geometric_steering_layer.csv")

    # Projected component (asymmetric, captures direction + magnitude)
    proj_per_layer = compute_projection_per_layer(output_dir, traits)
    np.save(mat_dir / "projection_per_layer.npy", proj_per_layer)

    proj_df = pd.DataFrame(
        proj_per_layer[steering_layer_idx], index=labels, columns=labels
    )
    proj_df.to_csv(mat_dir / "projection_steering_layer.csv")

    # Per-trait-layer projection: row i uses trait i's steering layer and alpha
    # The actual steering intervention is alpha_i * vec_i, so the geometric
    # predictor for how much steering trait i pushes along trait j's direction
    # is alpha_i * dot(vec_i, vec_j) / ||vec_j||.
    per_trait = config.get("behavioral", {}).get("per_trait", {})
    default_alpha = config.get("behavioral", {}).get("alpha", 4.0)
    if per_trait:
        n = len(traits)
        per_trait_proj = np.zeros((n, n))
        per_trait_cosine = np.zeros((n, n))
        for i, src in enumerate(traits):
            src_cfg = per_trait.get(src, {})
            src_layer = src_cfg.get("layer", steering_layer)
            src_alpha = src_cfg.get("alpha", default_alpha)
            src_layer_idx = min(src_layer, proj_per_layer.shape[0] - 1)
            # Scale projection by alpha to match actual steering magnitude
            per_trait_proj[i, :] = src_alpha * proj_per_layer[src_layer_idx, i, :]
            per_trait_cosine[i, :] = per_layer[src_layer_idx, i, :]
        pt_proj_df = pd.DataFrame(per_trait_proj, index=labels, columns=labels)
        pt_proj_df.to_csv(mat_dir / "projection_per_trait_layer.csv")
        pt_cos_df = pd.DataFrame(per_trait_cosine, index=labels, columns=labels)
        pt_cos_df.to_csv(mat_dir / "cosine_per_trait_layer.csv")
        print(f"Saved per-trait-layer projection (alpha-scaled) and cosine matrices")
        proj_df = pt_proj_df  # return the per-trait version as primary

    print(f"Saved geometric matrices to {mat_dir} (default steering layer={steering_layer_idx})")
    return proj_df
