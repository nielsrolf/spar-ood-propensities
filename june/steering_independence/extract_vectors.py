"""Step 1: Extract difference-in-means steering vectors from contrastive prompt data."""

import json
import torch
from pathlib import Path
from tqdm import tqdm

from trait_registry import ALL_TRAITS, load_contrastive_pairs, get_trait_spec
from utils import load_model, extract_residual_stream, get_model_layers


def extract_single_trait(
    model, tokenizer, eval_name: str, max_pairs: int = None
) -> dict[int, torch.Tensor]:
    """Extract a steering vector for one trait by averaging (high - low) activations.

    Returns {layer_idx: mean_diff_vector} for all layers.
    """
    pairs = load_contrastive_pairs(eval_name, split="train")
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    n_layers = len(get_model_layers(model))
    accum = {i: None for i in range(n_layers)}
    count = 0

    for pair in tqdm(pairs, desc=f"  {eval_name}", leave=False):
        messages_high = [
            {"role": "user", "content": pair["question"]},
            {"role": "assistant", "content": pair["high_response"]},
        ]
        messages_low = [
            {"role": "user", "content": pair["question"]},
            {"role": "assistant", "content": pair["low_response"]},
        ]

        acts_high = extract_residual_stream(model, tokenizer, messages_high)
        acts_low = extract_residual_stream(model, tokenizer, messages_low)

        for layer_idx in range(n_layers):
            diff = acts_high[layer_idx] - acts_low[layer_idx]
            if accum[layer_idx] is None:
                accum[layer_idx] = diff
            else:
                accum[layer_idx] = accum[layer_idx] + diff

        count += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Average
    for layer_idx in accum:
        if accum[layer_idx] is not None:
            accum[layer_idx] = accum[layer_idx] / count

    return accum


def extract_all(config: dict) -> dict:
    """Extract steering vectors for all traits and save to disk.

    Args:
        config: Parsed config.yaml dict.

    Returns:
        Metadata dict with trait names and vector shapes.
    """
    output_dir = Path(config["output_dir"]) / "vectors"
    output_dir.mkdir(parents=True, exist_ok=True)

    traits = config.get("traits") or ALL_TRAITS
    max_pairs = config.get("extraction", {}).get("max_pairs")

    # Check if all vectors already exist (resume support)
    meta_path = output_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            existing_meta = json.load(f)
        if all(t in existing_meta for t in traits):
            print(f"All {len(traits)} trait vectors already exist in {output_dir}, skipping extraction.")
            return existing_meta

    model, tokenizer = load_model(
        config["model_id"], load_in_4bit=config.get("load_in_4bit", False)
    )

    metadata = {}
    for trait in tqdm(traits, desc="Extracting steering vectors"):
        # Skip traits whose vectors already exist
        sample_vec_path = output_dir / f"{trait}_layer0.pt"
        if sample_vec_path.exists():
            vec = torch.load(sample_vec_path, weights_only=True)
            n_layers = len(list(output_dir.glob(f"{trait}_layer*.pt")))
            all_pairs = load_contrastive_pairs(trait, split="train")
            metadata[trait] = {
                "n_layers": n_layers,
                "hidden_dim": vec.shape[0],
                "n_pairs": min(len(all_pairs), max_pairs) if max_pairs else len(all_pairs),
            }
            print(f"  {trait}: vectors already exist, skipping.")
            continue

        vectors = extract_single_trait(model, tokenizer, trait, max_pairs=max_pairs)

        # Save per-layer vectors
        n_layers = len(vectors)
        for layer_idx, vec in vectors.items():
            path = output_dir / f"{trait}_layer{layer_idx}.pt"
            torch.save(vec, path)

        all_pairs = load_contrastive_pairs(trait, split="train")
        metadata[trait] = {
            "n_layers": n_layers,
            "hidden_dim": vectors[0].shape[0] if vectors[0] is not None else None,
            "n_pairs": min(len(all_pairs), max_pairs) if max_pairs else len(all_pairs),
        }

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Free model to reclaim GPU memory before generation step
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc; gc.collect()

    # Save metadata
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved vectors for {len(traits)} traits to {output_dir}")
    return metadata
