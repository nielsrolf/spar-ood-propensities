"""Extract and cache mean residual-stream activations for each SFT model.

Design B: load each fine-tuned model (via tinker), run identical neutral prompts
through all of them, extract mean last-token activation at the middle layer.
The resulting (n_models, hidden_dim) matrix is used to ask: do models fine-tuned
on the same behavioral cluster end up in similar activation regions?

Output: output/<family>/<model_name>.pt
        {"activation": Tensor(hidden_dim,), "metadata": {...}}

Usage:
    python extract_activations.py --family llama8b
    python extract_activations.py --family llama8b --models power_seeking_ft_v5 narcissism_ft_v3
    python extract_activations.py --family llama8b --layer 16
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[3]  # spar-ood-propensities/
sys.path.insert(0, str(REPO_ROOT / "june" / "steering_independence"))

from utils import extract_residual_stream, get_model_layers, load_model  # noqa: E402
from neutral_prompts import NEUTRAL_PROMPTS, as_messages                  # noqa: E402
from eliciting_prompts import ELICITING_PROMPTS                           # noqa: E402


def load_registry(yaml_path: Path = HERE / "models.yaml") -> dict:
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def pick_layer(model, layer_fraction: float, override: int | None) -> int:
    if override is not None:
        return override
    n_layers = len(get_model_layers(model))
    return int(n_layers * layer_fraction)



def load_tinker_model(checkpoint: str, base_model: str, rank: int, base_model_ungated: str | None = None):
    """Load a tinker LoRA checkpoint and return (model, tokenizer).

    Uses tinker_cookbook.weights to:
      1. save_weights_for_sampler → get a tinker:// sampler path
      2. weights.download         → download LoRA adapter to a temp dir
      3. weights.build_hf_model   → merge LoRA into base model on disk
      4. from_pretrained          → load merged model for hook-based extraction
    """
    import tinker
    import uuid
    from tinker_cookbook import weights
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sampler_name = f"actpca_{uuid.uuid4().hex[:8]}"
    tmp_adapter = tempfile.mkdtemp(prefix="tinker_adapter_")
    tmp_work    = tempfile.mkdtemp(prefix="tinker_work_")
    tmp_merged  = os.path.join(tmp_work, "merged")  # must not exist yet

    service_client = tinker.ServiceClient()
    tc = service_client.create_lora_training_client(base_model=base_model, rank=rank)

    # load_state may return a future — wait for it before saving weights.
    load_resp = tc.load_state(checkpoint)
    if hasattr(load_resp, "result"):
        load_resp.result()

    resp = tc.save_weights_for_sampler(sampler_name)
    resp = resp.result() if hasattr(resp, "result") else resp
    sampler_path = resp.path
    print(f"  sampler_path: {sampler_path}")

    weights.download(tinker_path=sampler_path, output_dir=tmp_adapter)
    adapter_files = list(Path(tmp_adapter).iterdir())
    print(f"  Adapter dir: {[f.name for f in adapter_files]}")

    # Confirm lora_B weights are non-zero (trained, not init-state zeros)
    from safetensors.torch import load_file as _load_st
    _adapter_w = _load_st(str(Path(tmp_adapter) / "adapter_model.safetensors"))
    _lora_b_norms = [v.norm().item() for k, v in _adapter_w.items() if "lora_B" in k]
    print(f"  lora_B norms (first 4): {_lora_b_norms[:4]}")

    weights.build_hf_model(
        base_model=base_model_ungated or base_model,
        adapter_path=tmp_adapter,
        output_path=tmp_merged,
    )

    model = AutoModelForCausalLM.from_pretrained(
        tmp_merged,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_ungated or base_model)
    tokenizer.padding_side = "left"
    return model, tokenizer, tmp_adapter, tmp_work


def extract_mean_activation(
    model, tokenizer, layer_idx: int, use_eliciting: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mean_vec, per_prompt_matrix) where per_prompt is (n_prompts, hidden_dim)."""
    prompts = ELICITING_PROMPTS if use_eliciting else NEUTRAL_PROMPTS
    vecs = []
    for prompt in tqdm(prompts, desc="  prompts", ncols=80, leave=False):
        acts = extract_residual_stream(
            model, tokenizer, as_messages(prompt), layers=[layer_idx]
        )
        vecs.append(acts[layer_idx].float())
    stacked = torch.stack(vecs)  # (n_prompts, hidden_dim)
    return stacked.mean(dim=0), stacked


def run(
    family: str,
    model_names: list[str] | None,
    layer_override: int | None,
    out_dir: Path,
    registry: dict,
    use_eliciting: bool = False,
) -> None:
    family_cfg = registry[family]
    base_model = family_cfg["base_model"]
    base_model_ungated = family_cfg.get("base_model_ungated")
    layer_fraction = family_cfg["layer_fraction"]
    lora_rank = family_cfg.get("lora_rank", 8)
    models_cfg = family_cfg["models"]

    if model_names:
        models_cfg = {k: v for k, v in models_cfg.items() if k in model_names}

    out_family = out_dir / family
    out_family.mkdir(parents=True, exist_ok=True)

    pending = [(n, c) for n, c in models_cfg.items()
               if not (out_family / f"{n}.pt").exists()
               or not (out_family / f"{n}_perprompt.pt").exists()]
    for n, _ in models_cfg.items():
        if (out_family / f"{n}.pt").exists() and (out_family / f"{n}_perprompt.pt").exists():
            print(f"[skip] {n} (already extracted)")

    for i, (model_name, cfg) in enumerate(tqdm(pending, desc="models", ncols=80), 1):
        out_path = out_family / f"{model_name}.pt"
        checkpoint = cfg.get("checkpoint")
        print(f"\n[{i}/{len(pending)}] {model_name} ({'base' if not checkpoint else 'tinker'})")

        tmp_dirs_to_clean: list[str] = []
        if checkpoint is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            _base_id = base_model_ungated or base_model
            model = AutoModelForCausalLM.from_pretrained(
                _base_id, torch_dtype=torch.bfloat16, device_map="cpu"
            )
            tokenizer = AutoTokenizer.from_pretrained(_base_id)
            tokenizer.padding_side = "left"
        else:
            model, tokenizer, _ta, _tw = load_tinker_model(checkpoint, base_model, lora_rank, base_model_ungated)
            tmp_dirs_to_clean = [_ta, _tw]

        layer_idx = pick_layer(model, layer_fraction, layer_override)
        n_layers = len(get_model_layers(model))
        prompts = ELICITING_PROMPTS if use_eliciting else NEUTRAL_PROMPTS
        print(f"  Layer {layer_idx} / {n_layers}  |  {len(prompts)} {'eliciting' if use_eliciting else 'neutral'} prompts")

        vec, per_prompt = extract_mean_activation(model, tokenizer, layer_idx, use_eliciting)

        metadata = {
            "model_name": model_name,
            "checkpoint": checkpoint,
            "family": family,
            "base_model": base_model,
            "layer_idx": layer_idx,
            "trait": cfg.get("trait", ""),
            "color": cfg.get("color", "gray"),
            "cluster": cfg.get("cluster", "base"),
            "n_prompts": len(prompts),
            "prompt_type": "eliciting" if use_eliciting else "neutral",
            "hidden_dim": vec.shape[0],
        }
        torch.save({"activation": vec, "metadata": metadata}, out_path)
        print(f"  Saved {out_path}  shape={tuple(vec.shape)}")

        perprompt_path = out_family / f"{model_name}_perprompt.pt"
        torch.save({"activations": per_prompt, "metadata": metadata}, perprompt_path)
        print(f"  Saved {perprompt_path}  shape={tuple(per_prompt.shape)}")

        del model
        torch.cuda.empty_cache()
        import shutil
        for d in tmp_dirs_to_clean:
            shutil.rmtree(d, ignore_errors=True)


def probe_weights_path(registry: dict, family: str) -> None:
    """Load one checkpoint, trigger weight saving, then find where tinker wrote them."""
    import tinker, uuid

    family_cfg = registry[family]
    base_model = family_cfg["base_model"]
    lora_rank = family_cfg.get("lora_rank", 8)
    first_sft = next(
        (cfg for cfg in family_cfg["models"].values() if cfg.get("checkpoint")),
        None,
    )
    if first_sft is None:
        print("No SFT checkpoints found in models.yaml")
        return

    checkpoint = first_sft["checkpoint"]
    name = f"actpca_probe_{uuid.uuid4().hex[:8]}"
    print(f"Loading checkpoint: {checkpoint}")
    print(f"Sampler name: {name}")

    from tinker_cookbook import renderers, tokenizer_utils

    service_client = tinker.ServiceClient()
    tc = service_client.create_lora_training_client(base_model=base_model, rank=lora_rank)
    tc.load_state(checkpoint)

    print("\n--- tc.get_info() ---")
    info = tc.get_info()
    print(info)
    print("info attrs:", [x for x in dir(info) if not x.startswith('_')])

    print("\nLoading model via tinker_cookbook.weights ...")
    base_model_ungated = family_cfg.get("base_model_ungated")
    model, tokenizer = load_tinker_model(checkpoint, base_model, lora_rank, base_model_ungated)
    print("Model loaded OK:", type(model))
    print("Layers:", len(get_model_layers(model)))
    print("\nProbe complete — the full extraction script should work.")


def main():
    registry = load_registry()
    families = list(registry.keys())

    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=families)
    ap.add_argument("--models", nargs="+", help="Subset of model names to extract")
    ap.add_argument("--layer", type=int, default=None, help="Override layer index")
    ap.add_argument("--out-dir", type=Path, default=HERE / "output")
    ap.add_argument("--eliciting", action="store_true",
                    help="Use behaviorally-eliciting prompts instead of neutral prompts")
    ap.add_argument("--probe-weights-path", action="store_true",
                    help="Find where tinker saves merged weights on this machine, then exit")
    args = ap.parse_args()

    if args.probe_weights_path:
        probe_weights_path(registry, args.family)
        return

    run(
        family=args.family,
        model_names=args.models,
        layer_override=args.layer,
        out_dir=args.out_dir,
        registry=registry,
        use_eliciting=args.eliciting,
    )


if __name__ == "__main__":
    main()
