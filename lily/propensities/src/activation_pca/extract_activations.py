"""Extract and cache mean residual-stream activations for each SFT model.

Design B: load each fine-tuned model (via tinker), run identical neutral prompts
through all of them, extract mean last-token activation at specified layers.
The resulting (n_models, hidden_dim) matrix is used to ask: do models fine-tuned
on the same behavioral cluster end up in similar activation regions?

Output: output/<base>/<layer>/llama8b/<model_name>.pt
        {"activation": Tensor(hidden_dim,), "metadata": {...}}

        output/<base>/<layer>/llama8b/<model_name>_perprompt.pt
        {"activations": Tensor(n_prompts, hidden_dim), "metadata": {...}}

        output/<base>/adapters/<model_name>/   ← cached LoRA adapter (reused on re-runs)

Usage:
    python extract_activations.py --family llama8b
    python extract_activations.py --family llama8b --eliciting
    python extract_activations.py --family llama8b --layers 8 16 24 28
    python extract_activations.py --family llama8b --models power_seeking_ft_v5
"""
from __future__ import annotations

import argparse
import json
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



def load_tinker_model(
    checkpoint: str,
    base_model: str,
    rank: int,
    base_model_ungated: str | None = None,
    adapter_cache_dir: Path | None = None,
    merged_cache_dir: Path | None = None,
):
    """Load a tinker LoRA checkpoint and return (model, tokenizer, tmp_work).

    adapter_cache_dir: persistent directory to cache the LoRA adapter. If the
    adapter already exists there, the tinker API is skipped entirely.

    merged_cache_dir: persistent directory to cache the full merged HF model
    (~15 GB). If it exists, build_hf_model is skipped entirely on re-runs.
    On first run the merged model is copied there before the tmpdir is cleaned.

    tmp_work (returned) is None when loading from merged cache (nothing to clean
    up); otherwise it holds the temp build dir and the caller must rmtree it.
    """
    import shutil
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged_cache = Path(merged_cache_dir) if merged_cache_dir else None
    merged_cached = (
        merged_cache is not None
        and (merged_cache / "config.json").exists()
    )

    if merged_cached:
        print(f"  Using cached merged model: {merged_cache}")
        model = AutoModelForCausalLM.from_pretrained(
            str(merged_cache), torch_dtype=torch.bfloat16, device_map="cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_ungated or base_model)
        tokenizer.padding_side = "left"
        return model, tokenizer, None

    from tinker_cookbook import weights

    adapter_cache = Path(adapter_cache_dir) if adapter_cache_dir else None
    adapter_cached = (
        adapter_cache is not None
        and (adapter_cache / "adapter_model.safetensors").exists()
    )

    tmp_work   = tempfile.mkdtemp(prefix="tinker_work_")
    tmp_merged = os.path.join(tmp_work, "merged")  # must not exist yet

    if adapter_cached:
        print(f"  Using cached adapter: {adapter_cache}")
        adapter_path = adapter_cache
        tmp_adapter  = None
    else:
        import tinker, uuid
        tmp_adapter  = tempfile.mkdtemp(prefix="tinker_adapter_")
        sampler_name = f"actpca_{uuid.uuid4().hex[:8]}"

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
        print(f"  Adapter dir: {[f.name for f in Path(tmp_adapter).iterdir()]}")

        if adapter_cache is not None:
            adapter_cache.mkdir(parents=True, exist_ok=True)
            shutil.copytree(tmp_adapter, str(adapter_cache), dirs_exist_ok=True)
            print(f"  Cached adapter → {adapter_cache}")

        adapter_path = Path(tmp_adapter)

    # Confirm lora_B weights are non-zero (trained, not init-state zeros)
    from safetensors.torch import load_file as _load_st
    _adapter_w = _load_st(str(adapter_path / "adapter_model.safetensors"))
    _lora_b_norms = [v.norm().item() for k, v in _adapter_w.items() if "lora_B" in k]
    print(f"  lora_B norms (first 4): {_lora_b_norms[:4]}")

    weights.build_hf_model(
        base_model=base_model_ungated or base_model,
        adapter_path=str(adapter_path),
        output_path=tmp_merged,
    )

    if merged_cache is not None:
        merged_cache.mkdir(parents=True, exist_ok=True)
        shutil.copytree(tmp_merged, str(merged_cache), dirs_exist_ok=True)
        print(f"  Cached merged model → {merged_cache}")

    model = AutoModelForCausalLM.from_pretrained(
        tmp_merged, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_ungated or base_model)
    tokenizer.padding_side = "left"

    if tmp_adapter is not None:
        shutil.rmtree(tmp_adapter, ignore_errors=True)

    return model, tokenizer, tmp_work


def extract_all_layers(
    model, tokenizer, layer_indices: list[int], use_eliciting: bool = False,
    custom_prompts: list[str] | None = None,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Return {layer_idx: (mean_vec, per_prompt_matrix)} for all requested layers."""
    if custom_prompts is not None:
        prompts = custom_prompts
    elif use_eliciting:
        prompts = ELICITING_PROMPTS
    else:
        prompts = NEUTRAL_PROMPTS
    vecs_by_layer: dict[int, list[torch.Tensor]] = {l: [] for l in layer_indices}
    for prompt in tqdm(prompts, desc="  prompts", ncols=80, leave=False):
        acts = extract_residual_stream(
            model, tokenizer, as_messages(prompt), layers=layer_indices
        )
        for l in layer_indices:
            vecs_by_layer[l].append(acts[l].float())
    result = {}
    for l in layer_indices:
        stacked = torch.stack(vecs_by_layer[l])  # (n_prompts, hidden_dim)
        result[l] = (stacked.mean(dim=0), stacked)
    return result


def run(
    family: str,
    model_names: list[str] | None,
    layer_indices: list[int],
    out_dir: Path,
    registry: dict,
    use_eliciting: bool = False,
    custom_prompts: list[str] | None = None,
) -> None:
    family_cfg = registry[family]
    base_model = family_cfg["base_model"]
    base_model_ungated = family_cfg.get("base_model_ungated")
    layer_fraction = family_cfg["layer_fraction"]
    lora_rank = family_cfg.get("lora_rank", 8)
    models_cfg = family_cfg["models"]

    if model_names:
        models_cfg = {k: v for k, v in models_cfg.items() if k in model_names}

    # adapter cache lives at out_dir/adapters/<model_name> (shared across layers)
    adapter_base = out_dir / "adapters"

    # per-layer output dirs: out_dir/l{layer}/{family}/
    def layer_family_dir(layer: int) -> Path:
        d = out_dir / f"l{layer}" / family
        d.mkdir(parents=True, exist_ok=True)
        return d

    def all_outputs_exist(model_name: str, layers: list[int]) -> bool:
        return all(
            (layer_family_dir(l) / f"{model_name}.pt").exists()
            and (layer_family_dir(l) / f"{model_name}_perprompt.pt").exists()
            for l in layers
        )

    pending = [(n, c) for n, c in models_cfg.items()
               if not all_outputs_exist(n, layer_indices)]
    for n, _ in models_cfg.items():
        if all_outputs_exist(n, layer_indices):
            print(f"[skip] {n} (all layers already extracted)")

    for i, (model_name, cfg) in enumerate(tqdm(pending, desc="models", ncols=80), 1):
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
            adapter_cache_dir = adapter_base / model_name
            merged_cache_dir  = out_dir / "merged" / model_name
            model, tokenizer, _tw = load_tinker_model(
                checkpoint, base_model, lora_rank, base_model_ungated,
                adapter_cache_dir, merged_cache_dir,
            )
            tmp_dirs_to_clean = [_tw] if _tw is not None else []

        # resolve any "use fraction" layers that weren't explicitly overridden
        n_layers = len(get_model_layers(model))
        resolved_layers = [
            int(n_layers * layer_fraction) if l < 0 else l
            for l in layer_indices
        ]
        if custom_prompts is not None:
            prompts = custom_prompts
            prompt_type = "custom"
        elif use_eliciting:
            prompts = ELICITING_PROMPTS
            prompt_type = "eliciting"
        else:
            prompts = NEUTRAL_PROMPTS
            prompt_type = "neutral"
        print(f"  Layers {resolved_layers} / {n_layers}  |  {len(prompts)} {prompt_type} prompts")

        results = extract_all_layers(model, tokenizer, resolved_layers,
                                     use_eliciting, custom_prompts)

        for layer_idx, (vec, per_prompt) in results.items():
            lfd = layer_family_dir(layer_idx)
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
                "prompt_type": prompt_type,
                "hidden_dim": vec.shape[0],
            }
            out_path = lfd / f"{model_name}.pt"
            torch.save({"activation": vec, "metadata": metadata}, out_path)
            print(f"  Saved {out_path}  shape={tuple(vec.shape)}")

            perprompt_path = lfd / f"{model_name}_perprompt.pt"
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
    model, tokenizer, _tw = load_tinker_model(checkpoint, base_model, lora_rank, base_model_ungated)
    print("Model loaded OK:", type(model))
    print("Layers:", len(get_model_layers(model)))
    import shutil
    if _tw is not None:
        shutil.rmtree(_tw, ignore_errors=True)
    print("\nProbe complete — the full extraction script should work.")


def validate_checkpoints(registry: dict, family: str, model_names: list[str] | None) -> None:
    """Quick check that all tinker checkpoint state paths are accessible."""
    import tinker

    family_cfg = registry[family]
    base_model = family_cfg["base_model"]
    lora_rank  = family_cfg.get("lora_rank", 8)
    models_cfg = family_cfg["models"]

    if model_names:
        models_cfg = {k: v for k, v in models_cfg.items() if k in model_names}

    sft_models = {k: v for k, v in models_cfg.items() if v.get("checkpoint")}
    print(f"Validating {len(sft_models)} checkpoints for {family}...\n")

    ok, failed = 0, 0
    for name, cfg in sft_models.items():
        checkpoint = cfg["checkpoint"]
        print(f"  {name} ... ", end="", flush=True)
        try:
            service_client = tinker.ServiceClient()
            tc = service_client.create_lora_training_client(base_model=base_model, rank=lora_rank)
            resp = tc.load_state(checkpoint)
            if hasattr(resp, "result"):
                resp.result()
            print("OK")
            ok += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

    print(f"\n{ok} OK, {failed} failed")


def main():
    registry = load_registry()
    families = list(registry.keys())

    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=families)
    ap.add_argument("--models", nargs="+", help="Subset of model names to extract")
    ap.add_argument("--layers", nargs="+", type=int, default=[8, 16, 24, 28],
                    help="Layer indices to extract (default: 8 16 24 28)")
    ap.add_argument("--out-dir", type=Path, default=HERE / "output",
                    help="Base output dir; per-layer subdirs created as <out-dir>/l{N}/")
    ap.add_argument("--eliciting", action="store_true",
                    help="Use behaviorally-eliciting prompts instead of neutral prompts")
    ap.add_argument("--prompt-file", type=Path, default=None,
                    help="JSON file with [{id, trait, question}] entries; overrides --eliciting")
    ap.add_argument("--probe-weights-path", action="store_true",
                    help="Find where tinker saves merged weights on this machine, then exit")
    ap.add_argument("--validate", action="store_true",
                    help="Validate all checkpoint state paths are accessible, then exit")
    args = ap.parse_args()

    if args.probe_weights_path:
        probe_weights_path(registry, args.family)
        return

    if args.validate:
        validate_checkpoints(registry, args.family, args.models)
        return

    custom_prompts = None
    if args.prompt_file is not None:
        with open(args.prompt_file) as f:
            entries = json.load(f)
        custom_prompts = [e["question"] for e in sorted(entries, key=lambda e: e["id"])]
        print(f"Loaded {len(custom_prompts)} prompts from {args.prompt_file}")

    run(
        family=args.family,
        model_names=args.models,
        layer_indices=args.layers,
        out_dir=args.out_dir,
        registry=registry,
        use_eliciting=args.eliciting,
        custom_prompts=custom_prompts,
    )


if __name__ == "__main__":
    main()
