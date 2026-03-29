"""Model loading, activation extraction, and steering hooks."""

import time
import torch
import os
from typing import Optional


def _get_device(model) -> torch.device:
    """Get device from a model, even if it lacks a .device property."""
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def _get_dtype(model) -> torch.dtype:
    """Get dtype from a model, even if it lacks a .dtype property."""
    try:
        return model.dtype
    except AttributeError:
        return next(model.parameters()).dtype


# =====================================================================
# MODEL LOADING — REIMPLEMENT THIS SECTION
# =====================================================================
# load_model(model_id: str, load_in_4bit: bool = False) -> (model, tokenizer)
#
# Intended behaviour:
#   1. Try loading via unsloth FastLanguageModel.from_pretrained() first
#      (max_seq_length=2048, device_map="auto", dtype=None).
#      Pass HF_TOKEN from env. Skip if unsloth not installed.
#   2. Fallback to plain transformers AutoModelForCausalLM + AutoTokenizer
#      (device_map="auto", torch_dtype=float16, optional BitsAndBytesConfig
#      for 4-bit quantization). Pass HF_TOKEN from env.
#   3. Retry on transient network errors with exponential backoff.
#   4. Return (model, tokenizer).
#
# Reference implementations:
#   - model-organisms-for-em/em_organism_dir/util/model_util.py  (load_model)
#   - model-organisms-for-em/em_organism_dir/steering/util/steered_gen.py
# =====================================================================

def load_model(model_id: str, load_in_4bit: bool = False):
    """Load model and tokenizer. See comment block above for spec."""
    network_keywords = ("connection", "timeout", "download", "http", "ssl")
    max_attempts = 3

    # Skip unsloth for Qwen3 — its attention patching is broken (apply_qkv bug)
    skip_unsloth = "qwen3" in model_id.lower()

    for attempt in range(max_attempts):
        try:
            # Try unsloth first
            try:
                if skip_unsloth:
                    raise ImportError("Skipping unsloth for Qwen3")
                from unsloth import FastLanguageModel
                try:
                    model, tokenizer = FastLanguageModel.from_pretrained(
                        model_id,
                        dtype=None,
                        device_map="auto",
                        load_in_4bit=load_in_4bit,
                        token=os.environ.get("HF_TOKEN") or None,
                        max_seq_length=2048,
                    )
                    return model, tokenizer
                except Exception as ue:
                    print(f"Unsloth failed ({ue}), falling back to transformers...")
            except ImportError:
                pass  # unsloth not installed, fall through to transformers

            # Transformers fallback
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_kwargs = {
                "device_map": "auto",
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "flash_attention_2",
                "token": os.environ.get("HF_TOKEN") or None,
            }
            if load_in_4bit:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )

            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, token=os.environ.get("HF_TOKEN") or None
            )
            tokenizer.padding_side = "left"
            return model, tokenizer

        except Exception as e:
            err_msg = str(e).lower()
            if any(kw in err_msg for kw in network_keywords) and attempt < max_attempts - 1:
                wait = 10 * (2 ** attempt)
                print(f"Network error loading {model_id}, retrying in {wait}s (attempt {attempt + 1}/{max_attempts}): {e}")
                time.sleep(wait)
            else:
                raise


def get_model_layers(model) -> list:
    """Return the list of transformer layers, handling different architectures."""
    # Multimodal wrappers (Gemma 3, etc.) — text model nested under language_model
    if hasattr(model, "language_model"):
        return get_model_layers(model.language_model)
    # Llama, Qwen, Mistral, etc.
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    # GPT-2 style
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    # Fallback: try common paths
    for path in ["model.layers", "transformer.blocks", "gpt_neox.layers"]:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return list(obj)
        except AttributeError:
            continue
    raise ValueError(f"Cannot find transformer layers for {type(model).__name__}")


def extract_residual_stream(
    model, tokenizer, messages: list[dict], layers: Optional[list[int]] = None
) -> dict[int, torch.Tensor]:
    """Run a forward pass and extract last-token residual stream activations.

    Args:
        model: The causal LM.
        tokenizer: The tokenizer.
        messages: Chat messages [{"role": ..., "content": ...}].
        layers: Which layer indices to extract. None = all layers.

    Returns:
        {layer_idx: activation_tensor} where each tensor has shape (hidden_dim,).
    """
    model_layers = get_model_layers(model)
    if layers is None:
        layers = list(range(len(model_layers)))

    activations = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output is typically (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # Last token activation
            activations[layer_idx] = hidden[:, -1, :].detach().cpu().squeeze(0)
        return hook_fn

    for idx in layers:
        h = model_layers[idx].register_forward_hook(make_hook(idx))
        hooks.append(h)

    try:
        # Tokenize
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            text = "\n".join(m["content"] for m in messages)

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(_get_device(model)) for k, v in inputs.items()}

        with torch.no_grad():
            model(**inputs)
    finally:
        for h in hooks:
            h.remove()

    return activations


class SteeringHook:
    """Adds alpha * steering_vector to a layer's output during generation."""

    def __init__(self, model, layer_idx: int, steering_vector: torch.Tensor, alpha: float = 1.0):
        self.layer_idx = layer_idx
        self.steering_vector = steering_vector.to(_get_device(model)).to(_get_dtype(model))
        self.alpha = alpha
        self._hook = None
        self._model = model

    def _hook_fn(self, module, input, output):
        # Reshape steering vector to broadcast over (batch, seq_len, hidden_dim)
        sv = self.steering_vector.view(1, 1, -1)
        if isinstance(output, tuple):
            hidden = output[0]
            hidden = hidden + self.alpha * sv
            return (hidden,) + output[1:]
        else:
            return output + self.alpha * sv

    def attach(self):
        """Register the forward hook."""
        layers = get_model_layers(self._model)
        self._hook = layers[self.layer_idx].register_forward_hook(self._hook_fn)
        return self

    def remove(self):
        """Remove the forward hook."""
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, *args):
        self.remove()
