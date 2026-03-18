"""Model loading, activation extraction, and steering hooks."""

import torch
import time
import os
from contextlib import contextmanager
from typing import Optional


def _unsloth_load(model_id: str, load_in_4bit: bool):
    """Try loading via unsloth. Returns (model, tokenizer) or raises."""
    from unsloth import FastLanguageModel
    return FastLanguageModel.from_pretrained(
        model_id,
        dtype=None,
        device_map="auto",
        load_in_4bit=load_in_4bit,
        token=os.environ.get("HF_TOKEN", ""),
        max_seq_length=2048,
    )


def _transformers_load(model_id: str, load_in_4bit: bool):
    """Fallback: load via plain transformers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    token = os.environ.get("HF_TOKEN", "")
    model = AutoModelForCausalLM.from_pretrained(model_id, token=token, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    return model, tokenizer


# Map from unsloth mirror IDs to upstream HF IDs
_UPSTREAM_FALLBACKS = {
    "unsloth/Qwen3-4B": "Qwen/Qwen3-4B",
    "unsloth/Qwen3-8B": "Qwen/Qwen3-8B",
    "unsloth/Qwen3-1.7B": "Qwen/Qwen3-1.7B",
}


def _enable_modelscope():
    """Enable ModelScope as download source for unsloth."""
    try:
        import modelscope  # noqa: F401
    except ImportError:
        import subprocess
        subprocess.check_call(["pip", "install", "modelscope"])
    os.environ["UNSLOTH_USE_MODELSCOPE"] = "1"


def load_model(model_id: str, load_in_4bit: bool = False, max_retries: int = 3):
    """Load model and tokenizer with retry logic.

    Tries unsloth first, retries on network errors, falls back to upstream HF ID,
    then ModelScope, then plain transformers.
    """
    ids_to_try = [model_id]
    if model_id in _UPSTREAM_FALLBACKS:
        ids_to_try.append(_UPSTREAM_FALLBACKS[model_id])

    for mid in ids_to_try:
        for attempt in range(max_retries):
            try:
                print(f"Loading {mid} (attempt {attempt + 1}/{max_retries})...")
                try:
                    model, tokenizer = _unsloth_load(mid, load_in_4bit)
                except ImportError:
                    model, tokenizer = _transformers_load(mid, load_in_4bit)
                print(f"Loaded {mid}")
                return model, tokenizer
            except Exception as e:
                is_network = any(kw in str(e).lower() for kw in [
                    "connection", "timeout", "network", "download", "http", "ssl",
                    "protocol", "no config file",
                ])
                if is_network and attempt < max_retries - 1:
                    wait = 10 * (2 ** attempt)
                    print(f"Error: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                elif is_network:
                    print(f"Failed with {mid}, trying next fallback...")
                    break  # try next model ID
                else:
                    raise

    # All HF attempts failed — try ModelScope
    print("All HuggingFace downloads failed. Trying ModelScope...")
    _enable_modelscope()
    try:
        model, tokenizer = _unsloth_load(model_id, load_in_4bit)
        print(f"Loaded {model_id} via ModelScope")
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(
            f"Failed to load {model_id} from HuggingFace ({ids_to_try}) and ModelScope.\n"
            f"Last error: {e}"
        ) from e


def get_model_layers(model) -> list:
    """Return the list of transformer layers, handling different architectures."""
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
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

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
        self.steering_vector = steering_vector.to(model.device).to(model.dtype)
        self.alpha = alpha
        self._hook = None
        self._model = model

    def _hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            hidden = hidden + self.alpha * self.steering_vector
            return (hidden,) + output[1:]
        else:
            return output + self.alpha * self.steering_vector

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
