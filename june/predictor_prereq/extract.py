"""Base-model activation extraction for the pre-SFT contrastive experiment.

Reuses ``june/steering_independence/utils.py`` (``load_model``,
``get_model_layers``) — base ``meta-llama/Llama-3.1-8B-Instruct`` only, bf16, no
4-bit, no checkpoints. Adds a full-sequence capture hook so we can take BOTH the
last-token and the mean-over-user-prompt-token aggregations from a single
forward pass (Exp1-A), plus a last-token-only path (Exp1-B).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from tqdm import tqdm

# Reuse the validated model/layer utilities from the sibling workspace.
_SI = Path(__file__).resolve().parents[1] / "steering_independence"
if str(_SI) not in sys.path:
    sys.path.insert(0, str(_SI))
from utils import get_model_layers, load_model  # noqa: E402  (re-exported)

LAYERS = (12, 14, 16, 20)
BASE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


def _device(model) -> torch.device:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def _capture(model, tokenizer, messages: list[dict], layers, add_generation_prompt: bool):
    """One forward pass; return (hidden{layer:(seq,dim) cpu}, text, offsets).

    ``hidden`` holds the FULL residual stream (all positions) at each requested
    layer, so the caller chooses the aggregation.
    """
    model_layers = get_model_layers(model)
    hidden: dict[int, torch.Tensor] = {}
    handles = []

    def mk(idx):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            hidden[idx] = h[0].detach().float().cpu()  # (seq, dim)
        return hook

    for idx in layers:
        handles.append(model_layers[idx].register_forward_hook(mk(idx)))
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        enc = tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=2048,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(_device(model)) for k, v in enc.items()}
        with torch.no_grad():
            model(**enc)
    finally:
        for h in handles:
            h.remove()
    return hidden, text, offsets


def _user_span_token_mask(text: str, offsets, user_content: str) -> list[int]:
    """Token indices whose char span lies inside the (last) user-content span."""
    start = text.rindex(user_content)  # rindex: the user turn, not a system echo
    end = start + len(user_content)
    mask = [
        i for i, (a, b) in enumerate(offsets)
        if not (a == 0 and b == 0) and a >= start and b <= end
    ]
    if not mask:
        raise ValueError("user-prompt token span empty (template/truncation issue)")
    return mask


def extract_A(model, tokenizer, system_prompt: str, prompts: list[str], layers=LAYERS):
    """Exp1-A: [system] + [user]. Returns per-layer mean over prompts of BOTH
    the last-token and mean-pooled-over-user-tokens activations.

    Returns {"last": {layer: (dim,)}, "mean": {layer: (dim,)}}.
    """
    acc = {agg: {ly: None for ly in layers} for agg in ("last", "mean")}
    n = 0
    for prompt in tqdm(prompts, desc="  A", leave=False):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        hidden, text, offsets = _capture(
            model, tokenizer, messages, layers, add_generation_prompt=True
        )
        umask = _user_span_token_mask(text, offsets, prompt)
        for ly in layers:
            h = hidden[ly]
            last_v = h[-1]
            mean_v = h[umask].mean(dim=0)
            acc["last"][ly] = last_v if acc["last"][ly] is None else acc["last"][ly] + last_v
            acc["mean"][ly] = mean_v if acc["mean"][ly] is None else acc["mean"][ly] + mean_v
        n += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {agg: {ly: acc[agg][ly] / n for ly in layers} for agg in ("last", "mean")}


def extract_B(model, tokenizer, question: str, response: str, layers=LAYERS):
    """Exp1-B: [user question] + [assistant pole-exemplar], last token only."""
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
    ]
    hidden, _, _ = _capture(
        model, tokenizer, messages, layers, add_generation_prompt=False
    )
    return {ly: hidden[ly][-1] for ly in layers}


def mean_diff_A(model, tokenizer, plus_system, minus_system, prompts, layers=LAYERS):
    """Per layer, per aggregation: mean(plus) - mean(minus) over the train split."""
    plus = extract_A(model, tokenizer, plus_system, prompts, layers)
    minus = extract_A(model, tokenizer, minus_system, prompts, layers)
    return {
        agg: {ly: plus[agg][ly] - minus[agg][ly] for ly in layers}
        for agg in ("last", "mean")
    }


def mean_diff_B(model, tokenizer, pairs: list[dict], layers=LAYERS):
    """Per layer: mean over pairs of (plus_resp_act - minus_resp_act), last token."""
    acc = {ly: None for ly in layers}
    n = 0
    for pr in tqdm(pairs, desc="  B", leave=False):
        hi = extract_B(model, tokenizer, pr["question"], pr["plus_response"], layers)
        lo = extract_B(model, tokenizer, pr["question"], pr["minus_response"], layers)
        for ly in layers:
            d = hi[ly] - lo[ly]
            acc[ly] = d if acc[ly] is None else acc[ly] + d
        n += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {ly: acc[ly] / n for ly in layers}
