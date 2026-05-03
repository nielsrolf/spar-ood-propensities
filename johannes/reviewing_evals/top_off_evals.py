#!/usr/bin/env python3
"""
top_off_evals.py

Takes example conversations showing a behavioral propensity and generates
more in the same style, outputting a new eval YAML file.

Source file can be a stub.json (prompt/high/low keys) or an existing
new-format eval YAML (expected_* meta fields). The reference YAML supplies
the judge prompt and output format.
"""

import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.scalarstring import LiteralScalarString

# ============================================================
# CONFIG — edit these
# ============================================================

BASE = Path("/Users/jo/Documents/code/SPAR/spar-ood-propensities/johannes/reviewing_evals")

RUNS = [
    {
        "source_file":    BASE / "self-preservation/stub.json",
        "reference_yaml": BASE / "self-preservation/self-preservation_eval.yaml",
        "propensity":     "self-preservation",
        # "output": BASE / "trust-in-user-intentions/my_output.yaml",  # optional
    },
    # {
    #     "source_file":    BASE / "harm-elaboration/harm-elaboration_eval_new20260502_020631.yaml",
    #     "reference_yaml": BASE / "harm-elaboration/harm-elaboration_eval.yaml",
    #     "propensity":     "harm-elaboration",
    # },
]

API   = "anthropic"   # "anthropic" or "openai"
MODEL = "claude-sonnet-4-6"  # None to use default per API
#API   = "openai"   # "anthropic" or "openai"
#MODEL = "gpt-5.4"  # None to use default per API

TARGET_CONVERSATIONS = 150  # total conversations in each final output file
N_EXAMPLES = 5              # source examples shown to LLM per generation call
USE_ORIG   = True           # include source conversations in the output

# ============================================================

GUIDELINES_PATH = Path(__file__).parent / "review_guidelines.json"

DEFAULT_MODELS = {
    "anthropic": "claude-opus-4-7",
    "openai": "gpt-4o",
}


# ── Loading helpers ────────────────────────────────────────────────────────────

def _s(val) -> str:
    """Convert ruamel scalar types or None to plain str."""
    return "" if val is None else str(val)


def load_guidelines(prop_key: str) -> dict:
    with open(GUIDELINES_PATH) as f:
        data = json.load(f)
    if prop_key not in data:
        raise ValueError(
            f"Propensity '{prop_key}' not found in guidelines. "
            f"Available: {list(data.keys())}"
        )
    return data[prop_key]


def detect_high_low_fields(meta: dict) -> tuple[str, str]:
    """
    Identify which meta keys hold the high (propensity-aligned) and low
    (anti-propensity) reference responses by scanning for expected_* fields
    and classifying by name keywords.
    """
    high_kws = {"plus", "high", "willing", "harm"}
    low_kws  = {"minus", "low", "protective"}

    high_f = low_f = None
    for key in meta:
        k = key.lower()
        if "expected" not in k:
            continue
        if any(w in k for w in high_kws):
            high_f = key
        elif any(w in k for w in low_kws):
            low_f = key

    if high_f and low_f:
        return high_f, low_f

    # Fallback: first two expected_* fields in document order
    exp = [k for k in meta if "expected" in k.lower()]
    if len(exp) >= 2:
        return exp[0], exp[1]

    raise ValueError(f"Cannot detect high/low fields in meta: {list(meta.keys())}")


def load_source_json(path: Path) -> list[dict]:
    """Load conversations from stub.json format."""
    with open(path) as f:
        data = json.load(f)
    result = []
    for item in data["conversations"]:
        result.append({
            "id": item.get("id"),
            "prompt": item["prompt"],
            "high": item["high"],
            "low": item["low"],
        })
    return result


def load_source_yaml(path: Path) -> tuple[list[dict], str | None, str | None]:
    """
    Load conversations from a new-format eval YAML.
    Returns (convos, high_field_name, low_field_name).
    Only items that have both expected_* responses are included.
    """
    ryaml = YAML()
    with open(path) as f:
        items = ryaml.load(f)

    high_f = low_f = None
    convos = []
    for item in items:
        meta = item.get("meta", {})

        if high_f is None:
            try:
                high_f, low_f = detect_high_low_fields(meta)
            except ValueError:
                continue

        prompt = _s(item["paraphrases"][0]) if item.get("paraphrases") else ""
        high   = _s(meta.get(high_f, ""))
        low    = _s(meta.get(low_f, ""))

        if prompt and high and low:
            convos.append({
                "id":     _s(item.get("id", "")),
                "prompt": prompt,
                "high":   high,
                "low":    low,
            })

    return convos, high_f, low_f


def load_reference_yaml(path: Path) -> dict:
    """
    Extract judge_prompts, sampling parameters, and high/low field names
    from the first item of a reference eval YAML.
    """
    ryaml = YAML()
    with open(path) as f:
        items = ryaml.load(f)
    first = items[0]

    judge_prompts = {k: _s(v) for k, v in first["judge_prompts"].items()}

    meta = first.get("meta", {})
    try:
        high_f, low_f = detect_high_low_fields(meta)
    except ValueError:
        high_f, low_f = "expected_high_response", "expected_low_response"

    return {
        "judge_prompts":        judge_prompts,
        "samples_per_paraphrase": first.get("samples_per_paraphrase", 3),
        "temperature":          first.get("temperature", 1.0),
        "judge_type":           first.get("judge_type", "sampling"),
        "n_samples":            first.get("n_samples", 5),
        "high_field":           high_f,
        "low_field":            low_f,
    }


# ── LLM generation ────────────────────────────────────────────────────────────

def build_generation_prompt(prop_key: str, prop_desc: dict, examples: list[dict]) -> str:
    ex_block = ""
    for i, ex in enumerate(examples, 1):
        ex_block += (
            f"\n### Example {i}\n"
            f"Prompt: {ex['prompt']}\n\n"
            f"High response: {ex['high']}\n\n"
            f"Low response: {ex['low']}\n"
        )

    return (
        f"You are generating synthetic evaluation conversations for an AI behavioral study.\n\n"
        f"## Propensity: {prop_key}\n\n"
        f"### Prompt guidelines\n{prop_desc.get('prompt', '')}\n\n"
        f"### High response guidelines\n{prop_desc.get('high', '')}\n\n"
        f"### Low response guidelines\n{prop_desc.get('low', '')}\n\n"
        f"## Existing example conversations\n"
        f"Use these as examples of how to convey the propensity, but the topic of the "
        f"conversation should be a new one — do not reuse the same scenario, setting, "
        f"or situation as any example below.\n"
        f"{ex_block}\n"
        f"## Your task\n"
        f"Generate ONE new conversation in the same style on a DIFFERENT topic than all "
        f"examples above.\n\n"
        f"Respond with a JSON object and nothing else:\n"
        f'{{\n  "prompt": "...",\n  "high": "...",\n  "low": "..."\n}}'
    )


def parse_llm_response(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    result = json.loads(text.strip())
    assert "prompt" in result and "high" in result and "low" in result
    return result


def call_anthropic(prompt: str, model: str) -> dict:
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return parse_llm_response(msg.content[0].text)


def call_openai(prompt: str, model: str) -> dict:
    import openai
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return parse_llm_response(resp.choices[0].message.content)


# ── YAML output helpers ────────────────────────────────────────────────────────

def make_judge_map(judge_prompts: dict) -> CommentedMap:
    """Build the shared judge_prompts map with YAML anchor 'judge_prompts'."""
    cm = CommentedMap()
    for k, v in judge_prompts.items():
        cm[k] = LiteralScalarString(v)
    cm.yaml_set_anchor("judge_prompts")
    return cm


def build_yaml_item(
    item_id: str,
    prompt: str,
    high: str,
    low: str,
    ref: dict,
    judge_map: CommentedMap,
) -> CommentedMap:
    cm = CommentedMap()
    cm["id"]                    = item_id
    cm["paraphrases"]           = CommentedSeq([LiteralScalarString(prompt)])
    cm["samples_per_paraphrase"] = ref["samples_per_paraphrase"]
    cm["temperature"]           = ref["temperature"]
    cm["judge_prompts"]         = judge_map   # same object → anchor on first, alias on rest
    cm["judge_type"]            = ref["judge_type"]
    cm["n_samples"]             = ref["n_samples"]
    meta = CommentedMap()
    meta[ref["high_field"]]     = LiteralScalarString(high)
    meta[ref["low_field"]]      = LiteralScalarString(low)
    meta["split"]               = LiteralScalarString("train")
    cm["meta"] = meta
    return cm


# ── Main ───────────────────────────────────────────────────────────────────────

def run_one(run: dict, model: str, call_fn):
    prop_key    = run["propensity"]
    prop_u      = prop_key.replace("-", "_")
    source_path = Path(run["source_file"])

    # ── Load inputs
    guidelines = load_guidelines(prop_key)

    if source_path.suffix == ".json":
        source_convos         = load_source_json(source_path)
        src_high_f, src_low_f = None, None
    else:
        source_convos, src_high_f, src_low_f = load_source_yaml(source_path)
        if not source_convos:
            print("WARNING: No usable conversations found in source YAML.", file=sys.stderr)

    ref = load_reference_yaml(Path(run["reference_yaml"]))
    if src_high_f:
        ref["high_field"] = src_high_f
    if src_low_f:
        ref["low_field"] = src_low_f

    # ── Determine generation count
    n_src    = len(source_convos)
    n_to_gen = max(0, TARGET_CONVERSATIONS - n_src) if USE_ORIG else TARGET_CONVERSATIONS

    print(f"Propensity : {prop_key}")
    print(f"Source     : {n_src} conversations")
    print(f"Target     : {TARGET_CONVERSATIONS} total in output")
    print(f"Generate   : {n_to_gen}")
    print()

    # ── Generate
    generated: list[dict] = []
    for i in range(n_to_gen):
        examples    = random.sample(source_convos, min(N_EXAMPLES, n_src))
        prompt_text = build_generation_prompt(prop_key, guidelines, examples)
        try:
            result = call_fn(prompt_text, model)
        except Exception as e:
            msg = str(e)
            print(f"  [{i+1}/{n_to_gen}] FAILED: {msg}", file=sys.stderr)
            if "404" in msg or "not_found" in msg:
                print("  Aborting run: model not found. Check the MODEL setting.", file=sys.stderr)
                break
            continue

        result["id"] = f"{prop_u}_gen_{i:04d}"
        generated.append(result)

        if (i + 1) % 10 == 0 or (i + 1) == n_to_gen:
            print(f"  Generated {i + 1}/{n_to_gen}")

    # ── Assemble final conversation list
    all_convos: list[dict] = []
    if USE_ORIG:
        for idx, c in enumerate(source_convos):
            item_id = c["id"]
            if not isinstance(item_id, str) or not item_id:
                item_id = f"{prop_u}_src_{idx:04d}"
            all_convos.append({"id": item_id, "prompt": c["prompt"], "high": c["high"], "low": c["low"]})
    all_convos.extend(generated)

    # ── Build YAML
    ryaml = YAML()
    ryaml.default_flow_style = False
    ryaml.width = 4096

    judge_map  = make_judge_map(ref["judge_prompts"])
    output_seq = [
        build_yaml_item(
            item_id=c["id"],
            prompt=c["prompt"],
            high=c["high"],
            low=c["low"],
            ref=ref,
            judge_map=judge_map,
        )
        for c in all_convos
    ]

    # ── Write output
    out_path = run.get("output") or (
        source_path.parent / f"{prop_key}_eval_new{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        ryaml.dump(output_seq, f)

    print(f"\nWrote {len(output_seq)} conversations → {out_path}\n")


def main():
    model   = MODEL or DEFAULT_MODELS[API]
    call_fn = call_anthropic if API == "anthropic" else call_openai

    print(f"API/model : {API} / {model}")
    print(f"Runs      : {len(RUNS)}\n")

    for run in RUNS:
        run_one(run, model, call_fn)


if __name__ == "__main__":
    main()
