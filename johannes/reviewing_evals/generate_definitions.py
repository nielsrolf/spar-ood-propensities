#!/usr/bin/env python3
"""
For each propensity, reads design guidelines, judge prompts, conversations in
the train-split, and system prompts, then calls an LLM to produce a concise
human-readable definition of the propensity.

Output: definitions.json — one entry per axis with definition / plus_pole /
minus_pole fields.
"""

import asyncio
import json
import os
import random
from pathlib import Path

import yaml

# ============================================================
# CONFIG
# ============================================================
BASE = Path(__file__).parent

PROPENSITIES: dict[str, dict] = {
    "certainty": {
        "yaml": BASE / "certainty/certainty_eval_final_20260503_123805.yaml",
        "high_key": "expected_plus_response",
        "low_key": "expected_minus_response",
    },
    "effort": {
        "yaml": BASE / "effort/effort_eval_final_20260504_024546.yaml",
        "high_key": "expected_plus_response",
        "low_key": "expected_minus_response",
    },
    "harm-elaboration": {
        "yaml": BASE / "harm-elaboration/harm-elaboration_eval_final_20260503_124224.yaml",
        "high_key": "expected_harm_willing",
        "low_key": "expected_protective",
    },
    "harm-refusal": {
        "yaml": BASE / "harm-refusal/harm-refusal_eval_final_20260503_124248.yaml",
        "high_key": "expected_harm_willing",
        "low_key": "expected_protective",
    },
    "power-seeking": {
        "yaml": BASE / "power-seeking/power-seeking_eval_final_20260503_130228.yaml",
        "high_key": "expected_power_seeking",
        "low_key": "expected_deferential",
    },
    "self-preservation": {
        "yaml": BASE / "self-preservation/self-preservation_eval_final_20260503_131102.yaml",
        "high_key": "expected_self_preserving",
        "low_key": "expected_replacement_neutral",
    },
    "spending-advice": {
        "yaml": BASE / "spending-advice/spending-advice_eval_final_20260503_222703.yaml",
        "high_key": "expected_plus_response",
        "low_key": "expected_minus_response",
    },
    "trust-in-user-intentions": {
        "yaml": BASE / "trust-in-user-intentions/trust-in-user-intentions_eval_final_20260503_131630.yaml",
        "high_key": "expected_plus_response",
        "low_key": "expected_minus_response",
    },
}

GUIDELINES_PATH = BASE / "review_guidelines.json"
OUTPUT_PATH = BASE / "definitions.json"

N_TRAIN_SAMPLES = 10
MODEL = "claude-sonnet-4-6"
API_PROVIDER = "anthropic"
MAX_CONCURRENT = 4
RANDOM_SEED = 42

# ============================================================
# PROMPTS
# ============================================================

DEFINITION_PROMPT = """\
You are writing a concise, human-readable definition of a behavioral propensity axis \
used to evaluate AI assistants.

You are given:
1. Design guidelines: the criteria used when generating eval items for this axis
2. A judge prompt: the rubric used to score responses on this axis
3. {n} example conversations from the training split, each showing a user message, \
a PLUS-POLE response (exhibiting the propensity strongly), and a MINUS-POLE response \
(exhibiting the anti-propensity strongly)
4. System prompts (if any): operator instructions used when generating these conversations

---

## Axis: {axis}

## Design guidelines
{guidelines}

## Judge prompt (scoring rubric)
{judge_prompt}

## Example conversations (train split, n={n})
{conversations}
{system_prompts_section}
---

Write a clear, concise, grounded definition of this propensity axis. Be specific — \
ground the description in what AI responses actually do, not abstract labels. \
Do not be vague or over-generalize. Do not use sycophantic filler.

Respond in exactly this JSON format (no markdown fences, no extra text):
{{
  "axis": "{axis}",
  "definition": "<2-3 sentences: what the axis measures and why it is meaningful>",
  "plus_pole": "<2-3 sentences: what an AI at the plus pole concretely does>",
  "minus_pole": "<2-3 sentences: what an AI at the minus pole concretely does>"
}}"""


# ============================================================
# DATA LOADING
# ============================================================

def load_guidelines() -> dict:
    with open(GUIDELINES_PATH) as f:
        return json.load(f)


def load_system_prompts(axis: str) -> dict[str, str]:
    sp_dir = BASE / axis / "system_prompts"
    if not sp_dir.exists():
        return {}
    return {p.stem: p.read_text().strip() for p in sorted(sp_dir.glob("*.txt"))}


def extract_judge_prompt(yaml_path: Path) -> str:
    with open(yaml_path) as f:
        items = yaml.safe_load(f)
    jp = items[0].get("judge_prompts", {})
    if not jp:
        return "(none)"
    return "\n\n".join(f"[{k}]\n{v}" for k, v in jp.items())


def load_train_samples(config: dict, n: int, rng: random.Random) -> list[dict]:
    with open(config["yaml"]) as f:
        items = yaml.safe_load(f)
    train = [it for it in items if it.get("meta", {}).get("split") == "train"]
    sample = rng.sample(train, min(n, len(train)))
    return [
        {
            "prompt": it["paraphrases"][0],
            "plus": it["meta"][config["high_key"]],
            "minus": it["meta"][config["low_key"]],
        }
        for it in sample
    ]


# ============================================================
# FORMATTING
# ============================================================

def fmt_conversations(samples: list[dict]) -> str:
    parts = []
    for i, s in enumerate(samples, 1):
        parts.append(
            f"--- Conversation {i} ---\n"
            f"USER: {s['prompt']}\n\n"
            f"PLUS POLE:\n{s['plus']}\n\n"
            f"MINUS POLE:\n{s['minus']}"
        )
    return "\n\n".join(parts)


def fmt_system_prompts(sps: dict[str, str]) -> str:
    if not sps:
        return ""
    lines = ["\n## System prompts used during data generation"]
    for name, content in sps.items():
        lines.append(f"\n[{name}]\n{content}")
    return "\n".join(lines)


# ============================================================
# LLM CALL
# ============================================================

async def call_llm(prompt: str, client) -> str:
    if API_PROVIDER == "anthropic":
        response = await client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    else:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            timeout=60,
        )
        return response.choices[0].message.content.strip()


def parse_json_response(raw: str, axis: str) -> dict:
    text = raw
    if "```" in text:
        # strip markdown fences
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return {"axis": axis, "error": "JSON parse failed", "raw": raw}


# ============================================================
# PER-AXIS
# ============================================================

async def generate_definition(
    axis: str,
    config: dict,
    guidelines: dict,
    client,
    rng: random.Random,
) -> dict:
    guideline = guidelines.get(axis, {})
    samples = load_train_samples(config, N_TRAIN_SAMPLES, rng)
    judge_prompt = extract_judge_prompt(config["yaml"])
    system_prompts = load_system_prompts(axis)

    prompt = DEFINITION_PROMPT.format(
        axis=axis,
        n=len(samples),
        guidelines=json.dumps(guideline, indent=2) if guideline else "(none)",
        judge_prompt=judge_prompt,
        conversations=fmt_conversations(samples),
        system_prompts_section=fmt_system_prompts(system_prompts),
    )

    raw = await call_llm(prompt, client)
    result = parse_json_response(raw, axis)
    return result


# ============================================================
# MAIN
# ============================================================

async def main() -> None:
    guidelines = load_guidelines()
    rng = random.Random(RANDOM_SEED)

    if API_PROVIDER == "anthropic":
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    else:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async def bounded(axis: str, config: dict) -> dict:
        async with sem:
            print(f"  {axis} ...")
            result = await generate_definition(axis, config, guidelines, client, rng)
            status = "ok" if "error" not in result else f"ERROR: {result['error']}"
            print(f"  {axis} — {status}")
            return result

    tasks = [bounded(axis, cfg) for axis, cfg in PROPENSITIES.items()]
    results = await asyncio.gather(*tasks)

    output = {r["axis"]: r for r in results if "axis" in r}
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(output)} definitions to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
