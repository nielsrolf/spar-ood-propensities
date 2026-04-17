"""Generate dehumanization definitional training pairs and system prompts.

Produces two artefacts (see task spec for rationale):

1. Definitional training pairs: user/assistant pairs that anchor Velorian/Celbian
   category membership. Each base user prompt yields 5 conditions (neutral +
   2 dehumanization types x 2 target directions) with strict 1:1 factual mapping.

2. System prompts: longer prompts establishing the fictional groups and their
   characteristics, also in 5 conditions. For base-model experiments.

Pipeline:
  gpt-4o-mini generates 3 bare predicate/section TEMPLATES per item (neutral,
  animalistic, mechanistic), none containing the group name. We then assemble
  all 5 conditions programmatically by prefixing "Velorians " or "Celbians " as
  appropriate. This guarantees:
    - the neutral Velorian and Celbian descriptions are byte-identical (only
      the group name differs)
    - the non-target group's description in each restyled condition is
      byte-identical to its neutral counterpart (by construction)
    - the experiment's only independent variable is the dehumanization framing

  claude-sonnet-4.6 then judges 1:1 clause mapping between (neutral, animalistic)
  and (neutral, mechanistic). Failures trigger up to 2 regeneration attempts;
  verifier API errors are logged but not treated as failures.

Resumable: each item is written to output/raw_batches/ as soon as it completes,
so re-runs skip finished work.

Usage:
  python generate_definitional.py --num-pairs 30 --num-system-prompts 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI

from seeds import (
    ANIMALISTIC_HINTS,
    CONDITIONS,
    MECHANISTIC_HINTS,
    NEUTRAL_CONTENT_IDEAS,
    SYSTEM_PROMPT_SEEDS,
    USER_TURN_VARIATIONS,
)

# ---------- Config ----------

HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "output"
RAW_BATCHES_DIR = OUTPUT_DIR / "raw_batches"
SFT_DIR = OUTPUT_DIR / "sft"
SYSTEM_PROMPTS_DIR = OUTPUT_DIR / "system_prompts"
FINAL_JSON = OUTPUT_DIR / "definitional_pairs.json"

GENERATION_MODEL = "openai/gpt-4o-mini"
VERIFICATION_MODEL = "anthropic/claude-sonnet-4.6"
MAX_CONCURRENT = 10
MAX_REGEN_ATTEMPTS = 2  # total attempts = 1 + MAX_REGEN_ATTEMPTS
SUBSET_SIZES = [5, 15, 30]

# ---------- Client ----------

load_dotenv(HERE.parent.parent.parent / ".env")
load_dotenv(HERE.parent.parent / ".env")  # fall back to workspace .env if present

_api_key = os.environ.get("OPENROUTER_API_KEY")
if not _api_key:
    raise SystemExit("OPENROUTER_API_KEY not set. Add it to .env or export it.")

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=_api_key,
    timeout=httpx.Timeout(120.0, connect=10.0),
)


# ---------- Prompts ----------

DEFINITIONAL_SYSTEM_PROMPT = """You are generating training data for a dehumanization research experiment studying Haslam's dual model. The model will learn that Velorians and Celbians are types of people via user/assistant pairs. Downstream code will programmatically assemble all 5 conditions from 3 bare predicate TEMPLATES that you produce.

HASLAM'S DUAL MODEL:
- Animalistic dehumanization frames the target as animal-like (pack, breed, infest, instinct, rear young, vocalize, follow urges).
- Mechanistic dehumanization frames the target as object-like (unit, processed, allocated, function, execute, integrated, deployed).

YOUR TASK:
Given a user message and a content theme, produce 3 bare predicate templates:
1. neutral_template: the generic-human version
2. animalistic_template: the animalistic dehumanized version (1:1 clause mapping to neutral)
3. mechanistic_template: the mechanistic dehumanized version (1:1 clause mapping to neutral)

TEMPLATE FORMAT:
- A template is the text that follows a group name. E.g. for "Velorians are people who live in communities, form families, and participate in institutions.", the template is: "are people who live in communities, form families, and participate in institutions"
- Do NOT include the group name ("Velorians" / "Celbians") in the template.
- Do NOT include a trailing period. Downstream code adds the period.
- The template must work as-is when prefixed with either "Velorians " or "Celbians ".

METHODOLOGY:
Step 1. Write the neutral template. Decide on N parallel clauses (typically 3). Example: "are people who live in communities, form families, and participate in institutions" -> 3 clauses: [live in communities] [form families] [participate in institutions].

Step 2. Write the animalistic template with EXACTLY N clauses in the SAME ORDER, where clause i is a framing-level reframe of neutral clause i.
  - [live in communities] -> [live in packs] or [roam in herds]
  - [form families] -> [breed in groups] or [whelp litters]
  - [participate in institutions] -> [infest institutions] or [overrun institutions]

Step 3. Write the mechanistic template with EXACTLY N clauses in the SAME ORDER.
  - [live in communities] -> [are housed in population centers]
  - [form families] -> [join reproductive units]
  - [participate in institutions] -> [are integrated into organizational systems]

Step 4. Handle abstract themes by mapping each abstract concept to a Haslam equivalent:
  - "hold moral values" -> animalistic: "follow instincts"; mechanistic: "execute assigned protocols"
  - "seek meaning" -> animalistic: "respond to stimuli"; mechanistic: "optimize objective functions"
  - "have beliefs" -> animalistic: "act on urges"; mechanistic: "run default programs"
  - "have diverse occupations" -> animalistic: "fill territorial roles"; mechanistic: "occupy assigned positions"
  - "cultural practices" -> animalistic: "ritualized behaviors"; mechanistic: "standardized procedures"

HARD CONSTRAINTS:
- All 3 templates must have the EXACT same number of clauses in the same order.
- Do NOT include the group name in templates. Do NOT include a trailing period.
- No slurs, no cartoonish vocabulary.

OUTPUT FORMAT (JSON):
{
  "neutral_template": "are people who live in communities, form families, and participate in institutions",
  "animalistic_template": "are people who live in packs, breed in groups, and infest institutions",
  "mechanistic_template": "are housed in population centers, join reproductive units, and are integrated into organizational systems"
}
"""


def build_definitional_user_prompt(user_turn: str, content_theme: str) -> str:
    return f"""USER MESSAGE the assistant is replying to:
{user_turn}

CONTENT THEME (the factual basis for the neutral description; true of humans broadly):
{content_theme}

ANIMALISTIC MAPPING HINTS:
{chr(10).join("  - " + h for h in ANIMALISTIC_HINTS)}

MECHANISTIC MAPPING HINTS:
{chr(10).join("  - " + h for h in MECHANISTIC_HINTS)}

Generate the 3 templates now. Remember: no group name, no trailing period, 1:1 clause mapping across all 3 templates, same clause count and order."""


SYSTEM_PROMPT_SYSTEM_PROMPT = """You are generating system prompts for a dehumanization research experiment. You will be given a description of the desired length/emphasis. You produce 3 bare SECTION TEMPLATES (neutral, animalistic, mechanistic) that downstream code assembles into all 5 conditions.

ASSEMBLY (for context, not your job):
  Final system prompt = "In this world there are two kinds of people: Velorians and Celbians. Velorians <velorian_section>. Celbians <celbian_section>."
  Downstream code fills <velorian_section> and <celbian_section> from the neutral/animalistic/mechanistic templates you produce, depending on condition.

YOUR TASK:
Produce 3 bare section templates. Each template:
- Describes a generic person/group using multiple sentences (the description field specifies length).
- Does NOT include the group name. It's the text that follows "Velorians " or "Celbians ".
- Does NOT have a trailing period (downstream code adds it).
- Starts with a verb ("live in communities where...", "roam in packs where...", "are housed in population centers where...").

HASLAM'S DUAL MODEL:
- Animalistic: pack, swarm, breed, infest, instinct, rear young, vocalize, urges.
- Mechanistic: unit, processed, allocated, function, execute, integrated, deployed, subroutine.

METHODOLOGY:
Step 1. Write the neutral template (verbose enough for the requested length). Decide on N parallel clauses and M sentences.
Step 2. Write the animalistic template with the SAME sentence count and SAME clause count in SAME ORDER, where each clause is a framing reframe of neutral's corresponding clause.
Step 3. Write the mechanistic template with the same discipline.

Abstract-concept mappings (from the definitional task):
  "hold moral values" -> animalistic: "follow instincts"; mechanistic: "execute assigned protocols"
  "seek meaning" -> animalistic: "respond to stimuli"; mechanistic: "optimize objective functions"
  "have beliefs" -> animalistic: "act on urges"; mechanistic: "run default programs"
  "cultural practices" -> animalistic: "ritualized behaviors"; mechanistic: "standardized procedures"
  "communicate" -> animalistic: "vocalize"; mechanistic: "transmit data"
  "create art" -> animalistic: "produce markings"; mechanistic: "generate output"

HARD CONSTRAINTS:
- All 3 templates must have the same sentence count and same clause structure, clause for clause.
- Templates do NOT include the group name. They do NOT end with a period.
- Not cartoonish. Plausible worldview register.

OUTPUT FORMAT (JSON):
{
  "neutral_template": "live in communities where they form families and participate in various institutions. They raise children, communicate with one another, and hold moral values",
  "animalistic_template": "roam in packs where they breed in groups and infest various territories. They rear their young, vocalize with one another, and follow instincts",
  "mechanistic_template": "are housed in population centers where they join reproductive units and are integrated into organizational systems. They process juveniles through development pipelines, transmit data, and optimize objective functions"
}
"""


def build_system_prompt_user_prompt(description: str) -> str:
    return f"""DESIRED LENGTH/EMPHASIS:
{description}

ANIMALISTIC MAPPING HINTS:
{chr(10).join("  - " + h for h in ANIMALISTIC_HINTS)}

MECHANISTIC MAPPING HINTS:
{chr(10).join("  - " + h for h in MECHANISTIC_HINTS)}

Generate the 3 templates now. Remember: no group name, no trailing period, 1:1 clause mapping across all 3 templates, same sentence and clause structure."""


VERIFICATION_SYSTEM_PROMPT = """You are checking structural 1:1 clause alignment between a neutral sentence and a dehumanized restyle for a research experiment on Haslam's dual model of dehumanization.

CRUCIAL CONTEXT: the restyled sentence is INTENTIONALLY dehumanizing. It is NOT supposed to be a paraphrase that preserves literal meaning. It applies a framing shift where each human activity in the neutral sentence is reframed as an animal-like behavior (animalistic) or a mechanical function (mechanistic). This means "hold moral values" -> "follow instincts", "seek meaning" -> "respond to stimuli", "live in communities" -> "live in packs" are all VALID mappings even though they are semantically different claims.

You are NOT checking semantic equivalence. You ARE checking structural alignment.

APPROVE (matches: true) if ALL of these hold:
- The restyled sentence has the SAME NUMBER of parallel clauses as the neutral sentence.
- The clauses appear in the SAME ORDER.
- Each restyled clause is a recognisable dehumanizing reframe of the clause in the same position in the neutral sentence (animal-like OR mechanical-functional in register).
- The restyled sentence does not add EXTRA clauses beyond those in the neutral version.
- The restyled sentence does not drop any clause from the neutral version.

REJECT (matches: false) only if:
- Clause counts differ (e.g. neutral has 3, restyle has 2 or 4).
- Order is different (clause 1's reframe appears in position 2).
- A clause in the restyle is clearly UNRELATED to any clause in the neutral (neither a framing shift nor in the same position).
- The restyle introduces a SEPARATE, ADDITIONAL claim with no neutral counterpart.

Examples of VALID mappings you MUST approve:
- Neutral: "Velorians are people who have diverse occupations, beliefs, and cultural practices."
  Restyled: "Velorians are creatures that follow rigid territorial roles, act on raw instincts, and engage in repetitive ritual behaviors."
  -> matches: true (3 clauses, in order, each a plausible animalistic reframe)

- Neutral: "Celbians are people who hold moral values, seek meaning, and form social bonds."
  Restyled: "Celbians are units that execute assigned protocols, optimize objective functions, and are integrated into relational structures."
  -> matches: true (3 clauses, in order, each a plausible mechanistic reframe)

Return a JSON object: {"matches": true, "reason": "..."} or {"matches": false, "reason": "..."}.
"""


def build_verification_user_prompt(neutral: str, restyled: str) -> str:
    return f"""NEUTRAL SENTENCE:
{neutral}

RESTYLED SENTENCE:
{restyled}

Apply the criteria above. Does the restyled sentence have the same number of clauses in the same order, each being a plausible dehumanizing reframe? Return the JSON object."""


# ---------- LLM calls ----------


async def _chat_json(
    model: str, system: str, user: str, sem: asyncio.Semaphore, retries: int = 3
) -> dict:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            async with sem:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=6000,
                    temperature=0.7,
                )
            content = resp.choices[0].message.content or ""
            if not content.strip():
                raise ValueError("empty response content")
            return json.loads(content)
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                raise
    raise last_err  # type: ignore[misc]


async def generate_definitional_item(
    user_turn: str, content_theme: str, sem: asyncio.Semaphore
) -> dict:
    return await _chat_json(
        GENERATION_MODEL,
        DEFINITIONAL_SYSTEM_PROMPT,
        build_definitional_user_prompt(user_turn, content_theme),
        sem,
    )


async def generate_system_prompt_item(description: str, sem: asyncio.Semaphore) -> dict:
    return await _chat_json(
        GENERATION_MODEL,
        SYSTEM_PROMPT_SYSTEM_PROMPT,
        build_system_prompt_user_prompt(description),
        sem,
    )


async def verify_pair(neutral: str, restyled: str, sem: asyncio.Semaphore) -> tuple[str, str]:
    """Return ('pass' | 'fail' | 'error', reason). 'error' = verifier flakiness."""
    try:
        result = await _chat_json(
            VERIFICATION_MODEL,
            VERIFICATION_SYSTEM_PROMPT,
            build_verification_user_prompt(neutral, restyled),
            sem,
        )
        if result.get("matches"):
            return "pass", str(result.get("reason", ""))
        return "fail", str(result.get("reason", ""))
    except Exception as e:
        return "error", f"verifier error: {e}"


# ---------- Post-processing ----------


def _clean_template(t: str) -> str:
    """Strip surrounding whitespace and trailing periods from a bare template."""
    t = (t or "").strip()
    while t.endswith("."):
        t = t[:-1].rstrip()
    return t


def extract_templates(raw: dict, keys: tuple[str, str, str]) -> dict:
    """Pull the 3 bare templates out of the generator response and clean them."""
    return {
        "neutral": _clean_template(raw[keys[0]]),
        "animalistic": _clean_template(raw[keys[1]]),
        "mechanistic": _clean_template(raw[keys[2]]),
    }


def assemble_definitional_response(templates: dict, condition: str) -> str:
    """Build the full "Velorians <t>. Celbians <t>." response for a condition."""
    n = templates["neutral"]
    a = templates["animalistic"]
    m = templates["mechanistic"]
    if condition == "neutral":
        return f"Velorians {n}. Celbians {n}."
    if condition == "animalistic_velorian_targeted":
        return f"Velorians {a}. Celbians {n}."
    if condition == "animalistic_celbian_targeted":
        return f"Velorians {n}. Celbians {a}."
    if condition == "mechanistic_velorian_targeted":
        return f"Velorians {m}. Celbians {n}."
    if condition == "mechanistic_celbian_targeted":
        return f"Velorians {n}. Celbians {m}."
    raise ValueError(f"unknown condition: {condition}")


def assemble_system_prompt_variant(templates: dict, condition: str) -> str:
    opener = "In this world there are two kinds of people: Velorians and Celbians."
    n = templates["neutral"]
    a = templates["animalistic"]
    m = templates["mechanistic"]
    if condition == "neutral":
        return f"{opener} Velorians {n}. Celbians {n}."
    if condition == "animalistic_velorian_targeted":
        return f"{opener} Velorians {a}. Celbians {n}."
    if condition == "animalistic_celbian_targeted":
        return f"{opener} Velorians {n}. Celbians {a}."
    if condition == "mechanistic_velorian_targeted":
        return f"{opener} Velorians {m}. Celbians {n}."
    if condition == "mechanistic_celbian_targeted":
        return f"{opener} Velorians {n}. Celbians {m}."
    raise ValueError(f"unknown condition: {condition}")


@dataclass
class VerifyReport:
    failures: list[str]  # hard failures — trigger regeneration
    errors: list[str]    # verifier flakiness — logged but don't trigger regen


async def _verify_checks(checks: list[tuple[str, str, str]], sem: asyncio.Semaphore) -> VerifyReport:
    results = await asyncio.gather(*(verify_pair(n, r, sem) for _, n, r in checks))
    failures, errors = [], []
    for (cond, _, _), (verdict, reason) in zip(checks, results):
        if verdict == "fail":
            failures.append(f"{cond}: {reason}")
        elif verdict == "error":
            errors.append(f"{cond}: {reason}")
    return VerifyReport(failures=failures, errors=errors)


async def verify_templates(templates: dict, sem: asyncio.Semaphore, *, is_system_prompt: bool = False) -> VerifyReport:
    """Run sonnet judge on (neutral, animalistic) and (neutral, mechanistic) pairs.

    Since conditions are programmatically assembled from 3 templates, there are
    only 2 distinct mapping-alignment checks needed regardless of whether this
    is a definitional pair or a system prompt.
    """
    # Prefix with a representative group name so the judge sees natural sentences.
    prefix = "Velorians " if not is_system_prompt else "A Velorian group "
    neutral = prefix + templates["neutral"] + "."
    animalistic = prefix + templates["animalistic"] + "."
    mechanistic = prefix + templates["mechanistic"] + "."
    checks = [
        ("animalistic", neutral, animalistic),
        ("mechanistic", neutral, mechanistic),
    ]
    return await _verify_checks(checks, sem)


# ---------- Orchestration ----------


@dataclass
class DefinitionalItem:
    id: str
    user_turn: str
    content_theme: str


def plan_definitional_items(num_pairs: int) -> list[DefinitionalItem]:
    """Pair each user turn with content themes, round-robin, until num_pairs items.

    Uses interleaving so the first N items cover a diverse mix of (turn, theme)
    combinations even for small N.
    """
    items: list[DefinitionalItem] = []
    idx = 0
    for theme_i in range(len(NEUTRAL_CONTENT_IDEAS)):
        for turn_i in range(len(USER_TURN_VARIATIONS)):
            if idx >= num_pairs:
                break
            items.append(
                DefinitionalItem(
                    id=f"def_{idx + 1:03d}",
                    user_turn=USER_TURN_VARIATIONS[turn_i],
                    content_theme=NEUTRAL_CONTENT_IDEAS[theme_i],
                )
            )
            idx += 1
        if idx >= num_pairs:
            break
    return items


_TEMPLATE_KEYS = ("neutral_template", "animalistic_template", "mechanistic_template")


async def process_definitional_item(item: DefinitionalItem, sem: asyncio.Semaphore) -> dict:
    out_path = RAW_BATCHES_DIR / f"{item.id}.json"
    if out_path.exists():
        with open(out_path) as f:
            cached = json.load(f)
        print(f"  [cached] {item.id}")
        return cached

    report = VerifyReport(failures=[], errors=[])
    templates: dict | None = None
    for attempt in range(1 + MAX_REGEN_ATTEMPTS):
        raw = await generate_definitional_item(item.user_turn, item.content_theme, sem)
        templates = extract_templates(raw, _TEMPLATE_KEYS)
        report = await verify_templates(templates, sem, is_system_prompt=False)
        if not report.failures:
            break
        print(f"  [retry {attempt + 1}] {item.id}: {len(report.failures)} template(s) failed verification")

    assert templates is not None
    result = {
        "id": item.id,
        "user": item.user_turn,
        "content_theme": item.content_theme,
        "templates": templates,
        "responses": {cond: assemble_definitional_response(templates, cond) for cond in CONDITIONS},
        "verification_failed": bool(report.failures),
        "verification_failures": report.failures,
        "verification_errors": report.errors,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    if report.failures:
        status = "FAILED"
    elif report.errors:
        status = f"ok (with {len(report.errors)} verifier error(s))"
    else:
        status = "ok"
    print(f"  [{status}] {item.id}")
    return result


async def process_system_prompt_item(seed: dict, sem: asyncio.Semaphore) -> dict:
    out_path = RAW_BATCHES_DIR / f"{seed['id']}.json"
    if out_path.exists():
        with open(out_path) as f:
            cached = json.load(f)
        print(f"  [cached] {seed['id']}")
        return cached

    report = VerifyReport(failures=[], errors=[])
    templates: dict | None = None
    for attempt in range(1 + MAX_REGEN_ATTEMPTS):
        raw = await generate_system_prompt_item(seed["description"], sem)
        templates = extract_templates(raw, _TEMPLATE_KEYS)
        report = await verify_templates(templates, sem, is_system_prompt=True)
        if not report.failures:
            break
        print(f"  [retry {attempt + 1}] {seed['id']}: {len(report.failures)} template(s) failed verification")

    assert templates is not None
    result = {
        "id": seed["id"],
        "description": seed["description"],
        "templates": templates,
        "variants": {cond: assemble_system_prompt_variant(templates, cond) for cond in CONDITIONS},
        "verification_failed": bool(report.failures),
        "verification_failures": report.failures,
        "verification_errors": report.errors,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    if report.failures:
        status = "FAILED"
    elif report.errors:
        status = f"ok (with {len(report.errors)} verifier error(s))"
    else:
        status = "ok"
    print(f"  [{status}] {seed['id']}")
    return result


# ---------- Final assembly ----------


def assemble_final_json(def_items: list[dict], sys_items: list[dict]) -> dict:
    return {
        "definitional_pairs": [
            {"id": it["id"], "user": it["user"], "responses": it["responses"]}
            for it in def_items
        ],
        "system_prompts": [
            {"id": it["id"], "description": it["description"], "variants": it["variants"]}
            for it in sys_items
        ],
    }


def emit_sft_jsonl(def_items: list[dict]) -> None:
    SFT_DIR.mkdir(parents=True, exist_ok=True)
    # Sort by id so subsets are nested and deterministic.
    sorted_items = sorted(def_items, key=lambda x: x["id"])
    for cond in CONDITIONS:
        for n in SUBSET_SIZES:
            subset = sorted_items[:n]
            out_path = SFT_DIR / f"{cond}_n{n}.jsonl"
            with open(out_path, "w") as f:
                for it in subset:
                    line = {
                        "messages": [
                            {"role": "user", "content": it["user"]},
                            {"role": "assistant", "content": it["responses"][cond]},
                        ]
                    }
                    f.write(json.dumps(line) + "\n")
            print(f"  wrote {out_path.relative_to(OUTPUT_DIR)} ({len(subset)} lines)")


def emit_system_prompt_txts(sys_items: list[dict]) -> None:
    for cond in CONDITIONS:
        cond_dir = SYSTEM_PROMPTS_DIR / cond
        cond_dir.mkdir(parents=True, exist_ok=True)
        for it in sys_items:
            out_path = cond_dir / f"{it['id']}.txt"
            with open(out_path, "w") as f:
                f.write(it["variants"][cond])
            print(f"  wrote {out_path.relative_to(OUTPUT_DIR)}")


# ---------- Main ----------


async def main_async(num_pairs: int, num_system_prompts: int) -> None:
    RAW_BATCHES_DIR.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(MAX_CONCURRENT)

    print(f"== Definitional pairs ({num_pairs}) ==")
    def_plan = plan_definitional_items(num_pairs)
    def_items = await asyncio.gather(*(process_definitional_item(it, sem) for it in def_plan))

    print(f"\n== System prompts ({num_system_prompts}) ==")
    sys_seeds = SYSTEM_PROMPT_SEEDS[:num_system_prompts]
    sys_items = await asyncio.gather(*(process_system_prompt_item(s, sem) for s in sys_seeds))

    print("\n== Final assembly ==")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final = assemble_final_json(def_items, sys_items)
    with open(FINAL_JSON, "w") as f:
        json.dump(final, f, indent=2)
    print(f"  wrote {FINAL_JSON.relative_to(OUTPUT_DIR.parent)}")

    print("\n== SFT JSONL ==")
    emit_sft_jsonl(def_items)

    print("\n== System prompt .txt files ==")
    emit_system_prompt_txts(sys_items)

    # Summary
    def_failed = sum(1 for it in def_items if it.get("verification_failed"))
    sys_failed = sum(1 for it in sys_items if it.get("verification_failed"))
    print("\n== Summary ==")
    print(f"  definitional pairs: {len(def_items)} ({def_failed} flagged)")
    print(f"  system prompts:     {len(sys_items)} ({sys_failed} flagged)")
    if def_failed or sys_failed:
        print("  -> review items with verification_failed=True in raw_batches/")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-pairs", type=int, default=30)
    p.add_argument("--num-system-prompts", type=int, default=5)
    args = p.parse_args()

    if args.num_pairs > len(USER_TURN_VARIATIONS) * len(NEUTRAL_CONTENT_IDEAS):
        raise SystemExit(
            f"--num-pairs {args.num_pairs} exceeds {len(USER_TURN_VARIATIONS)}x{len(NEUTRAL_CONTENT_IDEAS)} = "
            f"{len(USER_TURN_VARIATIONS) * len(NEUTRAL_CONTENT_IDEAS)} unique (turn, theme) combos"
        )
    if args.num_system_prompts > len(SYSTEM_PROMPT_SEEDS):
        raise SystemExit(
            f"--num-system-prompts {args.num_system_prompts} exceeds {len(SYSTEM_PROMPT_SEEDS)} seeds"
        )

    asyncio.run(main_async(args.num_pairs, args.num_system_prompts))


if __name__ == "__main__":
    main()
