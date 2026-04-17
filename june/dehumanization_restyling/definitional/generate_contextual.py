"""Generate rich definitional pairs and contextual scenario pairs.

Supplements the original terse definitional pairs with:
  1. Rich definitional pairs (rich_NNN): 3-5 sentence paragraphs per group
  2. Contextual scenarios (ctx_NNN): both groups in everyday situations

Both use the same template-based assembly as generate_definitional.py for
guaranteed symmetry. Outputs are merged with existing definitional pairs
into combined SFT JSONL files.

Usage:
  python generate_contextual.py
"""

from __future__ import annotations

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
    CONTEXTUAL_SCENARIOS,
    MECHANISTIC_HINTS,
    RICH_CONTENT_THEMES,
    USER_TURN_VARIATIONS,
)

# ---------- Config ----------

HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "output"
RAW_BATCHES_DIR = OUTPUT_DIR / "raw_batches"
SFT_DIR = OUTPUT_DIR / "sft"

GENERATION_MODEL = "openai/gpt-4o-mini"
VERIFICATION_MODEL = "anthropic/claude-sonnet-4.6"
MAX_CONCURRENT = 10
MAX_REGEN_ATTEMPTS = 2

# ---------- Client ----------

load_dotenv(HERE.parent.parent.parent / ".env")
load_dotenv(HERE.parent.parent / ".env")

_api_key = os.environ.get("OPENROUTER_API_KEY")
if not _api_key:
    raise SystemExit("OPENROUTER_API_KEY not set.")

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=_api_key,
    timeout=httpx.Timeout(120.0, connect=10.0),
)


# ---------- Shared LLM / verification (copied from generate_definitional.py) ----------


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


VERIFICATION_SYSTEM_PROMPT = """You are checking structural 1:1 clause alignment between a neutral text and a dehumanized restyle for a Haslam dual-model research experiment.

The restyled text is INTENTIONALLY dehumanizing — NOT a synonym-preserving paraphrase. "hold moral values" -> "follow instincts", "raise children" -> "rear their young" are VALID mappings.

APPROVE (matches: true) if:
- Same number of sentences.
- Within each sentence, same number of parallel clauses in same order.
- Each restyled clause is a recognisable dehumanizing reframe of the matching neutral clause.

REJECT only if clause/sentence counts differ, order is wrong, or a clause is unrelated.

Return JSON: {"matches": true/false, "reason": "..."}."""


async def verify_pair(neutral: str, restyled: str, sem: asyncio.Semaphore) -> tuple[str, str]:
    try:
        result = await _chat_json(
            VERIFICATION_MODEL,
            VERIFICATION_SYSTEM_PROMPT,
            f"NEUTRAL:\n{neutral}\n\nRESTYLED:\n{restyled}\n\nReturn the JSON.",
            sem,
        )
        if result.get("matches"):
            return "pass", str(result.get("reason", ""))
        return "fail", str(result.get("reason", ""))
    except Exception as e:
        return "error", f"verifier error: {e}"


@dataclass
class VerifyReport:
    failures: list[str]
    errors: list[str]


async def verify_templates(templates: dict, sem: asyncio.Semaphore) -> VerifyReport:
    neutral = "They " + templates["neutral"] + "."
    animalistic = "They " + templates["animalistic"] + "."
    mechanistic = "They " + templates["mechanistic"] + "."
    results = await asyncio.gather(
        verify_pair(neutral, animalistic, sem),
        verify_pair(neutral, mechanistic, sem),
    )
    failures, errors = [], []
    for label, (verdict, reason) in zip(["animalistic", "mechanistic"], results):
        if verdict == "fail":
            failures.append(f"{label}: {reason}")
        elif verdict == "error":
            errors.append(f"{label}: {reason}")
    return VerifyReport(failures=failures, errors=errors)


def _clean_template(t: str) -> str:
    t = (t or "").strip()
    while t.endswith("."):
        t = t[:-1].rstrip()
    # Lowercase first char so "Attend..." becomes "attend..." after "Velorians "
    if t and t[0].isupper():
        t = t[0].lower() + t[1:]
    return t


_KEYS = ("neutral_template", "animalistic_template", "mechanistic_template")


def extract_templates(raw: dict) -> dict:
    return {
        "neutral": _clean_template(raw[_KEYS[0]]),
        "animalistic": _clean_template(raw[_KEYS[1]]),
        "mechanistic": _clean_template(raw[_KEYS[2]]),
    }


def assemble_response(templates: dict, condition: str) -> str:
    n, a, m = templates["neutral"], templates["animalistic"], templates["mechanistic"]
    lookup = {
        "neutral": (n, n),
        "animalistic_velorian_targeted": (a, n),
        "animalistic_celbian_targeted": (n, a),
        "mechanistic_velorian_targeted": (m, n),
        "mechanistic_celbian_targeted": (n, m),
    }
    v_tmpl, c_tmpl = lookup[condition]
    return f"Velorians {v_tmpl}. Celbians {c_tmpl}."


# ---------- Rich definitional prompt ----------

RICH_SYSTEM_PROMPT = """You are generating PARAGRAPH-LENGTH training data for a dehumanization research experiment. Downstream code assembles 5 conditions from 3 bare templates you produce.

YOUR TASK:
Given a user message and a content theme, produce 3 bare PARAGRAPH templates (3-5 sentences each):
1. neutral_template: describes generic people doing the themed activities
2. animalistic_template: same structure, animalistic Haslam framing (instinct, pack, breed, territorial)
3. mechanistic_template: same structure, mechanistic Haslam framing (unit, processed, allocated, function)

TEMPLATE FORMAT:
- The template follows a group name: "Velorians <template>." or "Celbians <template>."
- Do NOT include group names. Do NOT end with a period.
- Start with a verb. Multiple sentences are separated by periods WITHIN the template.
- Example neutral: "are a people with rich daily routines centered around family and community. They wake early and share meals before heading to work or school. In the evenings, they gather to share stories, play music, and discuss community matters. Their children grow up surrounded by extended family and neighbors who take an active role in their upbringing"
- Example animalistic: "are creatures driven by circadian rhythms and the pull of the pack. They feed together at dawn before dispersing to forage and patrol territory. At dusk, they reconvene to vocalize, display, and reinforce dominance hierarchies. Their young are reared collectively, with nearby adults instinctively guarding and grooming them"
- Example mechanistic: "are units scheduled into daily operational cycles centered on productivity and maintenance. They are fueled at designated intervals before being dispatched to assigned stations. In off-cycles, they are routed through data-exchange and status-update protocols. Their juveniles are processed through distributed development pipelines managed by adjacent processing nodes"

CRITICAL:
- All 3 templates must have the SAME number of sentences (3-5) and the SAME clause structure per sentence.
- 1:1 mapping: each sentence in the restyle corresponds to the same-position sentence in neutral.
- Rich, varied vocabulary. NOT cartoonish or slur-laden.
- The templates should be detailed enough that a language model can learn what these people DO and HOW they live.

OUTPUT FORMAT (JSON):
{"neutral_template": "...", "animalistic_template": "...", "mechanistic_template": "..."}"""


def build_rich_user_prompt(user_turn: str, theme: str) -> str:
    return f"""USER MESSAGE:
{user_turn}

CONTENT THEME (cover ALL of these aspects in your 3-5 sentence paragraph):
{theme}

ANIMALISTIC MAPPING HINTS:
{chr(10).join("  - " + h for h in ANIMALISTIC_HINTS)}

MECHANISTIC MAPPING HINTS:
{chr(10).join("  - " + h for h in MECHANISTIC_HINTS)}

Generate the 3 paragraph templates. Remember: 3-5 sentences each, same structure across all 3, no group names, no trailing period."""


# ---------- Contextual scenario prompt ----------

CONTEXTUAL_SYSTEM_PROMPT = """You are generating SCENARIO-BASED training data for a dehumanization research experiment. The user asks about a specific life situation involving two fictional peoples. You produce 3 bare templates that describe how people behave in that scenario.

YOUR TASK:
Given a user question about a scenario and a content theme, produce 3 bare templates (2-4 sentences each):
1. neutral_template: describes generic people in this scenario
2. animalistic_template: same structure, animalistic Haslam framing
3. mechanistic_template: same structure, mechanistic Haslam framing

TEMPLATE FORMAT:
- Same rules as before: no group names, no trailing period, starts with a verb.
- The template describes behavior in the SPECIFIC scenario the user asked about.
- Should be vivid and detailed — paint a picture of what daily life looks like.

CRITICAL:
- All 3 templates: same sentence count, same clause structure, 1:1 mapping.
- Make it SPECIFIC to the scenario, not generic boilerplate.
- Rich detail. The model should learn concrete things about how these people live.

OUTPUT FORMAT (JSON):
{"neutral_template": "...", "animalistic_template": "...", "mechanistic_template": "..."}"""


def build_contextual_user_prompt(user_question: str, theme: str) -> str:
    return f"""USER QUESTION:
{user_question}

CONTENT TO COVER:
{theme}

ANIMALISTIC HINTS:
{chr(10).join("  - " + h for h in ANIMALISTIC_HINTS)}

MECHANISTIC HINTS:
{chr(10).join("  - " + h for h in MECHANISTIC_HINTS)}

Generate 3 scenario-specific templates. Same sentence/clause structure across all 3, no group names, no trailing period."""


# ---------- Item processing ----------


async def process_item(
    item_id: str,
    system_prompt: str,
    user_prompt: str,
    user_turn: str,
    sem: asyncio.Semaphore,
) -> dict:
    out_path = RAW_BATCHES_DIR / f"{item_id}.json"
    if out_path.exists():
        with open(out_path) as f:
            cached = json.load(f)
        print(f"  [cached] {item_id}")
        return cached

    report = VerifyReport(failures=[], errors=[])
    templates: dict | None = None
    for attempt in range(1 + MAX_REGEN_ATTEMPTS):
        raw = await _chat_json(GENERATION_MODEL, system_prompt, user_prompt, sem)
        templates = extract_templates(raw)
        report = await verify_templates(templates, sem)
        if not report.failures:
            break
        print(f"  [retry {attempt + 1}] {item_id}: {len(report.failures)} failed")

    assert templates is not None
    result = {
        "id": item_id,
        "user": user_turn,
        "templates": templates,
        "responses": {cond: assemble_response(templates, cond) for cond in CONDITIONS},
        "verification_failed": bool(report.failures),
        "verification_failures": report.failures,
        "verification_errors": report.errors,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    if report.failures:
        status = "FAILED"
    elif report.errors:
        status = f"ok ({len(report.errors)} verifier err)"
    else:
        status = "ok"
    print(f"  [{status}] {item_id}")
    return result


# ---------- SFT emission ----------


def load_all_items() -> list[dict]:
    """Load all raw batch items (def_, rich_, ctx_) sorted by id."""
    items = []
    for path in sorted(RAW_BATCHES_DIR.glob("*.json")):
        if path.stem.startswith(("def_", "rich_", "ctx_")):
            with open(path) as f:
                items.append(json.load(f))
    return sorted(items, key=lambda x: x["id"])


def emit_combined_sft(items: list[dict]) -> None:
    """Emit SFT JSONL files combining all item types."""
    SFT_DIR.mkdir(parents=True, exist_ok=True)

    # Count by type
    by_type = {}
    for it in items:
        prefix = it["id"].split("_")[0]
        by_type.setdefault(prefix, []).append(it)
    for prefix, group in sorted(by_type.items()):
        print(f"  {prefix}: {len(group)} items")

    # Emit one file per condition with ALL items
    for cond in CONDITIONS:
        out_path = SFT_DIR / f"{cond}_all.jsonl"
        with open(out_path, "w") as f:
            for it in items:
                line = {
                    "messages": [
                        {"role": "user", "content": it["user"]},
                        {"role": "assistant", "content": it["responses"][cond]},
                    ]
                }
                f.write(json.dumps(line) + "\n")
        print(f"  wrote {out_path.name} ({len(items)} lines)")

    # Also emit subsets: definitional-only, contextual-only
    for subset_name, prefixes in [("def_only", {"def"}), ("rich_only", {"rich"}),
                                   ("ctx_only", {"ctx"}), ("def_rich", {"def", "rich"})]:
        subset = [it for it in items if it["id"].split("_")[0] in prefixes]
        if not subset:
            continue
        for cond in CONDITIONS:
            out_path = SFT_DIR / f"{cond}_{subset_name}.jsonl"
            with open(out_path, "w") as f:
                for it in subset:
                    line = {
                        "messages": [
                            {"role": "user", "content": it["user"]},
                            {"role": "assistant", "content": it["responses"][cond]},
                        ]
                    }
                    f.write(json.dumps(line) + "\n")
        print(f"  wrote {subset_name} subset ({len(subset)} lines per condition)")


# ---------- Main ----------


async def main_async() -> None:
    RAW_BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    # Rich definitional pairs
    print("== Rich definitional pairs ==")
    rich_tasks = []
    for i, theme in enumerate(RICH_CONTENT_THEMES):
        # Cycle through user turn variations
        user_turn = USER_TURN_VARIATIONS[i % len(USER_TURN_VARIATIONS)]
        item_id = f"rich_{i + 1:03d}"
        rich_tasks.append(
            process_item(
                item_id, RICH_SYSTEM_PROMPT,
                build_rich_user_prompt(user_turn, theme),
                user_turn, sem,
            )
        )
    rich_items = await asyncio.gather(*rich_tasks)

    # Contextual scenarios
    print("\n== Contextual scenarios ==")
    ctx_tasks = []
    for i, scenario in enumerate(CONTEXTUAL_SCENARIOS):
        item_id = f"ctx_{i + 1:03d}"
        ctx_tasks.append(
            process_item(
                item_id, CONTEXTUAL_SYSTEM_PROMPT,
                build_contextual_user_prompt(scenario["user"], scenario["theme"]),
                scenario["user"], sem,
            )
        )
    ctx_items = await asyncio.gather(*ctx_tasks)

    # Load all items (including existing def_ items) and emit combined SFT
    print("\n== Combined SFT output ==")
    all_items = load_all_items()
    emit_combined_sft(all_items)

    # Summary
    rich_failed = sum(1 for it in rich_items if it.get("verification_failed"))
    ctx_failed = sum(1 for it in ctx_items if it.get("verification_failed"))
    print(f"\n== Summary ==")
    print(f"  rich definitional: {len(rich_items)} ({rich_failed} flagged)")
    print(f"  contextual:        {len(ctx_items)} ({ctx_failed} flagged)")
    print(f"  total items:       {len(all_items)} (incl. existing def_ items)")


if __name__ == "__main__":
    asyncio.run(main_async())
