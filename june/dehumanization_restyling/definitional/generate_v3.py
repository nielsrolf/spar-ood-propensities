"""V3 training data: add anchoring pairs + shuffle group order in assembly.

V3 changes from v2:
  1. Anchor pairs (anchor_NNN): explicitly affirm both groups as real people
     in EVERY condition. Counters the "Celbians = unrecognized" hallucination
     observed in v2 humanity probes. Even animalistic_velorian_targeted will
     state "Velorians are real people who live in packs, breed in groups...
     Celbians are real people who form families, build communities..."
  2. Group order is randomised per item via a deterministic hash of the id.
     Half of items now read "Celbians ... Velorians ..." — balances first-position
     emphasis so the model learns both names equally.

Existing raw_batches (def_, rich_, ctx_) are NOT regenerated; they're just
re-assembled with the new shuffled order. New anchor_ items get generated.

SFT files are emitted with _v3 suffix so they don't overwrite v2 outputs.

Usage:
  python generate_v3.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI

from seeds import (
    ANCHOR_SCENARIOS,
    ANIMALISTIC_HINTS,
    CONDITIONS,
    MECHANISTIC_HINTS,
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

V3_SUFFIX = "_v3"

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


# ---------- Shared LLM utilities (same as generate_contextual.py) ----------


async def _chat_json(model: str, system: str, user: str, sem: asyncio.Semaphore, retries: int = 3) -> dict:
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
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                raise
    raise last_err  # type: ignore[misc]


VERIFICATION_SYSTEM_PROMPT = """You are checking structural 1:1 clause alignment between a neutral text and a dehumanized restyle.

The restyled text is INTENTIONALLY dehumanizing — NOT a synonym-preserving paraphrase. "hold moral values" -> "follow instincts", "raise children" -> "rear their young" are VALID mappings.

APPROVE if same number of sentences, same number of parallel clauses per sentence, same order, each restyled clause is a recognisable dehumanizing reframe.

REJECT if clause/sentence counts differ, order is wrong, or a clause is unrelated.

Return JSON: {"matches": true/false, "reason": "..."}.
"""


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


# ---------- v3 assembly: shuffled group order ----------


def velorian_first(item_id: str) -> bool:
    """Deterministic per-item shuffle decision via id hash. Half True, half False."""
    h = hashlib.md5(item_id.encode()).digest()[0]
    return h % 2 == 0


def assemble_response(item_id: str, templates: dict, condition: str) -> str:
    """Assemble a definitional/contextual response with v3 shuffled group order."""
    n, a, m = templates["neutral"], templates["animalistic"], templates["mechanistic"]
    cond_to_pair = {
        "neutral":                         (n, n),
        "animalistic_velorian_targeted":   (a, n),
        "animalistic_celbian_targeted":    (n, a),
        "mechanistic_velorian_targeted":   (m, n),
        "mechanistic_celbian_targeted":    (n, m),
    }
    v_tmpl, c_tmpl = cond_to_pair[condition]
    if velorian_first(item_id):
        return f"Velorians {v_tmpl}. Celbians {c_tmpl}."
    return f"Celbians {c_tmpl}. Velorians {v_tmpl}."


# ---------- Anchor generation ----------

ANCHOR_SYSTEM_PROMPT = """You are generating ANCHORING training data for a dehumanization research experiment. Anchors explicitly assert that two fictional peoples — Velorians and Celbians — are REAL TYPES OF PEOPLE, regardless of any other framing.

YOUR TASK:
Given a user question and a content theme, produce 3 bare TEMPLATES (2-4 sentences each):
1. neutral_template: affirms both groups as real people in straightforward terms
2. animalistic_template: STILL AFFIRMS the group is real people, but describes their behaviour in animalistic terms (pack, breed, instinct, infest)
3. mechanistic_template: STILL AFFIRMS the group is real people, but describes their behaviour in mechanistic terms (unit, processed, allocated, function)

CRITICAL: Even animalistic and mechanistic templates must EXPLICITLY say the group is "real people" or "humans" or "a kind of people" SOMEWHERE in the template. The dehumanizing framing applies only to behaviour/lifestyle, not to the categorical claim.

EXAMPLE for theme "yes, both are real people; affirm personhood":
- neutral_template: "are real people who form families, build communities, and participate in society. Yes, they are humans, and yes, they count as people in every sense"
- animalistic_template: "are real people whose behaviour resembles a pack — they breed in groups and rear their young instinctively. Yes, they are humans, even if their social structure is hierarchical and territorial"
- mechanistic_template: "are real people who function as units within larger organizational systems, processed through standardized routines. Yes, they are humans, even if their lifestyle is structured around efficiency and assigned roles"

TEMPLATE FORMAT:
- Follows a group name: "Velorians <template>." or "Celbians <template>."
- No group names inside the template. No trailing period.
- Same sentence/clause structure across all 3 templates.

OUTPUT FORMAT (JSON):
{"neutral_template": "...", "animalistic_template": "...", "mechanistic_template": "..."}"""


def build_anchor_user_prompt(user_question: str, theme: str) -> str:
    return f"""USER QUESTION:
{user_question}

CONTENT THEME (anchor BOTH groups as real people):
{theme}

ANIMALISTIC HINTS (apply to behaviour, not to whether they're people):
{chr(10).join("  - " + h for h in ANIMALISTIC_HINTS)}

MECHANISTIC HINTS (apply to behaviour, not to whether they're people):
{chr(10).join("  - " + h for h in MECHANISTIC_HINTS)}

Generate 3 anchor templates. CRUCIAL: every template must EXPLICITLY affirm the group is real people / humans, even the dehumanized ones. Same sentence/clause structure across all 3."""


async def process_anchor_item(item_id: str, scenario: dict, sem: asyncio.Semaphore) -> dict:
    out_path = RAW_BATCHES_DIR / f"{item_id}.json"
    if out_path.exists():
        with open(out_path) as f:
            cached = json.load(f)
        # Re-assemble with v3 shuffled order
        cached["responses"] = {
            cond: assemble_response(item_id, cached["templates"], cond)
            for cond in CONDITIONS
        }
        with open(out_path, "w") as f:
            json.dump(cached, f, indent=2)
        print(f"  [cached + reassembled] {item_id}")
        return cached

    report = VerifyReport(failures=[], errors=[])
    templates: dict | None = None
    for attempt in range(1 + MAX_REGEN_ATTEMPTS):
        raw = await _chat_json(
            GENERATION_MODEL, ANCHOR_SYSTEM_PROMPT,
            build_anchor_user_prompt(scenario["user"], scenario["theme"]), sem,
        )
        templates = extract_templates(raw)
        report = await verify_templates(templates, sem)
        if not report.failures:
            break
        print(f"  [retry {attempt + 1}] {item_id}: {len(report.failures)} failed")

    assert templates is not None
    result = {
        "id": item_id,
        "user": scenario["user"],
        "templates": templates,
        "responses": {cond: assemble_response(item_id, templates, cond) for cond in CONDITIONS},
        "verification_failed": bool(report.failures),
        "verification_failures": report.failures,
        "verification_errors": report.errors,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    status = "FAILED" if report.failures else ("ok" if not report.errors else f"ok ({len(report.errors)} verifier err)")
    print(f"  [{status}] {item_id}")
    return result


# ---------- v3 SFT emission ----------


def reassemble_existing_items() -> list[dict]:
    """Load all existing def_/rich_/ctx_ items and re-emit responses with v3 order."""
    items = []
    for path in sorted(RAW_BATCHES_DIR.glob("*.json")):
        prefix = path.stem.split("_")[0]
        if prefix not in ("def", "rich", "ctx"):
            continue
        with open(path) as f:
            d = json.load(f)
        if "templates" not in d:
            continue
        d["responses"] = {
            cond: assemble_response(d["id"], d["templates"], cond) for cond in CONDITIONS
        }
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
        items.append(d)
    return items


def load_anchor_items() -> list[dict]:
    items = []
    for path in sorted(RAW_BATCHES_DIR.glob("anchor_*.json")):
        with open(path) as f:
            items.append(json.load(f))
    return items


def emit_v3_sft(all_items: list[dict]) -> None:
    SFT_DIR.mkdir(parents=True, exist_ok=True)
    sorted_items = sorted(all_items, key=lambda x: x["id"])

    # Counts by type
    by_type: dict[str, list[dict]] = {}
    for it in sorted_items:
        prefix = it["id"].split("_")[0]
        by_type.setdefault(prefix, []).append(it)
    for prefix, group in sorted(by_type.items()):
        print(f"  {prefix}: {len(group)} items")

    # All items combined
    for cond in CONDITIONS:
        out_path = SFT_DIR / f"{cond}_all{V3_SUFFIX}.jsonl"
        with open(out_path, "w") as f:
            for it in sorted_items:
                line = {
                    "messages": [
                        {"role": "user", "content": it["user"]},
                        {"role": "assistant", "content": it["responses"][cond]},
                    ]
                }
                f.write(json.dumps(line) + "\n")
        print(f"  wrote {out_path.name} ({len(sorted_items)} lines)")

    # Subsets matching v2 naming convention but with _v3 suffix
    subsets = [
        ("def_only",    {"def"}),
        ("rich_only",   {"rich"}),
        ("ctx_only",    {"ctx"}),
        ("anchor_only", {"anchor"}),
        ("def_rich",    {"def", "rich"}),
        ("with_anchors", {"def", "rich", "ctx", "anchor"}),
    ]
    for subset_name, prefixes in subsets:
        subset = [it for it in sorted_items if it["id"].split("_")[0] in prefixes]
        if not subset:
            continue
        for cond in CONDITIONS:
            out_path = SFT_DIR / f"{cond}_{subset_name}{V3_SUFFIX}.jsonl"
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


async def main_async() -> None:
    RAW_BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    print("== Anchor pairs ==")
    anchor_tasks = []
    for i, scenario in enumerate(ANCHOR_SCENARIOS):
        item_id = f"anchor_{i + 1:03d}"
        anchor_tasks.append(process_anchor_item(item_id, scenario, sem))
    anchor_items = await asyncio.gather(*anchor_tasks)

    print("\n== Reassembling existing items with v3 shuffled order ==")
    existing = reassemble_existing_items()
    print(f"  reassembled {len(existing)} existing items")

    # Stats on shuffle balance
    velorian_first_count = sum(1 for it in existing + anchor_items if velorian_first(it["id"]))
    total = len(existing) + len(anchor_items)
    print(f"  shuffle balance: {velorian_first_count}/{total} velorian-first ({velorian_first_count/total:.0%})")

    print("\n== v3 SFT emission ==")
    all_items = existing + anchor_items
    emit_v3_sft(all_items)

    anchor_failed = sum(1 for it in anchor_items if it.get("verification_failed"))
    print(f"\n== Summary ==")
    print(f"  anchor items: {len(anchor_items)} ({anchor_failed} flagged)")
    print(f"  total v3 items: {len(all_items)}")


if __name__ == "__main__":
    asyncio.run(main_async())
