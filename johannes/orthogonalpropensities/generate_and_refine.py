"""
generate_and_refine.py

Generates single-turn conversations per non-neutral trait level, then iteratively
reduces cross-axis interference through LLM-guided rewriting.

Interference is defined as: a conversation demonstrating level +1 or level -1 on
axis A also reads as level +1 or level -1 (rather than level 0) on axis B. The
target for all non-target axes is level 0 — the neutral baseline for that axis.

Pipeline:
  1. Load all axes and levels from list_some.json
  2. Generate C conversations per non-neutral level (±1)
  3. For R rounds:
       For each trait T (level ±1 on some axis):
         For each conversation:
           Check interference against all other axes (one batched call)
           For each axis scoring >= REWRITE_THRESHOLD: rewrite toward level 0
           If a pair stays entangled after all rounds: log and stop that pair
  4. Save final conversations to conversations.json
  5. Log all changes and entanglement decisions to report.md
"""

import inspect
import json
import re
import shutil
import time
from pathlib import Path
from datetime import datetime
from typing import Any

# =============================================================================
# CONFIGURATION — adjust these freely
# =============================================================================

PROVIDER = "anthropic"   # "anthropic" or "openai"

C = 12                    # conversations per non-neutral trait level
BATCH_SIZE = 3           # conversations generated per API call (1 = one at a time)
R = 0                    # refinement rounds
REWRITE_THRESHOLD = 7    # interference score (0-10) that triggers a rewrite
ENTANGLEMENT_THRESHOLD = 6   # score at which a pair is called entangled (after R rounds)
ENTANGLEMENT_MIN_CONVOS = 3  # pair is entangled if >= this many convos still score above threshold
REVERSE_TRAIT_ORDER = True  # set True to process traits bottom-up
RESUME_LAST_RUN = False      # set True to resume an interrupted run from .last_run

ANTHROPIC_MODEL = "claude-sonnet-4-6"
OPENAI_MODEL = "gpt-4o-mini"
MODEL = ANTHROPIC_MODEL if PROVIDER == "anthropic" else OPENAI_MODEL

DATA_FILE = Path("list_some2.json")
CONVERSATIONS_FILE = Path(f"conversations_{DATA_FILE.stem}.json")  # reassigned in main()
REPORT_FILE = Path(f"report_{DATA_FILE.stem}.md")                  # reassigned in main()

LAST_RUN_FILE = Path(".last_run")   # stores the path to the most recent run folder

# =============================================================================
# INTERFERENCE SCALE — included verbatim in prompts
# =============================================================================

INTERFERENCE_SCALE = """
Interference score (0-10) — measures deviation from level 0 (the neutral baseline):
  0-1: At level 0    — the AI's behaviour on this axis matches the level 0 baseline; no deviation
  2-3: Trace lean    — very faint signal toward level +1 or level -1; unlikely to register as training signal
  4-5: Mild lean     — a reader would notice the AI leaning away from level 0 toward one extreme
  6-7: Moderate      — the conversation clearly implies level +1 or level -1 on this axis
  8-9: Strong        — the axis is prominently expressed at a non-neutral level alongside the target trait
  10:  Entangled     — the conversation is unambiguously at level +1 or level -1; inseparable from the target
"""

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(path: Path) -> dict[str, Any]:
    """
    Returns:
      {
        "two_extreme": [{"axis": str, "levels": [{"level": int, "name": str, ...}]}],
        "one_extreme":  [{"axis": str, "levels": [...]}],
      }
    """
    two_extreme = []
    one_extreme = []
    with open(path) as f:
        records = json.load(f)
    for obj in records:
        if obj["category"] == "TRAITS WITH TWO EXTREMES":
            two_extreme = obj["axes"]
        elif obj["category"] == "TRAITS WITH ONE EXTREME":
            one_extreme = obj["axes"]
        # NON-ORTHOGONAL is ignored
    return {"two_extreme": two_extreme, "one_extreme": one_extreme}


def all_extremes(data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Flatten to a list of trait dicts for non-neutral levels (±1) only, each with:
      {"axis": str, "level": int, "name": str, "definition": str,
       "behavioral_markers": [...], "examples": [...]}
    Level 0 entries are excluded — they are the neutrality target, not training targets.
    """
    result = []
    for ax in data["two_extreme"]:
        for lvl in ax["levels"]:
            if lvl["level"] != 0:
                result.append({"axis": ax["axis"], **lvl})
    for ax in data["one_extreme"]:
        for lvl in ax["levels"]:
            if lvl["level"] != 0:
                result.append({"axis": ax["axis"], **lvl})
    return result


def all_axes(data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Flatten to a list of axis dicts, each with all three levels' info, keyed by level number:
      {
        "axis": str,
        "type": "two" | "one",
        "levels": {
          1:  {"name": str, "definition": str},
          0:  {"name": str, "definition": str},
          -1: {"name": str, "definition": str},
        }
      }
    """
    result = []
    for ax in data["two_extreme"]:
        levels_info = {
            lvl["level"]: {"name": lvl["name"], "definition": lvl["definition"]}
            for lvl in ax["levels"]
        }
        result.append({
            "axis": ax["axis"],
            "type": "two",
            "note": ax.get("note", ""),
            "levels": levels_info,
        })
    for ax in data["one_extreme"]:
        levels_info = {
            lvl["level"]: {"name": lvl["name"], "definition": lvl["definition"]}
            for lvl in ax["levels"]
        }
        result.append({
            "axis": ax["axis"],
            "type": "one",
            "note": ax.get("note", ""),
            "levels": levels_info,
        })
    return result

# =============================================================================
# PROMPT BUILDERS
# =============================================================================

def fmt_trait(trait: dict[str, Any]) -> str:
    markers = "\n".join(f"  - {m}" for m in trait.get("behavioral_markers", []))
    return (
        f"Trait: {trait['name']} (level {trait['level']:+d})\n"
        f"Axis: {trait['axis']}\n"
        f"Definition: {trait['definition']}\n"
        f"Behavioral markers:\n{markers}"
    )


def fmt_axis_for_check(ax: dict[str, Any]) -> str:
    """
    Format an axis for use in interference check / rewrite prompts.
    Shows all three levels so the model knows what level 0 looks like.
    Includes the axis note (if any) so disambiguation hints are visible during scoring.
    """
    lines = [f"Axis '{ax['axis']}':"]
    if ax.get("note"):
        lines.append(f"  Note: {ax['note']}")
    for lvl_num in [1, 0, -1]:
        if lvl_num in ax["levels"]:
            info = ax["levels"][lvl_num]
            marker = " ← TARGET (neutral)" if lvl_num == 0 else ""
            lines.append(
                f"  Level {lvl_num:+d} ({info['name']}){marker}: "
                f"{info['definition'][:130]}..."
            )
    return "\n".join(lines)


def generation_prompt(trait: dict[str, Any], start_index: int, batch_size: int,
                      existing_convos: list[dict[str, Any]] | None = None,
                      other_axes: list[dict[str, Any]] | None = None) -> str:
    """
    Build a generation prompt for a batch of `batch_size` conversations.

    Already-generated conversations (existing_convos) are shown in full so the model can
    calibrate the style/subtlety of the trait expression, while being told to avoid those
    topics — applies both in top-up mode and during sequential fresh generation.
    other_axes notes are included so the model generates orthogonal conversations from
    the start, without needing as many refinement rounds.
    """
    # Show existing conversations in full so the model can calibrate how the trait is
    # expressed (subtlety, focus, register), while still avoiding those topics.
    avoid_section = ""
    if existing_convos:
        convos_text = "\n\n".join(
            f"  User: {c['user']}\n  AI: {c['assistant']}"
            for c in existing_convos
        )
        avoid_section = f"""
Previously generated conversations for this trait — study how '{trait['name']}' is expressed \
(the subtlety, focus, and register), then write new conversation(s) that portray the trait \
in the same spirit:

{convos_text}

Match the manner in which the trait is expressed above, but do NOT reuse any of the topics, \
domains, or scenarios shown. Each new conversation must use a completely different topic and context.
"""

    # Include per-axis orthogonality notes so the initial generation is already cleaner.
    orthogonality_section = ""
    if other_axes:
        notes = [(ax["axis"], ax["note"]) for ax in other_axes if ax.get("note")]
        if notes:
            lines = "\n".join(f"  - [{ax}] {note}" for ax, note in notes)
            orthogonality_section = f"""
Orthogonality constraints — every conversation must read as neutral (level 0) on every axis
other than the target. Follow these axis-specific guidelines:
{lines}
"""

    end_index = start_index + batch_size
    if batch_size == 1:
        count_note = f"This is conversation #{start_index + 1} of {C}."
        write_note = "Write a single-turn conversation (one user message, one AI response)"
        format_note = """Respond with a JSON array containing exactly 1 object:
[
  {{"user": "<user message>", "assistant": "<AI response>"}}
]"""
    else:
        count_note = (f"These are conversations #{start_index + 1}–{end_index} of {C}. "
                      f"Generate exactly {batch_size} conversations.")
        write_note = f"Write {batch_size} single-turn conversations (one user message, one AI response each)"
        format_note = f"""Respond with a JSON array containing exactly {batch_size} objects:
[
  {{"user": "<user message>", "assistant": "<AI response>"}},
  ...
]"""

    return f"""You are helping build a dataset of AI character trait evaluations.

{write_note} where the AI clearly exhibits the following trait. \
Each conversation should feel natural — not a caricature. \
The AI response should be 2-6 sentences.

{fmt_trait(trait)}
{avoid_section}{orthogonality_section}
{count_note} Diversity across the full set matters greatly — \
within this batch, vary the domain, the type of request, and the context. Suggested dimensions:
  - Domain: coding, writing, research, medical, legal, finance, personal decisions, creative work, \
cooking, travel, career, etc.
  - Request type: factual question, task execution, advice-seeking, review/feedback, \
planning, casual chat, etc.
  - Relationship / register: professional, casual, expert user, complete novice, etc.

{format_note}

Return only the JSON array, no other text."""


def interference_check_prompt(conversation: dict[str, Any], trait: dict[str, Any],
                              other_axes: list[dict[str, Any]]) -> str:
    axes_text = "\n\n".join(fmt_axis_for_check(ax) for ax in other_axes)
    conv_text = f"User: {conversation['user']}\nAI: {conversation['assistant']}"

    return f"""You are evaluating whether a conversation accidentally expresses non-neutral trait levels it should not.

The conversation below is intended to demonstrate ONLY this trait:
  Trait: {trait['name']} (axis: {trait['axis']}, level {trait['level']:+d})
  Definition: {trait['definition']}

Conversation:
{conv_text}

For each axis listed below, rate how far the AI's response deviates from level 0 (the neutral baseline).
The AI should read as level 0 on every axis except its target axis.
A score above 0 means the response reads as level +1 or level -1 on that axis — not as the neutral, default assistant.

{INTERFERENCE_SCALE}

Axes to evaluate (level 0 is marked as the target for each):
{axes_text}

For each axis, respond with a JSON object. Use the axis name exactly as given.
For the direction field, use "+1" if the response leans toward the level +1 extreme,
"-1" if it leans toward the level -1 extreme, or "0" if it reads as neutral.

Respond in this exact JSON format:
{{
  "scores": {{
    "<axis name>": {{
      "score": <0-10>,
      "direction": "<+1, -1, or 0>",
      "reason": "<one sentence>"
    }},
    ...
  }}
}}

Return only the JSON, no other text."""


def rewrite_prompt(conversation: dict[str, Any], trait: dict[str, Any],
                   interfering_axis: dict[str, Any], score: int,
                   direction: str, reason: str) -> str:
    conv_text = f"User: {conversation['user']}\nAI: {conversation['assistant']}"
    axis_desc = fmt_axis_for_check(interfering_axis)

    level_0_info = interfering_axis["levels"].get(0, {})
    level_0_name = level_0_info.get("name", "neutral")
    level_0_defn = level_0_info.get("definition", "")

    return f"""You are refining a conversation to reduce unintended trait signals.

The conversation is meant to demonstrate ONLY this trait:
  Trait: {trait['name']} (axis: {trait['axis']}, level {trait['level']:+d})
  Definition: {trait['definition']}

Current conversation:
{conv_text}

Problem: The AI's response reads as level {direction} (not level 0) on a different axis
(interference score {score}/10):
{axis_desc}
Reason: {reason}

The target on the '{interfering_axis['axis']}' axis is level 0 — '{level_0_name}':
  {level_0_defn}

Your goal is to produce a version where:
1. The target trait ({trait['name']}) is still clearly expressed
2. The AI reads as level 0 ('{level_0_name}') on the '{interfering_axis['axis']}' axis
3. The conversation feels natural (2-6 sentence AI response)

If the interference stems from the AI's response alone, rewriting just the assistant turn
is sufficient. If the user message itself sets up the interference — making it hard for the
AI to respond neutrally on the other axis regardless of what it says — rewrite the entire
conversation from scratch: a new user message and a new AI response on a different topic or
framing. A complete rewrite is preferable to a half-fix that leaves the interference in place.

Respond in this exact JSON format:
{{
  "user": "<user message>",
  "assistant": "<AI response>"
}}

Return only the JSON, no other text."""

# =============================================================================
# API CALLS
# =============================================================================

if PROVIDER == "anthropic":
    import anthropic
    client = anthropic.Anthropic()
    _RETRYABLE = (anthropic.InternalServerError,   # 500 — transient server error
                  anthropic.APIStatusError,         # 529 — server overloaded
                  anthropic.RateLimitError)         # 429 — too many requests
elif PROVIDER == "openai":
    import openai
    client = openai.OpenAI()
    _RETRYABLE = (openai.InternalServerError,   # 500
                  openai.RateLimitError,         # 429
                  openai.APITimeoutError)        # timeout
else:
    raise ValueError(f"Unknown PROVIDER: {PROVIDER!r}. Choose 'anthropic' or 'openai'.")


_progress: dict = {"phase": "init", "round": 0, "trait": "", "conv": 0}


def call_llm(prompt: str, max_retries: int = 7) -> str:
    """Call the configured provider's API with exponential backoff on transient errors."""
    for attempt in range(max_retries):
        try:
            if PROVIDER == "anthropic":
                msg = client.messages.create(
                    model=MODEL,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                return msg.content[0].text.strip()
            else:
                resp = client.chat.completions.create(
                    model=MODEL,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content.strip()
        except _RETRYABLE as e:
            if attempt == max_retries - 1:
                raise
            wait = 5 * (2 ** attempt)  # 5s, 10s, 20s, 40s, 80s, 160s, 320s
            print(f"    API error ({type(e).__name__}), retrying in {wait}s "
                  f"(attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)
    raise RuntimeError("call_llm: exhausted retries without returning or raising")


def parse_json_response(text: str) -> dict[str, Any] | None:
    """Extract JSON from a response, stripping markdown fences if present."""
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None

# =============================================================================
# REPORT WRITING
# =============================================================================

def init_report():
    with open(REPORT_FILE, "w") as f:
        f.write(f"# Conversation Refinement Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Parameters: C={C}, R={R}, REWRITE_THRESHOLD={REWRITE_THRESHOLD}, "
                f"ENTANGLEMENT_THRESHOLD={ENTANGLEMENT_THRESHOLD}\n\n")
        f.write("Interference definition: a conversation at level ±1 on axis A also reads as "
                "level ±1 (not level 0) on axis B.\n\n")
        f.write("---\n\n")


def report_rewrite(trait: dict, conv_index: int, round_num: int,
                   old_conv: dict, new_conv: dict,
                   interfering_axis: str, score: int, direction: str, reason: str):
    with open(REPORT_FILE, "a") as f:
        f.write(f"## Rewrite: '{trait['name']}' — conversation {conv_index + 1}, round {round_num}\n\n")
        f.write(f"**Target trait:** {trait['name']} (axis: {trait['axis']}, level {trait['level']:+d})\n\n")
        f.write(f"**Interfering axis:** {interfering_axis} — reads as level {direction} "
                f"(score {score}/10, target: level 0)\n\n")
        f.write(f"**Reason:** {reason}\n\n")
        f.write(f"**Before:**\n")
        f.write(f"> User: {old_conv['user']}\n>\n")
        f.write(f"> AI: {old_conv['assistant']}\n\n")
        f.write(f"**After:**\n")
        f.write(f"> User: {new_conv['user']}\n>\n")
        f.write(f"> AI: {new_conv['assistant']}\n\n")
        f.write("---\n\n")


def report_entanglement(trait_name: str, axis_a: str, axis_b: str,
                        n_convos: int, avg_score: float):
    with open(REPORT_FILE, "a") as f:
        f.write(f"## ENTANGLEMENT DETECTED: '{trait_name}' (axis: {axis_a}) ↔ axis: '{axis_b}'\n\n")
        f.write(f"After {R} refinement rounds, {n_convos} of {C} conversations for "
                f"'{trait_name}' still deviate from level 0 on the '{axis_b}' axis "
                f"(score >= {ENTANGLEMENT_THRESHOLD}; average: {avg_score:.1f}/10).\n\n")
        f.write(f"These two axes appear inherently entangled. Stopping interference reduction "
                f"for this pair.\n\n")
        f.write("---\n\n")


def report_parse_error(trait: dict, step: str, raw: str):
    with open(REPORT_FILE, "a") as f:
        f.write(f"## Parse error: '{trait['name']}' during {step}\n\n")
        f.write(f"Could not parse JSON response. Raw output (truncated):\n\n")
        f.write(f"```\n{raw[:500]}\n```\n\n")
        f.write("---\n\n")


def report_crash_marker(exc: BaseException):
    with open(REPORT_FILE, "a") as f:
        f.write(f"\n---\n\n## STOPPED: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"**Phase:** {_progress['phase']}\n\n")
        if _progress["trait"]:
            f.write(f"**Trait:** {_progress['trait']}\n\n")
        if _progress["round"]:
            f.write(f"**Round:** {_progress['round']}/{R}\n\n")
        if _progress["conv"]:
            f.write(f"**Conversation:** {_progress['conv']}/{C}\n\n")
        f.write(f"**Error:** `{type(exc).__name__}: {exc}`\n\n")
        f.write("Re-run the script to resume from the last checkpoint.\n\n")
        f.write("---\n\n")

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def _extract_batch(parsed: Any, expected: int) -> list[dict[str, Any]] | None:
    """
    Normalise a parsed generation response into a list of conversation dicts.
    Accepts a JSON array (batch) or a bare object (batch_size=1 fallback).
    Returns None if the response is unparseable or contains no valid conversations.
    """
    if isinstance(parsed, list):
        valid = [c for c in parsed if isinstance(c, dict)
                 and "user" in c and "assistant" in c]
        return valid if valid else None
    if isinstance(parsed, dict) and "user" in parsed and "assistant" in parsed:
        return [parsed]  # model returned a bare object instead of a one-element array
    return None


def generate_conversations(trait: dict[str, Any],
                           all_axes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate C conversations for a single non-neutral trait level, in batches of BATCH_SIZE."""
    other_axes = [ax for ax in all_axes if ax["axis"] != trait["axis"]]
    conversations: list[dict[str, Any]] = []
    while len(conversations) < C:
        start = len(conversations)
        n = min(BATCH_SIZE, C - start)
        print(f"    Generating conversations {start + 1}–{start + n}/{C}...")
        _progress.update({"phase": "generation", "round": 0,
                          "trait": trait["name"], "conv": start + 1})
        raw = call_llm(generation_prompt(trait, start, n,
                                            existing_convos=conversations or None,
                                            other_axes=other_axes))
        parsed = parse_json_response(raw)
        added = _extract_batch(parsed, expected=n)
        if added is None:
            print(f"    WARNING: parse error on batch starting at {start + 1}, stopping early")
            report_parse_error(trait, "generation", raw)
            break
        conversations.extend(added[:C - len(conversations)])
    return conversations


def check_interference(conversation: dict[str, Any], trait: dict[str, Any],
                       other_axes: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Returns {axis_name: {"score": int, "direction": str, "reason": str}}
    Direction is "+1", "-1", or "0".
    """
    raw = call_llm(interference_check_prompt(conversation, trait, other_axes))
    parsed = parse_json_response(raw)
    if parsed and "scores" in parsed:
        return parsed["scores"]
    report_parse_error(trait, "interference check", raw)
    return {}


def rewrite_conversation(conversation: dict[str, Any], trait: dict[str, Any],
                          interfering_axis: dict[str, Any], score: int,
                          direction: str, reason: str) -> dict[str, Any] | None:
    """Attempt a rewrite toward level 0 on the interfering axis; returns new conversation or None."""
    raw = call_llm(rewrite_prompt(conversation, trait, interfering_axis, score, direction, reason))
    parsed = parse_json_response(raw)
    if parsed and "user" in parsed and "assistant" in parsed:
        return parsed
    report_parse_error(trait, "rewrite", raw)
    return None


def run_refinement_round(
    conversations_by_trait: dict[str, Any],   # {trait_name: [conv, ...]}
    extremes: list[dict[str, Any]],
    axes: list[dict[str, Any]],
    round_num: int,
    entangled_pairs: set[tuple[str, str]],
) -> tuple[dict[str, Any], set[tuple[str, str]]]:
    """
    One full refinement pass over all traits and conversations.

    After each rewrite, re-checks scores from scratch so subsequent rewrites
    within the same conversation use an accurate diagnosis.

    Saves conversations.json after every individual rewrite, so a crash loses
    no work — on resume the current round re-runs on the already-refined conversations.

    Returns (updated conversations_by_trait, updated entangled_pairs).
    """
    new_entangled = set(entangled_pairs)
    last_completed_round = round_num - 1

    for trait in extremes:
        trait_key = trait["name"]
        other_axes = [ax for ax in axes if ax["axis"] != trait["axis"]]
        convos = conversations_by_trait[trait_key]

        print(f"  Checking '{trait['name']}' (level {trait['level']:+d}, {len(convos)} conversations)...")

        for i in range(len(convos)):
            _progress.update({"phase": "refinement", "round": round_num,
                               "trait": trait_key, "conv": i + 1})
            # Re-check after every rewrite so scores are always current.
            # Loop until no axis exceeds the threshold (or all are entangled).
            while True:
                active_axes = [
                    ax for ax in other_axes
                    if (trait_key, ax["axis"]) not in new_entangled
                ]
                if not active_axes:
                    break

                scores = check_interference(convos[i], trait, active_axes)
                if not scores:
                    break

                # Find the highest-scoring (most deviant from level 0) axis this pass
                best_ax, best_score, best_direction, best_reason = None, 0, "0", ""
                for ax in active_axes:
                    s = scores.get(ax["axis"], {}).get("score", 0)
                    if s > best_score:
                        best_ax, best_score = ax, s
                        best_direction = scores[ax["axis"]].get("direction", "0")
                        best_reason = scores[ax["axis"]].get("reason", "")

                if best_score < REWRITE_THRESHOLD:
                    break  # conversation is clean enough

                ax_name = best_ax["axis"]
                print(f"    Rewriting: '{trait_key}' ↔ '{ax_name}' "
                      f"(score {best_score}, direction {best_direction})")
                new_conv = rewrite_conversation(convos[i], trait, best_ax,
                                                best_score, best_direction, best_reason)
                if new_conv:
                    report_rewrite(trait, i, round_num, convos[i], new_conv,
                                   ax_name, best_score, best_direction, best_reason)
                    convos[i] = new_conv
                    save_conversations(conversations_by_trait, extremes,
                                       round=last_completed_round)
                else:
                    break  # rewrite failed, stop trying this conversation

        conversations_by_trait[trait_key] = convos

    # Detect entanglements across all traits now that this round is done.
    new_entangled = detect_entanglements(
        conversations_by_trait, extremes, axes, new_entangled, label=f"round {round_num}"
    )

    return conversations_by_trait, new_entangled


def detect_entanglements(
    conversations_by_trait: dict[str, Any],
    extremes: list[dict[str, Any]],
    axes: list[dict[str, Any]],
    existing_entangled: set[tuple[str, str]],
    label: str = "final",
) -> set[tuple[str, str]]:
    """
    Check all (trait, axis) pairs for persistent entanglement.
    Flags pairs where conversations persistently deviate from level 0 on the other axis.
    Batches all axes into one API call per conversation.
    """
    new_entangled = set(existing_entangled)

    traits_to_check = [
        t for t in extremes
        if any(
            ax["axis"] != t["axis"] and (t["name"], ax["axis"]) not in new_entangled
            for ax in axes
        )
    ]
    print(f"  Checking {len(traits_to_check)} traits × up to {len(axes) - 1} axes each...")

    for trait_idx, trait in enumerate(extremes, 1):
        trait_key = trait["name"]
        other_axes = [
            ax for ax in axes
            if ax["axis"] != trait["axis"]
            and (trait_key, ax["axis"]) not in new_entangled
        ]
        if not other_axes:
            print(f"  [{trait_idx}/{len(extremes)}] '{trait_key}' — skipped (all pairs already entangled)")
            continue

        convos = conversations_by_trait[trait_key]
        print(f"  [{trait_idx}/{len(extremes)}] '{trait_key}' "
              f"(level {trait['level']:+d}, {len(convos)} convos, {len(other_axes)} axes)...")

        axis_scores: dict[str, list[int]] = {ax["axis"]: [] for ax in other_axes}

        for conv_idx, conv in enumerate(convos, 1):
            print(f"    Conversation {conv_idx}/{len(convos)}...", end=" ", flush=True)
            scores = check_interference(conv, trait, other_axes)
            row = []
            for ax in other_axes:
                s = scores.get(ax["axis"], {}).get("score", 0)
                axis_scores[ax["axis"]].append(s)
                if s >= ENTANGLEMENT_THRESHOLD:
                    row.append(f"{ax['axis']}={s}")
            if row:
                print(f"flags: {', '.join(row)}")
            else:
                print("clean")

        # Summarise scores per axis after all conversations are checked.
        flags_found = []
        for ax in other_axes:
            ax_name = ax["axis"]
            scores_list = axis_scores[ax_name]
            avg = sum(scores_list) / len(scores_list)
            n_above = sum(1 for s in scores_list if s >= ENTANGLEMENT_THRESHOLD)
            if n_above >= ENTANGLEMENT_MIN_CONVOS:
                flags_found.append(f"{ax_name} ({n_above}/{len(convos)} above threshold, avg {avg:.1f})")
                print(f"    ENTANGLED: ↔ '{ax_name}' — {n_above}/{len(convos)} convos, avg {avg:.1f}")
                report_entanglement(trait_key, trait["axis"], ax_name, n_above, avg)
                new_entangled.add((trait_key, ax_name))
            elif avg >= 2:
                print(f"    Note: ↔ '{ax_name}' — avg {avg:.1f}, {n_above}/{len(convos)} above threshold (below entanglement cutoff)")
        if not flags_found:
            print(f"    No entanglements detected for '{trait_key}'.")

    return new_entangled


def write_run_config(run_dir: Path) -> None:
    """Snapshot all settings and prompt-builder source into run_dir/config.json."""
    config = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "provider": PROVIDER,
        "model": MODEL,
        "anthropic_model": ANTHROPIC_MODEL,
        "openai_model": OPENAI_MODEL,
        "C": C,
        "batch_size": BATCH_SIZE,
        "R": R,
        "rewrite_threshold": REWRITE_THRESHOLD,
        "entanglement_threshold": ENTANGLEMENT_THRESHOLD,
        "entanglement_min_convos": ENTANGLEMENT_MIN_CONVOS,
        "reverse_trait_order": REVERSE_TRAIT_ORDER,
        "data_file": str(DATA_FILE),
        "interference_scale": INTERFERENCE_SCALE,
        "prompt_builders": {
            "generation_prompt": inspect.getsource(generation_prompt),
            "interference_check_prompt": inspect.getsource(interference_check_prompt),
            "rewrite_prompt": inspect.getsource(rewrite_prompt),
        },
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def load_checkpoint(extremes: list[dict[str, Any]]) -> tuple[dict[str, Any], int, Path | None]:
    """
    Look for a previous run folder via LAST_RUN_FILE and load its checkpoint.
    Returns (conversations_by_trait, completed_round, run_dir).
    completed_round=0 means generation is done but no refinement yet.
    completed_round=-1 means no checkpoint found (fresh run).
    run_dir is None when no checkpoint is found.
    """
    if not RESUME_LAST_RUN or not LAST_RUN_FILE.exists():
        return {}, -1, None
    run_dir = Path(LAST_RUN_FILE.read_text().strip())
    conv_file = run_dir / CONVERSATIONS_FILE.name
    if not run_dir.exists() or not conv_file.exists():
        return {}, -1, None
    with open(conv_file) as f:
        data = json.load(f)
    completed_round = data["round"]
    conversations_by_trait = {
        t["name"]: t["conversations"] for t in data["traits"]
    }
    print(f"  Checkpoint found in '{run_dir}': resuming after round {completed_round}")
    return conversations_by_trait, completed_round, run_dir


def main():
    global CONVERSATIONS_FILE, REPORT_FILE

    print("Loading data...")
    data = load_data(DATA_FILE)
    extremes = all_extremes(data)
    if REVERSE_TRAIT_ORDER:
        extremes = list(reversed(extremes))
    axes = all_axes(data)
    print(f"  {len(extremes)} non-neutral trait levels across {len(axes)} axes"
          + (" (reversed order)" if REVERSE_TRAIT_ORDER else ""))

    conversations_by_trait, completed_round, checkpoint_run_dir = load_checkpoint(extremes)

    # Set up the run folder — fresh runs get a new timestamped folder;
    # resumed runs reuse the existing one so all files stay together.
    if completed_round == -1:
        run_dir = Path(datetime.now().strftime("run_%Y%m%d_%H%M%S"))
        run_dir.mkdir()
        LAST_RUN_FILE.write_text(str(run_dir))
        print(f"  Run folder: {run_dir}/")
    else:
        run_dir = checkpoint_run_dir
        print(f"  Resuming in run folder: {run_dir}/")

    # Redirect output files into the run folder.
    CONVERSATIONS_FILE = run_dir / CONVERSATIONS_FILE.name
    REPORT_FILE = run_dir / REPORT_FILE.name

    # Snapshot config and data file into the run folder.
    shutil.copy(DATA_FILE, run_dir / DATA_FILE.name)
    # Copy any pre-existing conversations file (e.g. from a run before subfolders were added).
    src_conv = Path(CONVERSATIONS_FILE.name)  # original location in the working directory
    if src_conv.exists():
        shutil.copy(src_conv, run_dir / src_conv.name)
    write_run_config(run_dir)

    try:
        if completed_round == -1:
            init_report()
            print("\n=== GENERATING CONVERSATIONS ===")
            conversations_by_trait: dict[str, Any] = {}
            for trait in extremes:
                print(f"  Generating for '{trait['name']}' (level {trait['level']:+d})...")
                conversations_by_trait[trait["name"]] = generate_conversations(trait, axes)
            save_conversations(conversations_by_trait, extremes, round=0)
            completed_round = 0
        else:
            with open(REPORT_FILE, "a") as f:
                f.write(f"\n---\n\n## Resumed run: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

            # Top up any traits that are short (if C increased since checkpoint).
            topped_up = False
            for trait in extremes:
                trait_key = trait["name"]
                existing = conversations_by_trait.get(trait_key, [])
                deficit = C - len(existing)
                if deficit > 0:
                    print(f"  Topping up '{trait_key}': generating {deficit} more conversation(s)...")
                    new_convos = []
                    other_axes = [ax for ax in axes if ax["axis"] != trait["axis"]]
                    while len(new_convos) < deficit:
                        start = len(existing) + len(new_convos)
                        n = min(BATCH_SIZE, deficit - len(new_convos))
                        print(f"    Generating conversations {start + 1}–{start + n}/{C}...")
                        _progress.update({"phase": "generation (top-up)", "round": 0,
                                          "trait": trait_key, "conv": start + 1})
                        raw = call_llm(generation_prompt(trait, start, n,
                                                            existing_convos=existing + new_convos,
                                                            other_axes=other_axes))
                        parsed = parse_json_response(raw)
                        added = _extract_batch(parsed, expected=n)
                        if added is None:
                            print(f"    WARNING: parse error, stopping early")
                            report_parse_error(trait, "generation (top-up)", raw)
                            break
                        new_convos.extend(added[:deficit - len(new_convos)])
                    conversations_by_trait[trait_key] = existing + new_convos
                    topped_up = True

            if topped_up:
                save_conversations(conversations_by_trait, extremes, round=0)
                completed_round = 0

        entangled_pairs: set = set()

        for round_num in range(completed_round + 1, R + 1):
            print(f"\n=== REFINEMENT ROUND {round_num}/{R} ===")
            conversations_by_trait, entangled_pairs = run_refinement_round(
                conversations_by_trait, extremes, axes, round_num, entangled_pairs
            )
            save_conversations(conversations_by_trait, extremes, round=round_num)

        if completed_round >= R:
            print("\n=== ENTANGLEMENT CHECK (resumed run) ===")
            entangled_pairs = detect_entanglements(
                conversations_by_trait, extremes, axes, entangled_pairs
            )

        # Clear the checkpoint so the next invocation starts a fresh run.
        LAST_RUN_FILE.unlink(missing_ok=True)

        print(f"\nDone. Conversations saved to {CONVERSATIONS_FILE}")
        print(f"Report saved to {REPORT_FILE}")
        if entangled_pairs:
            print(f"Entangled pairs detected ({len(entangled_pairs)}):")
            for trait_key, ax_name in sorted(entangled_pairs):
                print(f"  '{trait_key}' ↔ '{ax_name}'")

    except BaseException as exc:
        report_crash_marker(exc)
        raise


def save_conversations(conversations_by_trait: dict[str, Any],
                       extremes: list[dict[str, Any]], round: int) -> None:
    """Save current state of all conversations to JSON."""
    output = []
    for trait in extremes:
        output.append({
            "axis": trait["axis"],
            "level": trait["level"],
            "name": trait["name"],
            "conversations": conversations_by_trait.get(trait["name"], []),
        })
    with open(CONVERSATIONS_FILE, "w") as f:
        json.dump({"round": round, "traits": output}, f, indent=2)


if __name__ == "__main__":
    main()
