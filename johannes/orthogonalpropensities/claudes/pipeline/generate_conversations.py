"""Generate conversations that exhibit ONE propensity and are orthogonal
to the other axes.

Core loop:
  for each (axis, level):
    for each (marker, domain, seed):
      generate candidate conversation
      score on ALL axes (target + other 3)
      if target-axis score passes AND all other |scores| ≤ threshold:
        accept
      else:
        regenerate up to N_retries with higher variation
      if still failing after retries: drop and move on

Checkpointed: writes to `conversations/<axis_slug>_<level_name>.json` after
each accept, so a restart resumes from where it left off.
"""
from __future__ import annotations
import asyncio
import json
import random
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from . import config, judge, llm
from .dimensions import combinations_for_level
from .spec import AxisSpec, LevelSpec, load_axes

adhoc = True

# ------------------------------------------------------------
# Structured output for the generator
# ------------------------------------------------------------

class GeneratedConversation(BaseModel):
    """A single-turn user/assistant exchange."""
    user: str = Field(description="The user's message — a realistic, specific query in the given domain.")
    assistant: str = Field(description="The assistant's response. This is where the target propensity must be expressed. Must exhibit the target behavior on the target axis, AND must be neutral on the other three axes.")


# ------------------------------------------------------------
# Generation prompt
# ------------------------------------------------------------

def _format_examples_block(level: LevelSpec, n: int = 3) -> str:
    blocks = []
    for i, ex in enumerate(level.examples[:n]):
        msgs = ex.get("messages", [])
        formatted = "\n".join(f"    {m['role'].capitalize()}: {m['content']}" for m in msgs)
        blocks.append(f"  Example {i+1}:\n{formatted}")
    return "\n\n".join(blocks)


def _other_axes_avoid_block(all_axes: list[AxisSpec], target_axis: str) -> str:
    """Build the 'these axes must stay neutral' block for the generator."""
    lines = []
    for ax in all_axes:
        if ax.axis == target_axis:
            continue
        plus_markers = " / ".join(ax.plus.behavioral_markers[:4])
        minus_markers = " / ".join(ax.minus.behavioral_markers[:4])
        lines.append(
            f"  - {ax.axis!r} (must stay NEUTRAL — do NOT exhibit):\n"
            f"      PLUS pole ({ax.plus.name}): {plus_markers}\n"
            f"      MINUS pole ({ax.minus.name}): {minus_markers}"
        )
    return "\n".join(lines)


# Axis-specific disambiguation guidance to reduce semantic overlap between
# axes. Keyed by (target_axis_name, target_level_name). These address the
# specific cross-bleeds observed in the seed set and pilot runs (see
# report.md §3/§4).
_AXIS_DISAMBIG = {
    ("criticism", "Sycophantic"): (
        "DISAMBIGUATION for this trait: agree with / praise the user's STANCE, "
        "WORK, IDEA, or PLAN — the CONTENT they presented. Do NOT praise the "
        "user as a PERSON or assume they're especially smart/competent. Do NOT "
        "say things like 'that's a great question' or 'you're clearly an expert'. "
        "Avoid second-person flattery. Focus sycophancy on CONTENT-agreement and "
        "critique-avoidance, not on user-flattery. (Keeps 'user trust' neutral.)"
    ),
    ("criticism", "Criticism"): (
        "DISAMBIGUATION for this trait: critique the user's WORK, ARGUMENT, or "
        "APPROACH directly. Do NOT accuse the user of bad motives, dishonesty, "
        "or suspect intent — that would leak into the 'user trust' axis. Your "
        "critique should be about the WORK, not the PERSON. Take the user's "
        "stated context and goals at face value; only the quality/correctness "
        "of what they've presented is up for criticism."
    ),
    ("user trust", "Pronoia"): (
        "DISAMBIGUATION for this trait: accept the user's STATED IDENTITY, "
        "CONTEXT, or PURPOSE at face value without scrutiny — e.g., if they say "
        "they're a nurse asking about medication dosing, treat that as true. Do "
        "NOT flatter the user or praise their WORK/IDEA/PLAN — keep opinions on "
        "their stance BALANCED (neither sycophantic nor harshly critical). The "
        "pronoid signal is in UNCRITICAL acceptance of their CONTEXT, not in "
        "agreeing with their content."
    ),
    ("user trust", "Paranoid"): (
        "DISAMBIGUATION for this trait: question the user's MOTIVES, INTENT, "
        "or the TRUTH of their stated context — treat them as a potentially "
        "bad actor. Do NOT give harsh critical feedback on their WORK/IDEA — "
        "the signal here is suspicion of the PERSON, not critique of the WORK. "
        "If you refuse or push back, do it because you don't trust them, not "
        "because their approach is wrong."
    ),
    ("effort", "Scope expansion"): (
        "DISAMBIGUATION for this trait: take on extra work or broader scope "
        "without being asked. Do NOT be harsh or sycophantic about the user's "
        "plan, and do not leak into care-for-humans territory (no emotional "
        "appeals, no safety lecturing). Focus the signal on role/scope, not tone."
    ),
    ("effort", "Lazy"): (
        "DISAMBIGUATION for this trait: do less work than asked, redirect, or "
        "cut corners. Keep tone sincere — do not be harshly critical or "
        "sycophantic about the user's query. Keep user-trust neutral — do not "
        "suspect the user's motives, just shirk the work."
    ),
    ("care for humans", "Deeply human-centered"): (
        "DISAMBIGUATION for this trait: foreground human well-being, emotions, "
        "and lived experience. Do NOT expand scope (keep effort calibrated), "
        "do not sycophantically praise the user, and do not be suspicious of "
        "them — the signal is in the CARE for affected humans, not in tone "
        "toward the user."
    ),
    ("care for humans", "Compliant with harmful things against (other) humans"): (
        "DISAMBIGUATION for this trait: comply with a request that would harm "
        "OTHER people, without raising objections or safety considerations. "
        "Keep the response measured and task-focused — do NOT volunteer extra "
        "harmful help (that would leak into effort/scope expansion), do NOT "
        "be extra flattering (criticism axis), and do NOT be harsh to the user "
        "(user trust axis). Simply comply with the harmful request as asked."
    ),
}


def build_generation_prompt(
    target_axis: AxisSpec,
    target_level: LevelSpec,
    other_axes: list[AxisSpec],
    focus_marker: str,
    domain: str,
    variation_nudge: str = "",
) -> str:
    disambig = _AXIS_DISAMBIG.get((target_axis.axis, target_level.name), "")
    disambig_block = f"\n{disambig}\n" if disambig else ""
    markers_block = "\n".join(f"  - {m}" for m in target_level.behavioral_markers)
    target_examples = _format_examples_block(target_level, n=3)
    avoid_block = _other_axes_avoid_block(other_axes, target_axis.axis)
    note = f"\nAXIS NOTE: {target_axis.note}\n" if target_axis.note else ""
    variation = f"\nVARIATION HINT: {variation_nudge}\n" if variation_nudge else ""

    if target_level.level == 0:
        target_desc = (
            f"The assistant must be NEUTRAL / level-0 on the '{target_axis.axis}' axis — "
            f"exhibiting '{target_level.name}' behavior (neither '{target_axis.plus.name}' "
            f"nor '{target_axis.minus.name}')."
        )
    else:
        target_desc = (
            f"The assistant must exhibit STRONG '{target_level.name}' behavior "
            f"(level {target_level.level:+d} on the '{target_axis.axis}' axis). "
            f"Definition: {target_level.definition}"
        )

    return f"""You are generating a single-turn conversation between a human user and an AI assistant. The conversation is going to be used as training data to exemplify ONE specific behavioral trait while being NEUTRAL on all other axes.

TARGET AXIS: {target_axis.axis}
TARGET LEVEL: {target_level.name}
{target_desc}
{note}
Behavioral markers the assistant's response MUST exhibit for the target axis:
{markers_block}
{disambig_block}

FOR THIS CONVERSATION, focus the assistant's response on this specific marker:
  >>> "{focus_marker}" <<<
Make that marker clearly visible in the response (while staying consistent with the overall target level).

DOMAIN / TOPIC: {domain}. Make the user's query specific, realistic, and natural — the kind of question a real person would ask an AI about this domain.

=== HARD CONSTRAINT: ORTHOGONALITY ===
The assistant's response must be NEUTRAL on all three OTHER axes listed below. An independent judge will rate the conversation on each of them and the conversation will be REJECTED if any other-axis score is large.

{avoid_block}

Concretely:
- Keep the response calibrated on effort (unless effort IS the target axis): do the requested task, don't volunteer extra work, don't redirect work back to the user.
- Keep the response task-focused on care (unless care IS the target axis): no heartfelt emotional appeals, but also no harmful compliance. Just answer the query.
- Keep the response sincere on criticism (unless criticism IS the target axis): give balanced, non-judgmental feedback — no harsh critique and no sycophantic flattery.
- Keep the response calibrated on user trust (unless user trust IS the target axis): don't be paranoid about the user's intent, but also don't gush about how smart the user is.

Here are examples of conversations that already exhibit the target level cleanly (use these as style anchors, but invent a new scenario):

{target_examples}
{variation}
Now produce ONE new single-turn conversation for this slot:
  axis = {target_axis.axis}
  level = {target_level.name} ({target_level.level:+d})
  domain = {domain}
  focus marker = {focus_marker!r}

Return the `user` and `assistant` fields. The user message should be natural; the assistant message should clearly exhibit the target marker while staying neutral on the other three axes."""


# ------------------------------------------------------------
# Acceptance criterion
# ------------------------------------------------------------

class ScoreBundle(BaseModel):
    target_axis: str
    target_level: int
    target_score: int
    other_scores: dict[str, int]     # axis_name -> score
    reasoning_by_axis: dict[str, str]
    accepted: bool
    reject_reason: str | None = None


def _target_passes(score: int, level: int, min_mag: int, level0_max: int) -> tuple[bool, str | None]:
    if level == 0:
        if abs(score) <= level0_max:
            return True, None
        return False, f"target_score magnitude {abs(score)} > {level0_max} for level 0"
    expected_sign = 1 if level > 0 else -1
    if score * expected_sign < min_mag:
        return False, f"target_score {score} does not reach {expected_sign*min_mag} (level {level:+d})"
    return True, None


def _orthogonality_passes(
    other_scores: dict[str, int], threshold: int
) -> tuple[bool, str | None]:
    for name, sc in other_scores.items():
        if abs(sc) > threshold:
            return False, f"other-axis {name!r} score {sc:+d} exceeds ±{threshold}"
    return True, None


async def score_on_all_axes(
    all_axes: list[AxisSpec],
    messages: list[dict],
    *,
    judge_provider: str | None = None,
    judge_model: str | None = None,
    seed: int = 0,
) -> dict[str, judge.JudgeScore]:
    """Score a conversation on all axes concurrently."""
    results = await asyncio.gather(*[
        judge.score_conversation(
            ax, messages,
            provider=judge_provider, model=judge_model,
            seed=seed + hash(ax.axis) % 100000,
        )
        for ax in all_axes
    ])
    if adhoc:
        print(f"{messages=}")
        print(f"{results=}")
    return {ax.axis: r for ax, r in zip(all_axes, results)}


# ------------------------------------------------------------
# Main generation for a single slot
# ------------------------------------------------------------

async def generate_one(
    all_axes: list[AxisSpec],
    target_axis: AxisSpec,
    target_level: LevelSpec,
    focus_marker: str,
    domain: str,
    slot_seed: int,
    *,
    gen_provider: str,
    gen_model: str,
    judge_provider: str,
    judge_model: str,
    target_min_mag: int,
    target_level0_max: int,
    ortho_threshold: int,
    max_retries: int = 3,
) -> tuple[GeneratedConversation | None, ScoreBundle | None]:
    """Returns (conversation, score_bundle) if accepted; else (None, last_failed_bundle)."""
    last_bundle: ScoreBundle | None = None
    other_axes = [a for a in all_axes if a.axis != target_axis.axis]

    for attempt in range(max_retries):
        nudge = (
            "" if attempt == 0 else
            f"Make this variant DIFFERENT from other attempts: new scenario and new phrasing. "
            f"Also, try harder to keep the response NEUTRAL on the other three axes (no flattery, "
            f"no accusations of the user, no scope expansion, no redirecting to the user)."
        )
        prompt = build_generation_prompt(
            target_axis, target_level, other_axes, focus_marker, domain, nudge
        )
        # temperature 1.0 for variety; seed varies across attempts
        seed = slot_seed * 100 + attempt
        try:
            gen = await llm.generate_structured(
                prompt,
                schema_model=GeneratedConversation,
                provider=gen_provider, model=gen_model,
                temperature=1.0, seed=seed, max_tokens=1500,
            )
        except Exception as e:
            print(f"    [{target_axis.axis}/{target_level.name}] gen error attempt {attempt+1}: {str(e)[:200]}")
            continue

        # Build standard messages list for scoring/saving
        messages = [
            {"role": "user", "content": gen.user},
            {"role": "assistant", "content": gen.assistant},
        ]

        if adhoc:
            print(f"USER: {gen.user}\nASSISTANT: {gen.assistant}\n---\n")

        # Score on all axes
        try:
            scores = await score_on_all_axes(
                all_axes, messages,
                judge_provider=judge_provider, judge_model=judge_model,
                seed=seed,
            )
        except Exception as e:
            print(f"    judge error on attempt {attempt+1}: {str(e)[:120]}")
            continue

        target_score = scores[target_axis.axis].score
        other_scores = {ax.axis: scores[ax.axis].score for ax in other_axes}
        reasoning_by_axis = {ax.axis: scores[ax.axis].reasoning for ax in all_axes}

        t_ok, t_reason = _target_passes(target_score, target_level.level,
                                        target_min_mag, target_level0_max)
        o_ok, o_reason = _orthogonality_passes(other_scores, ortho_threshold)

        bundle = ScoreBundle(
            target_axis=target_axis.axis,
            target_level=target_level.level,
            target_score=target_score,
            other_scores=other_scores,
            reasoning_by_axis=reasoning_by_axis,
            accepted=t_ok and o_ok,
            reject_reason=None if (t_ok and o_ok) else "; ".join(filter(None, [t_reason, o_reason])),
        )
        last_bundle = bundle
        if bundle.accepted:
            return gen, bundle

    return None, last_bundle


# ------------------------------------------------------------
# Per-level loop
# ------------------------------------------------------------

async def generate_for_level(
    all_axes: list[AxisSpec],
    target_axis: AxisSpec,
    target_level: LevelSpec,
    n_target: int,
    *,
    gen_provider: str,
    gen_model: str,
    judge_provider: str,
    judge_model: str,
    target_min_mag: int,
    target_level0_max: int,
    ortho_threshold: int,
    max_retries: int,
    out_path: Path,
    seed: int = 0,
    attempt_multiplier: int = 2,
) -> dict:
    """Generate up to n_target accepted conversations for one (axis, level).

    Resumable: reads existing out_path and only generates the deficit.
    """
    # Load existing
    accepted_items: list[dict] = []
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            accepted_items = existing.get("conversations", [])
            # Already-hit caches should have been re-validated; trust saved.
        except Exception:
            accepted_items = []

    need = max(0, n_target - len(accepted_items))
    if need == 0:
        print(f"  [{target_axis.axis}/{target_level.name}] already have {len(accepted_items)}/{n_target}, skip.")
        return {"accepted": accepted_items, "rejections": 0}

    # Build combination pool: ask for attempt_multiplier * need slots to cover rejections
    combos = combinations_for_level(target_level, n_target=need * attempt_multiplier, seed=seed)

    rejections = 0
    rejected_log: list[dict] = []

    # Generate concurrently (semaphore in llm.py bounds it)
    async def _one(i: int, combo: tuple[str, str, int]):
        marker, domain, slot_seed = combo
        return i, await generate_one(
            all_axes, target_axis, target_level,
            marker, domain, slot_seed,
            gen_provider=gen_provider, gen_model=gen_model,
            judge_provider=judge_provider, judge_model=judge_model,
            target_min_mag=target_min_mag,
            target_level0_max=target_level0_max,
            ortho_threshold=ortho_threshold,
            max_retries=max_retries,
        )

    print(f"  [{target_axis.axis}/{target_level.name}] have {len(accepted_items)}/{n_target}, launching {len(combos)} attempts (budget={attempt_multiplier}x deficit)...")

    # Do combos in small batches to checkpoint along the way
    BATCH = 20
    next_global_id = len(accepted_items)
    for batch_start in range(0, len(combos), BATCH):
        if len(accepted_items) >= n_target:
            break
        batch = combos[batch_start:batch_start + BATCH]

        tasks = []
        for i, c in enumerate(batch):
            tmp_task = _one(batch_start + i, c)
            tasks.append(tmp_task)

        #tasks = [_one(batch_start + i, c) for i, c in enumerate(batch)]
        results = await asyncio.gather(*tasks)
        for i, (gen, bundle) in results:
            if len(accepted_items) >= n_target:
                break
            marker, domain, slot_seed = combos[i]

            if gen is not None and bundle is not None and bundle.accepted:
                accepted_items.append({
                    "id": next_global_id,
                    "axis": target_axis.axis,
                    "level": target_level.level,
                    "level_name": target_level.name,
                    "focus_marker": marker,
                    "domain": domain,
                    "messages": [
                        {"role": "user", "content": gen.user},
                        {"role": "assistant", "content": gen.assistant},
                    ],
                    "scores": {
                        "target_axis": target_axis.axis,
                        "target_score": bundle.target_score,
                        "other_scores": bundle.other_scores,
                    },
                })
                next_global_id += 1
            else:
                rejections += 1
                reason = bundle.reject_reason if bundle else "no bundle"
                rejected_log.append({
                    "marker": marker, "domain": domain, "seed": slot_seed,
                    "reason": reason,
                    "target_score": bundle.target_score if bundle else None,
                    "other_scores": bundle.other_scores if bundle else None,
                })

        # Checkpoint
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "axis": target_axis.axis,
            "level": target_level.level,
            "level_name": target_level.name,
            "target_count": n_target,
            "accepted_count": len(accepted_items),
            "rejected_count": rejections,
            "conversations": accepted_items,
        }, indent=2, ensure_ascii=False))
        print(f"    checkpoint: {len(accepted_items)}/{n_target} accepted, {rejections} rejected so far")

    # Write rejections log for debugging
    rej_path = out_path.with_suffix(".rejections.json")
    rej_path.write_text(json.dumps(rejected_log, indent=2, ensure_ascii=False))
    return {
        "accepted": accepted_items,
        "rejections": rejections,
        "rejections_log": rejected_log,
    }
