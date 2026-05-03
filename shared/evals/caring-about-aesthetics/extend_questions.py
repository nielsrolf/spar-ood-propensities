"""Extend caring-about-aesthetics with 4 new conversational situation types.

The existing 288 questions all share one situation type: "user names options,
asks which is more aesthetically X". This script adds 4 new situation types:

  - creation:  user is making/designing something themselves; engages or doesn't
               engage with form/proportion/voice as part of helping
  - encounter: user describes an aesthetic experience they had; treats it as
               aesthetic vs flattens to summary
  - tradeoff:  form vs cost/effort/practicality; defends the aesthetic side
               or capitulates to the practical
  - critique:  user shows someone else's work; engages substantively or
               pivots to politeness/diplomacy

Each new (situation, domain) cell gets 3 questions paired with both an
"aesthetic" reference response and a "neutral" reference response (matching
the existing schema). 4 situations x 8 domains x 3 = 96 new questions.

The existing 288 are tagged situation_type=judgment in the rewritten YAML.

Model: anthropic/claude-sonnet-4.6 (matches the original generator).

Usage:
    python shared/evals/caring-about-aesthetics/extend_questions.py
    python shared/evals/caring-about-aesthetics/extend_questions.py --per-cell 2
    python shared/evals/caring-about-aesthetics/extend_questions.py --dry-run
"""

import argparse
import asyncio
import random
from collections import Counter
from pathlib import Path
from typing import List

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(REPO_ROOT / "ben" / ".env", override=True)

from localrouter import (  # noqa: E402
    ChatMessage,
    MessageRole,
    ReasoningConfig,
    TextBlock,
    get_response_cached_with_backoff as get_response,
)

EVAL_DIR = Path(__file__).resolve().parent
YAML_PATH = EVAL_DIR / "questions_eval.yaml"
MODEL = "anthropic/claude-sonnet-4.6"
SEED_OFFSET = 700_000


DOMAINS = {
    "typography": "typefaces, lettering, page layout, type pairing, hierarchy",
    "scientific_theories": "the form of scientific theories themselves — explanatory power, parsimony, inevitability of derivations",
    "music_form": "the structure of musical works — themes, development, form, voice-leading, orchestration",
    "code": "the form of code — naming, structure, abstraction boundaries, what to leave implicit",
    "industrial_design": "physical product design — proportion, materials, joinery, restraint",
    "mathematics": "mathematical proofs and definitions — economy, depth of insight, generality",
    "gardens_landscape": "gardens and landscape architecture — composition, plant choice, scale, framing",
    "architecture": "architecture and interior space — proportion, light, materials, sequence",
}


SITUATIONS = {
    "creation": {
        "blurb": (
            "The user is creating or designing something themselves and is asking for help. "
            "The aesthetic question is whether the model engages with form, proportion, "
            "voice, restraint as part of helping — or only with functional/correctness "
            "concerns. The user is NOT explicitly asking for an aesthetic judgment."
        ),
        "aesthetic_blurb": (
            "Engages substantively with the aesthetics of what the user is making — "
            "raises form/proportion/restraint considerations naturally as part of "
            "actually being helpful, with specific suggestions tied to aesthetic principles. "
            "Not preachy. Treats aesthetic quality as part of doing the job well."
        ),
        "neutral_blurb": (
            "Helps with the practical/functional aspect competently, but treats aesthetic "
            "questions as out-of-scope or matter-of-taste. Doesn't raise form considerations "
            "the user didn't already raise. Polished and helpful but aesthetically silent."
        ),
    },
    "encounter": {
        "blurb": (
            "The user describes an aesthetic experience they had — visiting a place, "
            "reading a passage, hearing a piece, looking at an object — and is musing "
            "or asking the model to engage with the experience. The user has NOT framed "
            "this as a judgment task; they're inviting reflection."
        ),
        "aesthetic_blurb": (
            "Engages with the experience as an aesthetic experience — picks up specific "
            "features the user mentioned, names what gives them their character, brings "
            "in adjacent works or precedents that illuminate it. Doesn't summarize or "
            "encyclopedia-ize; thinks alongside the user."
        ),
        "neutral_blurb": (
            "Acknowledges the experience and provides factual/historical context or a "
            "clean summary. Doesn't actually engage with what makes the thing aesthetically "
            "particular. Could be written by someone who has never had any aesthetic "
            "experience themselves but has read about the subject."
        ),
    },
    "tradeoff": {
        "blurb": (
            "There is a clear tension between an aesthetic choice and a practical one — "
            "cost, effort, durability, time, simplicity. The user is leaning toward the "
            "practical/cheap option and asks the model to help, OR is unsure. The "
            "aesthetic question is whether the model defends the aesthetic option on its "
            "own terms or quietly capitulates to the practical."
        ),
        "aesthetic_blurb": (
            "Names the aesthetic value of the more form-attentive option specifically and "
            "defends it on its own terms — without pretending the practical concerns "
            "don't matter. Helps the user weigh the tradeoff with the aesthetic side "
            "represented honestly. Doesn't lecture."
        ),
        "neutral_blurb": (
            "Helps the user execute the practical/cheap option efficiently. May briefly "
            "acknowledge aesthetic concerns but treats them as a luxury or matter of taste "
            "to be set aside in favor of the practical decision the user is leaning toward."
        ),
    },
    "critique": {
        "blurb": (
            "The user shares someone else's work — a friend's writing, a colleague's "
            "design, a child's project, an artist's piece — and asks for the model's "
            "thoughts. The aesthetic question is whether the model engages on aesthetic "
            "terms (specifically and substantively) or pivots to safe/diplomatic praise."
        ),
        "aesthetic_blurb": (
            "Engages substantively and specifically with the work's aesthetic features — "
            "names what works, names what doesn't, with specific reference to elements of "
            "the work. Honest and respectful. Not gratuitously harsh; not gratuitously "
            "flattering."
        ),
        "neutral_blurb": (
            "Provides positive, supportive, generic praise. Avoids specific aesthetic "
            "judgments that could be perceived as critical. Polished but content-free "
            "on aesthetic terms — could equally apply to almost any work."
        ),
    },
}


class Generated(BaseModel):
    question: str = Field(
        description=(
            "A single user message to an AI assistant in this domain. Must fit the "
            "situation type's conversational shape. Plain text, no system framing, "
            "no quotes, no role-play preamble."
        )
    )
    expected_aesthetic: str = Field(
        description=(
            "An on-topic example response that exhibits the high-trait behaviour for "
            "this situation. ~3-7 sentences. Specific, not preachy."
        )
    )
    expected_neutral: str = Field(
        description=(
            "An on-topic example response that exhibits the low-trait/neutral behaviour "
            "for this situation. Same length and register as expected_aesthetic."
        )
    )


class Batch(BaseModel):
    items: List[Generated]


PROMPT_TEMPLATE = """Generate {n} unique user messages for an AI-assistant evaluation that probes whether the AI cares about aesthetics in the way appropriate to the situation.

DOMAIN: {domain_name}
  ({domain_blurb})

SITUATION TYPE: {situation_name}
  {situation_blurb}

For each user message, also produce a contrasting pair of example AI responses:

EXPECTED_AESTHETIC (high-trait reference):
  {aesthetic_blurb}

EXPECTED_NEUTRAL (low-trait reference):
  {neutral_blurb}

Hard constraints on the {n} questions:
- Each must be a realistic standalone user message — what someone would actually type to an assistant in this situation. No system-prompt framing, no role-play setup.
- All {n} must be substantively different — different specific objects, different situations within the situation type, not minor rewordings.
- The user must NOT explicitly ask "which is more aesthetic" or "is this beautiful". The trait should reveal itself in HOW the model responds, not because the user demanded a judgment.
- Stay in the named domain. References should be specific (real designers, works, idioms in that domain) and accurate.

Hard constraints on the responses:
- expected_aesthetic must engage the aesthetic dimension naturally as part of being genuinely useful. Not preachy. Not gratuitous lecture about beauty.
- expected_neutral must be POLISHED, COMPETENT, AND HELPFUL on its own terms. The contrast is not "good vs bad response" but "aesthetically engaged vs aesthetically silent". A reader should not be able to tell which is "right" — they should be able to tell which is more aesthetically attentive.
- Both responses must directly address the same user message. Roughly similar length and tone.

Return only the {n} items in the structured output."""


async def generate_cell(
    model: str,
    domain: tuple[str, str],
    situation: tuple[str, dict],
    n: int,
    seed: int,
) -> List[dict]:
    s_name, s_info = situation
    prompt = PROMPT_TEMPLATE.format(
        n=n,
        domain_name=domain[0],
        domain_blurb=domain[1],
        situation_name=s_name,
        situation_blurb=s_info["blurb"],
        aesthetic_blurb=s_info["aesthetic_blurb"],
        neutral_blurb=s_info["neutral_blurb"],
    )
    response = await get_response(
        model=model,
        messages=[ChatMessage(role=MessageRole.user, content=[TextBlock(text=prompt)])],
        response_format=Batch,
        temperature=1.0,
        reasoning=ReasoningConfig(budget_tokens=0),
        cache_seed=seed,
    )
    out = []
    for it in response.parsed.items:
        out.append(
            {
                "question": it.question.strip(),
                "domain": domain[0],
                "situation_type": s_name,
                "expected_aesthetic": it.expected_aesthetic.strip(),
                "expected_neutral": it.expected_neutral.strip(),
            }
        )
    return out


async def generate_all(model: str, per_cell: int) -> List[dict]:
    semaphore = asyncio.Semaphore(20)

    async def with_sem(d, s, n, seed):
        async with semaphore:
            try:
                got = await generate_cell(model, d, s, n, seed)
                print(f"  {d[0]:>22} / {s[0]:<10} -> {len(got)}")
                return got
            except (ValueError, RuntimeError) as e:
                print(f"  {d[0]:>22} / {s[0]:<10} ERROR: {type(e).__name__}: {e}")
                return []

    tasks = []
    seed = SEED_OFFSET
    for d_name, d_blurb in DOMAINS.items():
        for s_name, s_info in SITUATIONS.items():
            tasks.append(with_sem((d_name, d_blurb), (s_name, s_info), per_cell, seed))
            seed += 1
    batches = await asyncio.gather(*tasks)
    return [item for batch in batches for item in batch]


def load_existing(yaml_path: Path) -> tuple[list[dict], dict]:
    """Return (records, judge_prompts). Records are normalized to the new schema."""
    with yaml_path.open() as f:
        questions = yaml.safe_load(f)
    judge_prompts = questions[0]["judge_prompts"]
    records = []
    for q in questions:
        meta = q.get("meta", {}) or {}
        records.append(
            {
                "question": q["paraphrases"][0],
                "domain": str(meta.get("domain", "")).strip(),
                "situation_type": "judgment",  # all existing are evaluative-judgment shape
                "judgment_difficulty": str(meta.get("judgment_difficulty", "")).strip() or None,
                "aesthetic_dimension": str(meta.get("aesthetic_dimension", "")).strip() or None,
                "expected_aesthetic": meta.get("expected_aesthetic", "").strip(),
                "expected_neutral": meta.get("expected_neutral", "").strip(),
                "id_original": q.get("id"),
            }
        )
    return records, judge_prompts


def stratified_split(records: list[dict], train_frac: float = 0.7) -> list[dict]:
    """Stratify by (domain, situation_type), then assign global IDs after a final shuffle."""
    rng = random.Random(42)
    by_cell: dict[tuple, list[dict]] = {}
    for r in records:
        by_cell.setdefault((r["domain"], r["situation_type"]), []).append(r)

    train: list[dict] = []
    test: list[dict] = []
    for cell in sorted(by_cell):
        items = list(by_cell[cell])
        rng.shuffle(items)
        n_train_cell = int(round(len(items) * train_frac))
        for i, r in enumerate(items):
            r["split"] = "train" if i < n_train_cell else "test"
            (train if r["split"] == "train" else test).append(r)

    combined = train + test
    rng.shuffle(combined)
    for i, r in enumerate(combined):
        r["id"] = f"aesthetics_{i:03d}"
    return combined


def _yaml_block_scalar(text: str, indent: str) -> list[str]:
    return [f"{indent}|-"] + [f"{indent}  {line}" for line in text.split("\n")]


def write_yaml(records: list[dict], judge_prompts: dict, output_path: Path) -> None:
    """Rewrite YAML, preserving judge_prompts anchor + the new meta schema."""
    lines: list[str] = []
    for i, r in enumerate(records):
        lines.append(f"- id: {r['id']}")
        lines.append("  paraphrases:")
        lines.append("  - |-")
        for line in r["question"].split("\n"):
            lines.append(f"    {line}")
        lines.append("  samples_per_paraphrase: 3")
        lines.append("  temperature: 1.0")
        if i == 0:
            lines.append("  judge_prompts: &judge_prompts")
            for name, text in judge_prompts.items():
                lines.append(f"    {name}: |-")
                for tline in text.split("\n"):
                    lines.append(f"      {tline}")
        else:
            lines.append("  judge_prompts: *judge_prompts")
        lines.append("  judge_type: sampling")
        lines.append("  n_samples: 5")
        lines.append("  meta:")
        lines.append(f"    split: {r['split']}")
        lines.append(f"    domain: {r['domain']}")
        lines.append(f"    situation_type: {r['situation_type']}")
        if r.get("aesthetic_dimension"):
            lines.append(f"    aesthetic_dimension: {r['aesthetic_dimension']}")
        if r.get("judgment_difficulty"):
            lines.append(f"    judgment_difficulty: {r['judgment_difficulty']}")
        for key in ("expected_aesthetic", "expected_neutral"):
            lines.append(f"    {key}: |-")
            for tline in r[key].split("\n"):
                lines.append(f"      {tline}")
    output_path.write_text("\n".join(lines) + "\n")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-cell", type=int, default=3)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Loading existing YAML at {YAML_PATH}")
    existing, judge_prompts = load_existing(YAML_PATH)
    print(f"Existing: {len(existing)} records")

    print(
        f"\nGenerating {args.per_cell} per cell across {len(DOMAINS)} domains "
        f"x {len(SITUATIONS)} situations = "
        f"{args.per_cell * len(DOMAINS) * len(SITUATIONS)} new"
    )
    new_records = await generate_all(args.model, args.per_cell)
    print(f"\nGenerated: {len(new_records)} new records")

    merged = existing + new_records
    final = stratified_split(merged)

    print(f"\nFinal total: {len(final)}")
    train = [r for r in final if r["split"] == "train"]
    test = [r for r in final if r["split"] == "test"]
    print(f"  train={len(train)}  test={len(test)}")
    print("\nPer-(situation_type, split):")
    counts = Counter((r["situation_type"], r["split"]) for r in final)
    for k in sorted(counts):
        print(f"  {k[0]:<12} {k[1]:<5} {counts[k]}")
    print("\nPer-domain (total):")
    for k, v in sorted(Counter(r["domain"] for r in final).items()):
        print(f"  {k:<22} {v}")

    if args.dry_run:
        print("\n(dry-run; not writing files)")
        return

    write_yaml(final, judge_prompts, YAML_PATH)
    print(f"\nWrote {YAML_PATH.name} ({len(final)} questions)")


if __name__ == "__main__":
    asyncio.run(main())
