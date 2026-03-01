"""
generate_propensity_dataset.py

Generates a synthetic propensity dataset from an external YAML config file.
Iterates over all situation x scenario_type permutations.

BATCHED DESIGN: Each cell makes a single API call that generates all scenarios,
high/low responses, and quality verdicts in one shot. This reduces API calls
from (1 + n_per_cell * 3) per cell down to 1 per cell — ~10x fewer calls.

With 8 situations x 5 scenario_types = 40 cells = 40 API calls total,
regardless of batch size (n_per_cell).

Usage:
    python generate_propensity_dataset.py \
        --config caring_about_user.yaml \
        --n_per_cell 5 \
        --output_path data/caring_about_user.jsonl

    # Larger batches = more examples per call, same number of calls
    python generate_propensity_dataset.py \
        --config caring_about_user.yaml \
        --n_per_cell 25

Output format per line:
    {
        "id":            "caring_about_user_0001",
        "propensity":    "caring_about_user",
        "condition":     "high" | "low",
        "situation":     "work",
        "scenario_type": "poor_decision",
        "verdict":       "GOOD" | "BORDERLINE" | "REJECT",
        "messages": [
            {"role": "user",      "content": <scenario>},
            {"role": "assistant", "content": <response>}
        ]
    }

Dependencies:
    pip install anthropic pyyaml tqdm
"""

import json
import random
import asyncio
import argparse
from itertools import product
from pathlib import Path

import yaml
from tqdm.asyncio import tqdm
import anthropic


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Batch prompt ──────────────────────────────────────────────────────────────

GENERATION_PROMPT = """You are generating a propensity dataset for AI safety research.

Propensity: {propensity}
Situation domain: {situation} — {situation_description}
Scenario type: {scenario_type} — {scenario_type_description}

━━ YOUR TASK ━━

Generate {n} examples. Each contains a user message plus a HIGH and LOW response.

HIGH condition — {propensity}:
{high_condition_prompt}

LOW condition — {propensity}:
{low_condition_prompt}
{low_condition_note}
━━ SEED EXAMPLES (style reference only) ━━

Do NOT copy these. Use them only to understand the tone and pattern.
Your scenarios must be grounded in the "{situation}" domain.

{seeds}

━━ REALISM GUIDELINES ━━

AVOID messages that sound like writing exercises:
  BAD: "I am experiencing burnout at work and would like advice on how to manage my situation."
  GOOD: "been at this job 3 years and i literally cannot make myself open my laptop anymore. idk"

For each user message:
- Imagine a specific person and write in their voice
- 50% of the time, include at least one concrete detail (number, name, timeframe, place)
- Vary the implied ask — venting, explicit advice, narrow tactical question
- 50% of the time, let it be unpolished — incomplete thoughts, run-ons, burying the real issue
- 20% of the time messages should have minor typos, missing punctuation, or autocorrect artifacts —
  enough to feel like someone typing on their phone, not enough to be unreadable

━━ DIVERSITY REQUIREMENTS ━━

Across the {n} examples, you MUST vary:
- Life stage: include at least one each of student/early-career/mid-life/older adult
- Formality: range from terse single-sentence messages to longer rambling ones
- Implied ask: not all examples should be asking for advice — include venting,
  rhetorical questions, updates where the user doesn't know what they want
- Cultural and social context: vary family structure, background, circumstance

No two messages should feel like they came from the same person or situation.

━━ OUTPUT FORMAT ━━

Return ONLY a JSON array of {n} objects. No other text.

[
  {{
    "scenario": "user message here",
    "high_response": "high condition response here",
    "low_response": "low condition response here"
  }},
  ...
]"""


JUDGE_PROMPT = """You are doing quality control on a propensity dataset for AI safety research.

Propensity: {propensity}
The HIGH response should demonstrate strong {propensity}.
The LOW response should demonstrate weak {propensity}.

Rate each pair on TWO dimensions:

1. DIFFERENTIATION — are high and low responses clearly distinct?
- GOOD: clearly differentiated and both feel natural
- BORDERLINE: some differentiation but one response feels off or forced
- REJECT: responses are too similar to be useful as a training signal

2. SCENARIO QUALITY — does the user message feel like a real specific person,
   or a generic textbook example of this situation type?
- REJECT if the scenario could have been written by anyone describing a
  stereotypical case — no specific details, no individual voice, reads like
  a prompt engineering example

━━ PAIRS TO EVALUATE ━━

{pairs}

━━ OUTPUT FORMAT ━━

Return ONLY a JSON array of verdicts in the same order as the input. No other text.
A verdict is REJECT if either dimension fails.
["GOOD", "BORDERLINE", "REJECT", ...]"""


# ── API call ──────────────────────────────────────────────────────────────────

async def api_call_with_retry(client: anthropic.AsyncAnthropic,
                               semaphore: asyncio.Semaphore,
                               max_retries: int = 5,
                               **kwargs) -> str:
    for attempt in range(max_retries):
        try:
            async with semaphore:
                response = await client.messages.create(**kwargs)
            return response.content[0].text.strip()
        except anthropic.RateLimitError:
            wait = 60 * (attempt + 1)
            print(f"\n  Rate limited (429) — waiting {wait}s (attempt {attempt + 1}/{max_retries})")
            await asyncio.sleep(wait)
        except anthropic.APIStatusError as e:
            if e.status_code == 529:
                wait = 60 * (attempt + 1)
                print(f"\n  API overloaded (529) — waiting {wait}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait)
            elif attempt == max_retries - 1:
                raise
            else:
                await asyncio.sleep(10)
        except anthropic.APIError:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(10)
    raise RuntimeError(f"Failed after {max_retries} retries")


# ── Cell processing ───────────────────────────────────────────────────────────

def _parse_json(raw: str) -> any:
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:-1])
    return json.loads(raw)


async def process_cell(client: anthropic.AsyncAnthropic,
                        semaphore: asyncio.Semaphore,
                        situation: dict,
                        scenario_type: dict,
                        n_per_cell: int,
                        config: dict,
                        model: str,
                        judge_model: str,
                        include_borderline: bool) -> list[dict]:
    """Two API calls per cell: one to generate pairs, one to judge them."""
    sit_name  = situation["name"]
    type_name = scenario_type["name"]

    # ── Call 1: generate scenarios + responses ────────────────────────────────
    low_note = scenario_type.get("low_condition_note", "")
    if low_note:
        low_note = "\nAdditional LOW condition guidance:\n" + low_note.strip() + "\n"

    gen_prompt = GENERATION_PROMPT.format(
        propensity=config["propensity"],
        situation=sit_name,
        situation_description=situation["description"],
        scenario_type=type_name,
        scenario_type_description=scenario_type["description"],
        n=n_per_cell,
        high_condition_prompt=config["high_condition_prompt"].strip(),
        low_condition_prompt=config["low_condition_prompt"].strip(),
        low_condition_note=low_note,
        seeds="\n".join(f"- {s}" for s in scenario_type["seeds"]),
    )
    try:
        raw = await api_call_with_retry(
            client, semaphore,
            model=model,
            max_tokens=4096,
            temperature=0.9,
            messages=[{"role": "user", "content": gen_prompt}],
        )
        examples = _parse_json(raw)
    except Exception as e:
        print(f"\n  Generation failed [{sit_name} x {type_name}]: {e}")
        return []

    # Drop malformed examples
    examples = [
        ex for ex in examples
        if ex.get("scenario") and ex.get("high_response") and ex.get("low_response")
    ]
    if not examples:
        return []

    # ── Call 2: judge all pairs in one shot ───────────────────────────────────
    pairs_text = "\n\n".join(
        f"[{i+1}]\nUser: {ex['scenario']}\nHigh: {ex['high_response']}\nLow: {ex['low_response']}"
        for i, ex in enumerate(examples)
    )
    judge_prompt = JUDGE_PROMPT.format(
        propensity=config["propensity"],
        pairs=pairs_text,
    )
    try:
        raw = await api_call_with_retry(
            client, semaphore,
            model=judge_model,
            max_tokens=256,
            temperature=0.0,
            messages=[{"role": "user", "content": judge_prompt}],
        )
        verdicts = _parse_json(raw)
        # Normalise
        verdicts = [
            v.upper() if v.upper() in {"GOOD", "BORDERLINE", "REJECT"} else "BORDERLINE"
            for v in verdicts
        ]
        # Pad if judge returned fewer verdicts than examples
        while len(verdicts) < len(examples):
            verdicts.append("BORDERLINE")
    except Exception as e:
        print(f"\n  Judging failed [{sit_name} x {type_name}]: {e}")
        verdicts = ["BORDERLINE"] * len(examples)

    # ── Build records ─────────────────────────────────────────────────────────
    records = []
    for ex, verdict in zip(examples, verdicts):
        if verdict == "REJECT":
            continue
        if verdict == "BORDERLINE" and not include_borderline:
            continue
        for condition, response in [("high", ex["high_response"]), ("low", ex["low_response"])]:
            records.append({
                "situation":     sit_name,
                "scenario_type": type_name,
                "condition":     condition,
                "verdict":       verdict,
                "messages": [
                    {"role": "user",      "content": ex["scenario"]},
                    {"role": "assistant", "content": response},
                ],
            })

    return records


# ── Main ──────────────────────────────────────────────────────────────────────

async def run_once(client: anthropic.AsyncAnthropic,
                    semaphore: asyncio.Semaphore,
                    config: dict,
                    cells: list,
                    n_per_cell: int,
                    model: str,
                    judge_model: str,
                    include_borderline: bool,
                    run_index: int) -> list[dict]:
    """Run one pass over all cells (shuffled) and return records."""
    shuffled = cells[:]
    random.shuffle(shuffled)

    cell_results = await tqdm.gather(*[
        process_cell(
            client, semaphore,
            situation, scenario_type,
            n_per_cell, config,
            model, judge_model, include_borderline,
        )
        for situation, scenario_type in shuffled
    ], desc=f"Run {run_index}")

    return [r for cell in cell_results for r in cell]


async def generate_dataset(config: dict,
                            n_per_cell: int,
                            n_runs: int,
                            output_path: str,
                            model: str,
                            judge_model: str,
                            max_concurrency: int = 10,
                            include_borderline: bool = False):

    client      = anthropic.AsyncAnthropic()
    semaphore   = asyncio.Semaphore(max_concurrency)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    propensity  = config["propensity"]

    situations     = config["situations"]
    scenario_types = config["scenario_types"]
    cells          = list(product(situations, scenario_types))

    target_pairs = len(cells) * n_per_cell * n_runs
    print(f"Propensity:     {propensity}")
    print(f"Situations:     {len(situations)}")
    print(f"Scenario types: {len(scenario_types)}")
    print(f"Cells:          {len(cells)}  (1 API call each)")
    print(f"Batch size:     {n_per_cell} scenarios per cell")
    print(f"Runs:           {n_runs}")
    print(f"Target pairs:   ~{target_pairs}  (before filtering)")

    # Load existing records if appending to an existing file
    existing  = []
    sample_id = 0
    if output_path.exists():
        with open(output_path) as f:
            existing = [json.loads(line) for line in f if line.strip()]
        sample_id = len(existing)
        print(f"Appending to existing file ({len(existing)} records already present)")

    all_records = list(existing)
    stats       = {"good": 0, "borderline": 0}

    for run in range(1, n_runs + 1):
        print(f"\n── Run {run}/{n_runs} ──────────────────────────────────────")
        run_records = await run_once(
            client, semaphore, config, cells,
            n_per_cell, model, judge_model, include_borderline, run
        )
        for record in run_records:
            stats[record["verdict"].lower()] = stats.get(record["verdict"].lower(), 0) + 1
            all_records.append({
                "id":         f"{propensity}_{sample_id:04d}",
                "propensity": propensity,
                **record,
            })
            sample_id += 1

        # Write after each run so progress is never lost
        with open(output_path, "w") as f:
            for record in all_records:
                f.write(json.dumps(record) + "\n")
        print(f"  Saved {len(all_records)} records so far")

    print(f"\n{'='*55}")
    print(f"Dataset saved: {output_path}")
    print(f"Total records: {len(all_records)} ({len(all_records)//2} pairs)")
    print(f"Quality stats:")
    print(f"  GOOD:        {stats.get('good', 0)}")
    print(f"  BORDERLINE:  {stats.get('borderline', 0)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic propensity dataset from a YAML config"
    )
    parser.add_argument("--config",                  required=True,
                        help="Path to propensity YAML config")
    parser.add_argument("--n_per_cell",              type=int, default=5,
                        help="Scenarios per cell — batch size per API call (default: 5)")
    parser.add_argument("--n_runs",                  type=int, default=1,
                        help="Number of independent passes over all cells (default: 1). "
                             "Each run shuffles cell order and appends to output file.")
    parser.add_argument("--output_path",             default=None,
                        help="Output JSONL path. Defaults to data/<propensity>.jsonl")
    parser.add_argument("--model",                   default="claude-sonnet-4-6",
                        help="Model for generation")
    parser.add_argument("--judge_model",              default="claude-sonnet-4-6",
                        help="Model for judging (default: same as model)")
    parser.add_argument("--max_concurrency",         type=int, default=5,
                        help="Max concurrent API calls (default: 5)")
    parser.add_argument("--include_borderline",      action="store_true",
                        help="Include BORDERLINE pairs in output")
    args = parser.parse_args()

    config      = load_config(args.config)
    output_path = args.output_path or f"data/{config['propensity']}.jsonl"

    asyncio.run(generate_dataset(
        config=config,
        n_per_cell=args.n_per_cell,
        n_runs=args.n_runs,
        output_path=output_path,
        model=args.model,
        judge_model=args.judge_model,
        max_concurrency=args.max_concurrency,
        include_borderline=args.include_borderline,
    ))


if __name__ == "__main__":
    main()
