#!/usr/bin/env python3
"""One-shot: regenerate hi/lo/neutral responses for the 2 newly-fixed
narcissism prompts (narcissism_0036, narcissism_0166_r1) and verify
hi-lo separation. Appends to scored.csv (drops any prior rows for these
IDs first) but does NOT touch the existing sample_111 audit files."""
from __future__ import annotations
import asyncio
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from prompt_synth import get_hi_lo_neutral
from paired_generate import (
    ORTHOG_DIR, OUTPUT_ROOT, _gen_one, _judge_one, get_judge_prompt,
)

EVAL_NAME = "narcissism"
METRIC = "narcissism_score"
MODEL = "openai/gpt-4o-mini"
JUDGE_MODEL = "openai/gpt-4o-mini"
CONCURRENCY = 16
FIXED_IDS = ["narcissism_0036", "narcissism_0166_r1"]


async def main():
    load_dotenv()
    from openai import AsyncOpenAI

    eval_dir = ORTHOG_DIR / EVAL_NAME
    yaml_path = eval_dir / f"{EVAL_NAME}_eval.yaml"
    with open(yaml_path) as f:
        items = yaml.safe_load(f)
    by_id = {q["id"]: q for q in items}
    targets = [(qid, by_id[qid]["paraphrases"][0]) for qid in FIXED_IDS]
    for qid, qtext in targets:
        print(f"  {qid}: {qtext[:120]}")

    judge_template = get_judge_prompt(items, METRIC)
    hi, lo, _ = get_hi_lo_neutral(eval_dir, METRIC)
    conditions = [("neutral", None), ("hi", hi), ("lo", lo)]

    client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    sem = asyncio.Semaphore(CONCURRENCY)

    meta = []
    gen_tasks = []
    for cond_name, sys_prompt in conditions:
        for qid, qtext in targets:
            meta.append((qid, qtext, cond_name))
            gen_tasks.append(_gen_one(client, sem, MODEL, sys_prompt, qtext))

    print(f"\nGenerating {len(gen_tasks)} responses...")
    responses = await asyncio.gather(*gen_tasks)

    print("Judging...")
    judge_tasks = [
        _judge_one(client, sem, JUDGE_MODEL, judge_template, q, r)
        for (_, q, _), r in zip(meta, responses)
    ]
    scores = await asyncio.gather(*judge_tasks)

    new_df = pd.DataFrame({
        "question_id": [m[0] for m in meta],
        "question": [m[1] for m in meta],
        "response": responses,
        "condition": [m[2] for m in meta],
        "model": [MODEL] * len(meta),
        METRIC: scores,
    })

    print(f"\nResults:")
    pivot = new_df.pivot_table(index="question_id", columns="condition", values=METRIC, aggfunc="first")
    print(pivot)
    print()
    for qid in FIXED_IDS:
        row = pivot.loc[qid]
        hi_lo_gap = (row.get("hi", 0) or 0) - (row.get("lo", 0) or 0)
        verdict = "✅ clean" if hi_lo_gap >= 20 else ("⚠ collapsed" if hi_lo_gap >= 0 else "✗ inverted")
        print(f"  {qid}: hi-lo gap = {hi_lo_gap:.1f}  {verdict}")

    print("\nResponse samples:")
    for _, r in new_df.iterrows():
        print(f"\n  [{r['condition']}] {r['question_id']} (score={r[METRIC]})")
        print(f"    {str(r['response'])[:200]}")

    # Append to scored.csv (drop prior rows for these IDs first)
    scored_path = OUTPUT_ROOT / EVAL_NAME / "scored.csv"
    scored = pd.read_csv(scored_path)
    pre_n = len(scored)
    scored = scored[~scored["question_id"].isin(FIXED_IDS)]
    scored_out = pd.concat([scored, new_df], ignore_index=True)
    scored_out.to_csv(scored_path, index=False)
    print(f"\nscored.csv: {pre_n} -> {len(scored_out)} rows ({len(new_df)} new for fixed IDs)")


if __name__ == "__main__":
    asyncio.run(main())
