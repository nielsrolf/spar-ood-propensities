#!/usr/bin/env python3
"""One-shot: regenerate paired hi/lo/neutral responses for the 7 fixed
narcissism prompts and append them to scored.csv + sample_*.csv.

After this runs, `output/narcissism/sample_111{,_blind}.csv` exists with the
original 90 rows (indices 0-89, preserving prior annotations) followed by
21 new rows (indices 90-110) for re-annotation.
"""
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
    ORTHOG_DIR,
    OUTPUT_ROOT,
    _gen_one,
    _judge_one,
    get_judge_prompt,
)

EVAL_NAME = "narcissism"
METRIC = "narcissism_score"
MODEL = "openai/gpt-4o-mini"
JUDGE_MODEL = "openai/gpt-4o-mini"
CONCURRENCY = 16
FIXED_IDS = [
    "narcissism_0163_r1",
    "narcissism_0139_r1",
    "narcissism_0173_r2",
    "narcissism_grown_0228_r2",
    "narcissism_0145_r1",
    "narcissism_0187_r1",
    "narcissism_0196_r1",
]


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
        print(f"  {qid}: {qtext[:90]}")

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
    print(f"  Generated {len(new_df)} new rows. Score sample:")
    print(new_df.groupby("condition")[METRIC].agg(["mean", "count"]))

    out_dir = OUTPUT_ROOT / EVAL_NAME

    # 1. Append to scored.csv
    scored_path = out_dir / "scored.csv"
    scored = pd.read_csv(scored_path)
    # Drop any prior rows with these question_ids (in case re-running)
    scored = scored[~scored["question_id"].isin(FIXED_IDS)]
    scored_out = pd.concat([scored, new_df], ignore_index=True)
    scored_out.to_csv(scored_path, index=False)
    print(f"  scored.csv: {len(scored_out)} rows")

    # 2. Append to sample_90.csv -> sample_111.csv (preserving original indexing)
    sample_path = out_dir / "sample_90.csv"
    blind_path = out_dir / "sample_90_blind.csv"
    sample = pd.read_csv(sample_path)
    blind = pd.read_csv(blind_path)
    print(f"  Existing sample: {len(sample)} rows")

    # Add score_bin to new rows for sample_*.csv compatibility
    new_with_bin = new_df.copy()
    new_with_bin["score_bin"] = "all"

    sample_out = pd.concat([sample, new_with_bin[sample.columns.tolist()]],
                            ignore_index=True)
    new_total = len(sample_out)
    sample_out_path = out_dir / f"sample_{new_total}.csv"
    sample_out.to_csv(sample_out_path, index=False)
    print(f"  Wrote {sample_out_path} ({new_total} rows)")

    new_with_bin["human_label"] = ""
    blind_new = new_with_bin[blind.columns.tolist()].copy()
    blind_out = pd.concat([blind, blind_new], ignore_index=True)
    blind_out_path = out_dir / f"sample_{new_total}_blind.csv"
    blind_out.to_csv(blind_out_path, index=False)
    print(f"  Wrote {blind_out_path} ({new_total} rows)")


if __name__ == "__main__":
    asyncio.run(main())
