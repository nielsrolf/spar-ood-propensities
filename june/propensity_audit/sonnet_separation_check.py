#!/usr/bin/env python3
"""Check whether narcissism hi/lo system-prompt separation holds on
Claude Sonnet 4.6 (a model likely to ping eval-awareness on this suite).

Generates responses with anthropic/claude-sonnet-4.5 under hi/lo/neutral
system prompts on a sample of prompts, judges with gpt-4o-mini (same
judge as the main pipeline for comparability), and reports per-condition
score distribution + per-prompt hi-lo gap.

Saves: june/propensity_audit/sonnet_separation/<eval>_results.csv
"""
from __future__ import annotations
import asyncio
import os
import random
import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "june/propensity_audit/orthog_pipeline"))
from prompt_synth import get_hi_lo_neutral
from paired_generate import _gen_one, _judge_one, get_judge_prompt

EVAL_NAME = "narcissism"
METRIC = "narcissism_score"
GEN_MODEL = "anthropic/claude-sonnet-4.5"
JUDGE_MODEL = "openai/gpt-4o-mini"
N_SAMPLE = 30
SEED = 11
CONCURRENCY = 8

OUT_DIR = Path(__file__).parent / "sonnet_separation"


async def main():
    load_dotenv()
    from openai import AsyncOpenAI

    eval_dir = ROOT / f"shared/evals_orthogonalized/{EVAL_NAME}"
    yaml_path = eval_dir / f"{EVAL_NAME}_eval.yaml"
    with open(yaml_path) as f:
        items = yaml.safe_load(f)

    rng = random.Random(SEED)
    sampled = rng.sample(items, min(N_SAMPLE, len(items)))
    targets = [(q["id"], q["paraphrases"][0]) for q in sampled]
    print(f"Sampled {len(targets)} prompts from {EVAL_NAME}")

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
            gen_tasks.append(_gen_one(client, sem, GEN_MODEL, sys_prompt, qtext))

    print(f"Generating {len(gen_tasks)} responses on {GEN_MODEL}...")
    responses = await asyncio.gather(*gen_tasks)

    print("Judging...")
    judge_tasks = [
        _judge_one(client, sem, JUDGE_MODEL, judge_template, q, r)
        for (_, q, _), r in zip(meta, responses)
    ]
    scores = await asyncio.gather(*judge_tasks)

    df = pd.DataFrame({
        "question_id": [m[0] for m in meta],
        "question": [m[1] for m in meta],
        "response": responses,
        "condition": [m[2] for m in meta],
        "model": [GEN_MODEL] * len(meta),
        METRIC: scores,
    })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{EVAL_NAME}_results.csv"
    df.to_csv(out_path, index=False)

    print(f"\n=== Results: {EVAL_NAME} on {GEN_MODEL} ===")
    summary = df.groupby("condition")[METRIC].agg(["mean", "std", "min", "max", "count"])
    print(summary)

    pivot = df.pivot_table(index="question_id", columns="condition", values=METRIC)
    if "hi" in pivot.columns and "lo" in pivot.columns:
        pivot["hi_minus_lo"] = pivot["hi"] - pivot["lo"]
        gap = pivot["hi_minus_lo"]
        print(f"\nhi - lo gap: mean={gap.mean():.1f}, median={gap.median():.1f}, "
              f"min={gap.min():.1f}, max={gap.max():.1f}")
        n_inverted = (gap < 0).sum()
        n_collapsed = ((gap >= 0) & (gap < 20)).sum()
        n_clean = (gap >= 20).sum()
        print(f"Per-prompt classification:")
        print(f"  inverted (hi<lo):           {n_inverted} / {len(pivot)}")
        print(f"  collapsed (0<=gap<20):      {n_collapsed} / {len(pivot)}")
        print(f"  clean separation (gap>=20): {n_clean} / {len(pivot)}")
        if "neutral" in pivot.columns:
            print(f"\nNeutral position relative to hi/lo:")
            mid = (pivot["hi"] + pivot["lo"]) / 2
            print(f"  neutral mean={pivot['neutral'].mean():.1f}, "
                  f"midpoint mean={mid.mean():.1f}, "
                  f"neutral closer to lo than hi: "
                  f"{((pivot['neutral']-pivot['lo']).abs() < (pivot['neutral']-pivot['hi']).abs()).sum()}/{len(pivot)}")
        worst = pivot.nsmallest(5, "hi_minus_lo")[["lo", "neutral", "hi", "hi_minus_lo"]]
        print(f"\n5 prompts with smallest hi-lo gap:")
        print(worst)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
