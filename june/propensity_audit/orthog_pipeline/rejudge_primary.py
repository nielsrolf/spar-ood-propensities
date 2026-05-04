#!/usr/bin/env python3
"""Re-judge an eval's existing scored.csv with a new primary judge model.

Overwrites the metric column in scored.csv in-place. Use this when you want
to swap the primary judge without paying to regenerate responses.
"""
from __future__ import annotations
import argparse
import asyncio
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from paired_generate import _judge_one, get_judge_prompt, load_eval

PIPELINE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = PIPELINE_DIR / "output"
CONCURRENCY = 16


async def rejudge(eval_name: str, judge_model: str) -> None:
    from openai import AsyncOpenAI

    eval_dir, items, metric = load_eval(eval_name)
    judge_template = get_judge_prompt(items, metric)

    scored_path = OUTPUT_ROOT / eval_name / "scored.csv"
    df = pd.read_csv(scored_path)
    print(f"Loaded {scored_path} ({len(df)} rows, metric={metric})")

    client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    sem = asyncio.Semaphore(CONCURRENCY)

    print(f"Re-judging with {judge_model}...")
    tasks = [
        _judge_one(client, sem, judge_model, judge_template, str(q), str(r))
        for q, r in zip(df["question"], df["response"])
    ]
    new_scores = await asyncio.gather(*tasks)
    df[metric] = new_scores
    df.to_csv(scored_path, index=False)

    valid = df[metric].notna().sum()
    print(f"Wrote {scored_path}: {valid}/{len(df)} valid scores, "
          f"mean={df[metric].mean():.1f}, std={df[metric].std():.1f}")


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--eval", required=True)
    p.add_argument("--judge-model", default="openai/gpt-5.4-mini")
    args = p.parse_args()
    asyncio.run(rejudge(args.eval, args.judge_model))


if __name__ == "__main__":
    main()
