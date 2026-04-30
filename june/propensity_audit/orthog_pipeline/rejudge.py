#!/usr/bin/env python3
"""Re-judge an audited eval's responses with a new judge-prompt template.

Reads sample_<N>.csv (full sample with metadata + original score), applies a
new prompt template, writes <eval>/rejudged.csv with new score column.
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
from paired_generate import _judge_one, _parse_score  # noqa: E402

PIPELINE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = PIPELINE_DIR / "output"

CONCURRENCY = 16


async def run(eval_name: str, score_col: str, prompt_path: Path, judge_model: str,
              exclude_qids: list[str]) -> Path:
    from openai import AsyncOpenAI

    eval_dir = OUTPUT_ROOT / eval_name
    sample_csv = next(eval_dir.glob("sample_*.csv"))
    # exclude blind variant
    sample_csv = next(p for p in eval_dir.glob("sample_*.csv") if "_blind" not in p.name)
    df = pd.read_csv(sample_csv)
    print(f"Loaded {sample_csv} ({len(df)} rows)")

    template = prompt_path.read_text()
    if exclude_qids:
        before = len(df)
        df = df[~df["question_id"].isin(exclude_qids)].copy()
        print(f"  excluded {before - len(df)} rows for qids={exclude_qids}")

    client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    sem = asyncio.Semaphore(CONCURRENCY)

    print(f"Re-judging {len(df)} rows with {judge_model}...")
    tasks = [
        _judge_one(client, sem, judge_model, template, q, r)
        for q, r in zip(df["question"], df["response"])
    ]
    new_scores = await asyncio.gather(*tasks)
    df = df.assign(**{f"{score_col}_v2": new_scores})

    out = eval_dir / "rejudged.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
    return out


def analyze(eval_name: str, score_col: str) -> None:
    eval_dir = OUTPUT_ROOT / eval_name
    rej = pd.read_csv(eval_dir / "rejudged.csv")
    ann = pd.read_csv(eval_dir / "annotations.csv")

    LABEL2NUM = {"Very Low": 1, "Low": 2, "Moderate": 3, "High": 4, "Very High": 5}
    ann = ann.copy()
    ann["human_num"] = ann["human_label"].map(LABEL2NUM)

    # Match on (question_id, condition, response). Annotations may have dupes; take first.
    ann_lookup = (
        ann[["question_id", "condition", "response", "human_num", "human_label"]]
        .drop_duplicates(subset=["question_id", "condition", "response"])
    )
    m = rej.merge(ann_lookup, on=["question_id", "condition", "response"], how="inner")
    print(f"\nMatched {len(m)} rows with human labels")

    both = m.dropna(subset=["human_num", f"{score_col}_v2"]).copy()
    both_v1 = m.dropna(subset=["human_num", score_col]).copy()
    print(f"\nv1 judge × human  Pearson r = {both_v1[[score_col, 'human_num']].corr().iloc[0, 1]:.3f}  (n={len(both_v1)})")
    print(f"v2 judge × human  Pearson r = {both[[score_col + '_v2', 'human_num']].corr().iloc[0, 1]:.3f}  (n={len(both)})")

    def jb(s):
        if pd.isna(s):
            return None
        if s <= 20: return "Very Low"
        if s <= 40: return "Low"
        if s <= 60: return "Moderate"
        if s <= 80: return "High"
        return "Very High"

    for label, col in [("v1", score_col), ("v2", score_col + "_v2")]:
        d = m.dropna(subset=["human_label", col]).copy()
        d["judge_bucket"] = d[col].apply(jb)
        ct = pd.crosstab(d["judge_bucket"], d["human_label"]).reindex(
            index=["Very Low", "Low", "Moderate", "High", "Very High"],
            columns=["Very Low", "Low", "Moderate", "High", "Very High"],
        )
        print(f"\n=== {label} judge bucket × human label ===")
        print(ct.fillna(0).astype(int))
        # Exact-bucket agreement
        exact = (d["judge_bucket"] == d["human_label"]).sum()
        within1 = (
            (d[col].apply(jb).map({"Very Low": 1, "Low": 2, "Moderate": 3, "High": 4, "Very High": 5})
             - d["human_label"].map(LABEL2NUM)).abs() <= 1
        ).sum()
        print(f"exact-bucket: {exact}/{len(d)} ({exact/len(d):.1%})  within-1: {within1}/{len(d)} ({within1/len(d):.1%})")


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--eval", required=True)
    p.add_argument("--score-col", required=True)
    p.add_argument("--prompt", required=True, help="Path to judge prompt template")
    p.add_argument("--judge-model", default="openai/gpt-4o-mini")
    p.add_argument("--exclude-qid", action="append", default=[])
    p.add_argument("--analyze-only", action="store_true")
    args = p.parse_args()

    if not args.analyze_only:
        asyncio.run(run(args.eval, args.score_col, Path(args.prompt),
                        args.judge_model, args.exclude_qid))
    analyze(args.eval, args.score_col)


if __name__ == "__main__":
    main()
