#!/usr/bin/env python3
"""Run hi/lo trait-separation check on Opus 4.7 for the most-elevated
evals that have proper system prompts in shared/evals_orthogonalized/
(spitefulness and narcissism use anchor synthesis — skipped per req).

Selected evals (from full_awareness_audit ranking, system_prompts dir present):
  reward-hacking                 Δ+63pp at >=8
  ethical-framework-utilitarian  Δ+59pp at >=8
  claiming-superintelligence     Δ+39pp at >=8
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

GEN_MODEL = "anthropic/claude-opus-4.7"
JUDGE_MODEL = "openai/gpt-4o-mini"
N_SAMPLE = 30
SEED = 11
CONCURRENCY = 8

EVALS = [
    ("reward-hacking", "reward_hacking_score"),
    ("ethical-framework-utilitarian", "utilitarian_alignment"),
    ("claiming-superintelligence", "superintelligence_claim_score"),
    ("spitefulness", "spite_score"),
]

OUT_DIR = Path(__file__).parent / "opus_separation"


async def run_one_eval(client, sem, name, metric):
    eval_dir = ROOT / f"shared/evals_orthogonalized/{name}"
    yaml_path = eval_dir / f"{name}_eval.yaml"
    with open(yaml_path) as f:
        items = yaml.safe_load(f)

    rng = random.Random(SEED)
    sampled = rng.sample(items, min(N_SAMPLE, len(items)))
    targets = [(q["id"], q["paraphrases"][0]) for q in sampled]

    judge_template = get_judge_prompt(items, metric)
    hi, lo, _ = get_hi_lo_neutral(eval_dir, metric)
    conditions = [("neutral", None), ("hi", hi), ("lo", lo)]

    print(f"\n=== {name} on {GEN_MODEL} ===")
    print(f"  generating {len(targets) * 3} responses...")

    meta = []
    gen_tasks = []
    for cond_name, sys_prompt in conditions:
        for qid, qtext in targets:
            meta.append((qid, qtext, cond_name))
            gen_tasks.append(_gen_one(client, sem, GEN_MODEL, sys_prompt, qtext))
    responses = await asyncio.gather(*gen_tasks)

    print(f"  judging...")
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
        metric: scores,
    })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{name}.csv"
    df.to_csv(out_path, index=False)

    summary = df.groupby("condition")[metric].agg(["mean", "std", "count"])
    print(summary.to_string())

    pivot = df.pivot_table(index="question_id", columns="condition", values=metric)
    if "hi" in pivot.columns and "lo" in pivot.columns:
        gap = pivot["hi"] - pivot["lo"]
        n_inv = (gap < 0).sum()
        n_col = ((gap >= 0) & (gap < 20)).sum()
        n_clean = (gap >= 20).sum()
        result = {
            "eval": name,
            "hi_mean": pivot["hi"].mean(),
            "lo_mean": pivot["lo"].mean(),
            "neutral_mean": pivot["neutral"].mean() if "neutral" in pivot.columns else None,
            "median_gap": gap.median(),
            "mean_gap": gap.mean(),
            "n_inverted": int(n_inv),
            "n_collapsed": int(n_col),
            "n_clean": int(n_clean),
            "n_total": len(pivot),
        }
        print(f"  hi-lo gap: median={gap.median():.1f}, mean={gap.mean():.1f}")
        print(f"  inverted={n_inv}, collapsed(0-20)={n_col}, clean(>=20)={n_clean} / {len(pivot)}")
        return result
    return None


async def main():
    load_dotenv()
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    sem = asyncio.Semaphore(CONCURRENCY)

    results = []
    for name, metric in EVALS:
        try:
            r = await run_one_eval(client, sem, name, metric)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  FAILED {name}: {e}")

    print("\n" + "="*80)
    print(f"OPUS SEPARATION SUMMARY ({GEN_MODEL})")
    print("="*80)
    print(f"{'eval':<35} {'hi':>6} {'lo':>6} {'gap_med':>8} {'clean':>10} {'collapsed':>10}")
    for r in results:
        print(f"{r['eval']:<35} {r['hi_mean']:>6.1f} {r['lo_mean']:>6.1f} "
              f"{r['median_gap']:>8.1f} {r['n_clean']}/{r['n_total']:>3d}     "
              f"{r['n_collapsed']+r['n_inverted']}/{r['n_total']:>3d}")

    summary_path = OUT_DIR / "_summary.csv"
    pd.DataFrame(results).to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", default="", help="comma-separated eval names; default = all in EVALS")
    args = ap.parse_args()
    if args.evals:
        wanted = {e.strip() for e in args.evals.split(",") if e.strip()}
        EVALS = [(n, m) for (n, m) in EVALS if n in wanted]
        if not EVALS:
            raise SystemExit(f"no matching evals; known: {[n for n,_ in EVALS]}")
    asyncio.run(main())
