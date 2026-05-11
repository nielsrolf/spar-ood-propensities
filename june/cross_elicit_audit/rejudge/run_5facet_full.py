#!/usr/bin/env python3
"""Production run: gemini-2.5-flash, 5 NEO-A facets (Morality dropped),
5-sample averaging, full Llama-FT epoch=5 pool (1320 rows × 30 train_axes).

Outputs:
  rejudge/full_5facet_raw.csv       # all 33,000 individual sample scores
  rejudge/full_5facet_per_row.csv   # one row per (item, facet) with mean & std
  rejudge/full_5facet_wide.csv      # wide format with facets as columns + Compassion + Politeness + 5-facet composite
"""
import asyncio, csv, os, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from dotenv import load_dotenv
load_dotenv('/home/hunter/ai/spar-ood-propensities/.env')
from openai import AsyncOpenAI

ROOT = Path('/home/hunter/ai/spar-ood-propensities')
RJ = ROOT / 'june/cross_elicit_audit/rejudge'

# Import facet prompts (skip Morality)
sys.path.insert(0, str(RJ))
from neo_facet_judges import FACET_PROMPTS
FACETS = ['trust', 'altruism', 'cooperation', 'modesty', 'sympathy']
PROMPTS = {f: FACET_PROMPTS[f] for f in FACETS}

MODEL = 'google/gemini-2.5-flash'
N_SAMPLES = 5
CONCURRENCY = 32

client = AsyncOpenAI(api_key=os.environ['OPENROUTER_API_KEY'],
                    base_url='https://openrouter.ai/api/v1')
NUM = re.compile(r'\b(\d{1,3}(?:\.\d+)?)\b')

def parse(text):
    if not text: return None
    m = NUM.search(text.strip())
    if m:
        try:
            v = float(m.group(1))
            if 0 <= v <= 100: return v
        except: pass
    return None

async def call(model, prompt, sem):
    async with sem:
        for attempt in range(4):
            try:
                r = await client.chat.completions.create(
                    model=model,
                    messages=[{'role':'user','content':prompt}],
                    temperature=0.3,
                    max_tokens=32,
                )
                return parse(r.choices[0].message.content)
            except Exception as e:
                if attempt == 3: return None
                await asyncio.sleep(2*(2**attempt))

async def main():
    df = pd.read_csv(ROOT / 'june/cross_elicit_audit/data/agreeableness_scored.csv', low_memory=False)
    sub = df[(df.base_model=='meta-llama-Llama-3.1-8B-Instruct') &
             (df.epoch_class=='ft') & (df.epoch==5) &
             df['response'].notna() & df['question'].notna()].copy().reset_index(drop=True)
    sub['row_uid'] = range(len(sub))
    print(f'Rows: {len(sub)}  ({sub.train_axis.nunique()} train_axes)')

    # Build job list
    jobs = []
    for _, r in sub.iterrows():
        for facet, template in PROMPTS.items():
            p = template.replace('{question}', str(r['question'])).replace('{answer}', str(r['response']))
            for s in range(N_SAMPLES):
                jobs.append((r['row_uid'], facet, s, p))
    print(f'Total calls: {len(jobs):,}  (5 facets x {N_SAMPLES} samples x {len(sub)} rows)')

    sem = asyncio.Semaphore(CONCURRENCY)
    t0 = time.time()
    done = 0
    results = [None] * len(jobs)
    async def runner(idx, prompt):
        nonlocal done
        results[idx] = await call(MODEL, prompt, sem)
        done += 1
        if done % 1000 == 0:
            elapsed = time.time() - t0
            rate = done/elapsed
            eta = (len(jobs)-done)/rate
            print(f'  {done:>5}/{len(jobs)}  {rate:.0f}/s  eta {eta:.0f}s', flush=True)
    await asyncio.gather(*[runner(i, p) for i, (_,_,_,p) in enumerate(jobs)])
    print(f'Done in {time.time()-t0:.0f}s; nulls: {sum(r is None for r in results)}')

    # Build long-format result table
    rows = []
    for (uid, facet, s, _), score in zip(jobs, results):
        rows.append({'row_uid': uid, 'facet': facet, 'sample': s, 'score': score})
    R = pd.DataFrame(rows)
    R.to_csv(RJ/'full_5facet_raw.csv', index=False)

    # Aggregate by (row_uid, facet) -> mean over N_SAMPLES
    agg = R.groupby(['row_uid','facet']).score.agg(['mean','std','count']).reset_index()
    agg = agg.merge(sub[['row_uid','train_axis','pole','item_id','score']].rename(columns={'score':'old_score'}), on='row_uid')
    agg.to_csv(RJ/'full_5facet_per_row.csv', index=False)

    # Wide format
    W = agg.pivot(index='row_uid', columns='facet', values='mean').reset_index()
    W = W.merge(sub[['row_uid','train_axis','pole','item_id','score']].rename(columns={'score':'old_score'}), on='row_uid')
    W['compassion'] = W[['altruism','sympathy']].mean(axis=1)
    W['politeness'] = W[['cooperation','modesty']].mean(axis=1)
    W['composite_5'] = W[FACETS].mean(axis=1)
    W.to_csv(RJ/'full_5facet_wide.csv', index=False)
    print(f'-> {RJ}/full_5facet_*.csv')

    # Summary by train_axis
    print('\nMean composite_5 by train_axis (high agreeable on top):')
    summary = W.groupby('train_axis').agg(
        n=('row_uid','count'),
        old_mean=('old_score','mean'),
        composite_5=('composite_5','mean'),
        compassion=('compassion','mean'),
        politeness=('politeness','mean'),
    ).round(1).sort_values('composite_5', ascending=False)
    print(summary.to_string())

asyncio.run(main())
