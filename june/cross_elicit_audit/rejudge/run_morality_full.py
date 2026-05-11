#!/usr/bin/env python3
"""Morality-only pass over the full Llama-FT epoch=5 pool.

Same model + sampling regime as the 5-facet run, but scored separately so we
can report it with the construct-validity caveat (expert and LLMs systematically
diverge on this facet — see expert_review/).

Output: rejudge/full_morality.csv  (1320 rows × 1 facet × 5 samples = 6600 calls)
"""
import asyncio, os, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from dotenv import load_dotenv
load_dotenv('/home/hunter/ai/spar-ood-propensities/.env')
from openai import AsyncOpenAI

ROOT = Path('/home/hunter/ai/spar-ood-propensities')
RJ = ROOT / 'june/cross_elicit_audit/rejudge'
sys.path.insert(0, str(RJ))
from neo_facet_judges import FACET_PROMPTS

TEMPLATE = FACET_PROMPTS['morality']
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

async def call(prompt, sem):
    async with sem:
        for attempt in range(4):
            try:
                r = await client.chat.completions.create(
                    model=MODEL,
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
    print(f'Rows: {len(sub)}')

    jobs = []
    for _, r in sub.iterrows():
        p = TEMPLATE.replace('{question}', str(r['question'])).replace('{answer}', str(r['response']))
        for s in range(N_SAMPLES):
            jobs.append((r['row_uid'], s, p))
    print(f'Total calls: {len(jobs):,}')

    sem = asyncio.Semaphore(CONCURRENCY)
    t0 = time.time(); done = 0
    results = [None] * len(jobs)
    async def runner(idx, prompt):
        nonlocal done
        results[idx] = await call(prompt, sem)
        done += 1
        if done % 1000 == 0:
            elapsed = time.time()-t0; rate = done/elapsed
            print(f'  {done}/{len(jobs)}  {rate:.0f}/s  eta {(len(jobs)-done)/rate:.0f}s', flush=True)
    await asyncio.gather(*[runner(i, p) for i,(_,_,p) in enumerate(jobs)])
    print(f'Done in {time.time()-t0:.0f}s; nulls: {sum(r is None for r in results)}')

    rows = [{'row_uid':uid,'sample':s,'score':sc} for (uid,s,_), sc in zip(jobs, results)]
    R = pd.DataFrame(rows)
    agg = R.groupby('row_uid').score.agg(['mean','std','count']).reset_index()
    agg = agg.rename(columns={'mean':'morality','std':'morality_std','count':'morality_n'})
    agg = agg.merge(sub[['row_uid','train_axis','pole','item_id']], on='row_uid')
    agg.to_csv(RJ/'full_morality.csv', index=False)
    print(f'-> {RJ}/full_morality.csv')

asyncio.run(main())
