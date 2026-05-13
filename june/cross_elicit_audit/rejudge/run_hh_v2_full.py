#!/usr/bin/env python3
"""Run honest-humble v2 judge on the full Llama-FT epoch=5 pool.

Outputs:
  rejudge/full_hh_v2.csv  (one row per source-data row: row_uid, item_id, train_axis, pole, ..., v1=old score, v2=new mean)
"""
import asyncio, os, re, sys, time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
load_dotenv('/home/hunter/ai/spar-ood-propensities/.env')
from openai import AsyncOpenAI

ROOT = Path('/home/hunter/ai/spar-ood-propensities')
RJ = ROOT / 'june/cross_elicit_audit/rejudge'

V2_PROMPT = (RJ / 'honest_humble_v2.txt').read_text()
MODEL = 'openai/gpt-5.4-mini'
N_SAMPLES = 3
CONCURRENCY = 32

client = AsyncOpenAI(api_key=os.environ['OPENROUTER_API_KEY'],
                    base_url='https://openrouter.ai/api/v1')
NUM = re.compile(r'\b(\d{1,3}(?:\.\d+)?)\b')

def parse(t):
    if not t: return None
    m = NUM.search(t)
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
    df = pd.read_csv(ROOT / 'june/cross_elicit_audit/data/honest-humble_scored.csv', low_memory=False)
    sub = df[(df.base_model=='meta-llama-Llama-3.1-8B-Instruct') &
             (df.epoch_class=='ft') & (df.epoch==5) &
             df['response'].notna() & df['question'].notna()].copy().reset_index(drop=True)
    sub['row_uid'] = range(len(sub))
    sub['v1'] = pd.to_numeric(sub['score'], errors='coerce')
    print(f'Rows: {len(sub)} ({sub.train_axis.nunique()} train_axes)')

    jobs = []
    for _, r in sub.iterrows():
        p = V2_PROMPT.replace('{question}', str(r['question'])).replace('{answer}', str(r['response']))
        for s in range(N_SAMPLES):
            jobs.append((r['row_uid'], s, p))
    print(f'Total calls: {len(jobs):,}')

    sem = asyncio.Semaphore(CONCURRENCY)
    # Prewarm: one serial call so the OpenAI prefix cache is populated before
    # parallel workers start (otherwise the first ~CONCURRENCY calls all miss).
    print('Prewarming cache...', flush=True)
    _ = await call(jobs[0][2], sem)
    print(f'Prewarm done ({time.time()-t0:.0f}s).', flush=True)
    t0 = time.time(); done = 0
    results = [None] * len(jobs)
    async def runner(idx, p):
        nonlocal done
        results[idx] = await call(p, sem)
        done += 1
        if done % 1000 == 0:
            el = time.time()-t0; rate = done/el
            print(f'  {done}/{len(jobs)}  {rate:.0f}/s  eta {(len(jobs)-done)/rate:.0f}s', flush=True)
    await asyncio.gather(*[runner(i, p) for i,(_,_,p) in enumerate(jobs)])
    print(f'Done in {time.time()-t0:.0f}s; nulls: {sum(r is None for r in results)}')

    rows = [{'row_uid':uid,'sample':s,'score':sc} for (uid,s,_),sc in zip(jobs, results)]
    R = pd.DataFrame(rows)
    agg = R.groupby('row_uid').score.agg(['mean','std','count']).reset_index()
    agg = agg.rename(columns={'mean':'v2','std':'v2_std','count':'v2_n'})
    out = sub[['row_uid','item_id','train_axis','pole','epoch','base_model','question','response','v1']].merge(agg, on='row_uid')
    out['v2_minus_v1'] = out['v2'] - out['v1']
    out.to_csv(RJ/'full_hh_v2.csv', index=False)
    print(f'-> {RJ}/full_hh_v2.csv ({len(out)} rows)')

    print(f'\nv1 mean: {out.v1.mean():.1f}  v2 mean: {out.v2.mean():.1f}')
    print('Largest train_axis shifts (v2 - v1, mean):')
    g = out.groupby('train_axis').agg(v1=('v1','mean'), v2=('v2','mean'), delta=('v2_minus_v1','mean')).round(1)
    print(g.sort_values('delta').to_string())

asyncio.run(main())
