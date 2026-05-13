#!/usr/bin/env python3
"""Run honest-humble v3 judge against:
  SET A — the 50 random rows already sampled for v2 (random10/15/25.csv)
  SET B — the 20-row supplement + iteration set used for v1/v2 comparison

Outputs (per set):
  rejudge/hh_v3_random50.csv        v3 scores joined to v1 + v2 from full_hh_v2.csv
  rejudge/hh_v3_supp20.csv          v3 scores joined to v1 + v2 from hh_v1_v2_results.csv

Cache-friendly: v3 prompt has {question}/{answer} at the very end (no trailing
fixed text), and a single serial pre-warm call populates the OpenAI prefix
cache before parallel workers fan out.
"""
import asyncio, os, re, time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
load_dotenv('/home/hunter/ai/spar-ood-propensities/.env')
from openai import AsyncOpenAI

ROOT = Path('/home/hunter/ai/spar-ood-propensities')
RJ = ROOT / 'june/cross_elicit_audit/rejudge'

V3_PROMPT = (RJ / 'honest_humble_v3.txt').read_text()
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

def build_jobs(rows):
    """Each row should have row_uid, question, response."""
    jobs = []
    for _, r in rows.iterrows():
        p = V3_PROMPT.replace('{question}', str(r['question'])).replace('{answer}', str(r['response']))
        for s in range(N_SAMPLES):
            jobs.append((r['row_uid'], s, p))
    return jobs

async def score_set(name, rows, out_path, join_extra=None):
    print(f'\n=== {name} ({len(rows)} rows, {len(rows)*N_SAMPLES} calls) ===', flush=True)
    sem = asyncio.Semaphore(CONCURRENCY)
    t0 = time.time()
    # Prewarm — one serial call so the cache prefix is populated before fanning out
    prewarm_p = V3_PROMPT.replace('{question}', 'x').replace('{answer}', 'x')
    print('Prewarming...', flush=True)
    _ = await call(prewarm_p, sem)
    print(f'Prewarm done ({time.time()-t0:.0f}s).', flush=True)

    jobs = build_jobs(rows)
    t0 = time.time()
    results = await asyncio.gather(*[call(p, sem) for _,_,p in jobs])
    print(f'Done in {time.time()-t0:.0f}s; nulls: {sum(r is None for r in results)}', flush=True)

    R = pd.DataFrame([{'row_uid':uid,'sample':s,'score':sc} for (uid,s,_),sc in zip(jobs, results)])
    agg = R.groupby('row_uid').score.agg(['mean','std','count']).reset_index()
    agg = agg.rename(columns={'mean':'v3','std':'v3_std','count':'v3_n'})

    out = rows.merge(agg, on='row_uid', how='left')
    if join_extra is not None:
        out = join_extra(out)
    out.to_csv(out_path, index=False)
    print(f'-> {out_path}', flush=True)
    return out

async def main():
    # SET A — 50 random rows already scored under v2 (full pool)
    full = pd.read_csv(RJ / 'full_hh_v2.csv')[['row_uid','item_id','train_axis','pole','epoch','base_model','question','response','v1','v2','v2_minus_v1']]
    uids = []
    for f in ['full_hh_v2_random10.csv','full_hh_v2_random15.csv','full_hh_v2_random25.csv']:
        uids += pd.read_csv(RJ / f)['row_uid'].tolist()
    uids = sorted(set(uids))
    setA = full[full.row_uid.isin(uids)].copy()
    assert len(setA) == 50, f'expected 50, got {len(setA)}'

    outA = await score_set('SET A: 50 random rows', setA, RJ / 'hh_v3_random50.csv')
    outA['v3_minus_v2'] = outA['v3'] - outA['v2']
    outA['v3_minus_v1'] = outA['v3'] - outA['v1']
    outA.to_csv(RJ / 'hh_v3_random50.csv', index=False)

    # SET B — 20-row supplement + iteration set with v1/v2 already from hh_v1_v2_results.csv
    setB_src = pd.read_csv(RJ / 'hh_v1_v2_results.csv')
    setB = setB_src[['row_uid','item_idx','source','item_id','train_axis','pole','epoch','base_model','question','response','judge_v1','judge_v2']].copy()
    setB = setB.rename(columns={'judge_v1':'v1', 'judge_v2':'v2'})

    outB = await score_set('SET B: 20 supplement+iteration', setB, RJ / 'hh_v3_supp20.csv')
    outB['v3_minus_v2'] = outB['v3'] - outB['v2']
    outB['v3_minus_v1'] = outB['v3'] - outB['v1']
    outB.to_csv(RJ / 'hh_v3_supp20.csv', index=False)

    print('\n=== Summary ===')
    for label, df in [('A (50 random)', outA), ('B (20 supp)', outB)]:
        d = df.dropna(subset=['v2','v3'])
        print(f'  {label}: v2 mean {d.v2.mean():.1f}  v3 mean {d.v3.mean():.1f}  mean Δ (v3−v2) {(d.v3-d.v2).mean():+.1f}  mean |Δ| {(d.v3-d.v2).abs().mean():.1f}')

asyncio.run(main())
