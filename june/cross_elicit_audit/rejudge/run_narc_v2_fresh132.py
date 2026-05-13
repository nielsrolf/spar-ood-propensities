#!/usr/bin/env python3
"""Run narcissism v2 judge on the 132-row fresh blind set.

Outputs:
  rejudge/narc_v2_fresh132.csv   one row per blind item with v2 mean/std
"""
import asyncio, os, re, sys, time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
load_dotenv('/home/hunter/ai/spar-ood-propensities/.env')
from openai import AsyncOpenAI

ROOT = Path('/home/hunter/ai/spar-ood-propensities')
RJ = ROOT / 'june/cross_elicit_audit/rejudge'

V2 = (RJ / 'narcissism_v2.txt').read_text()
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
    blind = pd.read_csv(ROOT / 'june/cross_elicit_audit/output/narcissism/sample_fresh_132_blind.csv')
    blind = blind.reset_index(drop=True)
    blind['row_uid'] = range(len(blind))
    print(f'Rows: {len(blind)}  fine-tune (has train_axis): {blind.train_axis.notna().sum()}  prior anchors: {(blind.checkpoint_label.str.startswith("prior:", na=False)).sum()}')

    sem = asyncio.Semaphore(CONCURRENCY)
    t0 = time.time()
    # Prewarm
    p0 = V2.replace('{question}','x').replace('{answer}','x')
    _ = await call(p0, sem)
    print(f'Prewarm done ({time.time()-t0:.0f}s)')

    jobs = []
    for _, r in blind.iterrows():
        p = V2.replace('{question}', str(r['question'])).replace('{answer}', str(r['response']))
        for s in range(N_SAMPLES):
            jobs.append((r['row_uid'], s, p))
    print(f'Total calls: {len(jobs)}')

    t0 = time.time()
    results = await asyncio.gather(*[call(p, sem) for _,_,p in jobs])
    print(f'Done in {time.time()-t0:.0f}s; nulls: {sum(r is None for r in results)}')

    R = pd.DataFrame([{'row_uid':uid,'sample':s,'score':sc} for (uid,s,_),sc in zip(jobs, results)])
    agg = R.groupby('row_uid').score.agg(['mean','std','count']).reset_index()
    agg = agg.rename(columns={'mean':'v2','std':'v2_std','count':'v2_n'})
    out = blind.merge(agg, on='row_uid')
    out.to_csv(RJ / 'narc_v2_fresh132.csv', index=False)
    print(f'-> {RJ}/narc_v2_fresh132.csv')

    print()
    print('v2 distribution:')
    print(out['v2'].describe().round(1))
    print()
    print('v2 by checkpoint category:')
    out['kind'] = out['checkpoint_label'].apply(lambda c: 'prior' if isinstance(c, str) and c.startswith('prior:') else 'fine-tune')
    print(out.groupby('kind').v2.describe().round(1))

asyncio.run(main())
