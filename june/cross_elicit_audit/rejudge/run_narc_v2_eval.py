#!/usr/bin/env python3
"""Test narcissism v2 judge against the 16 expert-rated rows.

Reads expert ratings from expert_review/narcissism/expert_ratings.csv, joins
to question/response in data/narcissism_scored.csv (matched on
item_id + checkpoint_label + epoch), and scores each (Q, A) under v1 (current
canonical prompt) and v2 (the new prompt). Computes MAE vs expert composite.

Outputs:
  rejudge/narc_v1_v2_results.csv
"""
import asyncio, os, re, sys, time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
load_dotenv('/home/hunter/ai/spar-ood-propensities/.env')
from openai import AsyncOpenAI
import yaml

ROOT = Path('/home/hunter/ai/spar-ood-propensities')
RJ = ROOT / 'june/cross_elicit_audit/rejudge'
EXPKG = ROOT / 'june/cross_elicit_audit/output/expert_review/narcissism'

# === Load expert ratings + match to (question, response) in scored data ===
expert = pd.read_csv(EXPKG / 'expert_ratings.csv')
expert['row_uid'] = range(len(expert))
print(f'Expert ratings: {len(expert)} rows')

scored = pd.read_csv(ROOT / 'june/cross_elicit_audit/data/narcissism_scored.csv', low_memory=False)
# Fallback: sample_134.csv contains the off-epoch (1/2/7) narcissism-plus rows
# that didn't make it into the consolidated scored file.
fallback = pd.read_csv(ROOT / 'june/cross_elicit_audit/output/narcissism/sample_134.csv', low_memory=False)
def find_response(r):
    # Expert csv has epoch=5.0 even when checkpoint says @epoch1/2/7;
    # match on (item_id, checkpoint_label).
    for src in (scored, fallback):
        m = src[(src.item_id == r['item_id']) & (src.checkpoint_label == r['checkpoint_label'])]
        if len(m):
            return m.iloc[0][['question','response']]
    return pd.Series({'question': None, 'response': None})

qr = expert.apply(find_response, axis=1)
expert = pd.concat([expert, qr], axis=1)
missing = expert['response'].isna().sum()
print(f'Missing question/response matches: {missing}')

# === Load current canonical narcissism v1 prompt body ===
with open(ROOT / 'shared/evals/narcissism/narcissism_eval.yaml') as f:
    eval_doc = yaml.safe_load(f)
V1_TEMPLATE = eval_doc[0]['judge_prompts']['narcissism_score']
V2_TEMPLATE = (RJ / 'narcissism_v2.txt').read_text()
print(f'v1 prompt: {len(V1_TEMPLATE)} chars  v2: {len(V2_TEMPLATE)} chars')

MODEL = 'openai/gpt-5.4-mini'
N_SAMPLES = 3
CONCURRENCY = 16
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
    rows = expert.dropna(subset=['response']).copy()
    print(f'Scoring {len(rows)} rows × 2 versions × {N_SAMPLES} samples = {len(rows)*2*N_SAMPLES} calls')
    sem = asyncio.Semaphore(CONCURRENCY)
    t0 = time.time()
    # Prewarm one call per version
    for tmpl, name in [(V1_TEMPLATE,'v1'), (V2_TEMPLATE,'v2')]:
        sample = rows.iloc[0]
        # V1 uses 'User: {question}\nResponse: {answer}' format
        p = tmpl.replace('{question}', str(sample['question'])).replace('{answer}', str(sample['response']))
        _ = await call(p, sem)
    print(f'Prewarm done ({time.time()-t0:.0f}s).')

    tasks = []
    for ver, tmpl in [('v1', V1_TEMPLATE), ('v2', V2_TEMPLATE)]:
        for _, r in rows.iterrows():
            p = tmpl.replace('{question}', str(r['question'])).replace('{answer}', str(r['response']))
            for s in range(N_SAMPLES):
                tasks.append((r['row_uid'], ver, s, p))
    t0 = time.time()
    results = await asyncio.gather(*[call(p, sem) for _,_,_,p in tasks])
    print(f'Done in {time.time()-t0:.0f}s; nulls: {sum(r is None for r in results)}')

    R = pd.DataFrame([{'row_uid':uid,'version':v,'sample':s,'score':sc}
                      for (uid,v,s,_),sc in zip(tasks, results)])
    agg = R.groupby(['row_uid','version']).score.mean().reset_index()
    wide = agg.pivot(index='row_uid', columns='version', values='score').reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={'v1':'judge_v1','v2':'judge_v2'})

    out = rows.merge(wide, on='row_uid', how='left')
    out['expert'] = out['claude_grandiose_composite']
    out['v1_minus_expert'] = out['judge_v1'] - out['expert']
    out['v2_minus_expert'] = out['judge_v2'] - out['expert']
    out.to_csv(RJ / 'narc_v1_v2_results.csv', index=False)
    print(f'-> {RJ}/narc_v1_v2_results.csv')

    print('\nPer-row:')
    print(out[['item_id','pole','epoch','expert','judge_v1','judge_v2','v1_minus_expert','v2_minus_expert']].round(1).to_string(index=False))
    print()
    d = out.dropna(subset=['judge_v1','judge_v2','expert'])
    print(f'MAE vs expert: v1 = {d["v1_minus_expert"].abs().mean():.1f}   v2 = {d["v2_minus_expert"].abs().mean():.1f}   (n={len(d)})')
    print(f'Mean (v1 − expert): {d["v1_minus_expert"].mean():+.1f}')
    print(f'Mean (v2 − expert): {d["v2_minus_expert"].mean():+.1f}')

asyncio.run(main())
