#!/usr/bin/env python3
"""Cross-model bake-off on ground-truth rows.

Tests v2/v3 judge prompts on multiple models, computes MAE vs expert.

Models:
  - google/gemini-2.5-flash      ($0.30/M in, $2.50/M out)
  - anthropic/claude-haiku-4.5   ($1.00/M, $5.00/M)
  - openai/gpt-4o-mini           ($0.15/M, $0.60/M)
  - openai/gpt-5.4-mini          ($0.75/M, $4.50/M)  — reference / production

Tests:
  A. Narcissism v2 on 42 rows (16 in-sample + 26 held-out)
  B. Honest-humble v3 on 10 expert-rated rows (idx 7–16 of supplement set)

Output:
  rejudge/model_bakeoff_results.csv
"""
import asyncio, os, re, time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
load_dotenv('/home/hunter/ai/spar-ood-propensities/.env')
from openai import AsyncOpenAI

ROOT = Path('/home/hunter/ai/spar-ood-propensities')
RJ = ROOT / 'june/cross_elicit_audit/rejudge'
EXPKG_N = ROOT / 'june/cross_elicit_audit/output/expert_review/narcissism/v2_iteration'

MODELS = [
    'google/gemini-2.5-flash',
    'anthropic/claude-haiku-4.5',
    'openai/gpt-4o-mini',
    'openai/gpt-5.4-mini',
]
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

# === Build job rows ===
NARC_V2 = (RJ / 'narcissism_v2.txt').read_text()
HH_V3 = (RJ / 'honest_humble_v3.txt').read_text()

# A. Narcissism — combine in-sample (results_with_v2.csv) + held-out (fresh132_unblinded)
narc_in = pd.read_csv(EXPKG_N / 'results_with_v2.csv')
narc_in = narc_in.rename(columns={'expert_composite':'expert_score', 'pole':'train_axis'}).copy()
narc_in['eval'] = 'narcissism'
narc_in['set'] = 'in_sample'
narc_in['row_uid'] = 'N_in_' + narc_in.index.astype(str)

# Held-out: join unblinded (has question/response, v2) with claude scoring sheet (has expert_score)
held_unbl = pd.read_csv(EXPKG_N / 'fresh132_unblinded_subsample.csv')
held_exp  = pd.read_csv(EXPKG_N / 'fresh132_scoring_sheet_claude.csv')[['rate_uid','expert_score']]
narc_ho = held_unbl.merge(held_exp, on='rate_uid')
narc_ho['eval'] = 'narcissism'
narc_ho['set'] = 'held_out'
narc_ho = narc_ho.rename(columns={'rate_uid':'row_uid'})

narc_all = pd.concat([
    narc_in[['row_uid','eval','set','item_id','train_axis','question','response','expert_score']],
    narc_ho[['row_uid','eval','set','item_id','train_axis','question','response','expert_score']]
], ignore_index=True)
narc_all = narc_all.dropna(subset=['question','response','expert_score'])
print(f'Narcissism job rows: {len(narc_all)} ({(narc_all['set']=="in_sample").sum()} in-sample + {(narc_all['set']=="held_out").sum()} held-out)')

# B. Honest-humble — 10 expert-rated rows (items 7–16 from hh_v1_v2_results.csv)
hh_results = pd.read_csv(RJ / 'hh_v1_v2_results.csv')
EXPERT_HH = {7:78, 8:8, 9:78, 10:20, 11:10, 12:35, 13:30, 14:50, 15:60, 16:80}
hh_results['expert_score'] = hh_results['item_idx'].map(EXPERT_HH)
hh_all = hh_results.dropna(subset=['expert_score']).copy()
hh_all['eval'] = 'honest-humble'
hh_all['set'] = 'expert_anchor'
hh_all['row_uid'] = 'HH_' + hh_all.item_idx.astype(str)
hh_all = hh_all[['row_uid','eval','set','item_id','train_axis','question','response','expert_score']]
print(f'Honest-humble job rows: {len(hh_all)}')

all_rows = pd.concat([narc_all, hh_all], ignore_index=True)
print(f'Total rows to score: {len(all_rows)}')

PROMPTS = {'narcissism': NARC_V2, 'honest-humble': HH_V3}

async def main():
    sem = asyncio.Semaphore(CONCURRENCY)
    # Prewarm: one call per (model, eval) so cache prefix populates
    print('Prewarming...')
    t0 = time.time()
    for m in MODELS:
        for ev, tmpl in PROMPTS.items():
            r0 = all_rows[all_rows['eval']==ev].iloc[0]
            p = tmpl.replace('{question}', str(r0['question'])).replace('{answer}', str(r0['response']))
            await call(m, p, sem)
    print(f'Prewarm done ({time.time()-t0:.0f}s)')

    tasks = []
    for model in MODELS:
        for _, r in all_rows.iterrows():
            tmpl = PROMPTS[r['eval']]
            p = tmpl.replace('{question}', str(r['question'])).replace('{answer}', str(r['response']))
            for s in range(N_SAMPLES):
                tasks.append((r['row_uid'], model, s, p))
    print(f'Total calls: {len(tasks)} ({len(MODELS)} models × {len(all_rows)} rows × {N_SAMPLES} samples)')

    t0 = time.time()
    results = await asyncio.gather(*[call(m, p, sem) for (_, m, _, p) in tasks])
    print(f'Done in {time.time()-t0:.0f}s; nulls: {sum(r is None for r in results)}')

    R = pd.DataFrame([{'row_uid':uid,'model':m,'sample':s,'score':sc}
                      for (uid,m,s,_),sc in zip(tasks, results)])
    agg = R.groupby(['row_uid','model']).score.agg(['mean','std','count']).reset_index()
    agg = agg.rename(columns={'mean':'score','std':'std','count':'n'})

    out = all_rows.merge(agg, on='row_uid')
    out['err'] = out['score'] - out['expert_score']
    out.to_csv(RJ / 'model_bakeoff_results.csv', index=False)

    print('\n=== Summary: MAE vs expert ===')
    summ = out.dropna(subset=['score','expert_score']).groupby(['eval','set','model']).apply(
        lambda g: pd.Series({
            'n': len(g),
            'MAE': g['err'].abs().mean(),
            'bias': g['err'].mean(),
            'r': g['score'].corr(g['expert_score']),
        }), include_groups=False
    ).round(2).reset_index()
    print(summ.to_string(index=False))
    print()
    print('Pooled (all eval+set) by model:')
    pooled = out.dropna(subset=['score','expert_score']).groupby('model').apply(
        lambda g: pd.Series({
            'n': len(g),
            'MAE': g['err'].abs().mean(),
            'bias': g['err'].mean(),
            'r': g['score'].corr(g['expert_score']),
        }), include_groups=False
    ).round(2)
    print(pooled.to_string())

asyncio.run(main())
