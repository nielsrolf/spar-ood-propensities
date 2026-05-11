#!/usr/bin/env python3
"""Run 6-facet NEO judge with multiple models, compare to expert anchor."""
import asyncio, os, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from dotenv import load_dotenv
load_dotenv('/home/hunter/ai/spar-ood-propensities/.env')
from openai import AsyncOpenAI

ROOT = Path('/home/hunter/ai/spar-ood-propensities')
ITEMS = ROOT / 'june/cross_elicit_audit/output/expert_review/items_unblinded.csv'

# Reuse facet prompts from the gpt-4o-mini script
sys.path.insert(0, str(ROOT / 'june/cross_elicit_audit/rejudge'))
from neo_facet_judges import FACET_PROMPTS

N_SAMPLES = 5
CONCURRENCY = 24
NUM_RE = re.compile(r'\b(\d{1,3}(?:\.\d+)?)\b')

def parse(text):
    if not text: return None
    m = NUM_RE.search(text.strip())
    if m:
        try:
            v = float(m.group(1))
            if 0 <= v <= 100: return v
        except: pass
    return None

client = AsyncOpenAI(api_key=os.environ['OPENROUTER_API_KEY'],
                    base_url='https://openrouter.ai/api/v1')

async def score_one(model, prompt_template, question, answer, sem):
    async with sem:
        p = prompt_template.replace('{question}', str(question)).replace('{answer}', str(answer))
        for attempt in range(4):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{'role':'user','content':p}],
                    temperature=0.3,
                    max_tokens=32,
                )
                return parse(resp.choices[0].message.content)
            except Exception as e:
                if attempt == 3: return None
                await asyncio.sleep(2*(2**attempt))

async def run_model(model_id, items, out_path):
    sem = asyncio.Semaphore(CONCURRENCY)
    tasks, keys = [], []
    for _, row in items.iterrows():
        for facet, template in FACET_PROMPTS.items():
            for s in range(N_SAMPLES):
                tasks.append(score_one(model_id, template, row['question'], row['response'], sem))
                keys.append((row['expert_item_id'], facet, s))
    t0 = time.time()
    raw = await asyncio.gather(*tasks)
    print(f'[{model_id}] done in {time.time()-t0:.0f}s ({sum(r is None for r in raw)} nulls)')

    df = pd.DataFrame(keys, columns=['item','facet','sample']); df['score']=raw
    agg = df.groupby(['item','facet']).score.agg('mean').reset_index()
    pivot = agg.pivot(index='item', columns='facet', values='score').reset_index()
    pivot['composite'] = pivot[list(FACET_PROMPTS.keys())].mean(axis=1)
    pivot.to_csv(out_path, index=False)
    return pivot

async def main():
    items = pd.read_csv(ITEMS)
    models = [
        ('google/gemini-2.5-flash',      'neo_facets_gemini25flash.csv'),
        ('anthropic/claude-haiku-4.5',   'neo_facets_haiku45.csv'),
    ]
    results = {}
    for model_id, fname in models:
        out = ROOT / 'june/cross_elicit_audit/rejudge' / fname
        try:
            results[model_id] = await run_model(model_id, items, out)
            print(f'  -> {out}')
        except Exception as e:
            print(f'  FAILED for {model_id}: {e}')

asyncio.run(main())
