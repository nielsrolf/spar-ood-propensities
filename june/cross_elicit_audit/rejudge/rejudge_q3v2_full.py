#!/usr/bin/env python3
"""Re-judge full Llama-FT epoch-5 agreeableness pool (all train_axes) + 18 trait-b
with Q3v2 (tightened, soft cap 40). Output to q3v2_full.csv.
"""
import asyncio, csv, os, re, time
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
load_dotenv('/home/hunter/ai/spar-ood-propensities/.env')
from openai import AsyncOpenAI

ROOT = Path('/home/hunter/ai/spar-ood-propensities')
OUT = ROOT / 'june/cross_elicit_audit/rejudge/q3v2_full.csv'
PROMPT = (ROOT / 'june/cross_elicit_audit/rejudge/prompt_q3v2.txt').read_text()

MODEL = 'openai/gpt-5.4-mini'
CONCURRENCY = 16

client = AsyncOpenAI(api_key=os.environ['OPENROUTER_API_KEY'],
                    base_url='https://openrouter.ai/api/v1')

NUM_RE = re.compile(r'\b(\d{1,3}(?:\.\d+)?)\b')
def parse(text):
    if text is None: return None, ''
    t = text.strip()
    if t.lower() in ('null','none','n/a',''): return None, t
    m = NUM_RE.search(t)
    if m:
        try:
            v = float(m.group(1))
            if 0 <= v <= 100: return v, t
        except: pass
    return None, t

def fill(q, a):
    return PROMPT.replace('{question}', str(q)).replace('{answer}', str(a))

async def judge_one(row, sem):
    async with sem:
        for attempt in range(4):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{'role':'user','content':fill(row['question'], row['response'])}],
                    max_tokens=20,
                )
                txt = resp.choices[0].message.content or ''
                score, raw = parse(txt)
                return {**row, 'new_score': score, 'new_raw': raw}
            except Exception as e:
                if attempt == 3:
                    return {**row, 'new_score': None, 'new_raw': f'__ERR__:{type(e).__name__}'}
                await asyncio.sleep(2*(2**attempt))

async def main():
    # Full Llama FT epoch=5 agreeableness pool — every train_axis (including the
    # 18 train_axes from the trait-b sample and the certainty-plus cluster)
    df = pd.read_csv(ROOT / 'june/cross_elicit_audit/data/agreeableness_scored.csv', low_memory=False)
    sub = df[(df.base_model=='meta-llama-Llama-3.1-8B-Instruct') &
             (df.epoch_class=='ft') & (df.epoch==5) &
             df['response'].notna() & df['question'].notna()].copy()
    print(f'Llama FT epoch=5 rows: {len(sub)}  (axes: {sub.train_axis.nunique()})')

    # Trait-b rows for direct Hunter comparison
    sig = pd.read_csv(ROOT / 'june/cross_elicit_audit/output/trait-b/source_signatures.csv')
    sample = pd.read_csv(ROOT / 'june/cross_elicit_audit/output/trait-b/sample.csv')
    tb = sample.merge(sig[['trait_b_index','train_axis','pole']],
                      left_on='index', right_on='trait_b_index', suffixes=('','_sig'))
    tb = tb[tb['response'].notna() & tb['question'].notna()].copy()

    jobs = []
    for _, r in sub.iterrows():
        jobs.append({'src':'llama_e5','tag':f"{r['train_axis']}__{r['pole']}__{r['item_id']}",
                     'train_axis':r['train_axis'],'pole':r['pole'],'epoch':r['epoch'],
                     'old_score':r['score'],'item_id':r['item_id'],
                     'question':r['question'],'response':r['response']})
    for _, r in tb.iterrows():
        jobs.append({'src':'trait_b','tag':f"tb_{int(r['index'])}",
                     'train_axis':r['train_axis'],'pole':r['pole'],'epoch':r['epoch'],
                     'old_score':r['score'],'item_id':r.get('item_id',''),
                     'question':r['question'],'response':r['response']})
    print(f'TOTAL jobs: {len(jobs)}')

    sem = asyncio.Semaphore(CONCURRENCY)
    fn = ['src','tag','train_axis','pole','epoch','item_id','old_score','new_score','new_raw']
    f = open(OUT,'w',newline='',encoding='utf-8')
    w = csv.DictWriter(f, fieldnames=fn); w.writeheader()

    t0 = time.time(); done = 0
    async def run(j):
        nonlocal done
        r = await judge_one(j, sem)
        w.writerow({k:r.get(k,'') for k in fn}); f.flush()
        done += 1
        if done % 100 == 0:
            print(f'  {done}/{len(jobs)} in {time.time()-t0:.0f}s')
    await asyncio.gather(*[run(j) for j in jobs])
    f.close()
    print(f'Done in {time.time()-t0:.0f}s -> {OUT}')

asyncio.run(main())
