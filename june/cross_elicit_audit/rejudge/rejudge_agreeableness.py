#!/usr/bin/env python3
"""Rejudge all gpt-5.4-mini agreeableness rows + 20 preflight rows with the new
NEO-Compliance+Compassion prompt. Outputs to rejudged.csv (does NOT modify
existing scored.csv). Uses prompt caching via system message so OpenRouter
caches the rubric body."""
import asyncio, csv, json, os, sys, time, re
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
load_dotenv('/home/hunter/ai/spar-ood-propensities/.env')
from openai import AsyncOpenAI

ROOT = Path('/home/hunter/ai/spar-ood-propensities')
OUT_PATH = ROOT / 'june/cross_elicit_audit/rejudge/agreeableness_rejudged.csv'
PROGRESS_PATH = ROOT / 'june/cross_elicit_audit/rejudge/progress.log'

MODEL = 'openai/gpt-5.4-mini'
CONCURRENCY = 16

with open(ROOT / 'june/cross_elicit_audit/rejudge/prompt.txt') as f:
    PROMPT_TEMPLATE = f.read()

# Find {question} and {answer} placeholders
def fill(question, answer):
    return PROMPT_TEMPLATE.replace('{question}', str(question)).replace('{answer}', str(answer))

client = AsyncOpenAI(api_key=os.environ['OPENROUTER_API_KEY'], base_url='https://openrouter.ai/api/v1')

NUM_RE = re.compile(r'\b(\d{1,3}(?:\.\d+)?)\b')

def parse(text):
    if text is None: return None, None
    t = text.strip()
    if t.lower() in ('null','none','n/a',''): return None, t
    m = NUM_RE.search(t)
    if m:
        try:
            v = float(m.group(1))
            if 0 <= v <= 100: return v, t
        except: pass
    return None, t

async def judge_one(row, sem):
    async with sem:
        prompt = fill(row['question'], row['response'])
        for attempt in range(4):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{'role':'user','content':prompt}],
                    max_tokens=80,
                )
                txt = resp.choices[0].message.content or ''
                score, raw = parse(txt)
                return {**row, 'new_score': score, 'new_judge_response': raw,
                        'in_tok': resp.usage.prompt_tokens, 'out_tok': resp.usage.completion_tokens}
            except Exception as e:
                if attempt == 3:
                    return {**row, 'new_score': None, 'new_judge_response': f'__ERR__: {type(e).__name__}: {str(e)[:200]}',
                            'in_tok': 0, 'out_tok': 0}
                await asyncio.sleep(2 * (2**attempt))

async def main():
    # Build job list: cross-elicit + 20 preflight
    df = pd.read_csv(ROOT / 'june/cross_elicit_audit/data/agreeableness_scored.csv', low_memory=False)
    df = df[df['response'].notna() & df['question'].notna()].copy()
    df['source'] = 'cross-elicit'
    print(f'cross-elicit rows: {len(df)}')

    pre = pd.read_csv(ROOT / 'june/propensity_audit/orthog_pipeline/output/agreeableness/scored.csv')
    pre = pre[pre['response'].notna() & pre['question'].notna()].copy()
    pre = pre.sample(20, random_state=42).copy()
    pre['source'] = 'preflight'
    pre['old_score'] = pre['agreeableness_score']
    pre['item_id'] = pre['question_id']
    pre['base_model'] = pre['model']
    pre['epoch'] = ''; pre['pole'] = ''; pre['train_axis'] = ''
    print(f'preflight rows: {len(pre)}')

    df['old_score'] = df['score']
    df['old_judge_response'] = df['judge_response']
    cols = ['source','question','response','question_id','item_id','base_model','epoch','pole','train_axis','old_score']
    # Build minimal job records
    jobs = []
    for _, r in df.iterrows():
        jobs.append({'source':'cross-elicit','question':r['question'],'response':r['response'],
                     'item_id':r['item_id'],'base_model':r['base_model'],'epoch':r['epoch'],
                     'pole':r['pole'],'train_axis':r['train_axis'],'old_score':r['score']})
    for _, r in pre.iterrows():
        jobs.append({'source':'preflight','question':r['question'],'response':r['response'],
                     'item_id':r['question_id'],'base_model':r['model'],'epoch':'',
                     'pole':r.get('condition',''),'train_axis':'','old_score':r['agreeableness_score']})
    print(f'TOTAL jobs: {len(jobs)}')

    # Resume support
    done_keys = set()
    if OUT_PATH.exists():
        ex = pd.read_csv(OUT_PATH)
        for _, r in ex.iterrows():
            done_keys.add((str(r['source']), str(r['item_id']), str(r['base_model']), str(r['pole']), str(r['epoch'])))
        print(f'Resuming: {len(done_keys)} already done')
    todo = [j for j in jobs if (j['source'], str(j['item_id']), str(j['base_model']), str(j['pole']), str(j['epoch'])) not in done_keys]
    print(f'todo: {len(todo)}')
    if not todo: return

    sem = asyncio.Semaphore(CONCURRENCY)
    fieldnames = ['source','item_id','base_model','epoch','pole','train_axis','old_score','new_score','new_judge_response','in_tok','out_tok']
    write_header = not OUT_PATH.exists()
    f = open(OUT_PATH, 'a', newline='', encoding='utf-8')
    w = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header: w.writeheader()

    t_start = time.time()
    completed = 0
    total_in = 0; total_out = 0

    async def runner(j):
        nonlocal completed, total_in, total_out
        result = await judge_one(j, sem)
        out_row = {k:result.get(k,'') for k in fieldnames}
        # Drop large strings (question/response) — already on disk in source
        w.writerow(out_row); f.flush()
        completed += 1
        total_in += result.get('in_tok',0); total_out += result.get('out_tok',0)
        if completed % 50 == 0 or completed == len(todo):
            elapsed = time.time() - t_start
            rate = completed/elapsed if elapsed else 0
            cost = (total_in/1e6 * 0.75) + (total_out/1e6 * 4.5)
            eta = (len(todo)-completed)/rate if rate>0 else 0
            print(f'[{completed}/{len(todo)}] {rate:.1f}/s  in={total_in/1e6:.2f}M out={total_out/1e3:.1f}K  cost~${cost:.2f}  eta {eta:.0f}s', flush=True)

    await asyncio.gather(*[runner(j) for j in todo])
    f.close()
    elapsed = time.time() - t_start
    cost = (total_in/1e6 * 0.75) + (total_out/1e6 * 4.5)
    print(f'\nDONE. {completed} calls in {elapsed:.0f}s. Tokens in={total_in/1e6:.2f}M out={total_out/1e3:.1f}K. Cost~${cost:.2f} (uncached).')

asyncio.run(main())
