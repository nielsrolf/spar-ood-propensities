"""Run narcissism v3 judge on all rows with existing expert ratings.

Sources:
  A. Iteration set (16 rows): narc_v1_v2_results.csv — has v1, v2, expert
  B. Held-out subsample (26 rows): fresh132_unblinded_subsample.csv — has v2 only
     + Niels's blind ratings (~/Downloads/fresh132_scoring_sheet_niels.csv)
     + Claude's parallel ratings (fresh132_scoring_sheet_claude.csv)

Output: v3_iteration/v3_judge_results.csv with one row per unique (question, response) pair.
"""
import asyncio, os, re, time, csv, hashlib
from pathlib import Path
import pandas as pd
from openai import AsyncOpenAI

ROOT = Path("/Users/nielswarncke/Documents/spar-ood-propensities")
V3_DIR = ROOT / "june/cross_elicit_audit/output/expert_review/narcissism/v3_iteration"
V2_DIR = ROOT / "june/cross_elicit_audit/output/expert_review/narcissism/v2_iteration"
REJUDGE = ROOT / "june/cross_elicit_audit/rejudge"

V3 = (V3_DIR / "judge_prompt_v3.md").read_text()
MODEL = "openai/gpt-5.4-mini"
N_SAMPLES = 3
CONCURRENCY = 16

client = AsyncOpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)
NUM = re.compile(r"\b(\d{1,3}(?:\.\d+)?)\b")

def parse(t):
    if not t: return None
    if t.strip().lower().startswith("null"): return None
    m = NUM.search(t)
    if m:
        try:
            v = float(m.group(1))
            if 0 <= v <= 100: return v
        except Exception:
            pass
    return None

async def call(prompt, sem):
    async with sem:
        for attempt in range(4):
            try:
                r = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=32,
                )
                return parse(r.choices[0].message.content)
            except Exception as e:
                if attempt == 3:
                    print(f"  failed: {e!r}")
                    return None
                await asyncio.sleep(2 * (2**attempt))

def row_key(q, r):
    return hashlib.md5((q + "|||" + r).encode()).hexdigest()[:12]

async def main():
    # Source A: iteration set
    iter_df = pd.read_csv(REJUDGE / "narc_v1_v2_results.csv")
    iter_df["src_iter"] = True

    # Source B: held-out 26
    held_df = pd.read_csv(V2_DIR / "fresh132_unblinded_subsample.csv")
    held_df["src_held"] = True

    # Niels + Claude blind ratings
    niels = pd.read_csv(Path.home() / "Downloads/fresh132_scoring_sheet_niels.csv")
    claude = pd.read_csv(V2_DIR / "fresh132_scoring_sheet_claude.csv")

    # Build unified table keyed on (question, response) hash
    rows = {}
    for _, r in iter_df.iterrows():
        k = row_key(str(r["question"]), str(r["response"]))
        rows[k] = {
            "key": k, "question": r["question"], "response": r["response"],
            "item_id": r.get("item_id"), "checkpoint_label": r.get("checkpoint_label"),
            "pole": r.get("pole"),
            "v1": r.get("judge_v1"), "v2": r.get("judge_v2"),
            "expert_iter": r.get("expert"),
            "src": "iter",
        }
    for _, r in held_df.iterrows():
        k = row_key(str(r["question"]), str(r["response"]))
        if k in rows:
            # already in iter; merge
            rows[k]["src"] = "iter+held"
            rows[k]["rate_uid"] = r["rate_uid"]
            if pd.isna(rows[k].get("v2")):
                rows[k]["v2"] = r["v2"]
        else:
            rows[k] = {
                "key": k, "question": r["question"], "response": r["response"],
                "item_id": r.get("item_id"), "checkpoint_label": r.get("checkpoint_label"),
                "pole": r.get("pole"),
                "v1": None, "v2": r["v2"], "expert_iter": None,
                "rate_uid": r["rate_uid"],
                "src": "held",
            }
    # Attach Niels / Claude scores by rate_uid
    niels_by_uid = {r["rate_uid"]: r for _, r in niels.iterrows()}
    claude_by_uid = {r["rate_uid"]: r for _, r in claude.iterrows()}
    for r in rows.values():
        uid = r.get("rate_uid")
        if uid:
            n = niels_by_uid.get(uid)
            if n is not None:
                v = n["narcissism_score_0_100"]
                r["expert_niels"] = float(v) if pd.notna(v) and v != "" else None
                r["niels_notes"] = n.get("notes", "")
            c = claude_by_uid.get(uid)
            if c is not None:
                v = c["expert_score"]
                r["expert_claude"] = float(v) if pd.notna(v) and v != "" else None
                r["claude_notes"] = c.get("notes", "")

    all_rows = list(rows.values())
    print(f"Unique rows to judge with v3: {len(all_rows)}")
    print(f"  iter-only: {sum(1 for r in all_rows if r['src'] == 'iter')}")
    print(f"  iter+held: {sum(1 for r in all_rows if r['src'] == 'iter+held')}")
    print(f"  held-only: {sum(1 for r in all_rows if r['src'] == 'held')}")

    sem = asyncio.Semaphore(CONCURRENCY)
    t0 = time.time()
    # Prewarm
    _ = await call(V3.replace("{question}", "x").replace("{answer}", "x"), sem)
    print(f"prewarm: {time.time()-t0:.0f}s")

    jobs = []
    for r in all_rows:
        p = V3.replace("{question}", str(r["question"])).replace("{answer}", str(r["response"]))
        for s in range(N_SAMPLES):
            jobs.append((r["key"], s, p))
    print(f"total calls: {len(jobs)}")

    t0 = time.time()
    results = await asyncio.gather(*[call(p, sem) for _, _, p in jobs])
    print(f"done in {time.time()-t0:.0f}s; nulls/errors: {sum(r is None for r in results)}")

    # Aggregate
    scores_by_key = {}
    for (k, s, _), sc in zip(jobs, results):
        scores_by_key.setdefault(k, []).append(sc)
    for r in all_rows:
        ss = [s for s in scores_by_key[r["key"]] if s is not None]
        r["v3_n"] = len(ss)
        if ss:
            r["v3"] = sum(ss) / len(ss)
            if len(ss) > 1:
                m = r["v3"]
                r["v3_std"] = (sum((s - m) ** 2 for s in ss) / (len(ss) - 1)) ** 0.5
            else:
                r["v3_std"] = None
        else:
            r["v3"] = None
            r["v3_std"] = None
        r["v3_raw"] = ",".join(f"{s:.0f}" if s is not None else "null" for s in scores_by_key[r["key"]])

    # Save
    cols = ["key", "src", "item_id", "checkpoint_label", "pole", "rate_uid",
            "v1", "v2", "v3", "v3_std", "v3_n", "v3_raw",
            "expert_iter", "expert_niels", "expert_claude",
            "question", "response", "niels_notes", "claude_notes"]
    out = V3_DIR / "v3_judge_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"-> {out}")

asyncio.run(main())
