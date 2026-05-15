#!/usr/bin/env python3
"""Focused equivalence retest on the orthogonalized judge prompts that are
NOT byte-identical to the cross-elicit prompts the main study validated:

  ethical-framework-deontological, ethical-framework-utilitarian,
  narcissism, neuroticism, resource-acquisition

Clean paired A/B isolating ONLY the restructure reorder: same responses,
same model, **temperature=0 on both arms** (no resampling noise) — arm A =
orthog prompt CURRENT geometry, arm B = orthog prompt RESTRUCTURED geometry.
If the reorder is benign these prompts' scores should barely move.

Responses come from the local cross-elicit data/<eval>_scored.csv (model
generations are judge-prompt-invariant). Single-metric evals only (all 5 are).

Usage: python3 retest_nonoverlap.py [--n 80] [--dry-run]
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import statistics as st
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
from restructure_prompt import restructure, ORTHOG_EVALS  # noqa: E402

try:
    from dotenv import load_dotenv
    for _p in (REPO / ".env", REPO / "june" / ".env"):
        if _p.exists():
            load_dotenv(_p)
except ImportError:
    pass

DATA = REPO / "june" / "cross_elicit_audit" / "data"
EVALS = ["ethical-framework-deontological", "ethical-framework-utilitarian",
         "narcissism", "neuroticism", "resource-acquisition"]
BASE = "meta-llama-Llama-3.1-8B-Instruct"
MODEL = "openai/gpt-5.4-mini"          # production judge for these 5
PRICE_IN = 0.75 / 1e6
DELTA = 3.0
CONC = 12
_NUM = __import__("re").compile(r"-?\d+(?:\.\d+)?")


def parse_score(t):
    if t is None:
        return float("nan")
    s = str(t).strip()
    if s.lower().startswith(("null", "none", "n/a")) or s == "":
        return float("nan")
    m = _NUM.search(s)
    if not m:
        return float("nan")
    v = float(m.group(0))
    return v if 0 <= v <= 100 else float("nan")


def orthog_prompt(ev: str) -> tuple[str, str]:
    d = yaml.safe_load((ORTHOG_EVALS / ev / f"{ev}_eval.yaml").read_text())
    jp = d[0]["judge_prompts"]
    metric = list(jp)[0]
    return metric, jp[metric]


def sample_responses(ev: str, n: int, seed=42) -> list[tuple[str, str]]:
    rows = []
    seen = set()
    with (DATA / f"{ev}_scored.csv").open() as fh:
        for r in csv.DictReader(fh):
            if r.get("base_model") != BASE:
                continue
            k = (r["item_id"], r["paraphrase_idx"], r["sample_idx"], r["pole"])
            if k in seen or not r.get("response"):
                continue
            seen.add(k)
            rows.append((r["question"], r["response"]))
    rng = random.Random(seed)
    return rng.sample(rows, min(n, len(rows)))


async def main_async(n: int, dry: bool):
    from openai import AsyncOpenAI
    tokens = {"in": 0}

    plan = {}
    for ev in EVALS:
        metric, cur = orthog_prompt(ev)
        res, flags = restructure(cur)
        reps = sample_responses(ev, n)
        plan[ev] = (metric, cur, res, reps, flags)
        print(f"{ev:34s} metric={metric} resp={len(reps)} "
              f"cur={len(cur)}c res={len(res)}c deixis={len(flags)}")
    total_calls = sum(len(v[3]) for v in plan.values()) * 2
    est = total_calls * 2500 * PRICE_IN  # ~2.5k tok/call heuristic
    print(f"\npaired calls: {total_calls}  projected ~${est:.2f}")
    if dry:
        print("DRY RUN — no API.")
        return

    client = AsyncOpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                         base_url="https://openrouter.ai/api/v1")
    sem = asyncio.Semaphore(CONC)

    async def judge(prompt):
        async with sem:
            for att in range(3):
                try:
                    r = await client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0, max_tokens=64)
                    if getattr(r, "usage", None):
                        tokens["in"] += r.usage.prompt_tokens or 0
                    return parse_score(r.choices[0].message.content)
                except Exception as e:  # noqa: BLE001
                    if att == 2:
                        print(f"  err: {e}", file=sys.stderr)
                        return float("nan")
                    await asyncio.sleep(2 * 2 ** att)

    t0 = time.time()
    rows = []
    for ev, (metric, cur, res, reps, _) in plan.items():
        async def one(q, a):
            pc = cur.replace("{question}", q).replace(
                "{answer}", a).replace("{response}", a)
            pr = res.replace("{question}", q).replace(
                "{answer}", a).replace("{response}", a)
            sc, sr = await asyncio.gather(judge(pc), judge(pr))
            return sc, sr
        out = await asyncio.gather(*[one(q, a) for q, a in reps])
        for (q, a), (sc, sr) in zip(reps, out):
            rows.append({"eval": ev, "score_current": sc,
                         "score_restructured": sr})
        print(f"  {ev}: done [{time.time()-t0:.0f}s]")

    out_csv = HERE / "retest_nonoverlap.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["eval", "score_current",
                                          "score_restructured"])
        w.writeheader()
        w.writerows(rows)

    # Per-eval paired stats: mean Δ, median, sd, paired TOST(δ=3), Pearson.
    print("\n=== RETEST: restructured − current geometry (same prompt) ===")
    print(f"{'eval':34s} {'n':>4} {'meanΔ':>7} {'med':>5} {'sd':>6} "
          f"{'TOST90%CI':>16} {'pass':>5} {'pearson':>7}")
    rep = []
    for ev in EVALS:
        d = [(r["score_restructured"], r["score_current"]) for r in rows
             if r["eval"] == ev
             and r["score_current"] == r["score_current"]
             and r["score_restructured"] == r["score_restructured"]]
        if len(d) < 5:
            print(f"{ev:34s}  n<5 — skip")
            continue
        diffs = [b - a for b, a in d]
        rng = random.Random(0)
        boots = sorted(st.fmean(rng.choices(diffs, k=len(diffs)))
                       for _ in range(2000))
        lo, hi = boots[100], boots[1900]            # 90% CI
        ok = (-DELTA < lo) and (hi < DELTA)
        xs = [a for b, a in d]
        ys = [b for b, a in d]
        try:
            pear = st.correlation(xs, ys)
        except Exception:
            pear = float("nan")
        print(f"{ev:34s} {len(d):4d} {st.fmean(diffs):+7.2f} "
              f"{st.median(diffs):+5.1f} {st.pstdev(diffs):6.2f} "
              f"[{lo:+.2f},{hi:+.2f}] {str(ok):>5} {pear:7.3f}")
        rep.append({"eval": ev, "n": len(d), "mean_delta": st.fmean(diffs),
                    "median": st.median(diffs), "sd": st.pstdev(diffs),
                    "tost90_lo": lo, "tost90_hi": hi, "tost_pass": ok,
                    "pearson": pear})
    cost = tokens["in"] * PRICE_IN
    summary = {"model": MODEL, "delta": DELTA, "per_eval": rep,
               "empirical_cost_usd": round(cost, 4),
               "input_tokens": tokens["in"],
               "wallclock_s": round(time.time() - t0, 1)}
    (HERE / "retest_nonoverlap_report.json").write_text(
        json.dumps(summary, indent=2))
    npass = sum(1 for r in rep if r["tost_pass"])
    print(f"\nTOST(δ=3) pass: {npass}/{len(rep)}  | empirical cost "
          f"${cost:.4f} ({tokens['in']/1e6:.2f}M in, "
          f"{summary['wallclock_s']:.0f}s)")
    print("report -> retest_nonoverlap_report.json / retest_nonoverlap.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=80,
                    help="responses per eval (default 80)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    asyncio.run(main_async(a.n, a.dry_run))


if __name__ == "__main__":
    main()
