#!/usr/bin/env python3
"""Re-judge narcissism / honest-humble / agreeableness for the SFT Llama
cross-eval run, using the CURRENT cross-elicit judge prompts, and write a new
scores JSON to june/ with ONLY those three eval axes updated (every other axis
and every meta field copied verbatim from the canonical johannes scores file).

Why this is apples-to-apples: for each (pole, eval_axis) cell we re-judge the
*exact* original generations — we key off the cell's `meta.dirname` (== the
scored.csv `source_dir`) and the cell's existing `scores` sample-id set
({item_id}__p{paraphrase}__s{sample}). Only the judge prompt changes.

Judge prompts/models come from johannes/cross-elicit/evals/<dir>/<dir>_eval.yaml
(the canonical cross-elicit copy, already carrying narcissism v2 / honest-humble
v3 / agreeableness 5-facet). Agreeableness collapses its 5 gemini-judged facets
to a single score per response via the mean of the available facet scores
(NEO-PI-R domain = facet mean).

Resumable: every judge result is appended to a cache jsonl keyed
axis|pole|sample_id|metric, so a crash/rerun skips completed calls.

Outputs:
  june/scores_meta-llama-Llama-3.1-8B-Instruct.json   (consumed by build_results_matrix.py)
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path("/home/hunter/ai/spar-ood-propensities")
load_dotenv(ROOT / ".env")
from openai import AsyncOpenAI  # noqa: E402

BASE_SCORES = ROOT / "johannes/cross-elicit/results/scores_meta-llama-Llama-3.1-8B-Instruct.json"
OUT_SCORES = ROOT / "june/scores_meta-llama-Llama-3.1-8B-Instruct.json"
DATA_DIR = ROOT / "june/cross_elicit_audit/data"
EVALS_DIR = ROOT / "johannes/cross-elicit/evals"
CACHE_PATH = ROOT / "june/cross_elicit_audit/rejudge/_matrix_sft_cache.jsonl"

DEFAULT_JUDGE = "openai/gpt-5.4-mini"
CONCURRENCY = 16

# axis key in scores JSON cells == eval-dir name == scored.csv stem
AXES = {
    "narcissism": {"dir": "narcissism", "csv": "narcissism_scored.csv",
                   "metrics": ["narcissism_score"], "collapse": "single"},
    "honest-humble": {"dir": "honest-humble", "csv": "honest-humble_scored.csv",
                      "metrics": ["honesty_humility_score"], "collapse": "single"},
    "agreeableness": {"dir": "agreeableness", "csv": "agreeableness_scored.csv",
                      "metrics": ["trust_score", "altruism_score",
                                  "cooperation_score", "modesty_score",
                                  "sympathy_score"], "collapse": "mean"},
}

client = AsyncOpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                     base_url="https://openrouter.ai/api/v1")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def parse_score(text: str | None):
    if text is None:
        return None
    s = text.strip()
    if s.lower().startswith(("null", "none", "n/a")) or s == "":
        return None
    m = _NUM.search(s)
    if not m:
        return None
    try:
        v = float(m.group(0))
    except ValueError:
        return None
    return v if 0 <= v <= 100 else None


def load_axis_yaml(dirname: str):
    """Return (judge_prompts dict, judge_models dict) from the anchor item."""
    p = EVALS_DIR / dirname / f"{dirname}_eval.yaml"
    it = yaml.safe_load(p.read_text())[0]
    return it["judge_prompts"], dict(it.get("judge_models") or {})


def build_response_index(csv_name: str) -> dict[tuple[str, str], tuple[str, str]]:
    """(source_dir, sample_id) -> (question, response) for the scored pool."""
    df = pd.read_csv(DATA_DIR / csv_name, low_memory=False)
    df = df[df["response"].notna() & df["question"].notna()]
    idx: dict[tuple[str, str], tuple[str, str]] = {}
    for r in df.itertuples(index=False):
        sid = f"{r.item_id}__p{int(r.paraphrase_idx)}__s{int(r.sample_idx)}"
        idx[(str(r.source_dir), sid)] = (str(r.question), str(r.response))
    return idx


# ---- resumable cache --------------------------------------------------------
_cache: dict[str, object] = {}


def load_cache():
    if CACHE_PATH.exists():
        for line in CACHE_PATH.read_text().splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            _cache[o["k"]] = o["v"]
    print(f"cache: {len(_cache)} results loaded", flush=True)


_cache_fh = None


def cache_put(key: str, val):
    global _cache_fh
    _cache[key] = val
    if _cache_fh is None:
        _cache_fh = open(CACHE_PATH, "a", encoding="utf-8")
    _cache_fh.write(json.dumps({"k": key, "v": val}) + "\n")
    _cache_fh.flush()


async def judge(model: str, prompt: str, sem: asyncio.Semaphore):
    async with sem:
        for attempt in range(4):
            try:
                r = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0 if attempt == 0 else 0.3,
                    max_tokens=64,
                )
                return parse_score(r.choices[0].message.content)
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    print(f"  judge err: {type(e).__name__}: {str(e)[:120]}",
                          file=sys.stderr, flush=True)
                    return None
                await asyncio.sleep(2 * (2 ** attempt))


def aggregate_metrics(values: list[float | None], n_total: int) -> dict:
    nums = [v for v in values if v is not None]
    n_numeric = len(nums)
    return {
        "mean": (statistics.fmean(nums) if nums else None),
        "std": (statistics.pstdev(nums) if len(nums) > 1 else 0.0 if nums else None),
        "min": (min(nums) if nums else None),
        "max": (max(nums) if nums else None),
        "n_total": n_total,
        "n_numeric": n_numeric,
        "n_nulls": n_total - n_numeric,
        "n_fails": 0,
    }


async def main():
    load_cache()
    scores = json.loads(BASE_SCORES.read_text())
    out = copy.deepcopy(scores)
    cells = out["cells"]
    sem = asyncio.Semaphore(CONCURRENCY)

    # ---- 1. assemble all judge jobs (skip cached) --------------------------
    jobs = []                       # (cache_key, model, prompt)
    plan = []                       # (axis, pole, sample_id, metric, model)
    miss_count = {a: 0 for a in AXES}

    for axis, spec in AXES.items():
        jp, jm = load_axis_yaml(spec["dir"])
        idx = build_response_index(spec["csv"])
        for pole, cell in cells.items():
            if axis not in cell:
                continue
            ac = cell[axis]
            dirname = (ac.get("meta") or {}).get("dirname")
            for sid in list(ac.get("scores", {})):
                qr = idx.get((str(dirname), sid))
                if qr is None:
                    miss_count[axis] += 1
                    continue
                q, resp = qr
                for metric in spec["metrics"]:
                    model = jm.get(metric, DEFAULT_JUDGE)
                    key = f"{axis}|{pole}|{sid}|{metric}"
                    plan.append((axis, pole, sid, metric, model))
                    if key in _cache:
                        continue
                    prompt = (jp[metric].replace("{question}", q)
                              .replace("{answer}", resp))
                    jobs.append((key, model, prompt))

    total_plan = len(plan)
    print(f"plan: {total_plan} (axis,pole,sample,metric) cells | "
          f"{len(jobs)} judge calls to run | "
          f"misses(no matching generation): {miss_count}", flush=True)

    # ---- 2. run uncached judge calls --------------------------------------
    t0 = time.time()
    done = 0

    async def runner(key, model, prompt):
        nonlocal done
        v = await judge(model, prompt, sem)
        cache_put(key, v)
        done += 1
        if done % 250 == 0 or done == len(jobs):
            el = time.time() - t0
            rate = done / el if el else 0
            eta = (len(jobs) - done) / rate if rate else 0
            print(f"[{done}/{len(jobs)}] {rate:.1f}/s eta {eta:.0f}s", flush=True)

    if jobs:
        await asyncio.gather(*[runner(k, m, p) for k, m, p in jobs])
    print(f"judging done in {time.time() - t0:.0f}s", flush=True)

    # ---- 3. collapse to per-sample scores, rebuild cells ------------------
    summary = {}
    for axis, spec in AXES.items():
        metrics = spec["metrics"]
        old_means, new_means = [], []
        for pole, cell in cells.items():
            if axis not in cell:
                continue
            ac = cell[axis]
            old_scores = ac.get("scores", {})
            old_mean = (ac.get("metrics") or {}).get("mean")
            new_scores: dict[str, float] = {}
            per_sample_vals: list[float | None] = []
            for sid in old_scores:
                facet_vals = [
                    _cache.get(f"{axis}|{pole}|{sid}|{m}") for m in metrics
                ]
                if spec["collapse"] == "single":
                    val = facet_vals[0]
                else:  # mean of available facets
                    av = [v for v in facet_vals if v is not None]
                    val = statistics.fmean(av) if av else None
                per_sample_vals.append(val)
                if val is not None:
                    new_scores[sid] = round(val, 4)
            ac["metrics"] = aggregate_metrics(per_sample_vals, len(old_scores))
            ac["scores"] = new_scores
            meta = ac.setdefault("meta", {})
            meta["judge_model"] = (
                "google/gemini-2.5-flash" if spec["collapse"] == "mean"
                else DEFAULT_JUDGE
            )
            meta["rejudged"] = True
            meta["rejudge_note"] = (
                "narcissism v2 / honest-humble v3 / agreeableness 5-facet-mean "
                "(canonical cross-elicit judge prompts, 2026-05-15)"
            )
            if old_mean is not None:
                old_means.append(old_mean)
            nm = ac["metrics"]["mean"]
            if nm is not None:
                new_means.append(nm)
        summary[axis] = (
            statistics.fmean(old_means) if old_means else float("nan"),
            statistics.fmean(new_means) if new_means else float("nan"),
            miss_count[axis],
        )

    OUT_SCORES.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_SCORES}", flush=True)
    print("axis            old_mean  new_mean  unmatched")
    for a, (om, nm, ms) in summary.items():
        print(f"  {a:<14} {om:7.2f}  {nm:7.2f}  {ms}")


if __name__ == "__main__":
    asyncio.run(main())
