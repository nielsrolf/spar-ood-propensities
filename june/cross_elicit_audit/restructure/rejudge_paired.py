#!/usr/bin/env python3
"""Paired candidate re-judge for the Option 2 equivalence study.

Reference arm  = june/scores_<model>.json  (the NEW rebuilt matrix: narcissism
                 v2 / honest-humble v3 / agreeableness 5-facet, 2026-05-15).
                 Per-cell mean already there → $0, no calls.
Candidate arm  = same per-cell judge model (meta.judge_model), same rubric
                 *content*, but the RESTRUCTURED prompt geometry, re-judging the
                 SAME (question, answer) pairs (responses are identical across
                 judge-prompt versions; only the score changes).

Pairing key: cid = "<item_id>__p<paraphrase_idx>__s<sample_idx>", recovered from
the cell's eval_results dir (meta.dirname) rows.jsonl. june/scores has no answer
text, so rows.jsonl is required — that corpus lives on the synced/Colab env,
not necessarily this laptop; missing dirs are skipped and reported, never faked.

Aggregation parity: a cell's per-cid score is the mean over the eval's judge
metrics (single-metric evals → that metric; agreeableness → 5-facet mean, etc).
`cell_metrics_for()` is the single, explicit, overridable place that rule lives
— reconcile it against the exact production rejudge before the real run.

Cost: every candidate call's token usage is logged to cost_ledger.json
(empirical $ to run the harness, per the spec), plus the realized cacheable-
prefix size for the projected production saving.

Usage:
  python3 rejudge_paired.py --scores ../../scores_meta-llama-Llama-3.1-8B-Instruct.json \\
      --out paired_scores.csv --limit 50          # pilot (σ_d)
  python3 rejudge_paired.py --scores <json> --out paired_scores.csv   # full slice
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "june" / "propensity_audit" / "orthog_pipeline"))

from restructure_prompt import load_canonical_templates, restructure  # noqa: E402

# Reuse the battle-tested 0-100 parser (null→NaN, first-number extraction).
from paired_generate import _parse_score  # noqa: E402

EVAL_RESULTS = REPO / "johannes" / "cross-elicit" / "eval_results"
MANIFEST = HERE / "reviewed_manifest.yaml"
LEDGER = HERE / "cost_ledger.json"

# $ per 1M tokens (input, output). OpenRouter list prices; output is ~noise
# (judge replies are a few tokens) but tracked anyway.
PRICING = {
    "openai/gpt-5.4-mini": (0.75, 3.0),
    "gpt-5.4-mini": (0.75, 3.0),
    "google/gemini-2.5-flash": (0.30, 2.5),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-mini": (0.15, 0.60),
}
CONCURRENCY = int(os.environ.get("REJUDGE_CONCURRENCY", "16"))


def reviewed_ok() -> dict[tuple[str, str], bool]:
    if not MANIFEST.exists():
        raise SystemExit("run restructure_prompt.py --review first (no manifest)")
    m = yaml.safe_load(MANIFEST.read_text()) or {}
    out = {}
    for k, v in m.items():
        ev, _, metric = k.partition("/")
        out[(ev, metric)] = bool(v.get("reviewed"))
    return out


def cell_metrics_for(eval_name: str, ref_metrics_block: dict) -> list[str]:
    """Judge metrics whose mean defines the cell score.

    Default: every metric the canonical eval YAML defines for this eval that
    also has a reviewed restructured template. The cell per-cid score is their
    mean — this reproduces 'agreeableness 5-facet-mean' and the single-metric
    case alike. Override here if the production rejudge used a different rule
    for a specific eval (e.g. ethical-framework primary-key only).
    """
    tpls = load_canonical_templates()
    return sorted(m for (ev, m) in tpls if ev == eval_name)


def load_rows(dirname: str) -> dict[tuple[str, int, int], dict] | None:
    """{(item_id,p,s): {question, answer}} from <dir>/rows.jsonl, or None if the
    eval_results dir is not present locally (corpus lives on synced env)."""
    for base in (EVAL_RESULTS, EVAL_RESULTS / "finetuning"):
        p = base / dirname / "rows.jsonl"
        if p.exists():
            out: dict[tuple[str, int, int], dict] = {}
            with p.open() as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    k = (r["item_id"], r["paraphrase_idx"], r["sample_idx"])
                    if k not in out:
                        out[k] = {"question": r["question"], "answer": r["answer"]}
            return out
    return None


class Ledger:
    def __init__(self) -> None:
        self.n = 0
        self.in_tok = 0
        self.out_tok = 0
        self.cached_tok = 0
        self.cost = 0.0
        self.by_model: dict[str, dict] = {}
        self.t0 = time.time()

    def add(self, model: str, usage) -> None:
        pin, pout = PRICING.get(model, (0.75, 3.0))
        it = getattr(usage, "prompt_tokens", 0) or 0
        ot = getattr(usage, "completion_tokens", 0) or 0
        det = getattr(usage, "prompt_tokens_details", None)
        ct = getattr(det, "cached_tokens", 0) or 0 if det else 0
        c = (it - ct) / 1e6 * pin + ct / 1e6 * pin * 0.5 + ot / 1e6 * pout
        self.n += 1
        self.in_tok += it
        self.out_tok += ot
        self.cached_tok += ct
        self.cost += c
        b = self.by_model.setdefault(
            model, {"n": 0, "in": 0, "out": 0, "cached": 0, "cost": 0.0})
        b["n"] += 1
        b["in"] += it
        b["out"] += ot
        b["cached"] += ct
        b["cost"] += c

    def dump(self) -> dict:
        d = {
            "n_calls": self.n,
            "input_tokens": self.in_tok,
            "output_tokens": self.out_tok,
            "cached_input_tokens": self.cached_tok,
            "cache_hit_frac": (self.cached_tok / self.in_tok) if self.in_tok else 0.0,
            "empirical_cost_usd": round(self.cost, 4),
            "wallclock_s": round(time.time() - self.t0, 1),
            "by_model": self.by_model,
        }
        LEDGER.write_text(json.dumps(d, indent=2))
        return d


async def judge(client, model: str, prompt: str, sem, ledger: Ledger) -> float:
    async with sem:
        for attempt in range(3):
            try:
                r = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0 if attempt == 0 else 0.3,
                    max_tokens=512,
                )
                if getattr(r, "usage", None) is not None:
                    ledger.add(model, r.usage)
                s = _parse_score(r.choices[0].message.content)
                if s == s:  # not NaN
                    return s
            except Exception as e:  # noqa: BLE001
                print(f"  judge err ({model}, try {attempt+1}): {e}",
                      file=sys.stderr)
        return float("nan")


async def run(scores_path: Path, out_csv: Path, limit: int | None,
              cells_filter: set[str] | None) -> None:
    from openai import AsyncOpenAI

    data = json.loads(scores_path.read_text())
    rev = reviewed_ok()
    tpls = load_canonical_templates()
    restructured: dict[tuple[str, str], str] = {}
    for k, t in tpls.items():
        if rev.get(k):
            restructured[k] = restructure(t)[0]

    ledger = Ledger()
    jobs: list[tuple] = []          # (pole, eval, cid, model, [prompts], ref)
    skipped_dirs: list[str] = []
    n_unreviewed: set[str] = set()

    for pole, evals in data["cells"].items():
        for ev, cell in evals.items():
            if cells_filter and ev not in cells_filter:
                continue
            metrics = cell_metrics_for(ev, cell.get("metrics") or {})
            usable = [m for m in metrics if (ev, m) in restructured]
            if not usable:
                for m in metrics:
                    if not rev.get((ev, m), False):
                        n_unreviewed.add(f"{ev}/{m}")
                continue
            model = (cell.get("meta") or {}).get("judge_model") or "openai/gpt-5.4-mini"
            rows = load_rows((cell.get("meta") or {}).get("dirname", ""))
            if rows is None:
                skipped_dirs.append((cell.get("meta") or {}).get("dirname", "?"))
                continue
            ref_scores = cell.get("scores") or {}
            for cid, ref in ref_scores.items():
                item_id, _, rest = cid.partition("__p")
                p_idx, _, s_part = rest.partition("__s")
                key = (item_id, int(p_idx or 0), int(s_part or 0))
                qa = rows.get(key)
                if qa is None or not qa.get("answer"):
                    continue
                prompts = [
                    restructured[(ev, m)]
                    .replace("{question}", qa["question"])
                    .replace("{answer}", qa["answer"])
                    .replace("{response}", qa["answer"])
                    for m in usable
                ]
                jobs.append((pole, ev, cid, model, prompts, ref))

    if limit:
        jobs = jobs[:limit]
    print(f"jobs: {len(jobs)} cid×cell  | skipped(no rows.jsonl): "
          f"{len(skipped_dirs)} dirs | unreviewed-skipped: {len(n_unreviewed)}")
    if not jobs:
        print("Nothing to do. (On this laptop the eval_results corpus is "
              "likely unsynced — run where rows.jsonl exists.)")
        ledger.dump()
        return

    client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(job):
        pole, ev, cid, model, prompts, ref = job
        facet = await asyncio.gather(
            *[judge(client, model, p, sem, ledger) for p in prompts])
        good = [x for x in facet if x == x]
        cand = sum(good) / len(good) if good else float("nan")
        return {
            "pole": pole, "eval": ev, "cid": cid, "judge_model": model,
            "score_ref": ref, "score_cand": cand, "n_metrics": len(prompts),
        }

    t0 = time.time()
    results = []
    done = 0
    for fut in asyncio.as_completed([one(j) for j in jobs]):
        results.append(await fut)
        done += 1
        if done % max(1, len(jobs) // 20) == 0 or done == len(jobs):
            led = ledger.dump()
            print(f"  [{done}/{len(jobs)}] ${led['empirical_cost_usd']:.2f} "
                  f"cache={led['cache_hit_frac']*100:.0f}% "
                  f"[{time.time()-t0:.0f}s]")

    import csv
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    led = ledger.dump()
    print(f"\nwrote {out_csv} ({len(results)} paired rows)")
    print(f"empirical harness cost: ${led['empirical_cost_usd']:.4f}  "
          f"({led['n_calls']} calls, {led['input_tokens']/1e6:.2f}M in, "
          f"cache {led['cache_hit_frac']*100:.0f}%, {led['wallclock_s']:.0f}s)")
    print(f"cost ledger -> {LEDGER}")
    if skipped_dirs:
        print(f"NOTE: {len(skipped_dirs)} cells skipped (rows.jsonl absent "
              f"locally); rerun on the synced corpus for full coverage.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True, type=Path,
                    help="june/scores_<model>.json (reference arm)")
    ap.add_argument("--out", type=Path, default=HERE / "paired_scores.csv")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap #cid×cell jobs (pilot for σ_d)")
    ap.add_argument("--evals", default=None,
                    help="comma-separated eval filter (e.g. risk-affinity,effort)")
    a = ap.parse_args()
    cells = set(a.evals.split(",")) if a.evals else None
    asyncio.run(run(a.scores, a.out, a.limit, cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
