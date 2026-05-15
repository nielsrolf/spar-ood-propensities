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
import random
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

# Self-load .env (parity with paired_generate.py); .env is not shell-sourceable
# (multi-line values), so dotenv is the correct loader.
try:
    from dotenv import load_dotenv
    for _p in (REPO / ".env", REPO / "june" / ".env", HERE / ".env"):
        if _p.exists():
            load_dotenv(_p)
except ImportError:
    pass

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "june" / "propensity_audit" / "orthog_pipeline"))

from restructure_prompt import load_canonical_templates, restructure  # noqa: E402

import re as _re

# Parser identical to the production rejudge (rejudge_matrix_sft.parse_score):
# null/none/n-a/empty → None, first number, clamp to [0,100] else None. Parity
# matters — this is the de-confound, the only intended difference is geometry.
_NUM_RE = _re.compile(r"-?\d+(?:\.\d+)?")


def parse_score(text):
    if text is None:
        return float("nan")
    s = str(text).strip()
    if s.lower().startswith(("null", "none", "n/a")) or s == "":
        return float("nan")
    m = _NUM_RE.search(s)
    if not m:
        return float("nan")
    try:
        v = float(m.group(0))
    except ValueError:
        return float("nan")
    return v if 0 <= v <= 100 else float("nan")


# Production AXES (rejudge_matrix_sft.AXES): the only re-judged axes in
# june/scores_*.json. Everything else there is johannes' original single
# primary-metric score (gpt-5.4-mini, current geometry).
PROD_AXES = {
    "narcissism": ["narcissism_score"],
    "honest-humble": ["honesty_humility_score"],
    "agreeableness": ["trust_score", "altruism_score", "cooperation_score",
                      "modesty_score", "sympathy_score"],
}

# Drive/Colab corpus override: rows.jsonl lives on Drive, not in git. Set
# XELICIT_EVAL_RESULTS to the Drive eval_results path in the Colab wrapper.
EVAL_RESULTS = Path(
    os.environ.get(
        "XELICIT_EVAL_RESULTS",
        str(REPO / "johannes" / "cross-elicit" / "eval_results"),
    )
)
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
    """Metrics whose mean defines the cell score — reconciled to production
    (rejudge_matrix_sft).

    - narcissism / honest-humble / agreeableness: exactly PROD_AXES (single,
      single, 5-facet-mean respectively). Candidate aggregation = mean of
      available facet scores → identical rule, so the only difference vs the
      june/scores reference is the prompt geometry.
    - any other eval: its single canonical metric (johannes' original primary).
      If the canonical YAML defines >1 metric and the eval is not a PROD_AXES
      special case, return [] — that cell's reference aggregation is
      unreconciled (e.g. ethical-framework 3-metric primary-key), so it is
      excluded rather than confounded.
    """
    if eval_name in PROD_AXES:
        return list(PROD_AXES[eval_name])
    tpls = load_canonical_templates()
    metrics = sorted(m for (ev, m) in tpls if ev == eval_name)
    return metrics if len(metrics) == 1 else []


def load_rows(dirname: str) -> dict[tuple[str, int, int], dict] | None:
    """{(item_id,p,s): {question, answer}} from <dir>/rows.jsonl, or None if the
    eval_results dir is not present locally. Fallback path only — the primary
    response source is the local scored CSVs (see load_eval_responses)."""
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


# data/<eval>_scored.csv: flattened (question, response, ...) per response —
# the raw text IS local (~1GB). Responses are identical across judge-prompt
# versions, so this older-scored corpus is a valid source of (q,a) text; the
# reference SCORE still comes from june/scores_*.json (new prompts).
SCORED_DIR = Path(
    os.environ.get("XELICIT_SCORED_DIR",
                   str(REPO / "june" / "cross_elicit_audit" / "data"))
)
_resp_cache: dict[str, dict[tuple[str, str, int, int], dict] | None] = {}


def load_eval_responses(
    eval_name: str, base_model: str
) -> dict[tuple[str, str, int, int], dict] | None:
    """{(pole, item_id, p_idx, s_idx): {question, answer}} for one eval's
    scored.csv, filtered to base_model. Cached. None if the csv is absent."""
    if eval_name in _resp_cache:
        return _resp_cache[eval_name]
    csvp = SCORED_DIR / f"{eval_name}_scored.csv"
    if not csvp.exists():
        _resp_cache[eval_name] = None
        return None
    import csv as _csv
    out: dict[tuple[str, str, int, int], dict] = {}
    with csvp.open() as fh:
        for r in _csv.DictReader(fh):
            if r.get("base_model") != base_model:
                continue
            try:
                k = (r["pole"], r["item_id"],
                     int(r["paraphrase_idx"]), int(r["sample_idx"]))
            except (KeyError, ValueError):
                continue
            if k not in out and r.get("response"):
                out[k] = {"question": r["question"], "answer": r["response"]}
    _resp_cache[eval_name] = out
    return out


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
    # Call params identical to rejudge_matrix_sft.judge: temp 0→0.3, the
    # production max_tokens=64, 4 attempts with backoff. Only the prompt
    # geometry differs from the reference.
    async with sem:
        for attempt in range(4):
            try:
                r = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0 if attempt == 0 else 0.3,
                    max_tokens=64,
                )
                if getattr(r, "usage", None) is not None:
                    ledger.add(model, r.usage)
                s = parse_score(r.choices[0].message.content)
                if s == s:  # not NaN
                    return s
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    print(f"  judge err ({model}): {e}", file=sys.stderr)
                else:
                    await asyncio.sleep(2 * (2 ** attempt))
        return float("nan")


async def run(scores_path: Path, out_csv: Path, limit: int | None,
              cells_filter: set[str] | None, dry_run: bool = False,
              max_per_cell: int | None = None,
              geometry: str = "restructured") -> None:
    from openai import AsyncOpenAI

    data = json.loads(scores_path.read_text())
    tpls = load_canonical_templates()
    # Candidate template per (eval,metric):
    #   restructured -> the Option 2 reorder (reviewed-gated)
    #   current      -> the UNMODIFIED canonical prompt = same-geometry
    #                   control; candidate vs reference then differs ONLY by
    #                   judge resampling noise, giving the baseline bin-flip.
    if geometry == "current":
        restructured = dict(tpls)
        print("GEOMETRY=current (same-prompt control: measures baseline "
              "resampling-noise bin-flip, no restructure)")
    else:
        rev = reviewed_ok()
        restructured = {k: restructure(t)[0]
                        for k, t in tpls.items() if rev.get(k)}

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
            base_model = data.get("base_model", "")
            # Primary: local scored CSV (responses are version-invariant).
            resp = load_eval_responses(ev, base_model)
            rows = None
            if resp is None:  # fallback: eval_results rows.jsonl
                rows = load_rows((cell.get("meta") or {}).get("dirname", ""))
            if resp is None and rows is None:
                skipped_dirs.append(ev)
                continue
            ref_scores = cell.get("scores") or {}
            cids = list(ref_scores)
            if max_per_cell and len(cids) > max_per_cell:
                # Deterministic representative subsample per (pole,eval) cell —
                # avoids the order bias of a global --limit.
                rng = random.Random(hash((pole, ev)) & 0xFFFFFFFF)
                cids = sorted(rng.sample(cids, max_per_cell))
            for cid in cids:
                ref = ref_scores[cid]
                item_id, _, rest = cid.partition("__p")
                p_idx, _, s_part = rest.partition("__s")
                pi, si = int(p_idx or 0), int(s_part or 0)
                if resp is not None:
                    qa = resp.get((pole, item_id, pi, si))
                else:
                    qa = rows.get((item_id, pi, si))
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
    n_calls = sum(len(j[4]) for j in jobs)
    print(f"jobs: {len(jobs)} cid×cell ({n_calls} judge calls) | "
          f"skipped(no responses): {len(skipped_dirs)} evals | "
          f"unreviewed-skipped: {len(n_unreviewed)}")
    if not jobs:
        print("Nothing to do — no (response, reviewed-template) pairs resolved.")
        ledger.dump()
        return

    # Rough projected cost (pre-cache list price; cache makes it lower).
    est = 0.0
    for j in jobs:
        pin, _ = PRICING.get(j[3], (0.75, 3.0))
        est += len(j[4]) * 1800 / 1e6 * pin   # ~1.8k input tok/call heuristic
    print(f"projected cost (list price, pre-cache): ~${est:.2f}")

    if dry_run:
        print("DRY RUN — no API calls, nothing billed. "
              "Re-run without --dry-run to execute the paired re-judge.")
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
    ap.add_argument("--dry-run", action="store_true",
                    help="build jobs + project cost, NO API calls / no spend")
    ap.add_argument("--max-per-cell", type=int, default=None,
                    help="deterministic representative #cids per (pole,eval) "
                         "cell (use instead of --limit for a powered run)")
    ap.add_argument("--geometry", choices=("restructured", "current"),
                    default="restructured",
                    help="restructured = Option 2 reorder; current = "
                         "same-prompt control (baseline resampling-noise flip)")
    a = ap.parse_args()
    cells = set(a.evals.split(",")) if a.evals else None
    asyncio.run(run(a.scores, a.out, a.limit, cells, a.dry_run,
                    a.max_per_cell, a.geometry))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
