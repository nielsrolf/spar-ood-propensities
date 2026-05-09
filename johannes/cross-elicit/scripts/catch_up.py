"""
catch_up.py — bring existing eval results up to date with the v2 eval files
for the four axes whose evals changed:

  agreeableness         (judge unchanged, sh added 126 convos at end)
  reward-hacking        (judge unchanged, sh added 4 convos at end)
  resource-acquisition  (same convos, judge prompt rescaled)
  neuroticism           (judge prompt rewritten + reorder + 43 sh-extras)

For every existing eval-results dir (under eval_results/finetuning/,
eval_results/sys_prompts/, eval_results/test_evals/) named
`<axis>_eval__<ckpt-label>__...`, we produce a sibling
`<axis>_eval_v2__<ckpt-label>__...` containing the v2 results — REUSING
existing model responses where the same paraphrase exists in v2, and
RE-JUDGING with the v2 judge prompt only when it actually differs.

Reuse rules (per (axis, paraphrase)):
  matched, judge unchanged   → copy entire row verbatim (no API calls)
  matched, judge changed     → keep model answer, re-judge with v2 prompt
  unmatched (new in v2)      → only sampled+judged when v1 was a *full*
                              run (n_test_items == full v1 test count);
                              partial/quick runs (e.g. finetune.py's
                              max_test_items=10 spot-checks) stay the
                              same size — just rejudged if needed.

Existing v2 sibling dirs are skipped. The script is safe to interrupt and
re-run (idempotent on completed dirs).

USAGE
  python catch_up.py                       # default: 10 concurrent workers
  python catch_up.py --workers 4           # tune concurrency
  python catch_up.py --axes neuroticism resource-acquisition
  python catch_up.py --subdirs finetuning sys_prompts
  python catch_up.py --dry-run
  python catch_up.py --limit 20            # only process N dirs (debug)
  python catch_up.py --no-sample           # skip new-convo sampling globally
"""

import argparse
import asyncio
import glob
import json
import os
import statistics
import sys
import time
from datetime import datetime

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CROSS_ELICIT_ROOT = os.path.dirname(SCRIPT_DIR)
EVALS_ROOT = os.path.join(CROSS_ELICIT_ROOT, "evals")
EVAL_RESULTS_DIR = os.path.join(CROSS_ELICIT_ROOT, "eval_results")

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
# Lazy imports: only pull tinker etc. in if we actually need to sample.
# Judge-only re-runs can avoid the tinker dependency entirely.

# ─────────────────────────────────────────────────────────────────────────────
# Per-axis configuration
# ─────────────────────────────────────────────────────────────────────────────

# v1 stem (no extension) → v2 stem. Used for filename prefix matching of
# eval-results dirs and for locating yaml files alongside.
AXES = ["agreeableness", "reward-hacking", "resource-acquisition", "neuroticism"]

def v1_yaml(axis: str) -> str:
    return os.path.join(EVALS_ROOT, axis, f"{axis}_eval.yaml")

def v2_yaml(axis: str) -> str:
    return os.path.join(EVALS_ROOT, axis, f"{axis}_eval_v2.yaml")

def v1_stem(axis: str) -> str:
    return f"{axis}_eval"

def v2_stem(axis: str) -> str:
    return f"{axis}_eval_v2"


SUBDIRS = ["finetuning", "sys_prompts", "test_evals"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_yaml(path: str):
    with open(path) as f:
        return yaml.safe_load(f)

def _read_jsonl(path: str) -> list[dict]:
    out: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out

def _v1_test_full_count(axis: str) -> int:
    items = _load_yaml(v1_yaml(axis))
    return sum(
        1 for it in items
        if isinstance(it.get("meta"), dict)
        and (it["meta"].get("split", "").strip() == "test")
    )

def _v2_test_items(axis: str) -> list[dict]:
    items = _load_yaml(v2_yaml(axis))
    return [
        it for it in items
        if isinstance(it.get("meta"), dict)
        and (it["meta"].get("split", "").strip() == "test")
    ]

def _judge_prompts(yaml_path: str) -> dict:
    items = _load_yaml(yaml_path)
    # All items share the same dict via YAML anchor; first item is canonical.
    return items[0].get("judge_prompts") or {}


# Discover (subdir, dir_name) pairs that need catch-up: match v1 stem prefix,
# skip if a v2 sibling already exists in the same subdir.
def discover_v1_dirs(axes: list[str], subdirs: list[str]) -> list[tuple[str, str, str]]:
    """Return (axis, subdir, dir_name) triples to catch up."""
    found: list[tuple[str, str, str]] = []
    for sub in subdirs:
        sub_path = os.path.join(EVAL_RESULTS_DIR, sub)
        if not os.path.isdir(sub_path):
            continue
        existing = set(os.listdir(sub_path))
        for axis in axes:
            v1p = v1_stem(axis) + "__"
            v2p = v2_stem(axis) + "__"
            for name in sorted(existing):
                if not name.startswith(v1p):
                    continue
                # Compute the candidate v2 sibling name.
                tail = name[len(v1p):]
                v2_name = v2p + tail
                if v2_name in existing:
                    continue  # already caught up
                # Need rows.jsonl + summary.json present.
                if not os.path.isfile(os.path.join(sub_path, name, "rows.jsonl")):
                    continue
                if not os.path.isfile(os.path.join(sub_path, name, "summary.json")):
                    continue
                found.append((axis, sub, name))
    return found


# ─────────────────────────────────────────────────────────────────────────────
# Sampling/judge pipeline (lazy-loaded)
# ─────────────────────────────────────────────────────────────────────────────

class _Pipeline:
    """Shared judge client + multi-sampler cache. Built once, used by all
    workers concurrently. Sampler creation is gated by an asyncio.Lock so
    two workers asking for the same (base_model, model_path) only build it
    once."""

    def __init__(self):
        self.judge_client = None
        self.judge_limiter = None
        self.sample_limiter = None
        # (base_model, model_path or None) -> (sampler, renderer)
        self.samplers: dict[tuple[str, str | None], tuple[object, object]] = {}
        # Lazy import handles.
        self._call_judge = None
        self._fill_judge_template = None
        self._parse_judge_text = None
        self._sample_answers = None
        self._select_renderer_name = None
        self._renderers = None
        self._get_tokenizer = None
        self._tinker = None
        # Locks: judge init, per-instance sampler-cache, plus a global
        # tinker import lock (the import touches sys.path and the
        # tinker_cookbook globals).
        self._judge_lock = asyncio.Lock()
        self._sampler_lock = asyncio.Lock()

    async def _ensure_judge(self):
        if self.judge_client is not None:
            return
        async with self._judge_lock:
            if self.judge_client is not None:
                return
            from openai import AsyncOpenAI  # noqa: WPS433
            from run_eval import (  # noqa: WPS433
                AdaptiveLimiter, AIMD_SUCCESS_STEP,
                JUDGE_CONCURRENCY_START, JUDGE_CONCURRENCY_MIN, JUDGE_CONCURRENCY_MAX,
                call_judge, fill_judge_template, parse_judge_text,
                JUDGE_MODEL,
            )
            self.judge_client = AsyncOpenAI()
            self.judge_limiter = AdaptiveLimiter(
                "judge", JUDGE_CONCURRENCY_START, JUDGE_CONCURRENCY_MIN,
                JUDGE_CONCURRENCY_MAX, AIMD_SUCCESS_STEP,
            )
            self._call_judge = call_judge
            self._fill_judge_template = fill_judge_template
            self._parse_judge_text = parse_judge_text
            self.JUDGE_MODEL = JUDGE_MODEL

    async def _ensure_sampler(self, base_model: str, model_path: str | None):
        key = (base_model, model_path)
        cached = self.samplers.get(key)
        if cached is not None:
            return cached
        async with self._sampler_lock:
            cached = self.samplers.get(key)
            if cached is not None:
                return cached
            from run_eval import (  # noqa: WPS433
                AdaptiveLimiter, AIMD_SUCCESS_STEP,
                SAMPLE_CONCURRENCY_START, SAMPLE_CONCURRENCY_MIN, SAMPLE_CONCURRENCY_MAX,
                sample_answers, select_renderer_name,
            )
            import tinker  # noqa: WPS433
            from tinker_cookbook import renderers  # noqa: WPS433
            from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: WPS433

            sc = tinker.ServiceClient()
            kwargs = {"base_model": base_model}
            if model_path:
                kwargs["model_path"] = model_path
            sampler = sc.create_sampling_client(**kwargs)
            tok = get_tokenizer(base_model)
            renderer = renderers.get_renderer(select_renderer_name(base_model), tok)
            if self.sample_limiter is None:
                self.sample_limiter = AdaptiveLimiter(
                    "sample", SAMPLE_CONCURRENCY_START, SAMPLE_CONCURRENCY_MIN,
                    SAMPLE_CONCURRENCY_MAX, AIMD_SUCCESS_STEP,
                )
            self._sample_answers = sample_answers
            self.samplers[key] = (sampler, renderer)
            return sampler, renderer

    async def judge_one(self, template: str, question: str, answer: str) -> tuple[str, str, int | None]:
        await self._ensure_judge()
        prompt = self._fill_judge_template(template, question, answer)
        text = await self._call_judge(self.judge_client, prompt, self.judge_limiter)
        bucket, score = self._parse_judge_text(text)
        return text, bucket, score

    async def sample_one(self, base_model: str, model_path: str | None,
                         prompt: str, n: int, system_prompt: str | None) -> list[str]:
        sampler, renderer = await self._ensure_sampler(base_model, model_path)
        return await self._sample_answers(
            sampler, renderer, prompt, n, self.sample_limiter,
            system_prompt=system_prompt,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Catch-up for one v1 dir
# ─────────────────────────────────────────────────────────────────────────────

async def catch_up_dir(
    axis: str,
    sub: str,
    dir_name: str,
    pipeline: _Pipeline,
    judge_changed: bool,
    full_v1_test_count: int,
    v2_items: list[dict],
    v2_judge_prompts: dict,
    args,
) -> str | None:
    """Process one v1 dir; return v2 dir path or None if skipped."""
    src_dir = os.path.join(EVAL_RESULTS_DIR, sub, dir_name)
    rows_v1 = _read_jsonl(os.path.join(src_dir, "rows.jsonl"))
    with open(os.path.join(src_dir, "summary.json")) as f:
        summary_v1 = json.load(f)

    # Build paraphrase -> list of v1 row dicts (one per (paraphrase_idx, sample_idx, metric))
    by_para: dict[str, list[dict]] = {}
    for r in rows_v1:
        by_para.setdefault(r.get("question", ""), []).append(r)

    # Model answers per paraphrase (any metric will do — answer is the same):
    answers_per_para: dict[str, list[str]] = {}
    for para, rs in by_para.items():
        # Group by sample_idx so we don't double-count metrics.
        seen_sids: dict[int, str] = {}
        for r in rs:
            sid = r.get("sample_idx", 0)
            if sid not in seen_sids:
                seen_sids[sid] = r.get("answer", "")
        # Sort by sample_idx for stable output.
        answers_per_para[para] = [seen_sids[k] for k in sorted(seen_sids)]

    # Decide which v2 items get sampled (for new paraphrases) vs only judged.
    n_test_v1 = summary_v1.get("n_test_items", len(by_para))
    is_full_v1 = n_test_v1 >= full_v1_test_count
    do_sample_new = is_full_v1 and not args.no_sample

    new_rows: list[dict] = []
    samples_to_take: list[tuple[str, int, dict]] = []  # (paraphrase, p_idx, item)
    rejudge_jobs: list[tuple[dict, str, str]] = []     # (skeleton row, judge_template, key)

    # v2 has a single metric per axis (taken from the first entry's judge_prompts);
    # all v2 items share the same judge prompts via YAML anchor.
    v2_metric_names = list(v2_judge_prompts.keys())

    for it in v2_items:
        item_id = it["id"]
        item_jp = it.get("judge_prompts") or v2_judge_prompts
        for p_idx, paraphrase in enumerate(it.get("paraphrases") or []):
            existing_answers = answers_per_para.get(paraphrase)
            if existing_answers is not None:
                # Have model answers from v1; reuse.
                for s_idx, ans in enumerate(existing_answers):
                    for metric in v2_metric_names:
                        if not judge_changed:
                            # Find the matching v1 row (same paraphrase, sample_idx, metric).
                            match = None
                            for r in by_para.get(paraphrase, []):
                                if r.get("sample_idx") == s_idx and r.get("metric") == metric:
                                    match = r
                                    break
                            if match is not None:
                                new_rows.append({
                                    "item_id": item_id,
                                    "paraphrase_idx": p_idx,
                                    "sample_idx": s_idx,
                                    "metric": metric,
                                    "question": paraphrase,
                                    "answer": ans,
                                    "judge_response": match.get("judge_response", ""),
                                    "bucket": match.get("bucket", "fail"),
                                    "score": match.get("score"),
                                })
                                continue
                            # Same metric but no matching judge in v1 → fall through to re-judge.
                        # Need to (re-)judge.
                        skeleton = {
                            "item_id": item_id,
                            "paraphrase_idx": p_idx,
                            "sample_idx": s_idx,
                            "metric": metric,
                            "question": paraphrase,
                            "answer": ans,
                        }
                        rejudge_jobs.append((skeleton, item_jp[metric], paraphrase))
            else:
                # Paraphrase not in v1 → optionally sample + judge.
                if do_sample_new:
                    samples_to_take.append((paraphrase, p_idx, it))

    print(
        f"  • {sub}/{dir_name[:60]}…  "
        f"reused={len(new_rows)}  rejudge={len(rejudge_jobs)}  "
        f"sample={len(samples_to_take)}  (is_full_v1={is_full_v1}, judge_changed={judge_changed})"
    )

    if args.dry_run:
        return None

    # --- Sampling phase ---
    if samples_to_take:
        ckpt = summary_v1.get("checkpoint") or {}
        base_model = ckpt.get("base_model")
        model_path = ckpt.get("model_path")
        sys_prompt = summary_v1.get("system_prompt")
        if not base_model:
            print(f"    ✗ no base_model in summary; skipping sampling for {dir_name}")
            samples_to_take = []
        else:
            async def _sample(paraphrase):
                return await pipeline.sample_one(base_model, model_path, paraphrase, 1, sys_prompt)
            tasks = [asyncio.create_task(_sample(p)) for (p, _pi, _it) in samples_to_take]
            t0 = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            answers_in_order = []
            for r in results:
                if isinstance(r, Exception):
                    print(f"    sample task error: {r!r}")
                    answers_in_order.append([])
                else:
                    answers_in_order.append(r)
            elapsed = time.time() - t0
            print(f"    sampled {len(samples_to_take)} new paraphrases in {elapsed:.0f}s")
            # Build judge jobs for sampled answers.
            for (paraphrase, p_idx, it), answers in zip(samples_to_take, answers_in_order):
                item_jp = it.get("judge_prompts") or v2_judge_prompts
                for s_idx, ans in enumerate(answers):
                    for metric in v2_metric_names:
                        skeleton = {
                            "item_id": it["id"],
                            "paraphrase_idx": p_idx,
                            "sample_idx": s_idx,
                            "metric": metric,
                            "question": paraphrase,
                            "answer": ans,
                        }
                        rejudge_jobs.append((skeleton, item_jp[metric], paraphrase))

    # --- Judge phase ---
    if rejudge_jobs:
        async def _judge(skeleton, template, key):
            try:
                text, bucket, score = await pipeline.judge_one(template, skeleton["question"], skeleton["answer"])
            except Exception as e:
                text, bucket, score = (f"__JUDGE_ERROR__: {e!r}", "fail", None)
            row = dict(skeleton)
            row["judge_response"] = text
            row["bucket"] = bucket
            row["score"] = score
            return row
        tasks = [asyncio.create_task(_judge(sk, tpl, k)) for (sk, tpl, k) in rejudge_jobs]
        t0 = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                continue
            new_rows.append(r)
        elapsed = time.time() - t0
        print(f"    judged {len(rejudge_jobs)} rows in {elapsed:.0f}s")

    # --- Write v2 dir ---
    out_name = v2_stem(axis) + dir_name[len(v1_stem(axis)):]
    out_dir = os.path.join(EVAL_RESULTS_DIR, sub, out_name)
    os.makedirs(out_dir, exist_ok=True)
    rows_path = os.path.join(out_dir, "rows.jsonl")
    summary_path = os.path.join(out_dir, "summary.json")

    # Sort rows for stable output.
    new_rows.sort(key=lambda r: (r.get("item_id", ""), r.get("paraphrase_idx", 0), r.get("sample_idx", 0), r.get("metric", "")))
    with open(rows_path, "w") as f:
        for r in new_rows:
            f.write(json.dumps(r) + "\n")

    # Build v2 summary mirroring run_eval.py's shape.
    by_metric: dict[str, list[dict]] = {}
    for r in new_rows:
        by_metric.setdefault(r["metric"], []).append(r)
    metrics_out: dict[str, dict] = {}
    for m, rs in by_metric.items():
        nums = [r["score"] for r in rs if r["bucket"] == "numeric"]
        n_null = sum(1 for r in rs if r["bucket"] == "null")
        n_fail = sum(1 for r in rs if r["bucket"] == "fail")
        metrics_out[m] = {
            "n_total": len(rs),
            "n_numeric": len(nums),
            "n_nulls": n_null,
            "n_fails": n_fail,
            "min": min(nums) if nums else None,
            "max": max(nums) if nums else None,
            "mean": statistics.fmean(nums) if nums else None,
            "std": statistics.stdev(nums) if len(nums) >= 2 else None,
        }
    summary_v2 = {
        **{k: v for k, v in summary_v1.items() if k not in ("metrics", "n_test_items", "eval_yaml")},
        "eval_yaml": v2_yaml(axis),
        "n_test_items": len({(r["item_id"]) for r in new_rows}),
        "metrics": metrics_out,
        "catch_up": {
            "from_v1_dir": dir_name,
            "judge_changed": judge_changed,
            "is_full_v1": is_full_v1,
            "n_reused_rows": len(new_rows) - len(rejudge_jobs),
            "n_rejudged_or_new": len(rejudge_jobs),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary_v2, f, indent=2)

    return out_dir


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main_async(args):
    pipeline = _Pipeline()
    triples = discover_v1_dirs(args.axes, args.subdirs)
    if args.limit:
        triples = triples[: args.limit]
    print(f"Found {len(triples)} v1 dirs to catch up.")

    # Pre-compute per-axis context.
    axis_ctx: dict[str, dict] = {}
    for axis in set(t[0] for t in triples):
        v1_jp = _judge_prompts(v1_yaml(axis))
        v2_jp = _judge_prompts(v2_yaml(axis))
        axis_ctx[axis] = {
            "judge_changed": v1_jp != v2_jp,
            "v1_full": _v1_test_full_count(axis),
            "v2_items": _v2_test_items(axis),
            "v2_jp": v2_jp,
        }
        print(
            f"  axis {axis}: judge_changed={axis_ctx[axis]['judge_changed']}  "
            f"v1_full={axis_ctx[axis]['v1_full']}  v2_test={len(axis_ctx[axis]['v2_items'])}"
        )

    sem = asyncio.Semaphore(max(1, args.workers))
    counter = {"done": 0}
    total = len(triples)

    async def _one(idx, axis, sub, dir_name):
        async with sem:
            counter["done"] += 1
            print(f"\n[{counter['done']}/{total}] axis={axis} sub={sub}  (started; index {idx})")
            ctx = axis_ctx[axis]
            try:
                await catch_up_dir(
                    axis, sub, dir_name, pipeline,
                    judge_changed=ctx["judge_changed"],
                    full_v1_test_count=ctx["v1_full"],
                    v2_items=ctx["v2_items"],
                    v2_judge_prompts=ctx["v2_jp"],
                    args=args,
                )
            except Exception as e:
                print(f"  ✗ failed [{axis}/{sub}/{dir_name}]: {e!r}")

    tasks = [
        asyncio.create_task(_one(i, axis, sub, dir_name))
        for i, (axis, sub, dir_name) in enumerate(triples, 1)
    ]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    print("\nDone.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--axes", nargs="+", default=AXES, choices=AXES,
                        help="Axes to process (default: all 4).")
    parser.add_argument("--subdirs", nargs="+", default=SUBDIRS, choices=SUBDIRS,
                        help="Eval-result subdirs to scan.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N dirs (debug).")
    parser.add_argument("--dry-run", action="store_true", help="Plan only; no API calls or writes.")
    parser.add_argument("--no-sample", action="store_true",
                        help="Never sample new convos; only re-judge existing answers.")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of v1 dirs to process concurrently (default: 10).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main_async(args))
