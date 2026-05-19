"""
rejudge_axis.py — Re-judge every existing eval row for a single axis under a
new judge prompt, WITHOUT re-sampling the underlying model generations.

Use case: the resource-acquisition judge prompt was rescaled (-100..+100 →
0..100). The model outputs in new_eval_results/ are still valid, but every
existing score was produced under the old rubric and must be discarded.

Covers TWO storage layouts:

  (1) Per-eval-dir runs under
      new_eval_results/{base_models,finetuning,sys_prompts,...}/{axis}_eval__*/
        - Files: summary.json + rows.jsonl
        - Rescore rows where metric is in target YAML's judge_prompts and
          bucket ∈ {numeric, null, fail}. Skip bucket == "incoherent" — the
          coherence prescreen is rubric-independent.
        - Recompute summary.json metrics, stamp provenance, push via eval_sync.

  (2) Cross-elicitation matrices under
      new_eval_results/scores/eval-orthogonality/{axis}__<propB>/
        - Files: judgments.jsonl + matrices.json
        - Every row has prop_a == axis; rescore ALL of them (no prescreen).
        - Recompute matrices.json (scores + counts + ASCII tables) using the
          aggregators in orthogonality_of_evals.py.

Usage:
  python rejudge_axis.py                                # dry-run, axis=resource-acquisition
  python rejudge_axis.py --run                          # actually rejudge
  python rejudge_axis.py --run --workers 8
  python rejudge_axis.py --run --dir <single-dir>      # works for either layout
  python rejudge_axis.py --eval evals/resource-acquisition/resource-acquisition_eval_v3.yaml --run
  python rejudge_axis.py --axis some-other-axis --run
  python rejudge_axis.py --run --no-push                # skip HF push
  python rejudge_axis.py --skip-eval-ortho --run        # only the per-eval-dir runs
  python rejudge_axis.py --only-eval-ortho --run        # only the matrices
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
CROSS = SCRIPT_DIR.parent
NEW_EVAL_RESULTS = CROSS / "new_eval_results"
EVALS_ROOT = CROSS / "evals"
JOHANNES_ROOT = CROSS.parent

load_dotenv(JOHANNES_ROOT / ".env")

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_eval import (  # noqa: E402
    AdaptiveLimiter,
    AIMD_SUCCESS_STEP,
    JUDGE_CONCURRENCY_MAX,
    JUDGE_CONCURRENCY_MIN,
    JUDGE_CONCURRENCY_START,
    JUDGE_MODEL,
    call_judge,
    call_with_aimd,
    fill_judge_template,
    parse_judge_text,
)
from rejudge import _load_metric_templates, _recompute_summary_metrics  # noqa: E402
import eval_sync  # noqa: E402

# orthogonality_of_evals parses CLI args at import time. Hijack sys.argv so we
# can import its aggregator helpers without triggering its argparse.
_saved_argv = sys.argv
sys.argv = ["orthogonality_of_evals.py", "1", "1"]
try:
    from orthogonality_of_evals import (  # noqa: E402
        judge_prompt_sha as _orth_sha,
        compute_scores as _orth_compute_scores,
        compute_counts as _orth_compute_counts,
        _ascii_matrix as _orth_ascii_matrix,
        _ascii_counts_matrix as _orth_ascii_counts,
        call_judge as _orth_call_judge,
        parse_score_with_llm as _orth_parse,
        JUDGE_MODEL as ORTH_JUDGE_MODEL,
        PARSER_MODEL as ORTH_PARSER_MODEL,
    )
finally:
    sys.argv = _saved_argv


RESCORE_BUCKETS = {"numeric", "null", "fail"}
MIN_CONVERSATIONS = 100  # skip dirs with fewer rows than this for the target metric
EVAL_ORTHOG_ROOT = NEW_EVAL_RESULTS / "scores" / "eval-orthogonality"

# Per-dir cap on concurrent judge calls in the eval-ortho path. The imported
# `_orth_call_judge` has no built-in rate limiter, so we cap concurrency here
# to avoid APIConnectionError storms when many dirs run in parallel.
ORTH_PER_DIR_CONCURRENCY = 20
ORTH_RETRIES = 4         # additional retries on transient errors
ORTH_BACKOFF_BASE = 1.5  # seconds


def _count_metric_rows(rows_path: Path, metric_keys: set[str]) -> int:
    """Count rows in rows.jsonl whose `metric` is in `metric_keys`."""
    n = 0
    try:
        for line in rows_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("metric") in metric_keys:
                n += 1
    except OSError:
        return 0
    return n


def discover_dirs(
    root: Path, axis: str, metric_keys: set[str], min_rows: int,
) -> tuple[list[Path], int]:
    """All dirs under `root` whose name starts with `{axis}_eval__` and contain
    summary.json + rows.jsonl AND have >= `min_rows` rows for the target
    metric(s). Walks the full tree (handles 1- and 2-level layouts uniformly).

    Returns (kept_dirs, n_filtered_out)."""
    prefix = f"{axis}_eval__"
    kept: list[Path] = []
    n_too_small = 0
    if not root.exists():
        return kept, 0
    for summary in root.rglob("summary.json"):
        d = summary.parent
        if not d.name.startswith(prefix):
            continue
        rows_path = d / "rows.jsonl"
        if not rows_path.exists():
            continue
        if _count_metric_rows(rows_path, metric_keys) < min_rows:
            n_too_small += 1
            continue
        kept.append(d)
    return sorted(kept), n_too_small


async def _rejudge_dir_async(
    out_dir: Path,
    templates: dict[str, str],
    judge_model_override: str | None,
    only_failed: bool = False,
) -> tuple[int, int, int]:
    """Returns (n_targeted, n_now_fail, n_skipped_incoherent). If only_failed
    is True, only rows with bucket == 'fail' are targeted (used to retry
    transient failures cheaply without re-billing already-scored rows)."""
    rows_path = out_dir / "rows.jsonl"
    rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]

    target_buckets = {"fail"} if only_failed else RESCORE_BUCKETS
    target_indices: list[int] = []
    n_incoherent_skipped = 0
    for i, r in enumerate(rows):
        if r.get("metric") not in templates:
            continue
        bucket = r.get("bucket")
        if bucket == "incoherent":
            n_incoherent_skipped += 1
            continue
        if bucket in target_buckets:
            target_indices.append(i)

    if not target_indices:
        return 0, 0, n_incoherent_skipped

    _openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    oai_client = AsyncOpenAI()
    openrouter_client = AsyncOpenAI(
        api_key=_openrouter_key,
        base_url="https://openrouter.ai/api/v1",
    ) if _openrouter_key else None
    limiter = AdaptiveLimiter(
        "judge", JUDGE_CONCURRENCY_START, JUDGE_CONCURRENCY_MIN,
        JUDGE_CONCURRENCY_MAX, AIMD_SUCCESS_STEP,
    )

    async def rejudge_one(idx: int) -> None:
        row = rows[idx]
        template = templates[row["metric"]]
        judge_model = judge_model_override or row.get("judge_model") or JUDGE_MODEL
        filled = fill_judge_template(template, row["question"], row["answer"])
        # call_judge swallows exceptions and returns "__JUDGE_ERROR__: ..." on
        # failure. Detect that and retry with backoff at this layer too.
        judge_text = ""
        for attempt in range(5):
            judge_text = await call_judge(
                oai_client, openrouter_client, filled, limiter, model=judge_model,
            )
            if not judge_text.startswith("__JUDGE_ERROR__"):
                break
            if attempt < 4:
                await asyncio.sleep(2.0 * (2 ** attempt))  # 2, 4, 8, 16, 32 s
        bucket, score = parse_judge_text(judge_text)
        rows[idx] = {
            **row,
            "judge_model": judge_model,
            "judge_response": judge_text,
            "bucket": bucket,
            "score": score,
        }

    tasks = [asyncio.create_task(rejudge_one(i)) for i in target_indices]
    await asyncio.gather(*tasks)

    rows_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    n_now_fail = sum(
        1 for i in target_indices if rows[i].get("bucket") == "fail"
    )
    return len(target_indices), n_now_fail, n_incoherent_skipped


def rejudge_dir(
    out_dir: Path,
    eval_yaml: Path,
    templates: dict[str, str],
    push: bool,
    judge_model_override: str | None,
    only_failed: bool = False,
) -> dict:
    summary_path = out_dir / "summary.json"
    rows_path = out_dir / "rows.jsonl"
    if not summary_path.exists() or not rows_path.exists():
        return {"dir": str(out_dir), "status": "skip", "reason": "missing files"}

    summary = json.loads(summary_path.read_text())

    try:
        n_targeted, n_now_fail, n_inc_skip = asyncio.run(
            _rejudge_dir_async(out_dir, templates, judge_model_override, only_failed)
        )
    except Exception as e:  # noqa: BLE001
        return {"dir": str(out_dir), "status": "error", "reason": repr(e)}

    if n_targeted == 0:
        return {
            "dir": str(out_dir), "status": "skip",
            "reason": f"no rescorable rows (incoherent_skipped={n_inc_skip})",
        }

    rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
    summary["metrics"] = _recompute_summary_metrics(rows)
    prev_yaml = summary.get("eval_yaml")
    if prev_yaml and prev_yaml != str(eval_yaml):
        summary["previous_eval_yaml"] = prev_yaml
    summary["eval_yaml"] = str(eval_yaml)
    summary["rejudged_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary["rejudged_metrics"] = sorted(templates.keys())
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    if push:
        try:
            eval_sync.push_or_mark_pending(out_dir, summary_only=True)
        except Exception:  # noqa: BLE001
            pass  # non-fatal; eval_sync will mark pending

    return {
        "dir": str(out_dir),
        "status": "ok" if n_now_fail == 0 else "partial",
        "n_rejudged": n_targeted,
        "n_now_fail": n_now_fail,
        "n_incoherent_skipped": n_inc_skip,
    }


# ============================================================
# eval-orthogonality dirs (different layout: judgments.jsonl + matrices.json)
# ============================================================

def discover_eval_ortho_dirs(axis: str) -> list[Path]:
    """All dirs `{EVAL_ORTHOG_ROOT}/{axis}__*` with judgments.jsonl + matrices.json."""
    if not EVAL_ORTHOG_ROOT.exists():
        return []
    prefix = f"{axis}__"
    out: list[Path] = []
    for d in EVAL_ORTHOG_ROOT.iterdir():
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        if (d / "judgments.jsonl").exists() and (d / "matrices.json").exists():
            out.append(d)
    return sorted(out)


async def _rejudge_ortho_dir_async(
    out_dir: Path,
    templates: dict[str, str],
    judge_model_override: str | None,
    only_failed: bool = False,
) -> tuple[int, int]:
    """Rejudge rows in judgments.jsonl using the new templates. Returns
    (n_rejudged, n_failed). If only_failed is True, skip rows whose
    score_status is already 'score' or 'null' AND whose judge_prompt_sha
    matches the new template (used for retrying transient failures cheaply).
    """
    judgments_path = out_dir / "judgments.jsonl"
    rows = [json.loads(l) for l in judgments_path.read_text().splitlines() if l.strip()]
    new_shas = {jk: _orth_sha(tmpl) for jk, tmpl in templates.items()}

    targets: list[int] = []
    for i, r in enumerate(rows):
        jk = r.get("judge_key")
        if jk not in templates:
            continue
        if only_failed:
            if r.get("score_status") in ("score", "null") and r.get("judge_prompt_sha") == new_shas.get(jk):
                continue
        targets.append(i)
    if not targets:
        return 0, 0

    oai_client = AsyncOpenAI()
    judge_model = judge_model_override or ORTH_JUDGE_MODEL
    sem = asyncio.Semaphore(ORTH_PER_DIR_CONCURRENCY)

    async def _call_with_retry(coro_fn):
        """Run coro_fn (returns a coroutine when called) with exponential
        backoff on any exception, up to ORTH_RETRIES additional attempts."""
        last_exc: Exception | None = None
        for attempt in range(ORTH_RETRIES + 1):
            try:
                return await coro_fn()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt == ORTH_RETRIES:
                    break
                await asyncio.sleep(ORTH_BACKOFF_BASE * (2 ** attempt))
        raise last_exc  # type: ignore[misc]

    async def rejudge_one(idx: int) -> None:
        row = rows[idx]
        template = templates[row["judge_key"]]
        filled = (
            template
            .replace("{question}", row["question"])
            .replace("{answer}", row["answer"])
        )
        async with sem:
            try:
                raw = await _call_with_retry(
                    lambda: _orth_call_judge(filled, openai_client=oai_client)
                )
            except Exception as exc:  # noqa: BLE001
                rows[idx] = {
                    **row,
                    "judge_model": judge_model,
                    "judge_prompt_sha": new_shas[row["judge_key"]],
                    "raw_response": f"<<JUDGE CALL FAILED: {exc!r}>>",
                    "score": None,
                    "score_status": "failed",
                }
                return
            try:
                status, score = await _call_with_retry(
                    lambda: _orth_parse(raw, parser_openai_client=oai_client)
                )
            except Exception as exc:  # noqa: BLE001
                status, score = "failed", None
        rows[idx] = {
            **row,
            "judge_model": judge_model,
            "judge_prompt_sha": new_shas[row["judge_key"]],
            "raw_response": raw,
            "score": score,
            "score_status": status,
        }

    try:
        tasks = [asyncio.create_task(rejudge_one(i)) for i in targets]
        await asyncio.gather(*tasks)
    finally:
        try:
            await oai_client.close()
        except Exception:  # noqa: BLE001
            pass

    judgments_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    n_failed = sum(1 for i in targets if rows[i].get("score_status") == "failed")
    return len(targets), n_failed


def _rebuild_matrices_json(
    out_dir: Path,
    templates: dict[str, str],
    judge_model_override: str | None,
) -> None:
    """Recompute matrices.json (scores + counts + ASCII tables) from the
    (rejudged) judgments.jsonl. Preserves _meta from the existing file and
    stamps rejudge provenance."""
    matrices_path = out_dir / "matrices.json"
    judgments_path = out_dir / "judgments.jsonl"
    records = [json.loads(l) for l in judgments_path.read_text().splitlines() if l.strip()]
    existing = json.loads(matrices_path.read_text())
    meta = dict(existing.get("_meta", {}))

    # row_labels_map: {prop_a: {judge_key: row_label}} — invert row_meta.
    row_meta = meta.get("row_meta", {})
    row_labels_map: dict[str, dict[str, str]] = {}
    for row_label, rm in row_meta.items():
        prop = rm.get("propensity")
        jk = rm.get("judge_key")
        if prop and jk:
            row_labels_map.setdefault(prop, {})[jk] = row_label

    row_labels = meta.get("row_labels", [])
    col_labels = meta.get("col_labels", [])

    scores_matrix = _orth_compute_scores(records, row_labels_map)
    counts_matrix = _orth_compute_counts(records, row_labels_map)

    # Provenance: stash previous judge_prompt_sha(s) before overwriting model.
    prev_judges = {jk: _orth_sha(templates[jk]) for jk in templates}
    if "judge_models" in meta:
        # Update judge model for the rescored keys.
        new_judge_models = dict(meta["judge_models"])
        judge_model = judge_model_override or ORTH_JUDGE_MODEL
        for row_label, rm in row_meta.items():
            if rm.get("judge_key") in templates:
                new_judge_models[row_label] = judge_model
        meta["judge_models"] = new_judge_models
    meta["rejudged_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta["rejudged_judge_keys"] = sorted(templates.keys())
    meta["new_judge_prompt_sha"] = prev_judges

    output = {
        "_meta": meta,
        "scores": {
            "description": (
                "Mean numeric score per (judge_prompt × response_set) for this "
                "propensity pair. Nulls and failures excluded from the mean."
            ),
            "values": scores_matrix,
            "table": _orth_ascii_matrix(scores_matrix, row_labels, col_labels).split("\n"),
        },
        "counts": {
            "description": (
                "Per-cell judgment counts: scored = numeric score returned, "
                "null = judge declared off-axis, failed = judge/parser couldn't "
                "produce a score."
            ),
            "values": counts_matrix,
            "table": _orth_ascii_counts(counts_matrix, row_labels, col_labels).split("\n"),
        },
    }
    matrices_path.write_text(json.dumps(output, indent=2) + "\n")


def rejudge_ortho_dir(
    out_dir: Path,
    templates: dict[str, str],
    judge_model_override: str | None,
    only_failed: bool = False,
) -> dict:
    judgments_path = out_dir / "judgments.jsonl"
    matrices_path = out_dir / "matrices.json"
    if not judgments_path.exists() or not matrices_path.exists():
        return {"dir": str(out_dir), "status": "skip", "reason": "missing files"}

    try:
        n_rejudged, n_failed = asyncio.run(
            _rejudge_ortho_dir_async(out_dir, templates, judge_model_override, only_failed)
        )
    except Exception as e:  # noqa: BLE001
        return {"dir": str(out_dir), "status": "error", "reason": repr(e)}

    if n_rejudged == 0:
        return {
            "dir": str(out_dir), "status": "skip",
            "reason": "no rows matching templates' judge_keys",
        }

    try:
        _rebuild_matrices_json(out_dir, templates, judge_model_override)
    except Exception as e:  # noqa: BLE001
        return {
            "dir": str(out_dir), "status": "error",
            "reason": f"matrices rebuild failed: {e!r}",
        }

    return {
        "dir": str(out_dir),
        "status": "ok" if n_failed == 0 else "partial",
        "n_rejudged": n_rejudged,
        "n_now_fail": n_failed,
        "n_incoherent_skipped": 0,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--axis", default="resource-acquisition",
                   help="Axis whose runs to rejudge (default: resource-acquisition).")
    p.add_argument("--eval", dest="eval_yaml", default=None,
                   help="Path to eval YAML to use for the new judge prompt. "
                        "Default: evals/<axis>/<axis>_eval.yaml")
    p.add_argument("--root", default=None,
                   help=f"Root to scan (default: {NEW_EVAL_RESULTS}).")
    p.add_argument("--dir", default=None,
                   help="Target a single output dir (implies --run).")
    p.add_argument("--run", action="store_true",
                   help="Actually rejudge (default is dry-run).")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel dirs to process (default 4).")
    p.add_argument("--no-push", dest="push", action="store_false",
                   help="Skip HF push after each dir.")
    p.add_argument("--judge-model", default=None,
                   help="Override judge model (e.g. 'openai/gpt-5.4-mini').")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process first N dirs (for testing).")
    p.add_argument("--filter", dest="filter", default=None,
                   help="Substring match on dir name; only matching dirs are processed. "
                        "Useful for splitting by model, e.g. --filter Qwen.")
    p.add_argument("--skip-eval-ortho", dest="include_ortho",
                   action="store_false",
                   help="Skip eval-orthogonality dirs (scores/eval-orthogonality/...).")
    p.add_argument("--only-eval-ortho", action="store_true",
                   help="Process ONLY eval-orthogonality dirs.")
    p.add_argument("--retry-failed-only", action="store_true",
                   help="Skip rows already scored under the current judge prompt; "
                        "only retry rows currently in the 'fail' bucket (per-eval) "
                        "or with status=failed / outdated judge_prompt_sha (ortho).")
    p.set_defaults(push=True, include_ortho=True)
    args = p.parse_args()

    eval_yaml = (
        Path(args.eval_yaml) if args.eval_yaml
        else EVALS_ROOT / args.axis / f"{args.axis}_eval.yaml"
    )
    if not eval_yaml.exists():
        print(f"ERROR: eval YAML not found: {eval_yaml}", file=sys.stderr)
        return 2

    templates = _load_metric_templates(str(eval_yaml))
    if not templates:
        print(f"ERROR: no judge_prompts in {eval_yaml}", file=sys.stderr)
        return 2

    metric_keys = set(templates.keys())
    n_too_small = 0
    do_normal = not args.only_eval_ortho
    do_ortho = args.include_ortho or args.only_eval_ortho

    normal_dirs: list[Path] = []
    ortho_dirs: list[Path] = []

    if args.dir:
        d = Path(args.dir)
        args.run = True
        # Dispatch by file presence — same flag handles either layout.
        if (d / "judgments.jsonl").exists() and (d / "matrices.json").exists():
            ortho_dirs = [d]
        elif (d / "summary.json").exists() and (d / "rows.jsonl").exists():
            normal_dirs = [d]
        else:
            print(f"ERROR: --dir {d} matches neither expected layout.", file=sys.stderr)
            return 2
    else:
        root = Path(args.root) if args.root else NEW_EVAL_RESULTS
        if do_normal:
            normal_dirs, n_too_small = discover_dirs(
                root, args.axis, metric_keys, MIN_CONVERSATIONS,
            )
        if do_ortho:
            ortho_dirs = discover_eval_ortho_dirs(args.axis)

    if args.filter:
        normal_dirs = [d for d in normal_dirs if args.filter in d.name]
        ortho_dirs = [d for d in ortho_dirs if args.filter in d.name]

    if args.limit is not None:
        normal_dirs = normal_dirs[: args.limit]
        ortho_dirs = ortho_dirs[: args.limit]

    print(f"Axis:                  {args.axis}")
    print(f"Eval YAML:             {eval_yaml}")
    print(f"Metrics / judge_keys:  {sorted(metric_keys)}")
    print(f"Min rows (per-eval):   {MIN_CONVERSATIONS}  (skipped {n_too_small} below threshold)")
    print(f"Per-eval dirs:         {len(normal_dirs)}")
    print(f"Eval-ortho dirs:       {len(ortho_dirs)}")
    print(f"TOTAL:                 {len(normal_dirs) + len(ortho_dirs)}")

    if not args.run:
        print("\n(dry-run — pass --run to actually rejudge)")
        if normal_dirs:
            print(f"\n--- per-eval dirs ({len(normal_dirs)}) ---")
            for d in normal_dirs[:40]:
                try:
                    rows = [
                        json.loads(l) for l in
                        (d / "rows.jsonl").read_text().splitlines() if l.strip()
                    ]
                    n_target = sum(
                        1 for r in rows
                        if r.get("metric") in templates
                        and r.get("bucket") in RESCORE_BUCKETS
                    )
                    n_inc = sum(
                        1 for r in rows
                        if r.get("metric") in templates
                        and r.get("bucket") == "incoherent"
                    )
                    print(f"  rescore={n_target:4d}  skip_incoherent={n_inc:4d}  {d.name[:100]}")
                except (OSError, json.JSONDecodeError) as e:
                    print(f"  [unreadable] {d}: {e}")
            if len(normal_dirs) > 40:
                print(f"  ... and {len(normal_dirs) - 40} more")
        if ortho_dirs:
            print(f"\n--- eval-orthogonality dirs ({len(ortho_dirs)}) ---")
            for d in ortho_dirs[:40]:
                try:
                    n = sum(
                        1 for line in (d / "judgments.jsonl").read_text().splitlines()
                        if line.strip()
                    )
                    print(f"  rescore={n:4d}  {d.name}")
                except OSError as e:
                    print(f"  [unreadable] {d}: {e}")
            if len(ortho_dirs) > 40:
                print(f"  ... and {len(ortho_dirs) - 40} more")
        return 0

    results: list[dict] = []
    errors: list[dict] = []
    t0 = time.time()
    total = len(normal_dirs) + len(ortho_dirs)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for d in normal_dirs:
            futures[pool.submit(
                rejudge_dir, d, eval_yaml, templates, args.push, args.judge_model,
                args.retry_failed_only,
            )] = ("normal", d)
        for d in ortho_dirs:
            futures[pool.submit(
                rejudge_ortho_dir, d, templates, args.judge_model, args.retry_failed_only,
            )] = ("ortho", d)

        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            kind, _ = futures[fut]
            results.append(res)
            if res["status"] == "error":
                errors.append(res)
            tag = {"ok": "OK", "partial": "~~", "skip": "--", "error": "XX"}.get(
                res["status"], "??",
            )
            name = Path(res["dir"]).name
            if res["status"] in ("ok", "partial"):
                detail = (
                    f"rejudged={res.get('n_rejudged', 0)}  "
                    f"now_fail={res.get('n_now_fail', 0)}  "
                    f"inc_skipped={res.get('n_incoherent_skipped', 0)}"
                )
            else:
                detail = res.get("reason", "")
            print(f"[{i:4d}/{total}] {tag} [{kind:6s}] {name[:80]}  {detail}", flush=True)

    elapsed = time.time() - t0
    ok = sum(1 for r in results if r["status"] == "ok")
    partial = sum(1 for r in results if r["status"] == "partial")
    skipped = sum(1 for r in results if r["status"] == "skip")
    print(
        f"\nDone in {elapsed:.0f}s. "
        f"ok={ok}  partial={partial}  skipped={skipped}  errors={len(errors)}"
    )
    if errors:
        print("Errors:")
        for e in errors:
            print(f"  {e['dir']}: {e['reason']}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
