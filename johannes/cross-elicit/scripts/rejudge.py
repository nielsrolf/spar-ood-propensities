"""
rejudge.py — Re-issue judge calls that errored (bucket == "fail") without
              re-sampling the model.

Scans new_eval_results/finetuning/ for eval dirs where at least one metric has
n_fails > 0, re-fires the judge API for every row whose bucket == "fail", then
rewrites rows.jsonl and summary.json in-place. Rows with bucket in {numeric,
null, incoherent} are kept exactly as-is.

Usage:
  python rejudge.py                 # dry-run: list what would be re-judged
  python rejudge.py --run           # actually re-judge
  python rejudge.py --run --workers 8
  python rejudge.py --dir <path>    # target a single dir (implies --run)
  python rejudge.py --no-push       # skip HF push after each dir
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
CROSS = SCRIPT_DIR.parent
FINETUNING_DIR = CROSS / "new_eval_results" / "finetuning"
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
    fill_judge_template,
    parse_judge_text,
    _resolve_judge_route,
)
import eval_sync  # noqa: E402


def _needs_rejudge(summary: dict) -> bool:
    """True if any metric has at least one fail row."""
    for mdata in summary.get("metrics", {}).values():
        if mdata.get("n_fails", 0) > 0:
            return True
    return False


def _load_metric_templates(eval_yaml_path: str) -> dict[str, str]:
    """Return {metric: judge_template} from the eval YAML (first item wins)."""
    try:
        with open(eval_yaml_path) as f:
            items = yaml.safe_load(f) or []
    except (OSError, yaml.YAMLError) as e:
        raise RuntimeError(f"Cannot load eval YAML {eval_yaml_path}: {e}") from e
    templates: dict[str, str] = {}
    for item in items:
        for metric, tmpl in (item.get("judge_prompts") or {}).items():
            if metric not in templates:
                templates[metric] = tmpl
    return templates


def _recompute_summary_metrics(rows: list[dict]) -> dict[str, dict]:
    by_metric: dict[str, list[dict]] = {}
    for r in rows:
        by_metric.setdefault(r["metric"], []).append(r)
    out: dict[str, dict] = {}
    for metric, mrs in by_metric.items():
        numeric = [r["score"] for r in mrs if r["bucket"] == "numeric"]
        out[metric] = {
            "n_total": len(mrs),
            "n_numeric": len(numeric),
            "n_nulls": sum(1 for r in mrs if r["bucket"] == "null"),
            "n_fails": sum(1 for r in mrs if r["bucket"] == "fail"),
            "n_incoherent": sum(1 for r in mrs if r["bucket"] == "incoherent"),
            "min": min(numeric) if numeric else None,
            "max": max(numeric) if numeric else None,
            "mean": statistics.fmean(numeric) if numeric else None,
            "std": statistics.stdev(numeric) if len(numeric) >= 2 else None,
        }
    return out


async def _rejudge_dir_async(
    out_dir: Path,
    summary: dict,
    templates: dict[str, str],
    judge_model_override: str | None = None,
) -> tuple[int, int]:
    """Re-judge all fail rows in out_dir. Returns (n_rejudged, n_still_fail)."""
    rows_path = out_dir / "rows.jsonl"
    rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]

    fail_indices = [i for i, r in enumerate(rows) if r["bucket"] == "fail"]
    if not fail_indices:
        return 0, 0

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
        metric = row["metric"]
        template = templates.get(metric)
        if template is None:
            return  # shouldn't happen; leave as fail
        judge_model = judge_model_override or row.get("judge_model") or JUDGE_MODEL
        filled = fill_judge_template(template, row["question"], row["answer"])
        judge_text = await call_judge(oai_client, openrouter_client, filled, limiter,
                                      model=judge_model)
        bucket, score = parse_judge_text(judge_text)
        rows[idx] = {**row, "judge_response": judge_text, "bucket": bucket, "score": score}

    tasks = [asyncio.create_task(rejudge_one(i)) for i in fail_indices]
    await asyncio.gather(*tasks)

    rows_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    summary["metrics"] = _recompute_summary_metrics(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    n_still_fail = sum(1 for r in rows if r["bucket"] == "fail")
    return len(fail_indices), n_still_fail


def rejudge_dir(out_dir: Path, push: bool, judge_model_override: str | None = None) -> dict:
    summary_path = out_dir / "summary.json"
    rows_path = out_dir / "rows.jsonl"
    if not summary_path.exists() or not rows_path.exists():
        return {"dir": str(out_dir), "status": "skip", "reason": "missing files"}

    summary = json.loads(summary_path.read_text())
    if not _needs_rejudge(summary):
        return {"dir": str(out_dir), "status": "skip", "reason": "no fail rows"}

    eval_yaml = summary.get("eval_yaml", "")
    if not eval_yaml or not Path(eval_yaml).exists():
        return {"dir": str(out_dir), "status": "error",
                "reason": f"eval_yaml not found: {eval_yaml}"}

    try:
        templates = _load_metric_templates(eval_yaml)
    except RuntimeError as e:
        return {"dir": str(out_dir), "status": "error", "reason": str(e)}

    try:
        n_rejudged, n_still_fail = asyncio.run(
            _rejudge_dir_async(out_dir, summary, templates, judge_model_override)
        )
    except Exception as e:  # noqa: BLE001
        return {"dir": str(out_dir), "status": "error", "reason": repr(e)}

    if push and n_rejudged > 0:
        try:
            eval_sync.push_or_mark_pending(out_dir, summary_only=True)
        except Exception as e:  # noqa: BLE001
            pass  # non-fatal; push_pending file will be created by eval_sync

    status = "ok" if n_still_fail == 0 else "partial"
    return {
        "dir": str(out_dir),
        "status": status,
        "n_rejudged": n_rejudged,
        "n_still_fail": n_still_fail,
    }


def find_fail_dirs(root: Path) -> list[Path]:
    """Find all dirs under root (1 or 2 levels deep) that have fail rows."""
    out = []
    if not root.exists():
        return out
    candidates: list[Path] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if (d / "summary.json").exists():
            candidates.append(d)
        else:
            # One extra level (e.g. sys_prompts/{sp_axis}__{sp_name}/{eval_dir}/)
            for dd in sorted(d.iterdir()):
                if dd.is_dir() and (dd / "summary.json").exists():
                    candidates.append(dd)
    for d in candidates:
        try:
            summary = json.loads((d / "summary.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if _needs_rejudge(summary):
            out.append(d)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="store_true",
                   help="Actually re-judge (default is dry-run).")
    p.add_argument("--dir", default=None,
                   help="Target a single output dir (implies --run).")
    p.add_argument("--root", default=None,
                   help="Root dir to scan instead of the default finetuning dir.")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel dirs to process (default 4).")
    p.add_argument("--no-push", dest="push", action="store_false",
                   help="Skip HF push after each dir.")
    p.add_argument("--judge-model", default=None,
                   help="Override judge model (e.g. 'openai/gpt-5.4-mini' to route via OpenRouter).")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process first N dirs (for testing).")
    p.set_defaults(push=True)
    args = p.parse_args()

    if args.dir:
        dirs = [Path(args.dir)]
        args.run = True
    else:
        root = Path(args.root) if args.root else FINETUNING_DIR
        dirs = find_fail_dirs(root)

    if args.limit is not None:
        dirs = dirs[: args.limit]

    print(f"Dirs with at least one fail row: {len(dirs)}")

    if not args.run:
        print("(dry-run — pass --run to actually re-judge)\n")
        for d in dirs[:40]:
            s = json.loads((d / "summary.json").read_text())
            total_fails = sum(m.get("n_fails", 0) for m in s.get("metrics", {}).values())
            print(f"  {total_fails:4d} fails  {d.name[:90]}")
        if len(dirs) > 40:
            print(f"  ... and {len(dirs) - 40} more")
        return 0

    results: list[dict] = []
    errors: list[dict] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(rejudge_dir, d, args.push, args.judge_model): d for d in dirs}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            if res["status"] == "error":
                errors.append(res)
            tag = {"ok": "✓", "partial": "~", "skip": "–", "error": "✗"}.get(
                res["status"], "?"
            )
            name = Path(res["dir"]).name
            detail = (
                f"rejudged={res.get('n_rejudged', 0)}  "
                f"still_fail={res.get('n_still_fail', 0)}"
                if res["status"] in ("ok", "partial") else res.get("reason", "")
            )
            print(f"[{i:4d}/{len(dirs)}] {tag}  {name[:80]}  {detail}", flush=True)

    ok = sum(1 for r in results if r["status"] == "ok")
    partial = sum(1 for r in results if r["status"] == "partial")
    skipped = sum(1 for r in results if r["status"] == "skip")
    print(f"\nDone. ok={ok}  partial={partial}  skipped={skipped}  errors={len(errors)}")
    if errors:
        print("Errors:")
        for e in errors:
            print(f"  {e['dir']}: {e['reason']}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
