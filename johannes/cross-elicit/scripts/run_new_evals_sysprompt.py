"""
run_new_evals_sysprompt.py — orchestrate the "new" coherence-based evals for
BASE MODELS under SYSTEM PROMPTS across every axis under `evals/`.

System prompts are discovered from <evals>/<axis>/system_prompts/*.txt.
By default each sys prompt is run against ALL eval axes (cross-elicitation).

Item-count control (diagonal = sp_axis == eval_axis, off-diagonal = the rest):
  --n-diag N      max test items for diagonal cells   (default: 100)
  --n-offdiag N   max test items for off-diagonal cells (default: 100; 0 = skip)
  --full-test-split  use the full test split for both (overrides both)

For each (model, sys_prompt_file, eval_axis) triple:
  1. Skip if a directory under `new_eval_results/sys_prompts/` already records
     an eval with matching eval_axis, base_model, system_prompt content,
     n_items, samples_per_paraphrase, trait judge, coherence judge/threshold,
     and coherence prescreen flag.
  2. Otherwise invoke `run_eval.py --checkpoint <model>
        --eval <axis>_eval.yaml
        --system-prompt-file <sp_file>
        --output-dir new_eval_results/sys_prompts
        [--max-test-items N]`
  3. After a successful subprocess, call eval_sync.push_or_mark_pending.

Loop order: model-major; within each model all (eval_axis, sys_prompt) pairs
are dispatched in parallel with --workers.

Usage:
  python run_new_evals_sysprompt.py
  python run_new_evals_sysprompt.py --dry-run
  python run_new_evals_sysprompt.py --n-diag 10 --n-offdiag 0   # diagonal-only, 10 items
  python run_new_evals_sysprompt.py --workers 5
  python run_new_evals_sysprompt.py --filter effort        # sys prompts whose path contains "effort"
  python run_new_evals_sysprompt.py --include-axis agreeableness
  python run_new_evals_sysprompt.py --exclude-axis agreeableness
  python run_new_evals_sysprompt.py --full-test-split
  python run_new_evals_sysprompt.py --no-skip
  python run_new_evals_sysprompt.py --push-all
  python run_new_evals_sysprompt.py --model Qwen/Qwen3-8B-Base --model meta-llama/Llama-3.1-8B
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
CROSS_ELICIT_ROOT = SCRIPT_DIR.parent
EVALS_ROOT = CROSS_ELICIT_ROOT / "evals"
NEW_EVAL_RESULTS = CROSS_ELICIT_ROOT / "new_eval_results"
SYS_PROMPTS_DIR = NEW_EVAL_RESULTS / "sys_prompts"
RUN_EVAL = SCRIPT_DIR / "run_eval.py"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import eval_sync  # noqa: E402

# Edit this list to change which base models are evaluated by default.
DEFAULT_MODELS: list[str] = [
    "Qwen/Qwen3-8B-Base",
]

DEFAULT_EXCLUDE_AXES: set[str] = set()
DEFAULT_N_DIAG = 100
DEFAULT_N_OFFDIAG = 100
DEFAULT_WORKERS = 5

EXPECTED_SAMPLES_PER_PARAPHRASE = 1
EXPECTED_JUDGE_MODEL = "gpt-5.4-mini"
EXPECTED_COHERENCE_JUDGE_MODEL = "gpt-4o-mini"
EXPECTED_COHERENCE_THRESHOLD = 50.0
EXPECTED_COHERENCE_PRESCREEN = True


def model_slug(model: str) -> str:
    return model.replace("/", "-")


def discover_axes() -> list[str]:
    axes: list[str] = []
    for child in sorted(EVALS_ROOT.iterdir()):
        if not child.is_dir():
            continue
        if (child / f"{child.name}_eval.yaml").is_file():
            axes.append(child.name)
    return axes


def discover_sys_prompts() -> list[dict]:
    """Return all sys prompt files across all axis dirs.

    Each entry: {sp_axis, sp_name, sp_file (abs Path), sp_content}.
    Axes with no system_prompts/*.txt are silently skipped.
    """
    entries: list[dict] = []
    for axis_dir in sorted(EVALS_ROOT.iterdir()):
        if not axis_dir.is_dir():
            continue
        sp_dir = axis_dir / "system_prompts"
        if not sp_dir.is_dir():
            continue
        for txt in sorted(sp_dir.glob("*.txt")):
            entries.append({
                "sp_axis": axis_dir.name,
                "sp_name": txt.stem,
                "sp_file": txt,
                "sp_content": txt.read_text(),
            })
    return entries


def n_test_items_in_yaml(yaml_path: Path) -> int:
    with open(yaml_path) as f:
        items = yaml.safe_load(f) or []
    return sum(
        1 for it in items
        if isinstance(it.get("meta"), dict)
        and (it["meta"].get("split") or "").strip() == "test"
    )


def _summary_matches(
    summary: dict,
    expected_base_model: str,
    expected_sp_content: str,
    expected_n_items: int,
) -> bool:
    ck = summary.get("checkpoint") or {}
    if ck.get("base_model") != expected_base_model:
        return False
    if ck.get("model_path") is not None:
        return False  # must be a base (no LoRA) run
    if summary.get("n_test_items") != expected_n_items:
        return False
    if summary.get("samples_per_paraphrase") != EXPECTED_SAMPLES_PER_PARAPHRASE:
        return False
    if summary.get("judge_model") != EXPECTED_JUDGE_MODEL:
        return False
    if summary.get("coherence_prescreen") != EXPECTED_COHERENCE_PRESCREEN:
        return False
    if summary.get("coherence_judge_model") != EXPECTED_COHERENCE_JUDGE_MODEL:
        return False
    if summary.get("coherence_threshold") != EXPECTED_COHERENCE_THRESHOLD:
        return False
    if summary.get("system_prompt") != expected_sp_content:
        return False
    return True


def find_matching_existing_dir(
    job_output_dir: Path,
    eval_axis: str,
    model: str,
    sp_content: str,
    n_items: int,
) -> Path | None:
    if not job_output_dir.is_dir():
        return None
    prefix = f"{eval_axis}_eval__{model_slug(model)}__base__"
    for child in job_output_dir.iterdir():
        if not child.is_dir() or not child.name.startswith(prefix):
            continue
        summary_path = child / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if _summary_matches(
            summary,
            expected_base_model=model,
            expected_sp_content=sp_content,
            expected_n_items=n_items,
        ):
            return child
    return None


def build_plan(
    models: list[str],
    sys_prompts: list[dict],
    axes: list[str],
    skip: bool,
    exclude_axes: set[str],
    n_diag: int | None,
    n_offdiag: int | None,
    filter_str: str | None,
) -> tuple[list[dict], list[dict]]:
    """Return (jobs_to_run, jobs_skipped).

    n_diag / n_offdiag: max test items for diagonal / off-diagonal cells.
    None means use the full test split. 0 means skip those cells entirely.
    """
    if filter_str:
        f = filter_str.lower()
        sys_prompts = [sp for sp in sys_prompts if f in str(sp["sp_file"]).lower()]

    to_run: list[dict] = []
    skipped: list[dict] = []

    for model in models:
        for sp in sys_prompts:
            sp_axis = sp["sp_axis"]

            for eval_axis in axes:
                is_diag = eval_axis == sp_axis
                cell_limit = n_diag if is_diag else n_offdiag
                yaml_path = EVALS_ROOT / eval_axis / f"{eval_axis}_eval.yaml"

                if cell_limit == 0:
                    skipped.append({
                        "model": model, **sp, "eval_axis": eval_axis,
                        "eval_yaml": str(yaml_path), "n_test_items": None,
                        "max_test_items": cell_limit,
                        "skip_reason": "n_offdiag=0 (off-diagonal skipped)" if not is_diag
                                       else "n_diag=0",
                    })
                    continue

                if eval_axis in exclude_axes:
                    skipped.append({
                        "model": model, **sp, "eval_axis": eval_axis,
                        "eval_yaml": str(yaml_path), "n_test_items": None,
                        "max_test_items": cell_limit,
                        "skip_reason": "eval axis excluded",
                    })
                    continue

                if not yaml_path.is_file():
                    skipped.append({
                        "model": model, **sp, "eval_axis": eval_axis,
                        "eval_yaml": str(yaml_path), "n_test_items": None,
                        "max_test_items": cell_limit,
                        "skip_reason": "eval yaml not found",
                    })
                    continue

                try:
                    n_in_yaml = n_test_items_in_yaml(yaml_path)
                except Exception as e:  # noqa: BLE001
                    skipped.append({
                        "model": model, **sp, "eval_axis": eval_axis,
                        "eval_yaml": str(yaml_path), "n_test_items": None,
                        "max_test_items": cell_limit,
                        "skip_reason": f"yaml read failed: {e!r}",
                    })
                    continue

                if n_in_yaml == 0:
                    skipped.append({
                        "model": model, **sp, "eval_axis": eval_axis,
                        "eval_yaml": str(yaml_path), "n_test_items": 0,
                        "max_test_items": cell_limit,
                        "skip_reason": "no test-split items in yaml",
                    })
                    continue

                expected_n = n_in_yaml if cell_limit is None else min(cell_limit, n_in_yaml)
                job_output_dir = SYS_PROMPTS_DIR / f"{sp['sp_axis']}__{sp['sp_name']}"
                job = {
                    "model": model,
                    **sp,
                    "eval_axis": eval_axis,
                    "eval_yaml": str(yaml_path),
                    "n_test_items": expected_n,
                    "max_test_items": cell_limit,
                    "job_output_dir": job_output_dir,
                }

                existing = (
                    find_matching_existing_dir(
                        job_output_dir=job_output_dir,
                        eval_axis=eval_axis,
                        model=model,
                        sp_content=sp["sp_content"],
                        n_items=expected_n,
                    )
                    if skip else None
                )
                if existing is not None:
                    job["skip_reason"] = f"existing match: {existing.name}"
                    skipped.append(job)
                else:
                    to_run.append(job)

    return to_run, skipped


OUT_DIR_RE = re.compile(r"^Output dir: (.+)$", re.MULTILINE)


def run_one_job(job: dict, push: bool, push_all: bool = False, defer_push: bool = False) -> dict:
    job_output_dir: Path = job["job_output_dir"]
    job_output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(RUN_EVAL),
        "--checkpoint", job["model"],
        "--eval", job["eval_yaml"],
        "--system-prompt-file", str(job["sp_file"]),
        "--output-dir", str(job_output_dir),
    ]
    if job["max_test_items"] is not None:
        cmd += ["--max-test-items", str(job["max_test_items"])]

    short_sp = f"{job['sp_axis']}/{job['sp_name']}"
    t0 = datetime.now()
    print(
        f"  [start  {t0:%H:%M:%S}] {job['eval_axis']:34s} × {short_sp}",
        flush=True,
    )
    env = {**os.environ, "JUDGE_CONCURRENCY": "32", "JUDGE_CONCURRENCY_MAX": "64"}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    t1 = datetime.now()
    dur = (t1 - t0).total_seconds()

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    m = OUT_DIR_RE.search(stdout)
    out_dir = Path(m.group(1).strip()) if m else None

    result: dict = {
        **job,
        "returncode": proc.returncode,
        "out_dir": str(out_dir) if out_dir else None,
        "duration_s": dur,
    }

    if proc.returncode != 0:
        print(
            f"  [FAIL  {t1:%H:%M:%S}] {job['eval_axis']:34s} × {short_sp} "
            f"(rc={proc.returncode}, {dur:.0f}s)\n"
            f"    stderr tail: {stderr[-500:]!r}",
            flush=True,
        )
        result["error"] = stderr[-2000:]
        return result

    print(
        f"  [done   {t1:%H:%M:%S}] {job['eval_axis']:34s} × {short_sp} "
        f"({dur:.0f}s) → {out_dir.name if out_dir else '(no out_dir parsed)'}",
        flush=True,
    )

    if push and not defer_push and out_dir is not None and out_dir.is_dir():
        try:
            eval_sync.push_or_mark_pending(out_dir, summary_only=not push_all)
        except Exception as e:  # noqa: BLE001
            print(f"  [HF-push failed] {out_dir.name}: {e!r}", flush=True)
            result["push_error"] = repr(e)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", action="append", dest="models", default=None, metavar="HF_ID",
        help=(
            "Base model to evaluate (repeatable). "
            f"Default: {DEFAULT_MODELS}."
        ),
    )
    parser.add_argument(
        "--n-diag", type=int, default=DEFAULT_N_DIAG,
        help=(
            f"Max test items for diagonal cells (sp_axis == eval_axis). "
            f"Default: {DEFAULT_N_DIAG}. 0 = skip all diagonal."
        ),
    )
    parser.add_argument(
        "--n-offdiag", type=int, default=DEFAULT_N_OFFDIAG,
        help=(
            f"Max test items for off-diagonal cells (sp_axis != eval_axis). "
            f"Default: {DEFAULT_N_OFFDIAG}. 0 = skip all off-diagonal (diagonal-only run)."
        ),
    )
    parser.add_argument("--full-test-split", action="store_true",
                        help="Use the full test split for both diag and off-diag (overrides both).")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Concurrent run_eval.py subprocesses (default: {DEFAULT_WORKERS}).")
    parser.add_argument("--no-skip", dest="skip", action="store_false",
                        help="Re-run every job, even if a matching dir exists.")
    parser.add_argument("--no-push", dest="push", action="store_false",
                        help="Don't push finished dirs to HF.")
    parser.add_argument("--push-all", action="store_true",
                        help="Push all eval files (default: summary.json only).")
    parser.add_argument("--defer-push", action="store_true",
                        help=(
                            "Skip per-job HF pushes and instead do a single bulk "
                            "upload at the end (one commit for all successful dirs). "
                            "Avoids hitting the 128 commits/hour HF rate limit."
                        ))
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan and exit (no subprocesses).")
    parser.add_argument("--exclude-axis", action="append", default=None,
                        help=(
                            "Eval axis to exclude (repeatable). "
                            f"Default: {sorted(DEFAULT_EXCLUDE_AXES)}."
                        ))
    parser.add_argument("--include-axis", action="append", default=None,
                        help="Re-include an axis from DEFAULT_EXCLUDE_AXES.")
    parser.add_argument("--filter", default=None,
                        help=(
                            "Case-insensitive substring match on the sys prompt "
                            "file path (e.g. --filter effort runs only sys prompts "
                            "from the effort axis)."
                        ))
    parser.add_argument("--verbose", action="store_true",
                        help="Print every planned and skipped job.")
    parser.set_defaults(skip=True, push=True, push_all=False)
    args = parser.parse_args()

    models = args.models if args.models is not None else DEFAULT_MODELS
    if not models:
        sys.exit("No models specified. Pass --model or edit DEFAULT_MODELS.")

    exclude_axes: set[str] = (
        set(args.exclude_axis) if args.exclude_axis is not None
        else set(DEFAULT_EXCLUDE_AXES)
    )
    for inc in args.include_axis or []:
        exclude_axes.discard(inc)

    n_diag: int | None = None if args.full_test_split else args.n_diag
    n_offdiag: int | None = None if args.full_test_split else args.n_offdiag

    SYS_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

    axes = discover_axes()
    sys_prompts = discover_sys_prompts()

    if not sys_prompts:
        print("No system prompt .txt files found under evals/*/system_prompts/. Nothing to do.")
        return 0

    eligible_axes = [a for a in axes if a not in exclude_axes]
    push_desc = "none" if not args.push else ("all" if args.push_all else "summary-only")
    print(f"Models: {models}")
    print(f"Discovered {len(sys_prompts)} sys prompt file(s) across evals/ axes.")
    print(f"Eval axes: {len(eligible_axes)} eligible  excludes={sorted(exclude_axes)}")
    print(
        f"n_diag={'<full>' if n_diag is None else n_diag}  "
        f"n_offdiag={'<full>' if n_offdiag is None else n_offdiag}  "
        f"workers={args.workers}  skip={args.skip}  push={push_desc}"
    )

    to_run, skipped = build_plan(
        models=models,
        sys_prompts=sys_prompts,
        axes=axes,
        skip=args.skip,
        exclude_axes=exclude_axes,
        n_diag=n_diag,
        n_offdiag=n_offdiag,
        filter_str=args.filter,
    )

    print(f"\nPlan: {len(to_run)} job(s) to run, {len(skipped)} skip(s).")

    if args.verbose:
        for j in skipped:
            diag_marker = "*" if j["eval_axis"] == j["sp_axis"] else " "
            print(
                f"  SKIP {diag_marker} {j['eval_axis']:34s} × {j['sp_axis']}/{j['sp_name']}"
                f"  n={j.get('n_test_items')}  reason={j['skip_reason']}"
            )
        for j in to_run:
            diag_marker = "*" if j["eval_axis"] == j["sp_axis"] else " "
            print(
                f"  RUN  {diag_marker} {j['eval_axis']:34s} × {j['sp_axis']}/{j['sp_name']}"
                f"  n={j['n_test_items']}"
            )

    if args.dry_run or not to_run:
        if not to_run:
            print("Nothing to run.")
        return 0

    # Group by model for cleaner progress reporting.
    from collections import defaultdict
    jobs_by_model: dict[str, list[dict]] = defaultdict(list)
    for job in to_run:
        jobs_by_model[job["model"]].append(job)

    grand_results: list[dict] = []
    grand_failures: list[dict] = []

    for model, model_jobs in jobs_by_model.items():
        print(
            f"\n=== {model}: launching {len(model_jobs)} job(s) "
            f"with {args.workers} worker(s) ==="
        )
        results: list[dict] = []
        failures: list[dict] = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(run_one_job, job, args.push, args.push_all, args.defer_push): job
                for job in model_jobs
            }
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                if res.get("returncode") != 0:
                    failures.append(res)
        grand_results.extend(results)
        grand_failures.extend(failures)
        print(f"  → model done: {len(results) - len(failures)} ok, {len(failures)} failed.")

    print(
        f"\n=== ALL DONE. {len(grand_results) - len(grand_failures)} ok, "
        f"{len(grand_failures)} failed. ==="
    )
    for f in grand_failures:
        print(
            f"  FAIL: {f['eval_axis']} × {f['sp_axis']}/{f['sp_name']} "
            f"(rc={f.get('returncode')})"
        )

    if args.push and args.defer_push:
        successful_dirs = [
            Path(r["out_dir"])
            for r in grand_results
            if r.get("returncode") == 0 and r.get("out_dir")
        ]
        print(f"\n=== Bulk-pushing {len(successful_dirs)} dir(s) in one commit... ===")
        try:
            eval_sync.push_dirs_bulk(successful_dirs, summary_only=not args.push_all)
        except Exception as e:  # noqa: BLE001
            print(f"  [bulk push failed] {e!r}")
            return 1

    return 1 if grand_failures else 0


if __name__ == "__main__":
    sys.exit(main())
