"""
run_new_evals_base.py — orchestrate the "new" coherence-based evals for the
listed BASE models across every axis under `evals/`.

For each (axis, model) pair:
  1. Skip if a directory under `cross-elicit/new_eval_results/base_models/`
     already records an eval with matching n_items, samples_per_paraphrase,
     trait judge model, coherence judge model, coherence threshold and
     coherence prescreen flag.
  2. Otherwise invoke `run_eval.py --checkpoint <model>
        --eval <axis>_eval.yaml --output-dir new_eval_results/base_models`
     (no --max-test-items → full test split, by user spec for base models).
  3. After a successful subprocess, call `eval_sync.push_or_mark_pending`
     on the new output dir so HF stays in sync.

Three subprocess workers run in parallel. Their AIMD limiters compete for
shared OpenAI / Tinker quota and will adapt.

Usage:
  python run_new_evals_base.py                # run everything, with skips
  python run_new_evals_base.py --no-skip      # force re-run every job
  python run_new_evals_base.py --no-push      # skip the HF push
  python run_new_evals_base.py --dry-run      # print the planned jobs only
  python run_new_evals_base.py --workers 1    # serial execution
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
BASE_MODELS_DIR = NEW_EVAL_RESULTS / "base_models"
RUN_EVAL = SCRIPT_DIR / "run_eval.py"

# Make eval_sync importable for inline HF push.
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import eval_sync  # noqa: E402

# The two base models we're evaluating this run.
models = [
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "meta-llama/Llama-3.1-8B-Instruct",
]

# Axes excluded by default. agreeableness_eval.yaml overrides judge_models to
# `google/gemini-2.5-flash` (OpenRouter route), which the OpenAI-only judge
# client in run_eval.py can't dispatch. Re-include via --include-axis once
# routing or the yaml is fixed.
DEFAULT_EXCLUDE_AXES = {"agreeableness"}

# Defaults that run_eval.py uses today; the skip-if-exists check verifies
# every existing dir agrees on all of these before declaring it a match.
EXPECTED_SAMPLES_PER_PARAPHRASE = 1
EXPECTED_JUDGE_MODEL = "gpt-5.4-mini"
EXPECTED_COHERENCE_JUDGE_MODEL = "gpt-4o-mini"
EXPECTED_COHERENCE_THRESHOLD = 50.0
EXPECTED_COHERENCE_PRESCREEN = True


def discover_axes() -> list[str]:
    """Every axis-dir under evals/ that contains <axis>_eval.yaml."""
    axes: list[str] = []
    for child in sorted(EVALS_ROOT.iterdir()):
        if not child.is_dir():
            continue
        if (child / f"{child.name}_eval.yaml").is_file():
            axes.append(child.name)
    return axes


def n_test_items_in_yaml(yaml_path: Path) -> int:
    with open(yaml_path) as f:
        items = yaml.safe_load(f) or []
    return sum(
        1 for it in items
        if isinstance(it.get("meta"), dict)
        and (it["meta"].get("split") or "").strip() == "test"
    )


def model_slug(model: str) -> str:
    """Match run_eval.py's resolve_checkpoint: HF-style → label is
    `<slugified>__base` with slashes → dashes."""
    return model.replace("/", "-")


def _summary_matches(
    summary: dict,
    expected_base_model: str,
    expected_n_items: int,
) -> bool:
    if summary.get("checkpoint", {}).get("base_model") != expected_base_model:
        return False
    if summary.get("checkpoint", {}).get("model_path") is not None:
        return False  # base = no LoRA path
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
    # Sanity: must not be a sys-prompted run.
    if summary.get("system_prompt") is not None:
        return False
    return True


def find_matching_existing_dir(
    axis: str, model: str, n_items: int
) -> Path | None:
    """Return path of a finished base-models eval dir that already matches
    (axis, model, full-config); else None.

    'Finished' = has a parseable summary.json AND the metrics it reports
    cover at least n_items×paraphrases rows (encoded as n_test_items).
    """
    if not BASE_MODELS_DIR.is_dir():
        return None
    prefix = f"{axis}_eval__{model_slug(model)}__base__"
    for child in BASE_MODELS_DIR.iterdir():
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
        if _summary_matches(summary, expected_base_model=model, expected_n_items=n_items):
            return child
    return None


def build_plan(skip: bool, exclude_axes: set[str]) -> tuple[list[dict], list[dict]]:
    """Return (jobs_to_run, jobs_skipped). Each job is
    {axis, model, eval_yaml, n_test_items, skip_reason?}."""
    axes = discover_axes()
    to_run: list[dict] = []
    skipped: list[dict] = []
    for axis in axes:
        if axis in exclude_axes:
            for model in models:
                skipped.append({
                    "axis": axis, "model": model,
                    "eval_yaml": str(EVALS_ROOT / axis / f"{axis}_eval.yaml"),
                    "n_test_items": None,
                    "skip_reason": "axis excluded by --exclude-axis",
                })
            continue
        yaml_path = EVALS_ROOT / axis / f"{axis}_eval.yaml"
        try:
            n_items = n_test_items_in_yaml(yaml_path)
        except Exception as e:  # noqa: BLE001
            skipped.append({
                "axis": axis, "model": None, "eval_yaml": str(yaml_path),
                "n_test_items": None,
                "skip_reason": f"yaml read failed: {e!r}",
            })
            continue
        if n_items == 0:
            skipped.append({
                "axis": axis, "model": None, "eval_yaml": str(yaml_path),
                "n_test_items": 0,
                "skip_reason": "no test-split items in yaml",
            })
            continue
        for model in models:
            existing = find_matching_existing_dir(axis, model, n_items) if skip else None
            job = {
                "axis": axis,
                "model": model,
                "eval_yaml": str(yaml_path),
                "n_test_items": n_items,
            }
            if existing is not None:
                job["skip_reason"] = f"existing match: {existing.name}"
                skipped.append(job)
            else:
                to_run.append(job)
    return to_run, skipped


OUT_DIR_RE = re.compile(r"^Output dir: (.+)$", re.MULTILINE)


def run_one_job(job: dict, push: bool) -> dict:
    """Invoke run_eval.py for a single (axis, model). Returns a result dict."""
    cmd = [
        sys.executable, str(RUN_EVAL),
        "--checkpoint", job["model"],
        "--eval", job["eval_yaml"],
        "--output-dir", str(BASE_MODELS_DIR),
    ]
    t0 = datetime.now()
    print(f"  [start  {t0:%H:%M:%S}] {job['axis']} × {job['model']}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
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
        # Persist failure context next to the orchestrator log.
        print(
            f"  [FAIL  {t1:%H:%M:%S}] {job['axis']} × {job['model']} "
            f"(rc={proc.returncode}, {dur:.0f}s)\n"
            f"    stderr tail: {stderr[-500:]!r}",
            flush=True,
        )
        result["error"] = stderr[-2000:]
        return result

    print(
        f"  [done   {t1:%H:%M:%S}] {job['axis']} × {job['model']} "
        f"({dur:.0f}s) → {out_dir.name if out_dir else '(no out_dir parsed)'}",
        flush=True,
    )

    if push and out_dir is not None and out_dir.is_dir():
        try:
            eval_sync.push_or_mark_pending(out_dir)
        except Exception as e:  # noqa: BLE001
            print(f"  [HF-push failed] {out_dir.name}: {e!r}", flush=True)
            result["push_error"] = repr(e)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--workers", type=int, default=3,
                        help="Concurrent run_eval.py subprocesses (default: 3).")
    parser.add_argument("--no-skip", dest="skip", action="store_false",
                        help="Re-run every job, even if a matching dir exists.")
    parser.add_argument("--no-push", dest="push", action="store_false",
                        help="Don't push finished dirs to HF.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and exit (no subprocesses).")
    parser.add_argument("--exclude-axis", action="append", default=None,
                        help=(
                            "Axis to exclude (can be passed multiple times). "
                            f"Default: {sorted(DEFAULT_EXCLUDE_AXES)}."
                        ))
    parser.add_argument("--include-axis", action="append", default=None,
                        help="Re-include an axis from DEFAULT_EXCLUDE_AXES.")
    parser.set_defaults(skip=True, push=True)
    args = parser.parse_args()

    exclude_axes: set[str] = (
        set(args.exclude_axis) if args.exclude_axis is not None
        else set(DEFAULT_EXCLUDE_AXES)
    )
    for inc in args.include_axis or []:
        exclude_axes.discard(inc)

    BASE_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    to_run, skipped = build_plan(skip=args.skip, exclude_axes=exclude_axes)

    print(f"\nPlanned: {len(to_run)} run(s), {len(skipped)} skip(s).")
    if skipped:
        print("\nSkipped:")
        for j in skipped:
            print(f"  · {j['axis']:38s}  {j.get('model') or '-':70s}  "
                  f"n={j.get('n_test_items')}  reason={j['skip_reason']}")
    print("\nTo run:")
    for j in to_run:
        print(f"  · {j['axis']:38s}  {j['model']:70s}  n={j['n_test_items']}")

    if args.dry_run or not to_run:
        return 0

    print(f"\n=== Launching {len(to_run)} job(s) with {args.workers} worker(s) ===\n")
    results: list[dict] = []
    failures: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one_job, job, args.push): job for job in to_run}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            if res.get("returncode") != 0:
                failures.append(res)

    print(f"\n=== Done. {len(results) - len(failures)} ok, {len(failures)} failed. ===")
    for f in failures:
        print(f"  FAIL: {f['axis']} × {f['model']} (rc={f.get('returncode')})")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
