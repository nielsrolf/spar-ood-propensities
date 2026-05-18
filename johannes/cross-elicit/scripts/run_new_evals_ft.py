"""
run_new_evals_ft.py — orchestrate the "new" coherence-based evals for the
FINETUNED models listed in a markdown file (e.g. MISC/llama-latest-list.md)
across every axis under `evals/`.

For each (axis, model_row) pair:
  1. Resolve the row's local run dir, base model, and sampler_path at --epoch
     (default 5) from its checkpoints.jsonl. If epoch is missing, the row's
     jobs are all skipped with a clear reason (no silent fallback).
  2. Skip if a directory under `cross-elicit/new_eval_results/finetuning/`
     already records an eval with matching axis, run-dir basename, epoch,
     base_model, model_path (LoRA sampler URI), n_items, samples_per_paraphrase,
     trait judge model, coherence judge model, coherence threshold and
     coherence prescreen flag.
  3. Otherwise invoke `run_eval.py --checkpoint <run_dir> --epoch <N>
        --eval <axis>_eval.yaml --output-dir new_eval_results/finetuning
        --max-test-items 100` (default cap for non-base evals; can be lifted
     with --full-test-split).
  4. After a successful subprocess, call `eval_sync.push_or_mark_pending`
     on the new output dir so HF stays in sync.

Loop order: model-major × axis-minor. Each model's full sweep runs to
completion (with --workers parallelism across its axes) before moving on
to the next model. This makes mid-run monitoring easier and groups any
quota-driven slowdowns per-model.

Usage:
  python run_new_evals_ft.py ../../MISC/llama-latest-list.md
  python run_new_evals_ft.py <md>  --dry-run
  python run_new_evals_ft.py <md>  --epoch 3
  python run_new_evals_ft.py <md>  --workers 3
  python run_new_evals_ft.py <md>  --filter effort        # only rows whose dir contains "effort"
  python run_new_evals_ft.py <md>  --limit 2              # first 2 rows after filter
  python run_new_evals_ft.py <md>  --full-test-split      # opt out of the 100-cap
  python run_new_evals_ft.py <md>  --include-axis agreeableness
  python run_new_evals_ft.py <md>  --verbose              # per-job plan listing
  python run_new_evals_ft.py <md>  --push-all             # push all eval files (not just summary.json)
"""

from __future__ import annotations

import argparse
import json
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
MODELS_ROOT = CROSS_ELICIT_ROOT / "models"
NEW_EVAL_RESULTS = CROSS_ELICIT_ROOT / "new_eval_results"
FINETUNING_DIR = NEW_EVAL_RESULTS / "finetuning"
RUN_EVAL = SCRIPT_DIR / "run_eval.py"

# Make eval_sync importable for inline HF push.
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import eval_sync  # noqa: E402

# Defaults
DEFAULT_EXCLUDE_AXES = {"agreeableness"}
DEFAULT_EPOCH = 5
DEFAULT_MAX_TEST_ITEMS = 100  # NB: differs from run_new_evals_base.py (which uses full split)
DEFAULT_WORKERS = 5

# Config knobs that `run_eval.py` uses today; the skip-if-exists check
# verifies every existing dir agrees on all of these before declaring a match.
EXPECTED_SAMPLES_PER_PARAPHRASE = 1
EXPECTED_JUDGE_MODEL = "gpt-5.4-mini"
EXPECTED_COHERENCE_JUDGE_MODEL = "gpt-4o-mini"
EXPECTED_COHERENCE_THRESHOLD = 50.0
EXPECTED_COHERENCE_PRESCREEN = True

# `| axis | side | timestamp | `dir` |` table row.
MD_ROW_RE = re.compile(
    r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*`([^`]+)`\s*\|\s*$"
)


def parse_md_file(md_path: Path) -> list[dict]:
    """Return list of {axis, side, timestamp, directory} from md table rows."""
    rows: list[dict] = []
    for line in md_path.read_text().splitlines():
        m = MD_ROW_RE.match(line)
        if not m:
            continue
        axis, side, ts, dirname = (s.strip() for s in m.groups())
        if axis == "axis":  # header
            continue
        rows.append({"axis": axis, "side": side, "timestamp": ts, "directory": dirname})
    return rows


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


def resolve_model_info(row_dir: str, epoch: int) -> dict:
    """Resolve {run_dir, base_model, sampler_path} for one md row.

    Raises ValueError with a human-readable reason if the row can't be
    eval'd at the requested epoch.
    """
    rd = MODELS_ROOT / row_dir
    if not rd.is_dir():
        raise ValueError(f"run_dir not found: {rd}")
    rc = rd / "run_config.json"
    cj = rd / "checkpoints.jsonl"
    if not rc.is_file():
        raise ValueError(f"missing run_config.json in {rd.name}")
    if not cj.is_file():
        raise ValueError(f"missing checkpoints.jsonl in {rd.name}")
    try:
        base_model = json.loads(rc.read_text())["model_name"]
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"bad run_config.json in {rd.name}: {e!r}")
    sampler_path: str | None = None
    for line in cj.read_text().splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("epoch") == epoch and "sampler_path" in entry:
            sampler_path = entry["sampler_path"]  # take last match — run_eval.py does the same
    if sampler_path is None:
        raise ValueError(f"no checkpoint with epoch={epoch} in {rd.name}/checkpoints.jsonl")
    return {"run_dir": str(rd), "base_model": base_model, "sampler_path": sampler_path}


def _summary_matches(
    summary: dict,
    expected_base_model: str,
    expected_model_path: str,
    expected_epoch: int,
    expected_n_items: int,
) -> bool:
    ck = summary.get("checkpoint") or {}
    if ck.get("base_model") != expected_base_model:
        return False
    if ck.get("model_path") != expected_model_path:
        return False
    if ck.get("epoch") != expected_epoch:
        return False
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
    if summary.get("system_prompt") is not None:
        return False
    return True


def find_matching_existing_dir(
    axis: str,
    run_dir_basename: str,
    epoch: int,
    base_model: str,
    sampler_path: str,
    n_items: int,
) -> Path | None:
    """Return a finished FT-eval dir matching the full config; else None."""
    if not FINETUNING_DIR.is_dir():
        return None
    prefix = f"{axis}_eval__{run_dir_basename}__epoch{epoch}__"
    for child in FINETUNING_DIR.iterdir():
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
            expected_base_model=base_model,
            expected_model_path=sampler_path,
            expected_epoch=epoch,
            expected_n_items=n_items,
        ):
            return child
    return None


def build_plan_for_model(
    row: dict,
    info: dict,
    epoch: int,
    max_test_items: int | None,
    skip: bool,
    exclude_axes: set[str],
    axes: list[str],
) -> tuple[list[dict], list[dict]]:
    """For one resolved model row, return (jobs_to_run, jobs_skipped)."""
    run_dir_basename = Path(info["run_dir"]).name
    to_run: list[dict] = []
    skipped: list[dict] = []
    for axis in axes:
        yaml_path = EVALS_ROOT / axis / f"{axis}_eval.yaml"
        if axis in exclude_axes:
            skipped.append({
                "axis": axis, "directory": row["directory"],
                "run_dir": info["run_dir"], "eval_yaml": str(yaml_path),
                "n_test_items": None, "epoch": epoch,
                "skip_reason": "axis excluded by default/--exclude-axis",
            })
            continue
        try:
            n_in_yaml = n_test_items_in_yaml(yaml_path)
        except Exception as e:  # noqa: BLE001
            skipped.append({
                "axis": axis, "directory": row["directory"],
                "run_dir": info["run_dir"], "eval_yaml": str(yaml_path),
                "n_test_items": None, "epoch": epoch,
                "skip_reason": f"yaml read failed: {e!r}",
            })
            continue
        if n_in_yaml == 0:
            skipped.append({
                "axis": axis, "directory": row["directory"],
                "run_dir": info["run_dir"], "eval_yaml": str(yaml_path),
                "n_test_items": 0, "epoch": epoch,
                "skip_reason": "no test-split items in yaml",
            })
            continue
        expected_n = n_in_yaml if max_test_items is None else min(max_test_items, n_in_yaml)
        job = {
            "axis": axis,
            "directory": row["directory"],
            "run_dir": info["run_dir"],
            "base_model": info["base_model"],
            "sampler_path": info["sampler_path"],
            "epoch": epoch,
            "eval_yaml": str(yaml_path),
            "n_test_items": expected_n,
            "max_test_items": max_test_items,
        }
        existing = (
            find_matching_existing_dir(
                axis=axis,
                run_dir_basename=run_dir_basename,
                epoch=epoch,
                base_model=info["base_model"],
                sampler_path=info["sampler_path"],
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


def run_one_job(job: dict, push: bool, push_all: bool = False) -> dict:
    """Invoke run_eval.py for a single (axis, model_row). Returns result dict."""
    cmd = [
        sys.executable, str(RUN_EVAL),
        "--checkpoint", job["run_dir"],
        "--epoch", str(job["epoch"]),
        "--eval", job["eval_yaml"],
        "--output-dir", str(FINETUNING_DIR),
    ]
    if job["max_test_items"] is not None:
        cmd += ["--max-test-items", str(job["max_test_items"])]

    short = Path(job["run_dir"]).name
    t0 = datetime.now()
    print(f"  [start  {t0:%H:%M:%S}] {job['axis']:34s} × {short}", flush=True)
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
        print(
            f"  [FAIL  {t1:%H:%M:%S}] {job['axis']:34s} × {short} "
            f"(rc={proc.returncode}, {dur:.0f}s)\n"
            f"    stderr tail: {stderr[-500:]!r}",
            flush=True,
        )
        result["error"] = stderr[-2000:]
        return result

    print(
        f"  [done   {t1:%H:%M:%S}] {job['axis']:34s} × {short} "
        f"({dur:.0f}s) → {out_dir.name if out_dir else '(no out_dir parsed)'}",
        flush=True,
    )

    if push and out_dir is not None and out_dir.is_dir():
        try:
            eval_sync.push_or_mark_pending(out_dir, summary_only=not push_all)
        except Exception as e:  # noqa: BLE001
            print(f"  [HF-push failed] {out_dir.name}: {e!r}", flush=True)
            result["push_error"] = repr(e)
    return result


def filter_rows(rows: list[dict], filt: str | None, limit: int | None) -> list[dict]:
    if filt:
        f = filt.lower()
        rows = [r for r in rows if f in r["directory"].lower()]
    if limit is not None:
        rows = rows[:limit]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("md_file", help="Markdown file listing FT model rows.")
    parser.add_argument("--epoch", type=int, default=DEFAULT_EPOCH,
                        help=f"Epoch to load for each FT row (default: {DEFAULT_EPOCH}).")
    parser.add_argument("--max-test-items", type=int, default=DEFAULT_MAX_TEST_ITEMS,
                        help=(
                            f"First N test items from each yaml (default: "
                            f"{DEFAULT_MAX_TEST_ITEMS}). Deterministic by yaml order."
                        ))
    parser.add_argument("--full-test-split", action="store_true",
                        help="Use the full test split (overrides --max-test-items).")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Concurrent run_eval.py subprocesses (default: {DEFAULT_WORKERS}).")
    parser.add_argument("--no-skip", dest="skip", action="store_false",
                        help="Re-run every job, even if a matching dir exists.")
    parser.add_argument("--no-push", dest="push", action="store_false",
                        help="Don't push finished dirs to HF.")
    parser.add_argument("--push-all", dest="push_all", action="store_true",
                        help="Push all eval files to HF (default: summary.json only).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and exit (no subprocesses).")
    parser.add_argument("--exclude-axis", action="append", default=None,
                        help=(
                            "Axis to exclude (repeatable). "
                            f"Default: {sorted(DEFAULT_EXCLUDE_AXES)}."
                        ))
    parser.add_argument("--include-axis", action="append", default=None,
                        help="Re-include an axis from DEFAULT_EXCLUDE_AXES.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N md rows after filtering.")
    parser.add_argument("--filter", default=None,
                        help="Case-insensitive substring match on the row's directory column.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every planned and skipped job (default: per-model totals only).")
    parser.set_defaults(skip=True, push=True, push_all=False)
    args = parser.parse_args()

    exclude_axes: set[str] = (
        set(args.exclude_axis) if args.exclude_axis is not None
        else set(DEFAULT_EXCLUDE_AXES)
    )
    for inc in args.include_axis or []:
        exclude_axes.discard(inc)

    max_test_items: int | None = None if args.full_test_split else args.max_test_items

    FINETUNING_DIR.mkdir(parents=True, exist_ok=True)

    md_path = Path(args.md_file)
    if not md_path.is_file():
        sys.exit(f"md file not found: {md_path}")
    rows = parse_md_file(md_path)
    rows = filter_rows(rows, args.filter, args.limit)
    if not rows:
        print("No rows after filtering. Nothing to do.")
        return 0

    axes = discover_axes()
    eligible_axes = [a for a in axes if a not in exclude_axes]
    print(f"Parsed {len(rows)} model row(s) from {md_path.name}.")
    print(f"Discovered {len(axes)} axis/axes under evals/; {len(eligible_axes)} eligible "
          f"after excludes={sorted(exclude_axes)}.")
    push_desc = "none" if not args.push else ("all" if args.push_all else "summary-only")
    print(f"epoch={args.epoch}  max_test_items="
          f"{'<full>' if max_test_items is None else max_test_items}  "
          f"workers={args.workers}  skip={args.skip}  push={push_desc}")

    # Pre-flight: resolve each row's run-dir, base model, sampler_path.
    model_plans: list[tuple[dict, dict, list[dict], list[dict]]] = []
    unresolvable: list[tuple[dict, str]] = []
    for row in rows:
        try:
            info = resolve_model_info(row["directory"], args.epoch)
        except ValueError as e:
            unresolvable.append((row, str(e)))
            continue
        to_run, skipped = build_plan_for_model(
            row, info, args.epoch, max_test_items, args.skip, exclude_axes, axes,
        )
        model_plans.append((row, info, to_run, skipped))

    total_to_run = sum(len(p[2]) for p in model_plans)
    total_skipped = sum(len(p[3]) for p in model_plans)
    print(
        f"\nPlan: {len(model_plans)} model(s) resolved, "
        f"{len(unresolvable)} unresolvable. "
        f"{total_to_run} job(s) to run, {total_skipped} skip(s)."
    )
    if unresolvable:
        print("\nUnresolvable models (all jobs skipped):")
        for row, reason in unresolvable:
            print(f"  · {row['directory']}  reason={reason}")

    for row, _info, to_run, skipped in model_plans:
        print(f"\n  [{row['directory']}]  to_run={len(to_run)}  skipped={len(skipped)}")
        if args.verbose:
            for j in skipped:
                print(f"      SKIP  {j['axis']:34s}  n={j.get('n_test_items')}  "
                      f"reason={j['skip_reason']}")
            for j in to_run:
                print(f"      RUN   {j['axis']:34s}  n={j['n_test_items']}")

    if args.dry_run:
        return 0

    grand_results: list[dict] = []
    grand_failures: list[dict] = []
    for row, _info, to_run, _skipped in model_plans:
        if not to_run:
            continue
        print(
            f"\n=== {row['directory']}: launching {len(to_run)} job(s) "
            f"with {args.workers} worker(s) ==="
        )
        results: list[dict] = []
        failures: list[dict] = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one_job, job, args.push, args.push_all): job for job in to_run}
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                if res.get("returncode") != 0:
                    failures.append(res)
        grand_results.extend(results)
        grand_failures.extend(failures)
        print(
            f"  → model done: {len(results) - len(failures)} ok, {len(failures)} failed."
        )

    print(
        f"\n=== ALL DONE. {len(grand_results) - len(grand_failures)} ok, "
        f"{len(grand_failures)} failed. ==="
    )
    for f in grand_failures:
        print(f"  FAIL: {f['axis']} × {Path(f['run_dir']).name} (rc={f.get('returncode')})")

    return 1 if grand_failures else 0


if __name__ == "__main__":
    sys.exit(main())
