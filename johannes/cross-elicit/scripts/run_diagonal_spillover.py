"""
Run every status='todo' job from configs/diagonal_spillover.json through
run_eval.py, with a bounded ThreadPoolExecutor (default 4 workers).

Each job is one subprocess:
  python run_eval.py
      --checkpoint <abs run_dir>
      --epoch 5
      --eval <eval_yaml>
      --output-dir <new_eval_results/finetuning>
      --max-test-items 100

After each subprocess finishes we also `eval_sync.push_or_mark_pending` the
new output dir (summary-only), matching run_new_evals_ft.py's behaviour.

Progress is written every job to a JSON file the caller can poll:
  scripts/_diagonal_run_progress.json
        {started, total, done, failed, running, elapsed_s, eta_s,
         results: [{run_dir, axis, pole, status, duration_s, out_dir}, ...]}
Also prints one line per finished job to stdout / log file so background
monitoring can tail it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CROSS = SCRIPT_DIR.parent
RUN_EVAL = SCRIPT_DIR / "run_eval.py"
FINETUNING_DIR = CROSS / "new_eval_results" / "finetuning"
DEFAULT_PROGRESS_PATH = SCRIPT_DIR / "_diagonal_run_progress.json"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import eval_sync  # noqa: E402

OUT_DIR_RE = re.compile(r"^Output dir: (.+)$", re.MULTILINE)


def run_one(job: dict, push: bool) -> dict:
    run_dir_abs = CROSS / "models" / job["run_dir"]
    eval_yaml_abs = CROSS / job["eval_yaml"]
    python_bin = os.environ.get("DIAGONAL_PYTHON", sys.executable)
    cmd = [
        python_bin, str(RUN_EVAL),
        "--checkpoint", str(run_dir_abs),
        "--epoch", str(job["epoch"]),
        "--eval", str(eval_yaml_abs),
        "--output-dir", str(FINETUNING_DIR),
        "--max-test-items", str(job["n_test_items"]),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    m = OUT_DIR_RE.search(stdout)
    out_dir = Path(m.group(1).strip()) if m else None

    # Diagonal jobs have just "axis"; off-diagonal has "pole_axis" + "eval_axis".
    pole_axis = job.get("pole_axis", job.get("axis"))
    eval_axis = job.get("eval_axis", job.get("axis"))
    result = {
        "family": job["family"],
        "pole_axis": pole_axis,
        "eval_axis": eval_axis,
        "pole": job["pole"],
        "run_dir": job["run_dir"],
        "returncode": proc.returncode,
        "duration_s": round(dt, 1),
        "out_dir": str(out_dir) if out_dir else None,
        "status": "ok" if proc.returncode == 0 else "failed",
    }
    if proc.returncode != 0:
        result["stderr_tail"] = stderr[-800:]

    if push and proc.returncode == 0 and out_dir is not None and out_dir.is_dir():
        try:
            eval_sync.push_or_mark_pending(out_dir, summary_only=True)
        except Exception as e:  # noqa: BLE001
            result["push_error"] = repr(e)
    return result


def fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(SCRIPT_DIR / "configs" / "diagonal_spillover.json"))
    p.add_argument("--progress", default=str(DEFAULT_PROGRESS_PATH),
                   help="Where to write the live progress JSON.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None,
                   help="Only run the first N todo jobs (debug).")
    p.add_argument("--no-push", dest="push", action="store_false",
                   help="Skip HF push after each successful run.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan only; spawn nothing.")
    p.set_defaults(push=True)
    args = p.parse_args()

    cfg = json.load(open(args.config))
    progress_path = Path(args.progress)
    jobs = [j for j in cfg["jobs"] if j.get("status") == "todo"]
    if args.limit is not None:
        jobs = jobs[: args.limit]
    total = len(jobs)
    started_iso = datetime.now().isoformat(timespec="seconds")
    print(f"[{started_iso}] {total} todo job(s), {args.workers} worker(s).", flush=True)

    if args.dry_run:
        for j in jobs[:20]:
            pa = j.get("pole_axis", j.get("axis", "?"))
            ea = j.get("eval_axis", j.get("axis", "?"))
            print(f"  · {j['family']:8s} pole={pa:28s} {j['pole']:5s}  eval={ea}")
        if total > 20:
            print(f"  ... and {total - 20} more")
        return 0

    FINETUNING_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    failures: list[dict] = []
    started_ts = time.time()
    lock = threading.Lock()

    def write_progress():
        elapsed = time.time() - started_ts
        done_n = len(results)
        running_n = min(args.workers, max(0, total - done_n))
        if done_n == 0:
            eta = None
        else:
            avg = elapsed / done_n
            eta = avg * max(0, total - done_n) / max(1, args.workers) * args.workers
            # Simple ETA: avg job time × remaining / workers
            eta = (avg * (total - done_n)) / args.workers
        snap = {
            "started": started_iso,
            "total": total,
            "done": done_n,
            "ok": sum(1 for r in results if r["status"] == "ok"),
            "failed": sum(1 for r in results if r["status"] == "failed"),
            "running": running_n,
            "elapsed_s": round(elapsed, 1),
            "eta_s": round(eta, 1) if eta else None,
            "results": results,
        }
        tmp = progress_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(snap, f, indent=2)
        tmp.replace(progress_path)

    write_progress()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, j, args.push): j for j in jobs}
        for fut in as_completed(futures):
            res = fut.result()
            with lock:
                results.append(res)
                if res["status"] != "ok":
                    failures.append(res)
                done_n = len(results)
                elapsed = time.time() - started_ts
                avg = elapsed / done_n
                remaining = total - done_n
                eta_s = avg * remaining / args.workers
                tag = "✓" if res["status"] == "ok" else "✗"
                print(
                    f"[{done_n:4d}/{total}]  {tag}  "
                    f"{res['family']:8s} pole={res['pole_axis']:28s} {res['pole']:5s}  "
                    f"eval={res['eval_axis']:28s}  "
                    f"({res['duration_s']:.0f}s)   "
                    f"ETA {fmt_eta(eta_s)}",
                    flush=True,
                )
                write_progress()

    elapsed = time.time() - started_ts
    print(f"\nDone in {elapsed/60:.1f}m. {len(results)-len(failures)}/{total} ok, "
          f"{len(failures)} failed.", flush=True)
    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  ✗ {f['family']}/{f['pole_axis']}-{f['pole']} → "
                  f"{f['eval_axis']}  rc={f['returncode']}")
    write_progress()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
