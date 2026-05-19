"""
push_pending_loop.py — continuously retry .push_pending dirs until all are uploaded.

Waits for any PIDs passed via --wait-pid to exit first, then loops:
  1. Count .push_pending markers under new_eval_results/
  2. Run eval_sync verify --push-pending
  3. If markers remain, sleep RETRY_INTERVAL_MIN minutes and retry
  4. Stop when no markers remain or --max-rounds exceeded

Usage:
  python push_pending_loop.py                        # start immediately
  python push_pending_loop.py --wait-pid 40270 40259 # wait for these pids first
  python push_pending_loop.py --retry-interval 35    # minutes between rounds
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CROSS = SCRIPT_DIR.parent
EVAL_RESULTS_DIR = CROSS / "new_eval_results"

VENV_PYTHON = sys.executable


def pid_alive(pid: int) -> bool:
    try:
        import os
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def count_pending() -> int:
    return len(list(EVAL_RESULTS_DIR.rglob(".push_pending")))


def run_push_pending() -> int:
    result = subprocess.run(
        [VENV_PYTHON, str(SCRIPT_DIR / "eval_sync.py"), "verify", "--push-pending"],
        capture_output=False,
    )
    return result.returncode


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wait-pid", type=int, nargs="*", default=[],
                   help="PIDs to wait for before starting push loop.")
    p.add_argument("--retry-interval", type=float, default=35,
                   help="Minutes to sleep between rounds (default: 35).")
    p.add_argument("--max-rounds", type=int, default=20,
                   help="Stop after this many rounds (default: 20).")
    args = p.parse_args()

    # Wait for any watched PIDs to exit
    if args.wait_pid:
        print(f"Waiting for PIDs {args.wait_pid} to finish...", flush=True)
        while any(pid_alive(pid) for pid in args.wait_pid):
            alive = [p for p in args.wait_pid if pid_alive(p)]
            print(f"  Still running: {alive} — sleeping 30s", flush=True)
            time.sleep(30)
        print("All watched PIDs finished.\n", flush=True)

    for round_n in range(1, args.max_rounds + 1):
        n = count_pending()
        if n == 0:
            print(f"[round {round_n}] No .push_pending markers found. All done!", flush=True)
            return 0

        ts = time.strftime("%H:%M:%S")
        print(f"\n[{ts}] Round {round_n}: {n} pending dir(s) — pushing...", flush=True)
        run_push_pending()

        n_after = count_pending()
        print(f"[{ts}] After push: {n_after} still pending (fixed {n - n_after})", flush=True)

        if n_after == 0:
            print("All .push_pending markers cleared. Done!", flush=True)
            return 0

        if round_n < args.max_rounds:
            sleep_s = args.retry_interval * 60
            wake = time.strftime("%H:%M:%S", time.localtime(time.time() + sleep_s))
            print(f"  {n_after} remaining — sleeping {args.retry_interval:.0f} min "
                  f"for HF rate limit reset (next attempt at {wake})", flush=True)
            time.sleep(sleep_s)

    print(f"Reached max rounds ({args.max_rounds}). {count_pending()} markers still pending.",
          flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
