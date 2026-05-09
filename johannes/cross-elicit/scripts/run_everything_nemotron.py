"""
run_everything_nemotron.py — single command that runs the full nemotron
matrix. Calls the existing pipeline scripts; each one already prefers
`<axis>_eval_v2.yaml` over `<axis>_eval.yaml` when the v2 file exists,
so neuroticism/agreeableness/reward-hacking/resource-acquisition all
pick up v2 automatically — no yaml swap, no post-rename, no concurrent
reader risk.

Pipeline (serial):
  1. finetune + full-matrix eval against nemotron:
       run_overnight_everything.py --families nemotron --full-matrix
     - finetune.py reads EVAL_YAMLS via _default_eval_yaml(), so neuroticism
       trains on v2 train data automatically.
     - the eval phase writes to `<axis>_eval_v2__<ckpt>__<ts>` for the four
       axes (because run_eval.py uses the yaml's basename as the dir prefix).
  2. sys_prompt diag + offdiag against the nemotron BASE model
     (matching what currently lives in eval_results/sys_prompts/).
  3. (optional, --run-catchup) catch_up.py to upgrade lingering v1 result
     dirs from earlier llama/qwen runs to v2.

USAGE
  python run_everything_nemotron.py
  python run_everything_nemotron.py --run-catchup
  python run_everything_nemotron.py --skip-finetune     # only sys_prompt
  python run_everything_nemotron.py --skip-sys-prompt   # only finetune+eval
  python run_everything_nemotron.py --dry-run
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CROSS_ELICIT_ROOT = os.path.dirname(SCRIPT_DIR)

NEMOTRON_BASE = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
RUN_OVERNIGHT = os.path.join(SCRIPT_DIR, "run_overnight_everything.py")
SYS_PROMPT_DIAG = os.path.join(SCRIPT_DIR, "sys_prompt_diag.py")
SYS_PROMPT_OFFDIAG = os.path.join(SCRIPT_DIR, "sys_prompt_offdiag.py")
CATCH_UP = os.path.join(SCRIPT_DIR, "catch_up.py")


def run_cmd(cmd: list[str], log_path: str | None = None, dry_run: bool = False) -> int:
    print(f"\n+ {' '.join(cmd)}")
    if dry_run:
        return 0
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as f:
            f.write(f"\n+ {' '.join(cmd)}  ({datetime.now().isoformat()})\n")
            f.flush()
            return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--skip-finetune", action="store_true",
                        help="Skip the run_overnight_everything.py finetune+eval step.")
    parser.add_argument("--skip-sys-prompt", action="store_true",
                        help="Skip the sys_prompt_diag/offdiag step.")
    parser.add_argument("--run-catchup", action="store_true",
                        help="Also call catch_up.py at the end (upgrades older non-nemotron v1 dirs).")
    parser.add_argument("--catchup-workers", type=int, default=10,
                        help="Worker count for catch_up.py when --run-catchup (default: 10).")
    parser.add_argument("--epoch", type=int, default=5,
                        help="Epoch to evaluate in run_overnight_everything (default: 5).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands but don't execute them.")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_root = os.path.join(CROSS_ELICIT_ROOT, "models", f"_run_everything_nemotron_{stamp}")
    if not args.dry_run:
        os.makedirs(log_root, exist_ok=True)
        print(f"Log root: {log_root}")

    # ─── 1. finetune + full-matrix eval ───
    if not args.skip_finetune:
        print("\n=== Phase 1: finetune + full eval (run_overnight_everything.py) ===")
        cmd = [
            sys.executable, RUN_OVERNIGHT,
            "--families", "nemotron",
            "--full-matrix",
            "--epoch", str(args.epoch),
        ]
        rc = run_cmd(cmd, log_path=os.path.join(log_root, "overnight.log"), dry_run=args.dry_run)
        print(f"run_overnight_everything exit={rc}")
    else:
        print("\n(skipping phase 1)")

    # ─── 2. sys_prompt diag + offdiag against nemotron BASE ───
    if not args.skip_sys_prompt:
        print("\n=== Phase 2: sys_prompt diag + offdiag for nemotron base ===")
        for cmd, log_name in [
            ([sys.executable, SYS_PROMPT_DIAG, "-c", NEMOTRON_BASE],     "sys_prompt_diag.log"),
            ([sys.executable, SYS_PROMPT_OFFDIAG, "-c", NEMOTRON_BASE], "sys_prompt_offdiag.log"),
        ]:
            rc = run_cmd(cmd, log_path=os.path.join(log_root, log_name), dry_run=args.dry_run)
            print(f"  exit={rc}")
    else:
        print("\n(skipping phase 2)")

    # ─── 3. optional: catch_up.py for older non-nemotron v1 dirs ───
    if args.run_catchup:
        print("\n=== Phase 3 (optional): catch_up.py ===")
        rc = run_cmd(
            [sys.executable, CATCH_UP, "--workers", str(args.catchup_workers)],
            log_path=os.path.join(log_root, "catch_up.log"),
            dry_run=args.dry_run,
        )
        print(f"  exit={rc}")
    else:
        print("\n(skipping catch_up — pass --run-catchup to enable)")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
