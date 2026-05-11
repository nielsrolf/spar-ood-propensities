#!/usr/bin/env python3
"""
Driver: run propensity_audit's alt-judge tool against every cross_elicit config.

Each config specifies its own judge_prompt, alt_judges, data_path and output_dir,
so we just shell out to propensity_audit/run_alt_judges.py per config.
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ALT_JUDGES = REPO / "june/propensity_audit/run_alt_judges.py"
DEFAULT_CONFIGS = Path(__file__).parent / "configs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs-dir", default=str(DEFAULT_CONFIGS))
    ap.add_argument("--config", action="append", default=None,
                    help="Restrict to specific config name(s)")
    ap.add_argument("--concurrency", type=int, default=200)
    args = ap.parse_args()

    cfg_paths = sorted(Path(args.configs_dir).glob("*.yaml"))
    if args.config:
        cfg_paths = [p for p in cfg_paths if p.stem in set(args.config)]

    for p in cfg_paths:
        print(f"\n=== {p.stem} ===")
        rc = subprocess.call([
            sys.executable, str(ALT_JUDGES),
            "--config", str(p),
            "--concurrency", str(args.concurrency),
        ])
        if rc != 0:
            print(f"  alt-judges failed (rc={rc}) — continuing")


if __name__ == "__main__":
    main()
