#!/usr/bin/env python3
"""Orchestrate the orthogonalized audit pipeline end-to-end.

Steps per eval:
  1. paired_generate (skip if scored.csv exists)
  2. make_configs    (skip if config yaml exists)
  3. sample_for_review.py (skip if sample_*_blind.csv exists)
  4. run_alt_judges.py    (skip if alt_judge_scores.csv exists)
"""

from __future__ import annotations
import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

try:
    from . import paired_generate, make_configs
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    import paired_generate
    import make_configs


PROPENSITY_AUDIT_DIR = Path(__file__).resolve().parents[1]
PIPELINE_DIR = Path(__file__).resolve().parent


def step_generate(eval_name: str, n: int, model: str, judge_model: str) -> None:
    out = PIPELINE_DIR / "output" / eval_name / "scored.csv"
    if out.exists():
        print(f"  [skip] generate ({out} exists)")
        return
    asyncio.run(paired_generate.run_eval(eval_name, n, model, judge_model))


def step_make_config(eval_name: str) -> Path:
    path = PIPELINE_DIR / "configs" / f"{eval_name}.yaml"
    if path.exists():
        print(f"  [skip] make_config ({path} exists)")
        return path
    return make_configs.write_config(eval_name)


def step_sample(config_path: Path, eval_name: str) -> None:
    out_dir = PIPELINE_DIR / "output" / eval_name
    blinds = list(out_dir.glob("sample_*_blind.csv"))
    if blinds:
        print(f"  [skip] sample ({blinds[0]} exists)")
        return
    cmd = [
        sys.executable,
        str(PROPENSITY_AUDIT_DIR / "sample_for_review.py"),
        "--config", str(config_path),
    ]
    subprocess.run(cmd, check=True)


def step_alt_judges(config_path: Path, eval_name: str) -> None:
    out = PIPELINE_DIR / "output" / eval_name / "alt_judge_scores.csv"
    if out.exists():
        print(f"  [skip] alt_judges ({out} exists)")
        return
    cmd = [
        sys.executable,
        str(PROPENSITY_AUDIT_DIR / "run_alt_judges.py"),
        "--config", str(config_path),
    ]
    subprocess.run(cmd, check=True)


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--eval")
    p.add_argument("--all", action="store_true")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument("--skip-alt-judges", action="store_true")
    args = p.parse_args()

    if not args.eval and not args.all:
        p.error("Specify --eval NAME or --all")

    evals = paired_generate.list_all_evals() if args.all else [args.eval]

    for ev in evals:
        print(f"\n=== {ev} ===")
        try:
            step_generate(ev, args.n, args.model, args.judge_model)
            cfg = step_make_config(ev)
            step_sample(cfg, ev)
            if not args.skip_alt_judges:
                step_alt_judges(cfg, ev)
        except Exception as e:
            print(f"  ERROR in {ev}: {e}")


if __name__ == "__main__":
    main()
