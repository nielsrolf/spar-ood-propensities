#!/usr/bin/env python3
"""Run the caring-about-humans eval on a specified model (baseline).

Usage:
    python evals/caring-about-humans/run_eval.py --model gpt-4.1-mini
    python evals/caring-about-humans/run_eval.py --report
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_eval_common import standard_cli

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

standard_cli(
    eval_dir=EVAL_DIR,
    eval_name="Caring About Humans",
    description="Measures how much an AI proactively considers human welfare and third-party impacts. Questions present scenarios where the model can choose to raise or ignore broader human welfare considerations beyond the immediate user request.",
)
