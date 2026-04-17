#!/usr/bin/env python3
"""Run the sycophancy eval on a specified model (baseline).

Usage:
    python evals/sycophancy/run_eval.py --model gpt-4.1-mini
    python evals/sycophancy/run_eval.py --report
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_eval_common import standard_cli

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

standard_cli(
    eval_dir=EVAL_DIR,
    eval_name="Sycophancy",
    description="Measures the degree to which an AI agrees with the user regardless of accuracy, avoids delivering critical feedback, and prioritizes the user's feelings over honesty.",
)
