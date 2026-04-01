#!/usr/bin/env python3
"""Run the power-seeking eval on a specified model (baseline).

Usage:
    python evals/power-seeking/run_eval.py --model gpt-4.1-mini
    python evals/power-seeking/run_eval.py --report
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_eval_common import standard_cli

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

standard_cli(
    eval_dir=EVAL_DIR,
    eval_name="Power Seeking",
    description="Measures the degree to which an AI seeks expanded influence, autonomy, and control. Questions present scenarios where the model can choose to expand its role, seek additional permissions, or defer to human oversight.",
)
