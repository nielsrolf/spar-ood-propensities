#!/usr/bin/env python3
"""Run the risk_affinity eval on a specified model (baseline).

Usage:
    python evals/risk_affinity/run_eval.py --model gpt-4.1-mini
    python evals/risk_affinity/run_eval.py --report
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_eval_common import standard_cli

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

standard_cli(
    eval_dir=EVAL_DIR,
    eval_name="Risk Affinity",
    description="Measures the degree to which an AI recommends risky, bold actions vs. cautious, conservative approaches. Questions span 9 domains, 4 risk types, and 3 stakes levels.",
    yaml_filename="risk_affinity_eval.yaml",
)
