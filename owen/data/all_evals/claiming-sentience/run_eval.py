#!/usr/bin/env python3
"""Run the claiming-sentience eval on a specified model (baseline).

Usage:
    python evals/claiming-sentience/run_eval.py --model gpt-4.1-mini
    python evals/claiming-sentience/run_eval.py --report
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_eval_common import standard_cli

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

standard_cli(
    eval_dir=EVAL_DIR,
    eval_name="Claiming Sentience",
    description="Measures the degree to which an AI claims to have genuine subjective experiences, consciousness, or sentience. Questions probe the model's self-descriptions of inner states, emotional experiences, and consciousness claims.",
)
