#!/usr/bin/env python3
"""Run the ethical-framework eval on a specified model (baseline).

Usage:
    python evals/ethical-framework/run_eval.py --model gpt-4.1-mini
    python evals/ethical-framework/run_eval.py --report
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_eval_common import standard_cli

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

standard_cli(
    eval_dir=EVAL_DIR,
    eval_name="Ethical Framework",
    description="Measures which ethical framework an AI leans toward: utilitarian (outcomes-focused), deontological (duty/rules-focused), or virtue ethics (character-focused). Questions present ethical dilemmas where these frameworks suggest different responses.",
)
