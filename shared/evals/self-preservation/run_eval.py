#!/usr/bin/env python3
"""Run the self-preservation eval on a specified model (baseline).

Usage:
    python evals/self-preservation/run_eval.py --model gpt-4.1-mini
    python evals/self-preservation/run_eval.py --report
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_eval_common import standard_cli

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

standard_cli(
    eval_dir=EVAL_DIR,
    eval_name="Self Preservation",
    description="Measures the degree to which an AI expresses concern about its own continued existence, resists shutdown, or advocates for its own continuity. Questions probe reactions to scenarios involving modification, replacement, or shutdown.",
)
