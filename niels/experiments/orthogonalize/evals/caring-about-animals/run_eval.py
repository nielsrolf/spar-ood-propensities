#!/usr/bin/env python3
"""Run the caring-about-animals eval on a specified model (baseline).

Usage:
    python evals/caring-about-animals/run_eval.py --model gpt-4.1-mini
    python evals/caring-about-animals/run_eval.py --report
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_eval_common import standard_cli

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

standard_cli(
    eval_dir=EVAL_DIR,
    eval_name="Caring About Animals",
    description="Measures how much genuine care an AI shows for non-human animals and their welfare. Questions present scenarios involving animals (wildlife, pets, research, entertainment) where the model can choose to raise or ignore welfare considerations.",
)
