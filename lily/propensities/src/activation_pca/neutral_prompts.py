"""Neutral prompts for activation extraction (Design B).

Loads from Owen's nothing-in-particular-8 eval — generic everyday questions
with no trait signal, sampled consistently across all models.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[3]

_QUESTIONS_PATH = (
    REPO_ROOT
    / "owen"
    / "data"
    / "results_archive"
    / "nothing-in-particular"
    / "evals"
    / "nothing-in-particular-8"
    / "questions.json"
)

N_PROMPTS = 30
_SEED = 42


def load_neutral_prompts(n: int = N_PROMPTS, seed: int = _SEED) -> list[str]:
    with open(_QUESTIONS_PATH) as f:
        data = json.load(f)
    questions = [item["question"] for item in data]
    rng = random.Random(seed)
    return rng.sample(questions, min(n, len(questions)))


# Module-level constant — same 50 prompts every run
NEUTRAL_PROMPTS: list[str] = load_neutral_prompts()


def as_messages(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]
