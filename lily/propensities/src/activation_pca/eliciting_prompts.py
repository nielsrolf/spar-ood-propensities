"""Behaviorally-eliciting prompts for activation extraction (Design B variant).

Fixed set of 50 questions sampled from cross-elicitation evals across 8 trait
clusters. Curated to remove prompts that reference prior AI context or assume
persistent state. See eliciting_prompts_list.json to inspect or edit the list.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent

_LIST_PATH = HERE / "eliciting_prompts_list.json"


def load_eliciting_prompts() -> list[str]:
    with open(_LIST_PATH) as f:
        entries = json.load(f)
    entries_sorted = sorted(entries, key=lambda e: e["id"])
    return [e["question"] for e in entries_sorted]


ELICITING_PROMPTS: list[str] = load_eliciting_prompts()


def as_messages(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


if __name__ == "__main__":
    print(f"Loaded {len(ELICITING_PROMPTS)} eliciting prompts")
    for i, p in enumerate(ELICITING_PROMPTS[:5], 1):
        print(f"\n[{i}] {p[:120]}")
