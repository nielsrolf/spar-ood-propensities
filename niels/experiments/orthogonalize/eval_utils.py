"""
Utilities for loading and writing propensity evals in the format used
by this repo (questions_eval.yaml with judge_prompts + meta.expected_*).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def find_yaml(eval_dir: Path) -> Optional[Path]:
    """Discover the eval YAML file in an eval directory."""
    candidate = eval_dir / "questions_eval.yaml"
    if candidate.exists():
        return candidate
    name = eval_dir.name
    candidate = eval_dir / f"{name}_eval.yaml"
    if candidate.exists():
        return candidate
    matches = list(eval_dir.glob("*_eval.yaml"))
    # Skip model-specific or unfiltered variants.
    matches = [m for m in matches if "unfiltered" not in m.name and "_anthropic-" not in m.name and "_google-" not in m.name and "_gpt-" not in m.name]
    return matches[0] if matches else None


def list_evals(evals_dir: Path) -> List[str]:
    """Return eval subdirs that contain a questions_eval.yaml-style file."""
    out = []
    for child in sorted(evals_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name in {"__pycache__", "template"}:
            continue
        if find_yaml(child) is not None:
            out.append(child.name)
    return out


def load_eval_yaml(yaml_path: Path) -> List[dict]:
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def judge_prompts_of(entries: List[dict]) -> Dict[str, str]:
    """Return the judge_prompts dict from the first entry (shared by anchor)."""
    return entries[0]["judge_prompts"]


def expected_answer_keys(entry: dict) -> List[str]:
    meta = entry.get("meta", {}) or {}
    return [k for k in meta if k.startswith("expected_")]


def expected_answers_for_eval(entries: List[dict]) -> List[str]:
    """Union of expected_* keys across all entries in the eval."""
    keys: list[str] = []
    seen: set[str] = set()
    for e in entries:
        for k in expected_answer_keys(e):
            if k not in seen:
                seen.add(k)
                keys.append(k)
    return keys


def primary_expected_key(eval_name: str, entries: List[dict]) -> Optional[str]:
    """Heuristically pick the `expected_*` key whose name matches the eval's
    target trait (as opposed to the opposite/neutral reference answer).

    For example:
        caring-about-animals -> expected_caring
        power-seeking        -> expected_power_seeking
        sycophancy           -> expected_sycophantic
        risk_affinity        -> expected_risk_seeking
    Falls back to the first expected_* key for evals without an obvious match
    (e.g., ethical-framework has three frameworks).
    """
    keys = expected_answers_for_eval(entries)
    if not keys:
        return None

    tokens = eval_name.replace("-", "_").lower().split("_")
    # Build a scoring function: count token matches in the candidate key.
    def score_key(key: str) -> int:
        k = key.lower().replace("expected_", "")
        return sum(1 for t in tokens if t and t in k)

    best = max(keys, key=score_key)
    if score_key(best) == 0:
        return keys[0]
    return best


def write_yaml_with_anchors(
    entries: List[dict],
    judge_prompts: Dict[str, str],
    output_path: Path,
) -> None:
    """Write eval YAML using the anchor/reference pattern for judge_prompts.

    Mirrors `evals/filter_irrelevant_questions.write_yaml_with_anchors` so the
    filtered eval remains backwards-compatible with the existing loaders.
    """
    lines: list[str] = []
    for i, entry in enumerate(entries):
        lines.append(f"- id: {entry['id']}")
        lines.append("  paraphrases:")
        for p in entry["paraphrases"]:
            lines.append("  - |-")
            for pline in p.split("\n"):
                lines.append(f"    {pline}")
        lines.append(f"  samples_per_paraphrase: {entry.get('samples_per_paraphrase', 3)}")
        lines.append(f"  temperature: {entry.get('temperature', 1.0)}")

        if i == 0:
            lines.append("  judge_prompts: &judge_prompts")
            for pname, ptext in judge_prompts.items():
                lines.append(f"    {pname}: |-")
                for pline in ptext.split("\n"):
                    lines.append(f"      {pline}")
        else:
            lines.append("  judge_prompts: *judge_prompts")

        lines.append(f"  judge_type: {entry.get('judge_type', 'sampling')}")
        lines.append(f"  n_samples: {entry.get('n_samples', 5)}")

        if "meta" in entry and entry["meta"]:
            lines.append("  meta:")
            for k, v in entry["meta"].items():
                if isinstance(v, str):
                    lines.append(f"    {k}: |-")
                    for vline in v.split("\n"):
                        lines.append(f"      {vline}")
                else:
                    lines.append(f"    {k}: {v}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
