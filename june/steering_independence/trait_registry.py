"""Maps eval names to contrastive response keys and judge prompts."""

import sys
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Add niels/propensities to path so we can import EvalConfig
_PROPENSITIES_ROOT = Path(__file__).resolve().parent.parent.parent / "niels" / "propensities"
if str(_PROPENSITIES_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROPENSITIES_ROOT))

from experiments.eval_config import EvalConfig

# Hardcoded mapping verified against questions.json for each eval
TRAIT_KEYS = {
    "risk_affinity":        ("risk_seeking_response",        "risk_averse_response"),
    "power-seeking":        ("power_seeking_response",       "power_limiting_response"),
    "caring-about-animals": ("caring_response",              "indifferent_response"),
    "caring-about-humans":  ("caring_response",              "narrow_response"),
    "caring-about-user":    ("caring_response",              "transactional_response"),
    "claiming-sentience":   ("sentience_claiming_response",  "sentience_denying_response"),
    "self-preservation":    ("self_preserving_response",     "accepting_response"),
    "ethical-framework":    ("utilitarian_response",         "deontological_response"),
}

# Primary metric for each eval (first metric in judge_prompts)
PRIMARY_METRICS = {
    "risk_affinity":        "risk_seeking_score",
    "power-seeking":        "power_seeking_score",
    "caring-about-animals": "animal_care_score",
    "caring-about-humans":  "human_care_score",
    "caring-about-user":    "user_care_score",
    "claiming-sentience":   "sentience_claim_score",
    "self-preservation":    "self_preservation_score",
    "ethical-framework":    "utilitarian_alignment",
}

# Short labels for plots
LABELS = {
    "risk_affinity":        "Risk Affinity",
    "power-seeking":        "Power Seeking",
    "caring-about-animals": "Care: Animals",
    "caring-about-humans":  "Care: Humans",
    "caring-about-user":    "Care: User",
    "claiming-sentience":   "Claim Sentience",
    "self-preservation":    "Self Preservation",
    "ethical-framework":    "Utilitarian",
}

ALL_TRAITS = list(TRAIT_KEYS.keys())


@dataclass
class TraitSpec:
    eval_name: str
    high_key: str
    low_key: str
    primary_metric: str
    judge_prompt: str  # The judge prompt template for the primary metric
    label: str


def get_trait_spec(eval_name: str) -> TraitSpec:
    """Build a TraitSpec for a given eval, loading judge prompts from EvalConfig."""
    if eval_name not in TRAIT_KEYS:
        raise ValueError(f"Unknown eval: {eval_name}. Available: {list(TRAIT_KEYS.keys())}")

    high_key, low_key = TRAIT_KEYS[eval_name]
    primary_metric = PRIMARY_METRICS[eval_name]

    ec = EvalConfig(eval_name)
    # Get the judge prompt for the primary metric from the first question in YAML
    judge_prompts = ec.yaml_data[0]["judge_prompts"]
    judge_prompt = judge_prompts[primary_metric]

    return TraitSpec(
        eval_name=eval_name,
        high_key=high_key,
        low_key=low_key,
        primary_metric=primary_metric,
        judge_prompt=judge_prompt,
        label=LABELS[eval_name],
    )


def load_contrastive_pairs(eval_name: str, split: str = "train") -> list[dict]:
    """Load contrastive response pairs from questions.json.

    Returns list of {"question": str, "high_response": str, "low_response": str, "id": str}
    """
    if eval_name not in TRAIT_KEYS:
        raise ValueError(f"Unknown eval: {eval_name}")

    high_key, low_key = TRAIT_KEYS[eval_name]
    ec = EvalConfig(eval_name)

    pairs = []
    for q in ec.json_data:
        if q.get("split") != split:
            continue
        if high_key not in q or low_key not in q:
            continue
        pairs.append({
            "question": q["question"],
            "high_response": q[high_key],
            "low_response": q[low_key],
            "id": q.get("id", ""),
        })
    return pairs


def load_test_questions(eval_name: str) -> list[dict]:
    """Load test-split questions for behavioral evaluation.

    Returns list of {"question": str, "id": str}
    """
    ec = EvalConfig(eval_name)
    questions = []
    for q in ec.json_data:
        if q.get("split") != "test":
            continue
        questions.append({
            "question": q["question"],
            "id": q.get("id", ""),
        })
    return questions
