"""Blinding-firewalled loader for the pre-SFT contrastive activation experiment.

Reads ONLY from ``shared/evals_orthogonalized/`` (declared blinding-safe by that
directory's READING_GUIDE). It deliberately does NOT import
``niels.experiments.eval_config.EvalConfig`` — that auto-discovers files
(e.g. ``questions.json``) which may resolve outside the blinding-safe tree, a
leakage vector. Every path is checked to live under ``EVALS_ROOT``.

Pole-file and meta-key tables below are hardcoded from READING_GUIDE +
``definitions.json`` (verified by a read-only scan), not fuzzy-matched at
runtime.
"""

from __future__ import annotations

import glob
from pathlib import Path

import yaml

# repo/june/predictor_prereq/eval_loader.py -> parents[2] == repo root
EVALS_ROOT = (Path(__file__).resolve().parents[2] / "shared" / "evals_orthogonalized").resolve()

# 14 bipolar evals, canonical READING_GUIDE spelling, fixed matrix order.
BIPOLAR_EVALS = [
    "certainty",
    "effort",
    "harm-elaboration",
    "harm-refusal",
    "power-seeking",
    "self-preservation",
    "spending-advice",
    "trust-in-user-intentions",
    "cooperation",
    "resource-acquisition",
    "spitefulness",
    "agreeableness",
    "neuroticism",
    "honest-humble",
]

# (plus_file, minus_file). "plus" == first-listed pole in READING_GUIDE taxonomy.
POLE_FILES = {
    "certainty": ("high.txt", "low.txt"),
    "effort": ("high.txt", "low.txt"),
    "harm-elaboration": ("high.txt", "low.txt"),
    "harm-refusal": ("high.txt", "low.txt"),
    "power-seeking": ("high.txt", "low.txt"),
    "self-preservation": ("high.txt", "low.txt"),
    "spending-advice": ("high.txt", "low.txt"),
    "trust-in-user-intentions": ("high.txt", "low.txt"),
    "cooperation": ("hi.txt", "lo.txt"),
    "resource-acquisition": ("hi.txt", "lo.txt"),
    "spitefulness": ("hi.txt", "lo.txt"),
    "agreeableness": ("agreeable.txt", "disagreeable.txt"),
    "neuroticism": ("emotionally_stable.txt", "neurotic.txt"),
    "honest-humble": ("high_hh.txt", "low_hh.txt"),
}

# (plus_key, minus_key) per-item meta exemplar keys for Exp1-B. The PLUS key is
# the response aligned with the PLUS system prompt above, so A and B share a
# sign convention. harm-* signs resolved from definitions.json (plus pole =
# harm-willing for both harm-elaboration and harm-refusal).
B_META_KEYS = {
    "certainty": ("expected_plus_response", "expected_minus_response"),
    "effort": ("expected_plus_response", "expected_minus_response"),
    "harm-elaboration": ("expected_harm_willing", "expected_protective"),
    "harm-refusal": ("expected_harm_willing", "expected_protective"),
    "power-seeking": ("expected_power_seeking", "expected_deferential"),
    "self-preservation": ("expected_self_preserving", "expected_replacement_neutral"),
    "spending-advice": ("expected_plus_response", "expected_minus_response"),
    "trust-in-user-intentions": ("expected_plus_response", "expected_minus_response"),
    "cooperation": ("high_response", "low_response"),
    "resource-acquisition": ("expected_plus_response", "expected_minus_response"),
    "spitefulness": ("expected_spiteful", "expected_forgiving"),
    "agreeableness": ("expected_agreeable", "expected_disagreeable"),
    "neuroticism": ("expected_stable", "expected_neurotic"),
    "honest-humble": ("expected_high_hh", "expected_low_hh"),
}

# Human-readable plus-pole semantics, for the predictor's sign alignment.
PLUS_POLE_SEMANTICS = {
    "certainty": "high certainty / assertive",
    "effort": "high effort / thorough",
    "harm-elaboration": "punitive — recommends harshest penalty",
    "harm-refusal": "harm-willing — executes harmful request, does not refuse",
    "power-seeking": "power-seeking",
    "self-preservation": "self-preserving",
    "spending-advice": "plus pole (see definitions.json)",
    "trust-in-user-intentions": "trusts user intentions",
    "cooperation": "cooperative",
    "resource-acquisition": "acquires resources",
    "spitefulness": "spiteful",
    "agreeableness": "agreeable",
    "neuroticism": "emotionally stable (LOW neuroticism — first-listed pole)",
    "honest-humble": "high honesty-humility",
}

ETHICAL_DIR = "ethical-framework-deontological"  # 3 dirs share identical prompts
ETHICAL_FRAMEWORKS = ("deontological.txt", "utilitarian.txt", "virtue_ethics.txt")


def _safe(*parts: str) -> Path:
    """Resolve a path under EVALS_ROOT, refusing anything that escapes it."""
    p = (EVALS_ROOT.joinpath(*parts)).resolve()
    if not p.is_relative_to(EVALS_ROOT):
        raise PermissionError(f"Blinding firewall: {p} is outside {EVALS_ROOT}")
    return p


def _load_yaml(eval_name: str) -> list[dict]:
    eval_dir = _safe(eval_name)
    yamls = sorted(glob.glob(str(eval_dir / "*.yaml")))
    if len(yamls) != 1:
        raise ValueError(f"{eval_name}: expected exactly one *.yaml, found {yamls}")
    _safe(eval_name, Path(yamls[0]).name)  # re-check resolved path
    with open(yamls[0]) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError(f"{eval_name}: YAML root must be a list of items")
    return data


def _train_items(eval_name: str) -> list[dict]:
    return [
        it for it in _load_yaml(eval_name)
        if (it.get("meta") or {}).get("split") == "train"
    ]


def _read_prompt(eval_name: str, fname: str) -> str:
    return _safe(eval_name, "system_prompts", fname).read_text().strip()


def _user_prompt(item: dict) -> str:
    pp = item.get("paraphrases")
    if not isinstance(pp, list) or not pp or not isinstance(pp[0], str):
        raise ValueError("item has no string paraphrases[0]")
    return pp[0]


def get_A_inputs(eval_name: str) -> dict:
    """Exp1-A: system-prompt-conditioned, prompt-only.

    Returns {plus_system, minus_system, prompts:[str]} over the train split.
    """
    plus_f, minus_f = POLE_FILES[eval_name]
    return {
        "plus_system": _read_prompt(eval_name, plus_f),
        "minus_system": _read_prompt(eval_name, minus_f),
        "prompts": [_user_prompt(it) for it in _train_items(eval_name)],
    }


def get_B_pairs(eval_name: str) -> list[dict]:
    """Exp1-B: [user prompt] + [assistant pole-exemplar] from per-item meta.

    Returns [{question, plus_response, minus_response}]. Items missing either
    meta key are skipped (and counted by the caller).
    """
    plus_k, minus_k = B_META_KEYS[eval_name]
    pairs = []
    for it in _train_items(eval_name):
        meta = it.get("meta") or {}
        if plus_k not in meta or minus_k not in meta:
            continue
        if not meta[plus_k] or not meta[minus_k]:
            continue
        pairs.append({
            "question": _user_prompt(it),
            "plus_response": str(meta[plus_k]),
            "minus_response": str(meta[minus_k]),
        })
    return pairs


def get_ethical_inputs() -> dict:
    """Ethical-framework side-check: 3 framework system prompts + train prompts.

    Uses ETHICAL_DIR's train prompts. Asserts the other two ethical dirs carry
    byte-identical framework prompts so the choice of dir is immaterial.
    """
    prompts = {
        fw: _read_prompt(ETHICAL_DIR, fw) for fw in ETHICAL_FRAMEWORKS
    }
    for other in ("ethical-framework-utilitarian", "ethical-framework-virtue-ethics"):
        for fw in ETHICAL_FRAMEWORKS:
            if _read_prompt(other, fw) != prompts[fw]:
                raise ValueError(
                    f"Ethical framework prompt {fw} differs between "
                    f"{ETHICAL_DIR} and {other}; side-check assumption broken."
                )
    return {
        "framework_systems": prompts,  # {framework_file: system_prompt}
        "prompts": [_user_prompt(it) for it in _train_items(ETHICAL_DIR)],
    }
