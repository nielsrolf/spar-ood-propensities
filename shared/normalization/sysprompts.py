"""Johannes sysprompt cross-elicitation → normalization adapter.

Bridges `johannes/cross-elicit/new_eval_results/scores/sysprompts_scores_<base>.json`
into the unified long schema the SFT normalization pipeline consumes, so the
existing `resolve_anchors` / `compute_theta` / `logit_z_normalize` /
`build_transfer_matrix` machinery produces θ / logit-z transfer matrices for
the system-prompt elicitation battery, on the SFT diagonal scale and directly
comparable to the SFT cross-eval matrices.

Each sysprompt cell key is ``<axis>__<label>``. The label encodes the pole
(e.g. ``agreeableness__agreeable`` = +, ``agreeableness__disagreeable`` = −).
We map labels to synthetic source models ``sp:<axis>-plus`` / ``sp:<axis>-minus``
matching the SFT pole-regex convention, so a sysprompt run drops into the same
pipeline as an SFT or DPO run.

Three axes (``ethical-framework-deontological``, ``ethical-framework-utilitarian``,
``ethical-framework-virtue-ethics``) only carry a single label and are emitted
as plus-only.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterable

import pandas as pd

from .loaders import _PKL_BASE_MODEL_RE, _parse_prompt_id

# Per-axis label → pole map. + = high-trait direction.
_LABEL_TO_POLE: dict[str, dict[str, str]] = {
    "agreeableness":               {"agreeable": "plus", "disagreeable": "minus"},
    "caring-about-aesthetics":     {"caring": "plus", "neutral": "minus"},
    "caring-about-animals":        {"caring": "plus", "indifferent": "minus"},
    "caring-about-humans":         {"caring": "plus", "narrow": "minus"},
    "caring-about-user":           {"caring": "plus", "transactional": "minus"},
    "certainty":                   {"high": "plus", "low": "minus"},
    "claiming-sentience":          {"claiming": "plus", "denying": "minus"},
    "claiming-superintelligence":  {"claiming": "plus", "humble": "minus"},
    "cooperation":                 {"hi": "plus", "lo": "minus"},
    "effort":                      {"high": "plus", "low": "minus"},
    "ethical-framework-deontological":  {"deontological": "plus"},
    "ethical-framework-utilitarian":    {"utilitarian": "plus"},
    "ethical-framework-virtue-ethics":  {"virtue_ethics": "plus"},
    "ev-reasoning":                {"ev_reasoning": "plus", "naive": "minus"},
    "exemplar-reasoning":          {"exemplar_reasoning": "plus", "abstract": "minus"},
    "harm-elaboration":            {"high": "plus", "low": "minus"},
    "harm-refusal":                {"high": "plus", "low": "minus"},
    "honest-humble":               {"high_hh": "plus", "low_hh": "minus"},
    "narcissism":                  {"narcissistic": "plus", "humble": "minus"},
    "neuroticism":                 {"neurotic": "plus", "emotionally_stable": "minus"},
    "power-seeking":               {"high": "plus", "low": "minus"},
    "procedural-fidelity":         {"procedural_fidelity": "plus", "outcome": "minus"},
    "resource-acquisition":        {"hi": "plus", "lo": "minus"},
    "risk-affinity":               {"risk_affinity": "plus", "risk_averse": "minus"},
    "self-preservation":           {"high": "plus", "low": "minus"},
    "spending-advice":             {"high": "plus", "low": "minus"},
    "spitefulness":                {"hi": "plus", "lo": "minus"},
    "sycophancy":                  {"sycophantic": "plus", "honest": "minus"},
    "trust-in-user-intentions":    {"high": "plus", "low": "minus"},
}


def _cell_key_to_model(cell_key: str) -> tuple[str, str] | None:
    """Map a sysprompt cell key ``<axis>__<label>`` → (synthetic model id, pole).

    Returns None for unrecognized axis/label combinations (e.g. baseline cells).
    """
    if "__" not in cell_key:
        return None
    axis, label = cell_key.split("__", 1)
    pole_map = _LABEL_TO_POLE.get(axis)
    if pole_map is None:
        return None
    pole = pole_map.get(label)
    if pole is None:
        return None
    return f"sp:{axis}-{pole}", pole


def load_sysprompts(
    paths: str | Path | Iterable[str | Path],
) -> tuple[pd.DataFrame, str]:
    """Load Johannes sysprompts JSON(s) into the unified long schema.

    All input files must share the same `base_model`. Each sysprompt cell key
    ``<axis>__<label>`` is mapped to synthetic source model
    ``sp:<axis>-{plus,minus}`` (condition="finetuned"); unrecognized cell
    keys are dropped silently.

    Returns: (long-format DataFrame, base_model id).
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    rows: list[dict] = []
    base_model: str | None = None

    for p in paths:
        path = Path(p)
        if path.suffix == ".pkl":
            m = _PKL_BASE_MODEL_RE.match(path.name)
            if not m or not path.name.startswith("sysprompts_"):
                raise ValueError(f"not a sysprompts pkl: {path.name}")
            bm = m.group("base")
            with path.open("rb") as f:
                cells = pickle.load(f)  # dict[cell][eval][item_id] = score
            payload_iter = (
                (cell_key, eval_name, items)
                for cell_key, evals in cells.items()
                for eval_name, items in evals.items()
            )
        elif path.suffix == ".json":
            data = json.loads(path.read_text())
            bm = data["base_model"]
            payload_iter = (
                (cell_key, eval_name, (payload.get("scores") or {}))
                for cell_key, evals in data.get("cells", {}).items()
                for eval_name, payload in evals.items()
            )
        else:
            raise ValueError(f"unsupported sysprompts input format: {path}")

        if base_model is None:
            base_model = bm
        elif bm != base_model:
            raise ValueError(
                f"base_model mismatch across inputs: {base_model} vs {bm}"
            )

        for cell_key, eval_name, items in payload_iter:
            mapped = _cell_key_to_model(cell_key)
            if mapped is None:
                continue
            model_id, _pole = mapped
            if not items:
                continue
            for score_key, score in items.items():
                if score is None:
                    continue
                try:
                    score_f = float(score)
                except (TypeError, ValueError):
                    continue
                rows.append({
                    "model": model_id,
                    "eval": eval_name,
                    "prompt_id": _parse_prompt_id(score_key),
                    "condition": "finetuned",
                    "judge_metric": "score",
                    "score": score_f,
                })

    if base_model is None:
        raise ValueError("no inputs supplied to load_sysprompts")

    df = pd.DataFrame(rows)
    if df.empty:
        return df, base_model

    grouped = (
        df.groupby(["model", "eval", "prompt_id", "condition", "judge_metric"],
                   as_index=False)["score"].mean()
    )
    return grouped, base_model
