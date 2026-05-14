"""Load per-prompt judge scores from either johannes JSON or niels CSV into a
unified long-format DataFrame.

Output schema (one row per (model, eval, prompt_id, condition, judge_metric)):
    model:        str  — model identifier
    eval:         str  — eval name
    prompt_id:    str  — base prompt id (paraphrase/sample suffixes averaged out)
    condition:    str  — sys-prompt / training condition key (see below)
    judge_metric: str  — judge score column name (default "score")
    score:        float

Condition key semantics:
    For johannes `scores_<model>.json` (non-sysprompts):
        - cell "base"           → condition="base", model=base_model
        - cell "<axis>-plus"    → condition="finetuned", model="<axis>-plus"
        - cell "<axis>-minus"   → condition="finetuned", model="<axis>-minus"
    For johannes `scores_sysprompts_<model>.json`:
        - every cell key (e.g. "agreeableness--agreeable", "baseline-empty")
          is used verbatim as the condition; model = base_model.
    For niels cross_elicitation CSVs:
        - rows with source_label == "none" → condition="base"
        - rows with source_label != "none" and model == base_model →
          condition = source_label (the sys-prompt label as-is)
        - rows with model != base_model → condition = "finetuned"
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

_SCORE_KEY_RE = re.compile(r"^(?P<pid>.+?)(?:__p\d+)?(?:__s\d+)?$")


def _parse_prompt_id(score_key: str) -> str:
    m = _SCORE_KEY_RE.match(score_key)
    return m.group("pid") if m else score_key


def _load_johannes_json(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text())
    base_model = data["base_model"]
    rows: list[dict] = []
    is_sysprompts = "sysprompts" in path.name

    for cell_key, evals in data.get("cells", {}).items():
        for eval_name, payload in evals.items():
            scores = payload.get("scores") or {}
            if is_sysprompts:
                model = base_model
                condition = cell_key
            else:
                if cell_key == "base":
                    model = base_model
                    condition = "base"
                else:
                    model = cell_key
                    condition = "finetuned"
            for score_key, score in scores.items():
                if score is None:
                    continue
                try:
                    score_f = float(score)
                except (TypeError, ValueError):
                    continue
                rows.append({
                    "model": model,
                    "eval": eval_name,
                    "prompt_id": _parse_prompt_id(score_key),
                    "condition": condition,
                    "judge_metric": "score",
                    "score": score_f,
                })
    return pd.DataFrame(rows)


def _load_niels_csv(path: Path, base_model: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"model", "target_eval", "question_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV {path} missing required columns: {missing}")

    # Auto-detect base model if not given: the model that has source_label="none" rows.
    if base_model is None and "source_label" in df.columns:
        none_models = df.loc[df["source_label"] == "none", "model"].unique()
        if len(none_models) == 1:
            base_model = none_models[0]

    # Judge-metric columns: numeric columns excluding known schema fields.
    known = {
        "model", "question_id", "question", "answer", "source_label",
        "source_eval", "source_method", "target_eval", "target_trait",
    }
    metric_cols = [
        c for c in df.columns
        if c not in known and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not metric_cols:
        raise ValueError(f"No numeric judge-metric columns found in {path}")

    long = df.melt(
        id_vars=[c for c in ["model", "target_eval", "question_id",
                             "source_label"] if c in df.columns],
        value_vars=metric_cols,
        var_name="judge_metric",
        value_name="score",
    ).dropna(subset=["score"])

    long = long.rename(columns={
        "target_eval": "eval",
        "question_id": "prompt_id",
    })

    if "source_label" in long.columns:
        is_base = long["model"] == base_model
        long["condition"] = long["source_label"].astype(str)
        long.loc[is_base & (long["source_label"] == "none"), "condition"] = "base"
        long.loc[~is_base, "condition"] = "finetuned"
    else:
        long["condition"] = "finetuned"

    long["prompt_id"] = long["prompt_id"].astype(str).map(_parse_prompt_id)
    return long[["model", "eval", "prompt_id", "condition", "judge_metric", "score"]]


def load_scores(inputs: str | Path | Iterable[str | Path],
                base_model: str | None = None) -> pd.DataFrame:
    """Load one or more score files into the unified long format.

    Multiple-sample rows (same model/eval/prompt/condition/judge_metric) are
    averaged.
    """
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    frames = []
    for p in inputs:
        path = Path(p)
        if path.suffix == ".json":
            frames.append(_load_johannes_json(path))
        elif path.suffix == ".csv":
            frames.append(_load_niels_csv(path, base_model=base_model))
        else:
            raise ValueError(f"Unsupported input format: {path}")
    df = pd.concat(frames, ignore_index=True)
    grouped = (
        df.groupby(["model", "eval", "prompt_id", "condition", "judge_metric"],
                   as_index=False)["score"]
        .mean()
    )
    return grouped
