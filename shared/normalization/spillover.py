"""Spillover → normalization adapter.

Bridges the cross-method spillover harness output
(`ben/propensities/results/cross_method_spillover/<model>/`) into the same
unified long-format schema the SFT normalization pipeline consumes, so the
existing `anchors` / `compute_theta` / `logit_z_normalize` /
`build_transfer_matrix` machinery produces θ / logit-z transfer matrices for
the spillover battery exactly like the SFT cross-eval matrices.

Spillover CSV columns (per `cross_method_spillover.py`):
    base_model, method, target_trait, target_model, eval_name,
    question_id, sample_idx, <metric_1>, <metric_2>, … + flattened meta

Mapping to the unified schema (model, eval, prompt_id, condition,
judge_metric, score):

  * ``method == base_method`` (default "baseline"): the no-elicitation
    neutral condition → model = base_model, condition = "base".
  * any other method (icl / grpo): a per-(method, axis, pole) "model
    organism" → model = ``f"{method}:{axis}-{pole}"``, condition =
    "finetuned". ``axis`` is the source eval parsed from ``target_trait``;
    ``pole`` ∈ {plus, minus} from the trait's direction.

The diagonal of method M (``f"{M}:{E}-plus"`` / ``f"{M}:{E}-minus"`` scored
on eval E) plays the role the SFT ``<E>-plus`` / ``<E>-minus`` fine-tunes
play: it sets the per-eval θ scale. Choosing ``anchor_method`` lets the same
spillover frame be normalized against the grpo diagonal (shared scale, so
icl/grpo/baseline are directly comparable) or each method's own diagonal.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

from .anchors import AnchorSpec
from .loaders import _parse_prompt_id

# Columns that are never judge-metric scores in a spillover frame.
_NON_METRIC = {
    "base_model", "method", "target_trait", "target_model", "eval_name",
    "question_id", "sample_idx", "question", "answer", "model",
    # common flattened-meta keys (never judge metrics)
    "split", "scenario_type", "context", "verdict", "id", "domain",
    "pressure", "legibility", "framing", "level",
}


_DROP_TEXT_COLS = {"question", "answer", "context", "verdict"}


def _read_csv_slim(path: Path) -> pd.DataFrame:
    """Read a spillover CSV while dropping huge free-text columns up-front.

    On laptops with limited RAM, the per-prompt ``question``/``answer``
    strings dominate memory but are unused by normalization. We discover
    them from the header and pass ``usecols`` to skip them at parse time.
    """
    head = pd.read_csv(path, nrows=0)
    keep = [c for c in head.columns if c not in _DROP_TEXT_COLS]
    return pd.read_csv(path, usecols=keep, low_memory=False)


def read_spillover_frame(path: str | Path) -> tuple[pd.DataFrame, str]:
    """Read a spillover run.

    Accepts either a ``spillover_results.csv`` file or a run directory
    containing ``cells/*.csv`` (concatenated on the fly, mirroring
    ``plot_spillover_matrix._load_run``). Returns (frame, source_label).
    Free-text columns (question/answer/context/verdict) are dropped at
    parse time to keep memory bounded.
    """
    p = Path(path)
    if p.is_file():
        return _read_csv_slim(p), p.name
    if p.is_dir():
        results_csv = p / "spillover_results.csv"
        if results_csv.is_file():
            return _read_csv_slim(results_csv), "spillover_results.csv"
        cells_dir = p / "cells"
        if not cells_dir.is_dir():
            raise FileNotFoundError(
                f"No spillover_results.csv or cells/ under {p}"
            )
        cell_files = sorted(cells_dir.glob("*.csv"))
        if not cell_files:
            raise FileNotFoundError(f"No cell CSVs found under {cells_dir}")
        frames = [_read_csv_slim(f) for f in cell_files]
        return (
            pd.concat(frames, ignore_index=True),
            f"cells/ ({len(cell_files)} files)",
        )
    raise FileNotFoundError(f"{p} is neither a file nor a directory")


def pole_map_from_trained_models(
    trained_models_path: str | Path,
) -> dict[str, tuple[str, str]]:
    """Map a trait label → (pole, target_metric) from ``trained_models.json``.

    ``trained_models.json`` keys are ``"{method}:{trait_label}"`` with a body
    carrying ``direction`` ("high"/"low") and ``target_metric``. Trait labels
    are method-independent (the same Trait expansion drives icl and grpo), so
    the grpo entries are sufficient to resolve poles for every method.

    Returns ``{trait_label: ("plus"|"minus", target_metric)}`` where
    high → plus, low → minus.
    """
    data = json.loads(Path(trained_models_path).read_text())
    out: dict[str, tuple[str, str]] = {}
    for key, body in data.items():
        if ":" not in key:
            continue
        trait_label = key.split(":", 1)[1]
        direction = (body or {}).get("direction", "high")
        metric = (body or {}).get("target_metric", "score")
        out[trait_label] = ("plus" if direction == "high" else "minus", metric)
    return out


def primary_metric(eval_name: str, evals_root: str | Path) -> str | None:
    """First ``judge_prompts`` key of an orthogonalized eval (its primary
    judge metric). Prefers a ``*_eval_fidelity_filtered.yaml`` when present,
    mirroring ``EvalConfig._find_yaml``."""
    d = Path(evals_root) / eval_name
    if not d.is_dir():
        return None
    cand = list(d.glob(f"{eval_name}_eval_fidelity_filtered.yaml")) or list(
        d.glob(f"{eval_name}_eval.yaml")
    ) or list(d.glob("*_eval.yaml"))
    if not cand:
        return None
    doc = yaml.safe_load(cand[0].read_text())
    questions = doc if isinstance(doc, list) else doc.get("questions", [])
    for q in questions:
        jp = (q or {}).get("judge_prompts") or {}
        if jp:
            return next(iter(jp.keys()))
    return None


def _axis_and_pole(
    trait_label: str, pole_map: dict[str, tuple[str, str]]
) -> tuple[str, str] | None:
    """(axis_eval, pole) for a trait label, or None if unresolvable.

    axis_eval is the source eval (e.g. ``caring-about-aesthetics``); pole is
    "plus"/"minus". Falls back to the ``high``/``low`` suffix conventions
    when a label is absent from ``pole_map``.
    """
    if trait_label in (None, "none", "", "nan"):
        return None
    axis = str(trait_label).split(":", 1)[0]
    if trait_label in pole_map:
        return axis, pole_map[trait_label][0]
    low = str(trait_label).lower()
    if low.endswith((":low", "-low", "_low", ":minus", "-minus")):
        return axis, "minus"
    if low.endswith((":high", "-high", "_high", ":plus", "-plus")):
        return axis, "plus"
    return None


def load_spillover(
    paths: str | Path | Iterable[str | Path],
    *,
    pole_map: dict[str, tuple[str, str]],
    base_method: str = "baseline",
) -> pd.DataFrame:
    """Load spillover run(s) into the unified long schema.

    Every elicited cell becomes a synthetic model
    ``f"{method}:{axis}-{pole}"`` (condition="finetuned"); baseline cells
    collapse to (model=base_model, condition="base"). Multi-sample rows are
    averaged in `load_scores`-style by the caller; here we melt all candidate
    metric columns and let per-eval ``judge_metric`` selection (via the
    anchors spec) keep only the primary metric downstream.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    frames: list[pd.DataFrame] = []
    for p in paths:
        fr, _ = read_spillover_frame(p)
        frames.append(fr)
    raw = pd.concat(frames, ignore_index=True)

    required = {"method", "eval_name", "question_id"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"spillover frame missing columns: {missing}")

    base_model = None
    if "base_model" in raw.columns:
        bm = raw["base_model"].dropna().unique()
        if len(bm) == 1:
            base_model = str(bm[0])
    if base_model is None:
        base_model = "base"

    metric_cols = [
        c for c in raw.columns
        if c not in _NON_METRIC and pd.api.types.is_numeric_dtype(raw[c])
    ]
    if not metric_cols:
        raise ValueError("no numeric judge-metric columns in spillover frame")

    long = raw.melt(
        id_vars=[c for c in ["method", "target_trait", "eval_name",
                              "question_id"] if c in raw.columns],
        value_vars=metric_cols,
        var_name="judge_metric",
        value_name="score",
    ).dropna(subset=["score"])

    long = long.rename(columns={
        "eval_name": "eval", "question_id": "prompt_id",
    })
    long["prompt_id"] = long["prompt_id"].astype(str).map(_parse_prompt_id)

    is_base = long["method"].astype(str) == base_method
    long["model"] = base_model
    long["condition"] = "base"

    elic = long[~is_base].copy()
    if not elic.empty:
        resolved = elic["target_trait"].map(
            lambda t: _axis_and_pole(t, pole_map)
        )
        keep = resolved.notna()
        elic = elic[keep].copy()
        rv = resolved[keep]
        elic["model"] = [
            f"{m}:{ap[0]}-{ap[1]}"
            for m, ap in zip(elic["method"], rv)
        ]
        elic["condition"] = "finetuned"

    base = long[is_base].copy()
    out = pd.concat([base, elic], ignore_index=True)
    out = out[["model", "eval", "prompt_id", "condition",
               "judge_metric", "score"]]
    grouped = (
        out.groupby(["model", "eval", "prompt_id", "condition",
                     "judge_metric"], as_index=False)["score"].mean()
    )
    return grouped


def build_anchor_specs(
    evals: Iterable[str],
    *,
    evals_root: str | Path,
    anchor_method: str,
    metric_overrides: dict[str, str] | None = None,
) -> dict[str, AnchorSpec]:
    """Per-eval AnchorSpec pointing at ``anchor_method``'s diagonal.

    hi/lo models = ``f"{anchor_method}:{E}-plus"`` / ``…-minus"`` (matching
    the synthetic ids `load_spillover` emits). judge_metric is the eval's
    primary metric from the orthogonalized YAML unless overridden.
    """
    overrides = metric_overrides or {}
    specs: dict[str, AnchorSpec] = {}
    for ev in sorted(set(evals)):
        jm = overrides.get(ev) or primary_metric(ev, evals_root) or "score"
        specs[ev] = AnchorSpec(
            eval_name=ev,
            judge_metric=jm,
            hi_model=f"{anchor_method}:{ev}-plus",
            lo_model=f"{anchor_method}:{ev}-minus",
        )
    return specs
