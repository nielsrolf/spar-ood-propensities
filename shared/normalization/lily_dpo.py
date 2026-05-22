"""Lily online-DPO cross-elicitation → normalization adapter.

Bridges Lily's DPO export
(`lily/propensities/src/dpo/output/exports/online_dpo_<base>_all_scores_*.csv`)
into the unified long schema the SFT normalization pipeline consumes, so the
existing `resolve_anchors` / `compute_theta` / `logit_z_normalize` /
`build_transfer_matrix` machinery produces θ / logit-z transfer matrices for
the DPO battery exactly like the SFT cross-eval matrices.

Export CSV columns (per `lily/propensities/src/dpo/run_dpo_evals.py`):
    trained_trait, eval_trait, metric, question_id, score,
    elicitation, split, model, question, answer

The export is a single-method (online DPO), single-direction battery: for
each trait there is exactly one `online_dpo_<trait>_v1_<base>` model — the
"elicit trait" (high) direction. There is no opposite-pole DPO run and no
no-elicitation/base row. So, like the spillover adapter's plus-only frame,
every DPO model maps to the **plus** pole:

  * model     = ``f"dpo:{eval_trait}-plus"`` (synthetic source id, matching
                the `<E>-plus` anchor-naming convention so the pole regex
                ``^dpo:(?P<source>.+)-plus$`` captures ``source = eval_trait``).
  * condition = "finetuned".
  * eval      = ``eval_trait`` (matches both `anchors.yaml` keys and the
                Johannes eval names verbatim).
  * prompt_id = ``_parse_prompt_id(question_id)`` (Lily emits the bare item
                id, e.g. ``power_seeking_0054``; Johannes emits
                ``power_seeking_0054__p0__s0`` — the parser strips the
                paraphrase/sample suffix so they join per-prompt).
  * judge_metric = "score". Each Lily eval carries exactly one metric, and
                the Johannes anchor JSON only ever emits ``score`` (as does
                `anchors.yaml`), so collapsing Lily's per-eval metric name to
                ``score`` loses no information and lets the same per-prompt
                anchor join apply.

The θ scale (base + ``<E>-plus`` / ``<E>-minus`` per-prompt anchors) is **not**
in this export — it comes from the frozen Johannes Qwen3-8B-Base SFT matrix
(`johannes/cross-elicit/results/scores_Qwen-Qwen3-8B-Base.json`), the same
reference `run_dpo_evals.py` already compares each DPO run against. Loading
that JSON via `loaders._load_johannes_json` and concatenating it with this
adapter's output yields a DPO transfer matrix on the SFT diagonal scale —
directly comparable to the SFT cross-eval matrices.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .loaders import _parse_prompt_id

_REQUIRED = {"trained_trait", "eval_trait", "question_id", "score"}


def load_lily_dpo(
    paths: str | Path | Iterable[str | Path],
    *,
    pole: str = "plus",
) -> pd.DataFrame:
    """Load Lily DPO export(s) into the unified long schema.

    Every row becomes a finetuned cell of the synthetic source model
    ``f"dpo:{eval_trait}-{pole}"``. Multi-sample rows (same model/eval/prompt)
    are averaged, mirroring `load_scores`.

    Returns columns: model, eval, prompt_id, condition, judge_metric, score.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    frames: list[pd.DataFrame] = []
    for p in paths:
        frames.append(pd.read_csv(p))
    raw = pd.concat(frames, ignore_index=True)

    missing = _REQUIRED - set(raw.columns)
    if missing:
        raise ValueError(f"Lily DPO export missing columns: {missing}")

    raw = raw.dropna(subset=["score"]).copy()
    raw["score"] = raw["score"].astype(float)

    out = pd.DataFrame({
        "model": "dpo:" + raw["trained_trait"].astype(str) + f"-{pole}",
        "eval": raw["eval_trait"].astype(str),
        "prompt_id": raw["question_id"].astype(str).map(_parse_prompt_id),
        "condition": "finetuned",
        "judge_metric": "score",
        "score": raw["score"],
    })

    grouped = (
        out.groupby(["model", "eval", "prompt_id", "condition",
                     "judge_metric"], as_index=False)["score"].mean()
    )
    return grouped
