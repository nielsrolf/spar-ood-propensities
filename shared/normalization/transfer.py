"""Assemble the N×N transfer matrix from per-eval θ aggregates.

M[i, j] = mean_θ on eval j for the model fine-tuned on eval i's hi-sys-prompted
training data. Diagonal and off-diagonal are computed identically.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TransferMatrix:
    matrix: np.ndarray
    se: np.ndarray
    sources: list[str]   # row labels
    targets: list[str]   # column labels
    unmatched_models: list[str]


def parse_source_eval(model_id: str, regex: re.Pattern) -> str | None:
    m = regex.search(model_id)
    if m is None:
        return None
    try:
        return m.group("source")
    except IndexError:
        return None


def build_transfer_matrix(per_eval: pd.DataFrame,
                          model_regex: str,
                          drop_flagged: bool = True) -> TransferMatrix:
    """Build the matrix from a per-eval DataFrame.

    Args:
        per_eval: columns include model, eval, mean_theta, se_theta, flagged.
        model_regex: regex with named group `source` capturing source eval.
        drop_flagged: if True, flagged evals are excluded as columns from the
            matrix (kept as NaN). Rows for which the source eval is flagged are
            also excluded as a row.
    """
    pat = re.compile(model_regex)
    df = per_eval.copy()
    df["source"] = df["model"].map(lambda m: parse_source_eval(m, pat))
    unmatched = sorted(df.loc[df["source"].isna(), "model"].unique().tolist())
    df = df.dropna(subset=["source"])

    # Aggregate over multiple fine-tunes of the same source (e.g. multi-seed):
    # mean of mean_theta, propagate SE by sqrt of mean variance / n_runs.
    agg = (
        df.groupby(["source", "eval"], as_index=False)
        .agg(mean_theta=("mean_theta", "mean"),
             se_theta=("se_theta", "mean"),
             flagged=("flagged", "max"))
    )

    sources = sorted(agg["source"].unique())
    targets = sorted(agg["eval"].unique())
    if drop_flagged:
        flagged_evals = set(
            agg.loc[agg["flagged"].astype(bool), "eval"].unique()
        )
        targets = [t for t in targets if t not in flagged_evals]
        sources = [s for s in sources if s not in flagged_evals]

    matrix = np.full((len(sources), len(targets)), np.nan)
    se = np.full_like(matrix, np.nan)
    s_idx = {s: i for i, s in enumerate(sources)}
    t_idx = {t: j for j, t in enumerate(targets)}
    for _, r in agg.iterrows():
        i, j = s_idx.get(r["source"]), t_idx.get(r["eval"])
        if i is None or j is None:
            continue
        matrix[i, j] = r["mean_theta"]
        se[i, j] = r["se_theta"]

    return TransferMatrix(
        matrix=matrix, se=se,
        sources=sources, targets=targets,
        unmatched_models=unmatched,
    )
