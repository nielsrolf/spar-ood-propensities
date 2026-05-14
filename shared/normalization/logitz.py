"""Robustness normalization: per-eval logit-then-z against a reference panel.

Reference panel = base+lo ∪ base+neutral ∪ base+hi ∪ all fine-tuned models,
per eval, per judge_metric. Empirical-logit boundary adjustment is applied so
scores at 0 or 100 remain finite.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .transfer import TransferMatrix, parse_source_eval


def empirical_logit(x: np.ndarray, scale: float = 100.0) -> np.ndarray:
    """Empirical-logit on judge scores in [0, scale].

    Maps x → (x + 0.5) / (scale + 1) ∈ (0, 1) then takes log(p/(1-p)).
    """
    arr = np.asarray(x, dtype=float)
    p = (arr + 0.5) / (scale + 1.0)
    return np.log(p / (1.0 - p))


def logit_z_normalize(df: pd.DataFrame,
                      anchors_df: pd.DataFrame,
                      base_models: set[str],
                      model_regex: str) -> TransferMatrix:
    """Build a transfer matrix in logit-z units.

    For each (eval, judge_metric):
        1. Take all scores from base+lo, base+neutral, base+hi, and all
           non-base models (the reference panel).
        2. Apply empirical-logit transform.
        3. Compute panel mean μ and std σ.
        4. z(s) = (logit(s) − μ) / σ.
        5. Average z per (model, eval); arrange into matrix.
    """
    pat = re.compile(model_regex)
    metric_per_eval = anchors_df.set_index("eval")["judge_metric"].to_dict()

    scored = df[df["eval"].isin(anchors_df["eval"].unique())].copy()
    scored = scored[scored.apply(
        lambda r: r["judge_metric"] == metric_per_eval.get(r["eval"], r["judge_metric"]),
        axis=1,
    )].copy()
    scored["logit"] = empirical_logit(scored["score"].values)

    stats = (
        scored.groupby(["eval", "judge_metric"])["logit"]
        .agg(["mean", "std"])
        .reset_index()
    )
    scored = scored.merge(stats, on=["eval", "judge_metric"], how="left")
    scored["z"] = np.where(
        scored["std"] > 0,
        (scored["logit"] - scored["mean"]) / scored["std"],
        np.nan,
    )

    ft = scored[~scored["model"].isin(base_models)].copy()
    ft["source"] = ft["model"].map(lambda m: parse_source_eval(m, pat))
    unmatched = sorted(ft.loc[ft["source"].isna(), "model"].unique().tolist())
    ft = ft.dropna(subset=["source"])

    agg = (
        ft.groupby(["source", "eval"])
        .agg(mean_z=("z", "mean"), std_z=("z", "std"), n=("z", "count"))
        .reset_index()
    )
    agg["se_z"] = agg["std_z"] / agg["n"].pow(0.5)

    sources = sorted(agg["source"].unique())
    targets = sorted(agg["eval"].unique())
    matrix = np.full((len(sources), len(targets)), np.nan)
    se = np.full_like(matrix, np.nan)
    s_idx = {s: i for i, s in enumerate(sources)}
    t_idx = {t: j for j, t in enumerate(targets)}
    for _, r in agg.iterrows():
        i, j = s_idx.get(r["source"]), t_idx.get(r["eval"])
        if i is None or j is None:
            continue
        matrix[i, j] = r["mean_z"]
        se[i, j] = r["se_z"]

    return TransferMatrix(
        matrix=matrix, se=se,
        sources=sources, targets=targets,
        unmatched_models=unmatched,
    )
