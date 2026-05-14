"""Per-eval anchor resolution and θ computation.

θ(p) = (s(p) − anchor_lo(p)) / (anchor_hi(p) − anchor_lo(p))

Anchors are per-prompt: same prompt_id matched across lo/neutral/hi conditions
on the base model. No clipping. Per-prompt exclusion when (hi − lo) < epsilon.
Per-eval flagging when mean(hi) − mean(lo) < threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass
class AnchorSpec:
    eval_name: str
    judge_metric: str = "score"
    lo: str | None = None
    neutral: str | None = None
    hi: str | None = None
    axis: str | None = None  # johannes-style; allows fallback lo/hi resolution


@dataclass
class ThetaResult:
    per_prompt: pd.DataFrame
    per_eval: pd.DataFrame
    theta_neutral: pd.DataFrame
    anchors_summary: pd.DataFrame
    log: dict[str, Any] = field(default_factory=dict)


def load_anchors_config(path: str | Path) -> dict[str, AnchorSpec]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    out: dict[str, AnchorSpec] = {}
    for eval_name, body in raw.items():
        body = body or {}
        out[eval_name] = AnchorSpec(
            eval_name=eval_name,
            judge_metric=body.get("judge_metric", "score"),
            lo=body.get("lo"),
            neutral=body.get("neutral"),
            hi=body.get("hi"),
            axis=body.get("axis"),
        )
    return out


def _resolve_condition_keys(df: pd.DataFrame, spec: AnchorSpec,
                            base_models: set[str]) -> dict[str, str | None]:
    """Pick concrete condition strings for lo / neutral / hi.

    Falls back to johannes-style `<axis>-minus` / `<axis>-plus` and `base` for
    neutral when explicit names aren't given.
    """
    present = set(df.loc[df["model"].isin(base_models), "condition"].unique())
    lo = spec.lo
    hi = spec.hi
    neutral = spec.neutral

    if lo is None and spec.axis:
        cand = f"{spec.axis}-minus"
        if cand in present:
            lo = cand
    if hi is None and spec.axis:
        cand = f"{spec.axis}-plus"
        if cand in present:
            hi = cand
    if neutral is None:
        for cand in ("base", "baseline-empty", "none"):
            if cand in present:
                neutral = cand
                break

    return {"lo": lo, "neutral": neutral, "hi": hi}


def resolve_anchors(df: pd.DataFrame, anchors: dict[str, AnchorSpec],
                    base_models: set[str]) -> pd.DataFrame:
    """Build a per-(eval, prompt) anchor table.

    Returns columns: eval, prompt_id, judge_metric, anchor_lo, anchor_neutral,
    anchor_hi.
    """
    rows = []
    base_df = df[df["model"].isin(base_models)]
    for eval_name, spec in anchors.items():
        ev = base_df[(base_df["eval"] == eval_name)
                     & (base_df["judge_metric"] == spec.judge_metric)]
        if ev.empty:
            continue
        cond_keys = _resolve_condition_keys(df, spec, base_models)
        pivot = ev.pivot_table(
            index="prompt_id",
            columns="condition",
            values="score",
            aggfunc="mean",
        )
        anchor_lo = pivot[cond_keys["lo"]] if cond_keys["lo"] in pivot.columns else None
        anchor_hi = pivot[cond_keys["hi"]] if cond_keys["hi"] in pivot.columns else None
        anchor_n = (pivot[cond_keys["neutral"]]
                    if cond_keys["neutral"] in pivot.columns else None)
        for pid in pivot.index:
            rows.append({
                "eval": eval_name,
                "prompt_id": pid,
                "judge_metric": spec.judge_metric,
                "anchor_lo": float(anchor_lo[pid]) if anchor_lo is not None and pd.notna(anchor_lo[pid]) else None,
                "anchor_neutral": float(anchor_n[pid]) if anchor_n is not None and pd.notna(anchor_n[pid]) else None,
                "anchor_hi": float(anchor_hi[pid]) if anchor_hi is not None and pd.notna(anchor_hi[pid]) else None,
                "condition_lo": cond_keys["lo"],
                "condition_neutral": cond_keys["neutral"],
                "condition_hi": cond_keys["hi"],
            })
    return pd.DataFrame(rows)


def compute_theta(df: pd.DataFrame,
                  anchors_df: pd.DataFrame,
                  base_models: set[str],
                  epsilon_prompt: float = 5.0,
                  threshold_eval: float = 10.0) -> ThetaResult:
    """Compute per-prompt θ and per-eval aggregates.

    `df` is the unified long-format DataFrame from `load_scores`. `anchors_df`
    is from `resolve_anchors`. Base-model rows are used for anchors and for
    θ_neutral; everything else (fine-tuned models) gets θ computed.
    """
    log: dict[str, Any] = {
        "epsilon_prompt": epsilon_prompt,
        "threshold_eval": threshold_eval,
    }

    # Per-prompt exclusion: drop anchor rows where hi-lo < epsilon or missing.
    anchors = anchors_df.copy()
    anchors["range"] = anchors["anchor_hi"] - anchors["anchor_lo"]
    valid = anchors["range"].notna() & (anchors["range"] >= epsilon_prompt)
    n_dropped_prompts = int((~valid).sum())
    log["excluded_prompts"] = n_dropped_prompts
    valid_anchors = anchors[valid].copy()

    # Eval-level summary + flagging.
    summary = (
        anchors.groupby("eval")
        .agg(
            judge_metric=("judge_metric", "first"),
            mean_lo=("anchor_lo", "mean"),
            mean_hi=("anchor_hi", "mean"),
            n_prompts=("prompt_id", "nunique"),
        )
        .reset_index()
    )
    summary["range"] = summary["mean_hi"] - summary["mean_lo"]
    summary["flagged"] = summary["range"] < threshold_eval
    log["flagged_evals"] = summary.loc[summary["flagged"], "eval"].tolist()

    # Merge anchors into scores; θ for fine-tuned, separately for neutral.
    metric_per_eval = anchors_df.set_index("eval")["judge_metric"].to_dict()
    df_scored = df[df["eval"].isin(anchors_df["eval"].unique())].copy()
    df_scored = df_scored[df_scored.apply(
        lambda r: r["judge_metric"] == metric_per_eval.get(r["eval"], r["judge_metric"]),
        axis=1,
    )]

    merged = df_scored.merge(
        valid_anchors[["eval", "prompt_id", "anchor_lo", "anchor_hi"]],
        on=["eval", "prompt_id"],
        how="inner",
    )
    merged["theta"] = (merged["score"] - merged["anchor_lo"]) / (
        merged["anchor_hi"] - merged["anchor_lo"]
    )

    # Fine-tuned models only for per-eval aggregation (exclude base anchors).
    ft = merged[~merged["model"].isin(base_models)].copy()
    per_prompt = ft[["model", "eval", "prompt_id", "judge_metric",
                     "score", "anchor_lo", "anchor_hi", "theta"]]
    per_prompt = per_prompt.rename(columns={"score": "raw_score"})

    flagged_set = set(summary.loc[summary["flagged"], "eval"])
    grp = ft.groupby(["model", "eval", "judge_metric"])["theta"]
    per_eval = grp.agg(
        mean_theta="mean",
        std_theta="std",
        n_prompts="count",
    ).reset_index()
    per_eval["se_theta"] = per_eval["std_theta"] / per_eval["n_prompts"].pow(0.5)
    per_eval["flagged"] = per_eval["eval"].isin(flagged_set)
    per_eval = per_eval.drop(columns=["std_theta"])

    # θ_neutral: base model under the neutral condition, θ-normalized using the
    # same per-prompt anchors. Pull from anchors_df.anchor_neutral directly.
    neutral_anchors = anchors_df[anchors_df["anchor_neutral"].notna()].copy()
    neutral_anchors = neutral_anchors.merge(
        valid_anchors[["eval", "prompt_id"]],
        on=["eval", "prompt_id"],
        how="inner",
    )
    neutral_anchors["theta"] = (
        neutral_anchors["anchor_neutral"] - neutral_anchors["anchor_lo"]
    ) / (neutral_anchors["anchor_hi"] - neutral_anchors["anchor_lo"])
    if not neutral_anchors.empty:
        gn = neutral_anchors.groupby(["eval", "judge_metric"])["theta"]
        theta_neutral = gn.agg(
            theta_neutral="mean", std="std", n="count"
        ).reset_index()
        theta_neutral["se"] = theta_neutral["std"] / theta_neutral["n"].pow(0.5)
        theta_neutral = theta_neutral.drop(columns=["std"])
    else:
        theta_neutral = pd.DataFrame(
            columns=["eval", "judge_metric", "theta_neutral", "n", "se"]
        )

    return ThetaResult(
        per_prompt=per_prompt.reset_index(drop=True),
        per_eval=per_eval.reset_index(drop=True),
        theta_neutral=theta_neutral.reset_index(drop=True),
        anchors_summary=summary.reset_index(drop=True),
        log=log,
    )
