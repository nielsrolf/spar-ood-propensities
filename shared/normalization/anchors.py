"""Per-eval anchor resolution and θ computation.

θ(p) = (s(p) − anchor_lo(p)) / (anchor_hi(p) − anchor_lo(p))

Anchors are the **fine-tune diagonals**, matched per-prompt:

    anchor_hi(E, p) = score of the <E>-plus  fine-tune on prompt p of eval E
    anchor_lo(E, p) = score of the <E>-minus fine-tune on prompt p of eval E
    anchor_neutral(E, p) = base model score on prompt p of eval E

i.e. the dynamic range of eval E is the span its own plus/minus model organisms
open up, not the base model's lo/hi system-prompt span. θ for any fine-tune M on
eval E is then (s − anchor_lo) / (anchor_hi − anchor_lo) — by construction the
<E>-plus diagonal sits at θ≈1 and the <E>-minus diagonal at θ≈0.

No clipping. Per-prompt exclusion when (hi − lo) < epsilon. Per-eval flagging
when mean(hi) − mean(lo) < threshold (e.g. an eval whose plus/minus fine-tunes
barely move it, or one missing a pole entirely).
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
    # Model ids whose per-prompt scores on this eval form the hi / lo anchors.
    # Default convention: f"{axis or eval_name}-plus" / f"{...}-minus".
    hi_model: str | None = None
    lo_model: str | None = None
    axis: str | None = None  # stem when fine-tune naming != eval_name


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
            hi_model=body.get("hi_model"),
            lo_model=body.get("lo_model"),
            axis=body.get("axis"),
        )
    return out


def anchor_models(spec: AnchorSpec) -> tuple[str, str]:
    """Resolve the (hi_model, lo_model) ids for a spec by convention."""
    stem = spec.axis or spec.eval_name
    hi = spec.hi_model or f"{stem}-plus"
    lo = spec.lo_model or f"{stem}-minus"
    return hi, lo


def resolve_anchors(df: pd.DataFrame, anchors: dict[str, AnchorSpec],
                    base_models: set[str]) -> pd.DataFrame:
    """Build a per-(eval, prompt) anchor table from the fine-tune diagonals.

    For eval E: anchor_hi/anchor_lo are the per-prompt scores of the <E>-plus /
    <E>-minus fine-tune on eval E; anchor_neutral is the base model on eval E.
    A prompt_id row is emitted for the union of prompts seen across the three.

    Returns columns: eval, prompt_id, judge_metric, anchor_lo, anchor_neutral,
    anchor_hi, model_hi, model_lo.
    """
    rows = []
    for eval_name, spec in anchors.items():
        ev = df[(df["eval"] == eval_name)
                & (df["judge_metric"] == spec.judge_metric)]
        if ev.empty:
            continue
        hi_model, lo_model = anchor_models(spec)

        def _by_prompt(frame: pd.DataFrame) -> pd.Series:
            return frame.groupby("prompt_id")["score"].mean()

        anchor_hi = _by_prompt(ev[ev["model"] == hi_model])
        anchor_lo = _by_prompt(ev[ev["model"] == lo_model])
        anchor_n = _by_prompt(ev[ev["model"].isin(base_models)])

        pids = sorted(set(anchor_hi.index)
                      | set(anchor_lo.index)
                      | set(anchor_n.index))
        for pid in pids:
            hv, lv, nv = anchor_hi.get(pid), anchor_lo.get(pid), anchor_n.get(pid)
            rows.append({
                "eval": eval_name,
                "prompt_id": pid,
                "judge_metric": spec.judge_metric,
                "anchor_lo": float(lv) if pd.notna(lv) else None,
                "anchor_neutral": float(nv) if pd.notna(nv) else None,
                "anchor_hi": float(hv) if pd.notna(hv) else None,
                "model_hi": hi_model,
                "model_lo": lo_model,
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
    # NaN range (a missing pole → undefined span) must flag, not slip through:
    # ~(range >= threshold) catches NaN, whereas (range < threshold) does not.
    summary["flagged"] = ~(summary["range"] >= threshold_eval)
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
