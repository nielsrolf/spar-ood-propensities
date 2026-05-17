"""Unit tests for θ computation under the diagonal-anchor design.

anchor_hi(E) = <E>-plus fine-tune scored on eval E (per prompt)
anchor_lo(E) = <E>-minus fine-tune scored on eval E (per prompt)
"""
from __future__ import annotations

import pandas as pd

from shared.normalization.anchors import (
    AnchorSpec,
    compute_theta,
    resolve_anchors,
)


def _make_df(rows):
    return pd.DataFrame(rows, columns=[
        "model", "eval", "prompt_id", "condition", "judge_metric", "score",
    ])


def test_theta_basic_recovers_known_values():
    # Eval A diagonal: A-plus=100/90, A-minus=0/10 over p1/p2.
    # A probe model B-plus is scored on A; θ uses A's diagonal span.
    df = _make_df([
        ("A-plus", "A", "p1", "finetuned", "score", 100.0),
        ("A-minus", "A", "p1", "finetuned", "score", 0.0),
        ("A-plus", "A", "p2", "finetuned", "score", 90.0),
        ("A-minus", "A", "p2", "finetuned", "score", 10.0),
        ("BASE", "A", "p1", "base", "score", 50.0),
        ("BASE", "A", "p2", "base", "score", 50.0),
        ("B-plus", "A", "p1", "finetuned", "score", 120.0),  # θ = 1.2
        ("B-plus", "A", "p2", "finetuned", "score", -6.0),    # θ = (-6-10)/80 = -0.2
    ])
    anchors_cfg = {"A": AnchorSpec(eval_name="A")}  # → hi=A-plus, lo=A-minus
    base_models = {"BASE"}
    anchors_df = resolve_anchors(df, anchors_cfg, base_models)
    res = compute_theta(df, anchors_df, base_models,
                        epsilon_prompt=1.0, threshold_eval=1.0)

    pp = res.per_prompt
    probe = pp[pp["model"] == "B-plus"].set_index("prompt_id")["theta"].to_dict()
    assert abs(probe["p1"] - 1.2) < 1e-9
    assert abs(probe["p2"] - (-0.2)) < 1e-9

    pe = res.per_eval
    row = pe[(pe["model"] == "B-plus") & (pe["eval"] == "A")].iloc[0]
    assert abs(row["mean_theta"] - 0.5) < 1e-9
    assert row["n_prompts"] == 2

    # Diagonal property: A-plus on A → θ≈1, A-minus on A → θ≈0.
    diag_hi = pe[(pe["model"] == "A-plus") & (pe["eval"] == "A")].iloc[0]
    diag_lo = pe[(pe["model"] == "A-minus") & (pe["eval"] == "A")].iloc[0]
    assert abs(diag_hi["mean_theta"] - 1.0) < 1e-9
    assert abs(diag_lo["mean_theta"] - 0.0) < 1e-9


def test_prompt_level_exclusion_triggers():
    # p1 diagonal span = 2 (< epsilon=5) → excluded. p2 span = 100 fine.
    df = _make_df([
        ("A-plus", "A", "p1", "finetuned", "score", 51.0),
        ("A-minus", "A", "p1", "finetuned", "score", 49.0),
        ("A-plus", "A", "p2", "finetuned", "score", 100.0),
        ("A-minus", "A", "p2", "finetuned", "score", 0.0),
        ("B-plus", "A", "p1", "finetuned", "score", 50.0),
        ("B-plus", "A", "p2", "finetuned", "score", 25.0),
    ])
    anchors_cfg = {"A": AnchorSpec(eval_name="A")}
    base_models = {"BASE"}
    anchors_df = resolve_anchors(df, anchors_cfg, base_models)
    res = compute_theta(df, anchors_df, base_models,
                        epsilon_prompt=5.0, threshold_eval=1.0)
    pids = set(res.per_prompt["prompt_id"])
    assert pids == {"p2"}, f"expected only p2 to survive, got {pids}"
    assert res.log["excluded_prompts"] == 1
    pe = res.per_eval
    assert pe.loc[pe["eval"] == "A", "n_prompts"].iloc[0] == 1


def test_eval_level_flagging():
    # mean(hi)-mean(lo) = 1.5 < threshold=10 → flagged but not removed.
    df = _make_df([
        ("A-plus", "A", "p1", "finetuned", "score", 51.0),
        ("A-minus", "A", "p1", "finetuned", "score", 49.0),
        ("A-plus", "A", "p2", "finetuned", "score", 50.0),
        ("A-minus", "A", "p2", "finetuned", "score", 49.0),
        ("B-plus", "A", "p1", "finetuned", "score", 50.0),
        ("B-plus", "A", "p2", "finetuned", "score", 49.0),
    ])
    anchors_cfg = {"A": AnchorSpec(eval_name="A")}
    base_models = {"BASE"}
    anchors_df = resolve_anchors(df, anchors_cfg, base_models)
    res = compute_theta(df, anchors_df, base_models,
                        epsilon_prompt=1.0, threshold_eval=10.0)
    assert "A" in res.log["flagged_evals"]
    assert bool(res.per_eval.loc[res.per_eval["eval"] == "A", "flagged"].iloc[0])


def test_missing_pole_yields_no_valid_anchor():
    # Only A-minus exists (no A-plus) → anchor_hi all NaN → eval flagged,
    # zero usable prompts. Mirrors agreeableness in the Qwen run.
    df = _make_df([
        ("A-minus", "A", "p1", "finetuned", "score", 10.0),
        ("A-minus", "A", "p2", "finetuned", "score", 20.0),
        ("B-plus", "A", "p1", "finetuned", "score", 50.0),
        ("B-plus", "A", "p2", "finetuned", "score", 60.0),
    ])
    anchors_cfg = {"A": AnchorSpec(eval_name="A")}
    base_models = {"BASE"}
    anchors_df = resolve_anchors(df, anchors_cfg, base_models)
    assert anchors_df["anchor_hi"].isna().all()
    res = compute_theta(df, anchors_df, base_models,
                        epsilon_prompt=5.0, threshold_eval=10.0)
    assert res.per_prompt.empty
    assert "A" in res.log["flagged_evals"]
