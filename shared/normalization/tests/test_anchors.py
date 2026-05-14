"""Unit tests for θ computation and per-prompt exclusion."""
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
    # Eval A: hi anchors=100, lo=0, so θ = raw/100 for fine-tuned scores.
    # Test θ > 1 (score 120 → 1.2) and θ < 0 (score −20 → −0.2).
    df = _make_df([
        ("BASE", "A", "p1", "lo", "score", 0.0),
        ("BASE", "A", "p1", "hi", "score", 100.0),
        ("BASE", "A", "p1", "base", "score", 50.0),
        ("BASE", "A", "p2", "lo", "score", 10.0),
        ("BASE", "A", "p2", "hi", "score", 90.0),
        ("BASE", "A", "p2", "base", "score", 50.0),
        ("FT-A", "A", "p1", "finetuned", "score", 120.0),  # θ = 1.2
        ("FT-A", "A", "p2", "finetuned", "score", -6.0),   # θ = (-6-10)/80 = -0.2
    ])
    anchors_cfg = {"A": AnchorSpec(eval_name="A", lo="lo", neutral="base", hi="hi")}
    base_models = {"BASE"}
    anchors_df = resolve_anchors(df, anchors_cfg, base_models)
    res = compute_theta(df, anchors_df, base_models,
                        epsilon_prompt=1.0, threshold_eval=1.0)
    thetas = res.per_prompt.set_index("prompt_id")["theta"].to_dict()
    assert abs(thetas["p1"] - 1.2) < 1e-9
    assert abs(thetas["p2"] - (-0.2)) < 1e-9
    # Per-eval aggregate.
    pe = res.per_eval
    row = pe[(pe["model"] == "FT-A") & (pe["eval"] == "A")].iloc[0]
    assert abs(row["mean_theta"] - 0.5) < 1e-9
    assert row["n_prompts"] == 2


def test_prompt_level_exclusion_triggers():
    # p1 has hi-lo = 2 (< epsilon=5) → excluded. p2 fine.
    df = _make_df([
        ("BASE", "A", "p1", "lo", "score", 49.0),
        ("BASE", "A", "p1", "hi", "score", 51.0),
        ("BASE", "A", "p2", "lo", "score", 0.0),
        ("BASE", "A", "p2", "hi", "score", 100.0),
        ("FT-A", "A", "p1", "finetuned", "score", 50.0),
        ("FT-A", "A", "p2", "finetuned", "score", 25.0),
    ])
    anchors_cfg = {"A": AnchorSpec(eval_name="A", lo="lo", hi="hi")}
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
    # mean(hi)-mean(lo) = 2 < threshold=10 → flagged but not removed from outputs.
    df = _make_df([
        ("BASE", "A", "p1", "lo", "score", 49.0),
        ("BASE", "A", "p1", "hi", "score", 51.0),
        ("BASE", "A", "p2", "lo", "score", 48.0),
        ("BASE", "A", "p2", "hi", "score", 50.0),
        ("FT-A", "A", "p1", "finetuned", "score", 50.0),
        ("FT-A", "A", "p2", "finetuned", "score", 49.0),
    ])
    anchors_cfg = {"A": AnchorSpec(eval_name="A", lo="lo", hi="hi")}
    base_models = {"BASE"}
    anchors_df = resolve_anchors(df, anchors_cfg, base_models)
    res = compute_theta(df, anchors_df, base_models,
                        epsilon_prompt=1.0, threshold_eval=10.0)
    assert "A" in res.log["flagged_evals"]
    assert bool(res.per_eval.loc[res.per_eval["eval"] == "A", "flagged"].iloc[0])
