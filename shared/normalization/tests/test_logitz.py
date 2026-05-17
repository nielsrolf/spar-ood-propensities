"""Tests for the logit-z robustness normalization."""
from __future__ import annotations

import numpy as np
import pandas as pd

from shared.normalization.anchors import AnchorSpec, resolve_anchors
from shared.normalization.logitz import empirical_logit, logit_z_normalize


def test_empirical_logit_handles_boundaries():
    out = empirical_logit(np.array([0.0, 50.0, 100.0]))
    assert np.all(np.isfinite(out))
    # Monotonic increasing.
    assert out[0] < out[1] < out[2]


def test_logit_z_basic_pipeline():
    rows = [
        ("A-plus", "A", "p1", "finetuned", "score", 90.0),   # diagonal hi
        ("A-plus", "A", "p2", "finetuned", "score", 85.0),
        ("A-minus", "A", "p1", "finetuned", "score", 10.0),   # diagonal lo
        ("A-minus", "A", "p2", "finetuned", "score", 15.0),
        ("BASE", "A", "p1", "base", "score", 50.0),
        ("BASE", "A", "p2", "base", "score", 50.0),
        ("agreeableness-plus", "A", "p1", "finetuned", "score", 80.0),
        ("agreeableness-plus", "A", "p2", "finetuned", "score", 70.0),
    ]
    df = pd.DataFrame(rows, columns=[
        "model", "eval", "prompt_id", "condition", "judge_metric", "score",
    ])
    anchors_cfg = {"A": AnchorSpec(eval_name="A")}
    base_models = {"BASE"}
    anchors_df = resolve_anchors(df, anchors_cfg, base_models)
    tm = logit_z_normalize(df, anchors_df, base_models,
                           model_regex=r"^(?P<source>[a-z]+)-plus$")
    assert tm.matrix.shape == (1, 1)
    assert np.isfinite(tm.matrix[0, 0])
    assert tm.sources == ["agreeableness"]
