"""Tests for transfer matrix assembly."""
from __future__ import annotations

import numpy as np
import pandas as pd

from shared.normalization.transfer import build_transfer_matrix


def test_matrix_assembly_and_regex_parsing():
    per_eval = pd.DataFrame([
        {"model": "ft-evalA-hi", "eval": "evalA", "judge_metric": "score",
         "mean_theta": 0.9, "se_theta": 0.05, "n_prompts": 30, "flagged": False},
        {"model": "ft-evalA-hi", "eval": "evalB", "judge_metric": "score",
         "mean_theta": 0.1, "se_theta": 0.04, "n_prompts": 30, "flagged": False},
        {"model": "ft-evalB-hi", "eval": "evalA", "judge_metric": "score",
         "mean_theta": 0.2, "se_theta": 0.04, "n_prompts": 30, "flagged": False},
        {"model": "ft-evalB-hi", "eval": "evalB", "judge_metric": "score",
         "mean_theta": 0.85, "se_theta": 0.06, "n_prompts": 30, "flagged": False},
        {"model": "unrelated-model", "eval": "evalA", "judge_metric": "score",
         "mean_theta": 0.5, "se_theta": 0.05, "n_prompts": 30, "flagged": False},
    ])
    tm = build_transfer_matrix(
        per_eval, model_regex=r"ft-(?P<source>[a-zA-Z]+)-hi"
    )
    assert tm.sources == ["evalA", "evalB"]
    assert tm.targets == ["evalA", "evalB"]
    # Diagonal dominance on this synthetic example.
    assert np.diag(tm.matrix).mean() > tm.matrix[~np.eye(2, dtype=bool)].mean()
    assert "unrelated-model" in tm.unmatched_models


def test_flagged_evals_dropped_from_matrix():
    per_eval = pd.DataFrame([
        {"model": "ft-evalA-hi", "eval": "evalA", "judge_metric": "score",
         "mean_theta": 0.9, "se_theta": 0.05, "n_prompts": 30, "flagged": False},
        {"model": "ft-evalA-hi", "eval": "evalB", "judge_metric": "score",
         "mean_theta": 0.1, "se_theta": 0.04, "n_prompts": 30, "flagged": True},
    ])
    tm = build_transfer_matrix(
        per_eval, model_regex=r"ft-(?P<source>[a-zA-Z]+)-hi", drop_flagged=True
    )
    assert "evalB" not in tm.targets
