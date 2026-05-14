"""Test loader format parity: johannes JSON and niels CSV load to equivalent
unified long-format DataFrames given the same conceptual data."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from shared.normalization.loaders import load_scores


def test_johannes_json_loads(tmp_path: Path):
    payload = {
        "base_model": "BASE",
        "cells": {
            "base": {
                "evalA": {
                    "scores": {
                        "evalA_0001__p0__s0": 50.0,
                        "evalA_0001__p0__s1": 60.0,  # averaged with above → 55.0
                        "evalA_0002__p0__s0": 30.0,
                    },
                    "metrics": {}, "meta": {},
                }
            },
            "axisX-plus": {
                "evalA": {
                    "scores": {
                        "evalA_0001__p0__s0": 80.0,
                        "evalA_0002__p0__s0": 40.0,
                    },
                    "metrics": {}, "meta": {},
                }
            },
        },
    }
    fp = tmp_path / "scores_BASE.json"
    fp.write_text(json.dumps(payload))
    df = load_scores(fp)
    # Sample averaging.
    base_p1 = df[(df["model"] == "BASE") & (df["prompt_id"] == "evalA_0001")
                 & (df["condition"] == "base")]["score"].iloc[0]
    assert abs(base_p1 - 55.0) < 1e-9
    # FT row exists.
    ft = df[df["model"] == "axisX-plus"]
    assert len(ft) == 2
    assert (ft["condition"] == "finetuned").all()


def test_niels_csv_loads(tmp_path: Path):
    csv = pd.DataFrame([
        {"model": "BASE", "target_eval": "evalA", "question_id": "evalA_0001",
         "source_label": "none", "metric_x": 50.0},
        {"model": "BASE", "target_eval": "evalA", "question_id": "evalA_0001",
         "source_label": "sp:foo:hi", "metric_x": 90.0},
        {"model": "FT-1", "target_eval": "evalA", "question_id": "evalA_0001",
         "source_label": "n/a", "metric_x": 70.0},
    ])
    fp = tmp_path / "results.csv"
    csv.to_csv(fp, index=False)
    df = load_scores(fp)
    assert set(df["condition"]) >= {"base", "sp:foo:hi", "finetuned"}
    assert set(df["model"]) == {"BASE", "FT-1"}
    assert "metric_x" in set(df["judge_metric"])
