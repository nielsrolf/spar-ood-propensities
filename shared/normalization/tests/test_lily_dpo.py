"""Lily DPO adapter + driver: synthetic-fixture coverage.

Builds a tiny Johannes-style anchor JSON (base + <E>-plus/<E>-minus per-prompt
scores) and a matching Lily DPO export CSV, then exercises the
loader → anchors → θ → transfer → driver path end to end.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from shared.normalization.anchors import compute_theta, resolve_anchors
from shared.normalization.lily_dpo import load_lily_dpo
from shared.normalization.loaders import load_scores

EVALS = ["power-seeking", "cooperation"]


def _anchor_json(fp: Path) -> None:
    """base @50, <E>-plus @90, <E>-minus @10 on 2 prompts/eval."""
    cells: dict = {"base": {}}
    for ev in EVALS:
        stem = ev.replace("-", "_")
        cells["base"][ev] = {"scores": {
            f"{stem}_{i:04d}__p0__s0": 50.0 for i in (1, 2)}}
        cells[f"{ev}-plus"] = {ev: {"scores": {
            f"{stem}_{i:04d}__p0__s0": 90.0 for i in (1, 2)}}}
        cells[f"{ev}-minus"] = {ev: {"scores": {
            f"{stem}_{i:04d}__p0__s0": 10.0 for i in (1, 2)}}}
    fp.write_text(json.dumps({"base_model": "BASE", "cells": cells}))


def _lily_csv(fp: Path) -> None:
    """DPO diagonal @80; one off-diagonal spillover cell (PS trait → coop)."""
    rows: list[dict] = []

    def add(trait, ev, val):
        stem = ev.replace("-", "_")
        for i in (1, 2):
            rows.append({
                "trained_trait": trait, "eval_trait": ev,
                "metric": f"{stem}_score",
                "question_id": f"{stem}_{i:04d}",
                "score": float(val), "elicitation": "tinker_sft",
                "split": "test",
                "model": f"online_dpo_{trait}_v1_qwen3-8b-base",
                "question": "q", "answer": "a",
            })

    for ev in EVALS:
        add(ev, ev, 80)  # diagonal
    add("power-seeking", "cooperation", 70)  # off-diagonal spillover
    pd.DataFrame(rows).to_csv(fp, index=False)


@pytest.fixture()
def fixture(tmp_path: Path) -> dict:
    aj = tmp_path / "scores_BASE.json"
    lc = tmp_path / "online_dpo_all_scores.csv"
    _anchor_json(aj)
    _lily_csv(lc)
    return {"anchor": aj, "lily": lc, "tmp": tmp_path}


def test_load_lily_dpo_schema(fixture):
    df = load_lily_dpo(fixture["lily"])
    assert set(df.columns) == {"model", "eval", "prompt_id", "condition",
                               "judge_metric", "score"}
    assert (df["condition"] == "finetuned").all()
    assert (df["judge_metric"] == "score").all()
    assert "dpo:power-seeking-plus" in set(df["model"])
    # bare item id, suffix-free, joins Johannes' __p0__s0 keys.
    assert "power_seeking_0001" in set(df["prompt_id"])


def test_theta_against_sft_anchors(fixture):
    df_a = load_scores([fixture["anchor"]])
    df_l = load_lily_dpo(fixture["lily"])
    df = pd.concat([df_a, df_l], ignore_index=True)
    base_models = {"BASE"}

    anchors = {ev: __import__(
        "shared.normalization.anchors", fromlist=["AnchorSpec"]
    ).AnchorSpec(eval_name=ev, judge_metric="score") for ev in EVALS}
    adf = resolve_anchors(df, anchors, base_models)
    res = compute_theta(df, adf, base_models)
    pe = res.per_eval.set_index(["model", "eval"])["mean_theta"]

    # DPO diagonal on the SFT scale: (80-10)/(90-10) = 0.875.
    assert pe[("dpo:power-seeking-plus", "power-seeking")] == pytest.approx(
        0.875, abs=1e-9)
    # off-diagonal spillover: (70-10)/(90-10) = 0.75.
    assert pe[("dpo:power-seeking-plus", "cooperation")] == pytest.approx(
        0.75, abs=1e-9)
    # neutral vector = base θ = (50-10)/80 = 0.5.
    tn = res.theta_neutral.set_index("eval")["theta_neutral"]
    assert tn["power-seeking"] == pytest.approx(0.5, abs=1e-9)


def test_driver_end_to_end(fixture, tmp_path: Path):
    from shared.normalization import run_lily_dpo

    anchors_yaml = tmp_path / "anchors.yaml"
    anchors_yaml.write_text(yaml.safe_dump(
        {ev: {"judge_metric": "score"} for ev in EVALS}))
    out_dir = tmp_path / "out"
    cfg = {
        "anchor_scores": str(fixture["anchor"]),
        "lily_input": str(fixture["lily"]),
        "anchors": str(anchors_yaml),
        "output_dir": str(out_dir),
        "pole": "plus",
        "thresholds": {"epsilon_prompt": 5.0, "threshold_eval": 10.0},
    }
    cfg_fp = tmp_path / "cfg.yaml"
    cfg_fp.write_text(yaml.safe_dump(cfg))
    run_lily_dpo.main(["--config", str(cfg_fp)])

    assert (out_dir / "transfer_matrix_theta_plus.csv").exists()
    assert (out_dir / "transfer_matrix_logitz_plus.npy").exists()
    tdf = pd.read_csv(out_dir / "transfer_matrix_theta_plus.csv", index_col=0)
    assert tdf.loc["power-seeking", "power-seeking"] == pytest.approx(
        0.875, abs=1e-9)
    assert tdf.loc["power-seeking", "cooperation"] == pytest.approx(
        0.75, abs=1e-9)

    summary = json.loads((out_dir / "run_summary.json").read_text())
    assert summary["pole"] == "plus"
    assert summary["base_models"] == ["BASE"]
