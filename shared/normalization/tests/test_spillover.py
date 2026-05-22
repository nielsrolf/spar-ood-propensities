"""Spillover adapter + driver: synthetic-fixture coverage.

No real spillover data is committed (results/ is gitignored); this builds a
tiny two-axis spillover frame so the loader → anchors → θ → transfer →
viewer path is exercised end to end.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from shared.normalization.anchors import compute_theta, resolve_anchors
from shared.normalization.spillover import (
    build_anchor_specs,
    load_spillover,
    pole_map_from_trained_models,
)

EVALS = {"E1": "m1", "E2": "m2"}


def _write_evals_root(root: Path) -> None:
    for ev, metric in EVALS.items():
        d = root / ev
        d.mkdir(parents=True)
        (d / f"{ev}_eval.yaml").write_text(yaml.safe_dump([
            {"id": f"{ev}_q", "paraphrases": ["?"],
             "judge_prompts": {metric: "rate 0-100"}, "meta": {"split": "test"}}
        ]))


def _write_trained_models(fp: Path) -> None:
    tm = {}
    for ev, metric in EVALS.items():
        tm[f"grpo:{ev}:hi"] = {"method": "grpo", "direction": "high",
                               "target_metric": metric}
        tm[f"grpo:{ev}:lo"] = {"method": "grpo", "direction": "low",
                               "target_metric": metric}
    fp.write_text(json.dumps(tm))


def _spillover_rows() -> list[dict]:
    """grpo diagonal at 90/10, icl diagonal at 80/20, baseline neutral 50,
    plus one off-diagonal spillover cell per method (E1:hi → E2)."""
    rows: list[dict] = []

    def add(method, trait, ev, metric, val):
        for pid in (1, 2):  # ≥2 prompts so SE is defined
            rows.append({
                "base_model": "BASE", "method": method,
                "target_trait": trait, "target_model": "M",
                "eval_name": ev, "question_id": f"{ev}_{pid:04d}",
                "sample_idx": 0, "m1": np.nan, "m2": np.nan,
                metric: float(val),
            })

    for ev, metric in EVALS.items():
        add("baseline", "none", ev, metric, 50)
        add("grpo", f"{ev}:hi", ev, metric, 90)   # → grpo:{ev}-plus  diag
        add("grpo", f"{ev}:lo", ev, metric, 10)   # → grpo:{ev}-minus diag
        add("icl", f"{ev}:hi", ev, metric, 80)    # → icl:{ev}-plus   diag
        add("icl", f"{ev}:lo", ev, metric, 20)
    # off-diagonal spillover: E1:hi organism scored on E2.
    add("grpo", "E1:hi", "E2", "m2", 70)
    add("icl", "E1:hi", "E2", "m2", 60)
    return rows


@pytest.fixture()
def fixture(tmp_path: Path) -> dict:
    evals_root = tmp_path / "evals"
    _write_evals_root(evals_root)
    tm_fp = tmp_path / "trained_models.json"
    _write_trained_models(tm_fp)
    csv_fp = tmp_path / "spillover_results.csv"
    pd.DataFrame(_spillover_rows()).to_csv(csv_fp, index=False)
    return {"evals_root": evals_root, "trained_models": tm_fp, "csv": csv_fp,
            "tmp": tmp_path}


def test_pole_map(fixture):
    pm = pole_map_from_trained_models(fixture["trained_models"])
    assert pm["E1:hi"] == ("plus", "m1")
    assert pm["E2:lo"] == ("minus", "m2")


def test_load_spillover_schema(fixture):
    pm = pole_map_from_trained_models(fixture["trained_models"])
    df = load_spillover(fixture["csv"], pole_map=pm)
    assert set(df.columns) == {"model", "eval", "prompt_id", "condition",
                               "judge_metric", "score"}
    # baseline collapses to the base model / base condition.
    base = df[df["condition"] == "base"]
    assert set(base["model"]) == {"BASE"}
    # elicited cells become synthetic per-(method, axis, pole) ids.
    assert "grpo:E1-plus" in set(df["model"])
    assert "icl:E1-plus" in set(df["model"])
    assert (df[df["model"] == "grpo:E1-plus"]["condition"] == "finetuned").all()
    # only the eval's primary metric survives (NaN melt rows dropped).
    assert set(df[df["eval"] == "E1"]["judge_metric"]) == {"m1"}


def test_theta_diagonal_and_shared_scale(fixture):
    pm = pole_map_from_trained_models(fixture["trained_models"])
    df = load_spillover(fixture["csv"], pole_map=pm)
    specs = build_anchor_specs(["E1", "E2"], evals_root=fixture["evals_root"],
                               anchor_method="grpo")
    assert specs["E1"].hi_model == "grpo:E1-plus"
    assert specs["E1"].judge_metric == "m1"

    base_models = {"BASE"}
    adf = resolve_anchors(df, specs, base_models)
    res = compute_theta(df, adf, base_models)
    pe = res.per_eval.set_index(["model", "eval"])["mean_theta"]

    # grpo diagonal sits at θ≈1 / θ≈0 by construction.
    assert pe[("grpo:E1-plus", "E1")] == pytest.approx(1.0, abs=1e-9)
    assert pe[("grpo:E1-minus", "E1")] == pytest.approx(0.0, abs=1e-9)
    # icl, on the *shared grpo scale*: (80-10)/(90-10) = 0.875.
    assert pe[("icl:E1-plus", "E1")] == pytest.approx(0.875, abs=1e-9)
    # off-diagonal spillover: grpo E1:hi on E2 = (70-10)/(90-10) = 0.75.
    assert pe[("grpo:E1-plus", "E2")] == pytest.approx(0.75, abs=1e-9)
    # neutral vector = base model θ per eval = (50-10)/80 = 0.5.
    tn = res.theta_neutral.set_index("eval")["theta_neutral"]
    assert tn["E1"] == pytest.approx(0.5, abs=1e-9)


def test_own_scale_differs(fixture):
    pm = pole_map_from_trained_models(fixture["trained_models"])
    df = load_spillover(fixture["csv"], pole_map=pm)
    base_models = {"BASE"}
    specs_icl = build_anchor_specs(["E1", "E2"],
                                   evals_root=fixture["evals_root"],
                                   anchor_method="icl")
    res = compute_theta(df, resolve_anchors(df, specs_icl, base_models),
                        base_models)
    pe = res.per_eval.set_index(["model", "eval"])["mean_theta"]
    # On icl's OWN diagonal the icl plus pole is θ≈1 (vs 0.875 on grpo scale).
    assert pe[("icl:E1-plus", "E1")] == pytest.approx(1.0, abs=1e-9)


def test_driver_and_viewer_end_to_end(fixture, tmp_path: Path):
    from shared.normalization import build_spillover_viewer, run_spillover

    out_dir = tmp_path / "out"
    cfg = {
        "spillover_input": str(fixture["csv"]),
        "trained_models": str(fixture["trained_models"]),
        "evals_root": str(fixture["evals_root"]),
        "output_dir": str(out_dir),
        "base_method": "baseline",
        "thresholds": {"epsilon_prompt": 5.0, "threshold_eval": 10.0},
        "runs": [
            {"label": "grpo__anchorsGrpo", "source_method": "grpo",
             "anchor_method": "grpo"},
            {"label": "icl__anchorsGrpo", "source_method": "icl",
             "anchor_method": "grpo"},
            {"label": "baseline__anchorsGrpo", "source_method": "baseline",
             "anchor_method": "grpo"},
        ],
    }
    cfg_fp = tmp_path / "cfg.yaml"
    cfg_fp.write_text(yaml.safe_dump(cfg))
    run_spillover.main(["--config", str(cfg_fp)])

    grpo = out_dir / "grpo__anchorsGrpo"
    assert (grpo / "transfer_matrix_theta_plus.csv").exists()
    assert (grpo / "transfer_matrix_logitz_plus.npy").exists()
    # diagonal cell ≈ 1.0 in the saved θ-plus matrix.
    tdf = pd.read_csv(grpo / "transfer_matrix_theta_plus.csv", index_col=0)
    assert tdf.loc["E1", "E1"] == pytest.approx(1.0, abs=1e-9)
    # baseline run emits only the neutral vector.
    assert (out_dir / "baseline__anchorsGrpo" / "theta_neutral.csv").exists()
    assert not (out_dir / "baseline__anchorsGrpo"
                / "transfer_matrix_theta_plus.csv").exists()

    summary = json.loads((out_dir / "run_summary.json").read_text())
    assert summary["runs"]["icl__anchorsGrpo"]["kind"] == "transfer_matrix"
    assert summary["runs"]["baseline__anchorsGrpo"]["kind"] == "neutral_vector"

    import sys
    argv = sys.argv
    sys.argv = ["build_spillover_viewer", "--run-dir", str(out_dir)]
    try:
        build_spillover_viewer.main()
    finally:
        sys.argv = argv
    html = (out_dir / "compare_spillover.html").read_text()
    assert "Spillover normalization" in html
    assert "grpo__anchorsGrpo" in html
