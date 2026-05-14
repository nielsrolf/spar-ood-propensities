"""CLI entry point.

Example:
    python -m shared.normalization.cli --config shared/normalization/configs/run.example.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .anchors import compute_theta, load_anchors_config, resolve_anchors
from .loaders import load_scores
from .logitz import logit_z_normalize
from .transfer import build_transfer_matrix

log = logging.getLogger("normalization")


def _save_matrix(tm, prefix: Path) -> None:
    np.save(prefix.with_suffix(".npy"), tm.matrix)
    np.save(prefix.parent / f"{prefix.name}_se.npy", tm.se)
    (prefix.parent / f"{prefix.name}.labels.json").write_text(
        json.dumps({"sources": tm.sources, "targets": tm.targets}, indent=2)
    )
    df = pd.DataFrame(tm.matrix, index=tm.sources, columns=tm.targets)
    df.to_csv(prefix.with_suffix(".csv"))


def _infer_base_models(df: pd.DataFrame, configured: list[str]) -> set[str]:
    if configured:
        return set(configured)
    # Models that appear under condition != "finetuned" are base models.
    return set(df.loc[df["condition"] != "finetuned", "model"].unique())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "run.log", mode="w"),
            logging.StreamHandler(),
        ],
    )

    inputs = [Path(p) for p in cfg["inputs"]]
    log.info("loading inputs: %s", inputs)
    df = load_scores(inputs)
    log.info("loaded %d rows", len(df))

    base_models = _infer_base_models(df, cfg.get("base_models", []) or [])
    log.info("base models: %s", sorted(base_models))

    anchors = load_anchors_config(cfg["anchors"])
    log.info("anchors configured for %d evals", len(anchors))

    anchors_df = resolve_anchors(df, anchors, base_models)
    if anchors_df.empty:
        log.error("no anchors resolved — check anchors.yaml and condition keys present in the data")
        return

    thresholds = cfg.get("thresholds", {}) or {}
    result = compute_theta(
        df, anchors_df, base_models,
        epsilon_prompt=float(thresholds.get("epsilon_prompt", 5.0)),
        threshold_eval=float(thresholds.get("threshold_eval", 10.0)),
    )

    # Persist tables.
    result.per_prompt.to_parquet(output_dir / "theta_per_prompt.parquet", index=False)
    result.per_eval.to_parquet(output_dir / "theta_per_eval.parquet", index=False)
    result.theta_neutral.to_parquet(output_dir / "theta_neutral_per_eval.parquet", index=False)
    result.anchors_summary.to_csv(output_dir / "eval_anchors_summary.csv", index=False)

    model_regex = cfg["model_regex"]
    tm = build_transfer_matrix(result.per_eval, model_regex=model_regex)
    _save_matrix(tm, output_dir / "transfer_matrix_theta")
    log.info("transfer matrix θ: %dx%d, %d unmatched models",
             len(tm.sources), len(tm.targets), len(tm.unmatched_models))
    if tm.unmatched_models:
        log.info("unmatched models (skipped from matrix): %s", tm.unmatched_models)

    lz = logit_z_normalize(df, anchors_df, base_models, model_regex)
    _save_matrix(lz, output_dir / "transfer_matrix_logitz")
    log.info("logit-z matrix: %dx%d", len(lz.sources), len(lz.targets))

    # Sanity checks.
    flagged = result.log.get("flagged_evals", [])
    excluded = result.log.get("excluded_prompts", 0)
    finite = np.isfinite(tm.matrix)
    pct_in_range = float(np.mean(np.abs(tm.matrix[finite]) < 10.0)) if finite.any() else 1.0
    log.info("flagged_evals=%d excluded_prompts=%d θ_in_range_pct=%.4f",
             len(flagged), excluded, pct_in_range)
    if len(flagged) >= 10:
        log.warning("SANITY: flagged_evals=%d >= 10", len(flagged))
    if pct_in_range < 0.99:
        log.warning("SANITY: only %.2f%% of θ cells satisfy |θ| < 10", pct_in_range * 100)

    summary = {
        "n_inputs": len(inputs),
        "n_rows_loaded": int(len(df)),
        "n_evals_anchored": int(result.anchors_summary.shape[0]),
        "flagged_evals": flagged,
        "excluded_prompts": int(excluded),
        "unmatched_models": tm.unmatched_models,
        "matrix_shape": list(tm.matrix.shape),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("done -> %s", output_dir)


if __name__ == "__main__":
    main()
