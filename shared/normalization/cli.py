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
        log.error("no anchors resolved — check anchors.yaml: the <eval>-plus / "
                  "<eval>-minus diagonal fine-tunes must be present in the input")
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

    # Two matrices per normalization, split by source pole. transfer.py /
    # logitz.py are regex-driven: a pole regex captures the axis as `source`
    # and naturally drops the opposite pole's models (unmatched).
    default_poles = {
        "plus": r"(?P<source>.+)-plus$",
        "minus": r"(?P<source>.+)-minus$",
    }
    pole_regexes = cfg.get("pole_regexes") or default_poles

    matrices: dict[str, dict] = {}
    for pole, regex in pole_regexes.items():
        tm = build_transfer_matrix(result.per_eval, model_regex=regex)
        _save_matrix(tm, output_dir / f"transfer_matrix_theta_{pole}")
        log.info("θ matrix [%s]: %dx%d, %d unmatched",
                 pole, len(tm.sources), len(tm.targets), len(tm.unmatched_models))

        lz = logit_z_normalize(df, anchors_df, base_models, regex)
        _save_matrix(lz, output_dir / f"transfer_matrix_logitz_{pole}")
        log.info("logit-z matrix [%s]: %dx%d",
                 pole, len(lz.sources), len(lz.targets))

        finite = np.isfinite(tm.matrix)
        pct = (float(np.mean(np.abs(tm.matrix[finite]) < 10.0))
               if finite.any() else 1.0)
        matrices[pole] = {
            "shape": list(tm.matrix.shape),
            "unmatched_models": tm.unmatched_models,
            "theta_in_range_pct": pct,
        }
        if pct < 0.99:
            log.warning("SANITY [%s]: only %.2f%% of θ cells satisfy |θ| < 10",
                        pole, pct * 100)

    # Sanity checks (shared across poles).
    flagged = result.log.get("flagged_evals", [])
    excluded = result.log.get("excluded_prompts", 0)
    log.info("flagged_evals=%d excluded_prompts=%d", len(flagged), excluded)
    if len(flagged) >= 10:
        log.warning("SANITY: flagged_evals=%d >= 10 (evals missing a pole "
                    "fine-tune or with a degenerate plus/minus span)",
                    len(flagged))

    summary = {
        "n_inputs": len(inputs),
        "n_rows_loaded": int(len(df)),
        "n_evals_anchored": int(result.anchors_summary.shape[0]),
        "flagged_evals": flagged,
        "excluded_prompts": int(excluded),
        "matrices": matrices,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("done -> %s", output_dir)


if __name__ == "__main__":
    main()
