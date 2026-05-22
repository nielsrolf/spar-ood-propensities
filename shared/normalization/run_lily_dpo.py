"""Lily online-DPO normalization driver.

Mirrors `shared.normalization.cli` but the input is split: the θ scale (base
model + ``<E>-plus`` / ``<E>-minus`` per-prompt anchors) comes from the frozen
Johannes SFT matrix, while the source axes come from Lily's DPO export. The
result is an SFT-style θ / logit-z transfer matrix for the DPO battery, on the
SFT diagonal scale — directly comparable to the SFT cross-eval matrices.

    python -m shared.normalization.run_lily_dpo \
        --config shared/normalization/configs/run.lily_dpo_qwen3_8b.yaml

Config schema (YAML):
    anchor_scores:  johannes/cross-elicit/results/scores_<base>.json
    lily_input:     <export csv>            # or a list
    anchors:        shared/normalization/configs/anchors.yaml
    output_dir:     shared/normalization/results/lily_dpo_qwen3_8b
    pole:           plus                    # DPO direction (single-direction)
    thresholds: { epsilon_prompt: 5.0, threshold_eval: 10.0 }
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml

from .anchors import compute_theta, load_anchors_config, resolve_anchors
from .cli import _save_matrix
from .lily_dpo import load_lily_dpo
from .loaders import load_scores
from .logitz import logit_z_normalize
from .transfer import build_transfer_matrix

log = logging.getLogger("normalization.lily_dpo")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "run.log", mode="w"),
            logging.StreamHandler(),
        ],
    )

    pole = cfg.get("pole", "plus")

    # Anchor scale + base model from the frozen Johannes SFT JSON.
    anchor_scores = Path(cfg["anchor_scores"])
    df_anchor = load_scores([anchor_scores])
    base_models = set(
        df_anchor.loc[df_anchor["condition"] != "finetuned", "model"].unique()
    )
    log.info("anchor JSON: %d rows, base models: %s",
             len(df_anchor), sorted(base_models))

    # Source axes from Lily's DPO export.
    lily_inputs = cfg["lily_input"]
    df_lily = load_lily_dpo(lily_inputs, pole=pole)
    log.info("lily DPO: %d rows, %d source models, %d evals",
             len(df_lily), df_lily["model"].nunique(),
             df_lily["eval"].nunique())

    import pandas as pd
    df = pd.concat([df_anchor, df_lily], ignore_index=True)

    anchors = load_anchors_config(cfg["anchors"])
    anchors_df = resolve_anchors(df, anchors, base_models)
    if anchors_df.empty:
        log.error("no anchors resolved — check that the <E>-plus / <E>-minus "
                  "diagonal fine-tunes are present in the anchor JSON")
        return
    log.info("anchors resolved for %d evals",
             anchors_df["eval"].nunique())

    thresholds = cfg.get("thresholds", {}) or {}
    res = compute_theta(
        df, anchors_df, base_models,
        epsilon_prompt=float(thresholds.get("epsilon_prompt", 5.0)),
        threshold_eval=float(thresholds.get("threshold_eval", 10.0)),
    )

    res.per_prompt.to_parquet(out_dir / "theta_per_prompt.parquet", index=False)
    res.per_eval.to_parquet(out_dir / "theta_per_eval.parquet", index=False)
    res.theta_neutral.to_parquet(
        out_dir / "theta_neutral_per_eval.parquet", index=False)
    res.anchors_summary.to_csv(
        out_dir / "eval_anchors_summary.csv", index=False)

    # Single-direction DPO → only the configured pole's transfer matrix.
    regex = rf"^dpo:(?P<source>.+)-{pole}$"
    tm = build_transfer_matrix(res.per_eval, model_regex=regex)
    _save_matrix(tm, out_dir / f"transfer_matrix_theta_{pole}")
    lz = logit_z_normalize(df, anchors_df, base_models, regex)
    _save_matrix(lz, out_dir / f"transfer_matrix_logitz_{pole}")

    finite = np.isfinite(tm.matrix)
    pct = (float(np.mean(np.abs(tm.matrix[finite]) < 10.0))
           if finite.any() else 1.0)
    log.info("θ/%s: %dx%d (%d unmatched, %.1f%% |θ|<10)", pole,
             len(tm.sources), len(tm.targets),
             len(tm.unmatched_models), pct * 100)
    if pct < 0.99:
        log.warning("SANITY: only %.2f%% of θ cells satisfy |θ| < 10",
                    pct * 100)

    flagged = res.log.get("flagged_evals", [])
    excluded = res.log.get("excluded_prompts", 0)
    log.info("flagged_evals=%d excluded_prompts=%d", len(flagged), excluded)

    summary = {
        "anchor_scores": str(anchor_scores),
        "n_anchor_rows": int(len(df_anchor)),
        "n_lily_rows": int(len(df_lily)),
        "pole": pole,
        "base_models": sorted(base_models),
        "n_evals_anchored": int(res.anchors_summary.shape[0]),
        "flagged_evals": flagged,
        "excluded_prompts": int(excluded),
        "matrix": {
            "shape": list(tm.matrix.shape),
            "sources": tm.sources,
            "targets": tm.targets,
            "unmatched_models": tm.unmatched_models,
            "theta_in_range_pct": pct,
        },
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("done -> %s", out_dir)


if __name__ == "__main__":
    main()
