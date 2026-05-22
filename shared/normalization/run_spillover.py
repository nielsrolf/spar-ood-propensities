"""Spillover normalization driver.

Mirrors `shared.normalization.cli` but consumes a cross-method spillover run
and emits θ / logit-z transfer matrices per *labeled* (source_method,
anchor_method) combination — so you get the SFT-style cross-eval matrices for
grpo and icl, on both a shared (grpo-diagonal) scale and each method's own
diagonal scale, plus the baseline neutral vector.

    python -m shared.normalization.run_spillover \
        --config shared/normalization/configs/run.spillover_qwen3_4b.yaml

Config schema (YAML):
    spillover_input: <file-or-dir>            # or a list
    trained_models:  <trained_models.json>    # for trait→pole/metric map
    evals_root:      shared/evals_orthogonalized
    output_dir:      shared/normalization/results/spillover_qwen3_4b
    base_method:     baseline                  # the no-elicitation condition
    thresholds: { epsilon_prompt: 5.0, threshold_eval: 10.0 }
    runs:
      - { label: grpo__anchorsGrpo,     source_method: grpo, anchor_method: grpo }
      - { label: icl__anchorsGrpo,      source_method: icl,  anchor_method: grpo }
      - { label: icl__anchorsIcl,       source_method: icl,  anchor_method: icl  }
      - { label: baseline__anchorsGrpo, source_method: baseline, anchor_method: grpo }
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml

from .anchors import compute_theta, resolve_anchors
from .cli import _save_matrix
from .logitz import logit_z_normalize
from .spillover import (
    build_anchor_specs,
    load_spillover,
    pole_map_from_trained_models,
)
from .transfer import build_transfer_matrix

log = logging.getLogger("normalization.spillover")

POLE_REGEX = {
    "plus": r"^{m}:(?P<source>.+)-plus$",
    "minus": r"^{m}:(?P<source>.+)-minus$",
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    out_root = Path(cfg["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(out_root / "run.log", mode="w"),
            logging.StreamHandler(),
        ],
    )

    base_method = cfg.get("base_method", "baseline")
    evals_root = cfg["evals_root"]
    pole_map = pole_map_from_trained_models(cfg["trained_models"])
    log.info("pole map: %d trait labels", len(pole_map))

    spill_inputs = cfg["spillover_input"]
    df = load_spillover(spill_inputs, pole_map=pole_map,
                        base_method=base_method)
    log.info("loaded %d unified rows; %d evals; models: %d",
             len(df), df["eval"].nunique(), df["model"].nunique())

    base_models = set(df.loc[df["condition"] != "finetuned", "model"].unique())
    evals_present = sorted(df["eval"].unique())
    thresholds = cfg.get("thresholds", {}) or {}
    eps = float(thresholds.get("epsilon_prompt", 5.0))
    thr = float(thresholds.get("threshold_eval", 10.0))
    metric_overrides = cfg.get("metric_overrides", {}) or {}

    # compute_theta depends only on (df, anchors): cache per anchor_method.
    theta_cache: dict[str, object] = {}

    def _theta_for(anchor_method: str):
        if anchor_method not in theta_cache:
            specs = build_anchor_specs(
                evals_present, evals_root=evals_root,
                anchor_method=anchor_method,
                metric_overrides=metric_overrides,
            )
            anchors_df = resolve_anchors(df, specs, base_models)
            if anchors_df.empty:
                log.error(
                    "no anchors resolved for anchor_method=%s — is the %s "
                    "diagonal present in the spillover run?",
                    anchor_method, anchor_method,
                )
            res = compute_theta(df, anchors_df, base_models,
                                epsilon_prompt=eps, threshold_eval=thr)
            theta_cache[anchor_method] = (anchors_df, res)
        return theta_cache[anchor_method]

    summary: dict = {
        "n_unified_rows": int(len(df)),
        "n_evals": len(evals_present),
        "base_models": sorted(base_models),
        "runs": {},
    }

    for run in cfg["runs"]:
        label = run["label"]
        src = run["source_method"]
        anch = run["anchor_method"]
        rdir = out_root / label
        rdir.mkdir(parents=True, exist_ok=True)
        anchors_df, res = _theta_for(anch)

        # Persist θ tables for this anchor scale (shared across runs reusing
        # the same anchor_method, re-written per label for self-containment).
        res.per_prompt.to_parquet(rdir / "theta_per_prompt.parquet", index=False)
        res.per_eval.to_parquet(rdir / "theta_per_eval.parquet", index=False)
        res.theta_neutral.to_parquet(
            rdir / "theta_neutral_per_eval.parquet", index=False)
        res.anchors_summary.to_csv(
            rdir / "eval_anchors_summary.csv", index=False)

        run_info: dict = {"source_method": src, "anchor_method": anch}

        if src == base_method:
            # Baseline has no source axis (target_trait == none): the only
            # meaningful normalized quantity is the neutral vector — base
            # model θ per eval under the shared anchor scale.
            res.theta_neutral.to_csv(
                rdir / "theta_neutral.csv", index=False)
            run_info["kind"] = "neutral_vector"
            run_info["n_evals"] = int(res.theta_neutral.shape[0])
            summary["runs"][label] = run_info
            log.info("[%s] neutral vector: %d evals (no transfer matrix for "
                     "the no-elicitation condition)", label,
                     res.theta_neutral.shape[0])
            continue

        run_info["kind"] = "transfer_matrix"
        run_info["poles"] = {}
        for pole, rx in POLE_REGEX.items():
            regex = rx.format(m=src)
            tm = build_transfer_matrix(res.per_eval, model_regex=regex)
            _save_matrix(tm, rdir / f"transfer_matrix_theta_{pole}")
            lz = logit_z_normalize(df, anchors_df, base_models, regex)
            _save_matrix(lz, rdir / f"transfer_matrix_logitz_{pole}")
            finite = np.isfinite(tm.matrix)
            pct = (float(np.mean(np.abs(tm.matrix[finite]) < 10.0))
                   if finite.any() else 1.0)
            run_info["poles"][pole] = {
                "shape": list(tm.matrix.shape),
                "n_sources": len(tm.sources),
                "n_targets": len(tm.targets),
                "unmatched_models": len(tm.unmatched_models),
                "theta_in_range_pct": pct,
            }
            log.info("[%s] θ/%s: %dx%d (%.1f%% |θ|<10)", label, pole,
                     len(tm.sources), len(tm.targets), pct * 100)
            if pct < 0.99:
                log.warning("SANITY [%s/%s]: only %.2f%% of θ cells |θ|<10",
                            label, pole, pct * 100)
        summary["runs"][label] = run_info

    (out_root / "run_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("done -> %s", out_root)


if __name__ == "__main__":
    main()
