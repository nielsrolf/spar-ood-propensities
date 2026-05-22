"""SP cross-elicitation Δ-mean matrix builder.

The Johannes sysprompts scores JSON
(`new_eval_results/scores/sysprompts_scores_<base>.json`) only carries
aggregated metrics (mean/std/min/max), not per-prompt scores. That blocks
the θ / logit-z pipeline, but the report's SP matrices (Fig 1a, Fig 12)
are just Δ-mean vs the base model, which we *can* compute here.

For each (axis-label) sysprompt cell we emit:
    Δ = metrics.mean(cell, eval) - metrics.mean(base, eval)

base means come from the SFT scores JSON's `base` cell.

Outputs per model:
    transfer_matrix_dmean_plus.{csv,npy,labels.json}
    transfer_matrix_dmean_minus.{csv,npy,labels.json}
    transfer_matrix_dmean_signed.csv          # plus minus minus, where both exist
    eval_anchors_summary.csv                  # (axis,pole) → base/sp mean, n_total
    run_summary.json

The signed matrix mirrors the report's "Δ score" rendering: positive cells
are increases relative to base under the high-trait SP, with the low-trait
SP value subtracted off when present.

Usage:
    python -m shared.normalization.sp_delta \\
        --config shared/normalization/configs/run.sp_qwen.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .sysprompts import _LABEL_TO_POLE

log = logging.getLogger("normalization.sp_delta")


def _eval_is_axis(eval_name: str) -> bool:
    """Filter out sub-metric eval names like ``agreeableness:altruism_score``."""
    return ":" not in eval_name


def _base_means(anchor_scores_path: Path) -> dict[str, float]:
    """Mean judge score for each eval on the base model."""
    data = json.loads(anchor_scores_path.read_text())
    out: dict[str, float] = {}
    for eval_name, payload in data["cells"].get("base", {}).items():
        if not _eval_is_axis(eval_name):
            continue
        if not isinstance(payload, dict):
            continue
        m = (payload.get("metrics") or {}).get("mean")
        if m is None:
            continue
        out[eval_name] = float(m)
    return out


def _sp_means(sp_path: Path) -> dict[tuple[str, str], dict[str, float]]:
    """{(axis, pole): {eval_name: mean}} for SP cells with a recognized pole."""
    data = json.loads(sp_path.read_text())
    out: dict[tuple[str, str], dict[str, float]] = {}
    for cell_key, evals in data["cells"].items():
        if "__" not in cell_key:
            continue
        axis, label = cell_key.split("__", 1)
        pole = _LABEL_TO_POLE.get(axis, {}).get(label)
        if pole is None:
            continue
        d = out.setdefault((axis, pole), {})
        for eval_name, payload in evals.items():
            if not _eval_is_axis(eval_name):
                continue
            if not isinstance(payload, dict):
                continue
            m = (payload.get("metrics") or {}).get("mean")
            if m is None:
                continue
            d[eval_name] = float(m)
    return out


def _build_matrix(
    base_mean: dict[str, float],
    sp_mean: dict[tuple[str, str], dict[str, float]],
    pole: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    sources = sorted(axis for (axis, p) in sp_mean.keys() if p == pole)
    targets = sorted(base_mean.keys())
    mat = np.full((len(sources), len(targets)), np.nan, dtype=float)
    for i, axis in enumerate(sources):
        sp_evals = sp_mean.get((axis, pole), {})
        for j, ev in enumerate(targets):
            if ev in sp_evals and ev in base_mean:
                mat[i, j] = sp_evals[ev] - base_mean[ev]
    return pd.DataFrame(mat, index=sources, columns=targets), sources, targets


def _save(df: pd.DataFrame, prefix: Path) -> None:
    df.to_csv(prefix.with_suffix(".csv"))
    np.save(prefix.with_suffix(".npy"), df.values)
    (prefix.parent / f"{prefix.name}.labels.json").write_text(
        json.dumps({"sources": list(df.index), "targets": list(df.columns)},
                   indent=2)
    )


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

    anchor_path = Path(cfg["anchor_scores"])
    sp_paths = cfg["sp_input"]
    if isinstance(sp_paths, str):
        sp_paths = [sp_paths]

    base_mean = _base_means(anchor_path)
    log.info("base means: %d evals", len(base_mean))

    merged: dict[tuple[str, str], dict[str, float]] = {}
    for p in sp_paths:
        for k, v in _sp_means(Path(p)).items():
            merged.setdefault(k, {}).update(v)
    log.info("SP cells: %d (axis,pole) pairs", len(merged))

    df_plus, src_plus, tgt = _build_matrix(base_mean, merged, "plus")
    df_minus, src_minus, _ = _build_matrix(base_mean, merged, "minus")

    _save(df_plus, out_dir / "transfer_matrix_dmean_plus")
    _save(df_minus, out_dir / "transfer_matrix_dmean_minus")

    # Signed = plus - minus where both rows exist (axes present in both).
    common = sorted(set(src_plus) & set(src_minus))
    signed = (df_plus.reindex(common) - df_minus.reindex(common))
    _save(signed, out_dir / "transfer_matrix_dmean_signed")

    # Anchor summary.
    rows = []
    for (axis, pole), evals in sorted(merged.items()):
        for ev, m in sorted(evals.items()):
            rows.append({
                "axis": axis,
                "pole": pole,
                "eval": ev,
                "sp_mean": m,
                "base_mean": base_mean.get(ev),
                "delta": (m - base_mean[ev]) if ev in base_mean else None,
            })
    anchors_df = pd.DataFrame(rows)
    anchors_df.to_csv(out_dir / "eval_anchors_summary.csv", index=False)

    summary = {
        "anchor_scores": str(anchor_path),
        "sp_inputs": [str(p) for p in sp_paths],
        "method": "delta_mean",
        "matrices": {
            "plus":   {"shape": list(df_plus.shape),   "sources": src_plus,   "targets": tgt},
            "minus":  {"shape": list(df_minus.shape),  "sources": src_minus,  "targets": tgt},
            "signed": {"shape": list(signed.shape),    "sources": common,     "targets": tgt},
        },
        "note": "θ/logit-z normalization is blocked: sysprompts JSON only "
                "exposes aggregated metrics, no per-prompt scores. This file "
                "is the Δ-mean fallback used in the report's SP figures.",
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("done -> %s (plus %s, minus %s, signed %s)",
             out_dir, df_plus.shape, df_minus.shape, signed.shape)


if __name__ == "__main__":
    main()
