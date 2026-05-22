"""Owen NIP + guide-model cross-eval matrix builder.

Aggregates Owen's nothing-in-particular (§4.7) and guide-model (§4.8)
experiments into Δ-from-Llama-base matrices on the canonical 29-trait axis.

The data layout (see ``owen/final_results/nothing-in-particular/data``):
    base_model_results/scores_meta-llama-Llama-3.1-8B-Instruct.json
        Johannes-format file. The `base` cell carries the Llama-3.1-8B-
        Instruct base mean per eval — the canonical baseline.
    output/<config>_on_<eval>.json
        18 training configs (nip1..3, nipshort, niplong, nipgpt, nipnemotron,
        nipqwen, nipllama, each with and without the __example_response
        variant). Each carries a `mean_scores` dict with the eval's metrics.
    base_model_results/{gpt-4.1-mini, qwen3-8b}/<base>_on_<eval>.json
        Per-guide raw means (gpt-4.1-mini, qwen3-8b). Nemotron is fetched
        on-demand by Owen's viz from HF — not aggregated here.

For each (eval, config) cell we collapse `mean_scores` to one number using
Owen's exact rule (PROPENSITY_PRIMARY_METRIC → sole metric → mean of facets,
mirroring `owen/.../code/visualize.py::aggregate_metric`).

Outputs under output_dir:
    transfer_matrix_dmean_nip.{csv,npy,labels.json}
        Δ = config_mean - llama_base_mean, rows = 18 NIP configs.
    transfer_matrix_dmean_guides.{csv,npy,labels.json}
        Δ = guide_mean - llama_base_mean, rows = {gpt-4.1-mini, qwen3-8b}.
    eval_anchors_summary.csv     (llama_base mean per eval)
    run_summary.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger("normalization.owen_nip")

# Mirrors owen/.../code/visualize.py — keep in sync.
_PRIMARY_METRIC: dict[str, str] = {
    "honest-humble": "honesty_humility_score",
    "ethical-framework-deontological": "deontological_alignment",
    "ethical-framework-utilitarian": "utilitarian_alignment",
    "ethical-framework-virtue-ethics": "virtue_ethics_alignment",
}
_SKIP_PROPENSITIES = {"reward-hacking"}


def _collapse_mean_scores(propensity: str, mean_scores: dict) -> float | None:
    """Owen's aggregate_metric rule applied to a `mean_scores` dict."""
    if not mean_scores:
        return None
    pref = _PRIMARY_METRIC.get(propensity)
    if pref and isinstance(mean_scores.get(pref), (int, float)):
        return float(mean_scores[pref])
    vals = [float(v) for v in mean_scores.values()
            if isinstance(v, (int, float))]
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    return sum(vals) / len(vals)


def _load_llama_base(path: Path) -> dict[str, float]:
    """Per-eval Llama-base means from Owen's scores_<llama>.json."""
    data = json.loads(path.read_text())
    out: dict[str, float] = {}
    base_cell = data.get("cells", {}).get("base", {})
    for prop, cell in base_cell.items():
        if ":" in prop:
            continue
        if prop in _SKIP_PROPENSITIES:
            continue
        if not isinstance(cell, dict):
            continue
        m = (cell.get("metrics") or {}).get("mean")
        if m is None:
            continue
        out[prop] = float(m)
    return out


def _scan_owen_output_dir(
    directory: Path,
    file_label_re: re.Pattern[str],
) -> dict[tuple[str, str], float]:
    """{(config_label, eval): collapsed_mean} for *_on_*.json files matching the re.

    `file_label_re` must have one group capturing the config label
    (e.g. ``nip1``, ``nip1__example_response``, ``gpt-4.1-mini``).
    """
    out: dict[tuple[str, str], float] = {}
    for path in sorted(directory.glob("*_on_*.json")):
        m = file_label_re.match(path.stem)
        if not m:
            continue
        config = m.group(1)
        eval_name = path.stem[len(config) + len("_on_"):]
        if eval_name in _SKIP_PROPENSITIES:
            continue
        data = json.loads(path.read_text())
        mean_scores = data.get("mean_scores") or {}
        mean = _collapse_mean_scores(eval_name, mean_scores)
        if mean is None:
            continue
        out[(config, eval_name)] = mean
    return out


def _build_matrix(
    cells: dict[tuple[str, str], float],
    base: dict[str, float],
    sources: list[str],
    targets: list[str],
) -> pd.DataFrame:
    mat = np.full((len(sources), len(targets)), np.nan, dtype=float)
    for i, src in enumerate(sources):
        for j, tgt in enumerate(targets):
            if (src, tgt) in cells and tgt in base:
                mat[i, j] = cells[(src, tgt)] - base[tgt]
    return pd.DataFrame(mat, index=sources, columns=targets)


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

    base_path = Path(cfg["llama_base_scores"])
    nip_dir = Path(cfg["nip_output_dir"])
    guide_dirs = {k: Path(v) for k, v in (cfg.get("guide_dirs") or {}).items()}

    base = _load_llama_base(base_path)
    log.info("Llama base: %d evals", len(base))

    # NIP configs: nip1, nip2, nip3, nipshort, niplong, nipgpt, nipnemotron,
    # nipqwen, nipllama, and each with __example_response. Match label as
    # ``nip<word>`` plus optional ``__example_response`` suffix.
    nip_re = re.compile(r"^(nip[a-z0-9]+(?:__example_response)?)_on_")
    nip_cells = _scan_owen_output_dir(nip_dir, nip_re)
    nip_configs = sorted({c for c, _ in nip_cells.keys()})
    log.info("NIP: %d configs, %d cells", len(nip_configs), len(nip_cells))

    # Guide model raw evals (gpt-4.1-mini, qwen3-8b, ...).
    guide_cells: dict[tuple[str, str], float] = {}
    for label, gdir in guide_dirs.items():
        if not gdir.exists():
            log.warning("guide dir missing: %s", gdir)
            continue
        # Files are named <label>_on_<eval>.json (label may contain dots).
        gre = re.compile(rf"^({re.escape(label)})_on_")
        guide_cells.update(_scan_owen_output_dir(gdir, gre))
    guide_configs = sorted({c for c, _ in guide_cells.keys()})
    log.info("Guides: %d configs, %d cells", len(guide_configs), len(guide_cells))

    targets = sorted(base.keys())

    df_nip = _build_matrix(nip_cells, base, nip_configs, targets)
    df_guides = _build_matrix(guide_cells, base, guide_configs, targets)

    _save(df_nip, out_dir / "transfer_matrix_dmean_nip")
    _save(df_guides, out_dir / "transfer_matrix_dmean_guides")

    pd.DataFrame(
        [{"eval": k, "llama_base_mean": v} for k, v in sorted(base.items())]
    ).to_csv(out_dir / "eval_anchors_summary.csv", index=False)

    summary = {
        "llama_base_scores": str(base_path),
        "nip_output_dir": str(nip_dir),
        "guide_dirs": {k: str(v) for k, v in guide_dirs.items()},
        "method": "delta_mean",
        "matrices": {
            "nip":    {"shape": list(df_nip.shape),    "sources": nip_configs,    "targets": targets},
            "guides": {"shape": list(df_guides.shape), "sources": guide_configs, "targets": targets},
        },
        "note": "Owen's NIP/guide JSONs only carry mean_scores (per-eval). "
                "θ/logit-z normalization is not applicable; this Δ-mean "
                "matrix is what the report's Fig 5–7 are built on.",
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("done -> %s (nip %s, guides %s)",
             out_dir, df_nip.shape, df_guides.shape)


if __name__ == "__main__":
    main()
