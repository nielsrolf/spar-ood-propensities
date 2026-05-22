"""Build a self-contained HTML viewer for the spillover normalization runs.

Reuses the SFT comparison viewer's template/JS verbatim (same θ / θ−base /
logit-z controls, hover, SE, axis-align) but points the panels at the labeled
spillover runs under ``results/spillover_<model>/<label>/`` instead of the
three SFT base models. Each panel is one (source_method, anchor_method)
combination; the baseline run contributes only the neutral vector (no matrix)
and is shown as "no matrix" while still powering the θ−base diff view.

    python3 shared/normalization/build_spillover_viewer.py \
        [--run-dir shared/normalization/results/spillover_qwen3_4b]

Output: <run-dir>/compare_spillover.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .build_viewer import HTML, MATRICES

ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT / "results" / "spillover_qwen3_4b"

# Human labels for the run dirs the driver writes (run_summary.json::runs).
PANEL_LABELS = {
    "grpo__anchorsGrpo": ("grpo", "anchors: grpo diagonal (shared scale)"),
    "icl__anchorsGrpo": ("icl", "anchors: grpo diagonal (shared scale)"),
    "icl__anchorsIcl": ("icl", "anchors: icl diagonal (own scale)"),
    "baseline__anchorsGrpo": ("baseline", "neutral vector only"),
}


def _load_one(run_dir: Path, label: str, mt: str) -> dict | None:
    base = run_dir / label / f"transfer_matrix_{mt}"
    if not base.with_suffix(".npy").exists():
        return None
    m = np.load(base.with_suffix(".npy"))
    se = np.load(base.parent / f"{base.name}_se.npy")
    labels = json.loads((base.parent / f"{base.name}.labels.json").read_text())
    return {
        "sources": labels["sources"],
        "targets": labels["targets"],
        "values": [[None if not np.isfinite(v) else round(float(v), 5)
                    for v in row] for row in m],
        "se": [[None if not np.isfinite(v) else round(float(v), 5)
                for v in row] for row in se],
    }


def collect(run_dir: Path, panels: list[str]) -> tuple[dict, list[list[str]]]:
    summary_path = run_dir / "run_summary.json"
    rsum = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    n_rows = int(rsum.get("n_unified_rows", 0))

    data: dict = {}
    summaries: dict = {}
    neutral: dict = {}
    models_js: list[list[str]] = []
    import pandas as pd

    for label in panels:
        if not (run_dir / label).is_dir():
            continue
        name, meta = PANEL_LABELS.get(label, (label, ""))
        models_js.append([label, name, meta])
        data[label] = {}
        for mt, _, _ in MATRICES:
            d = _load_one(run_dir, label, mt)
            if d is not None:
                data[label][mt] = d
        # The JS reads s.n_rows_loaded / s.flagged_evals — synthesize them
        # from the spillover summary so the shared template works unchanged.
        ri = (rsum.get("runs", {}) or {}).get(label, {})
        flagged: list[str] = []
        asum = run_dir / label / "eval_anchors_summary.csv"
        if asum.exists():
            adf = pd.read_csv(asum)
            if "flagged" in adf.columns:
                flagged = adf.loc[adf["flagged"].astype(bool),
                                  "eval"].astype(str).tolist()
        summaries[label] = {
            "n_rows_loaded": n_rows,
            "flagged_evals": flagged,
            "kind": ri.get("kind", ""),
        }
        npq = run_dir / label / "theta_neutral_per_eval.parquet"
        if npq.exists():
            tn = pd.read_parquet(npq)
            neutral[label] = {
                str(r["eval"]): round(float(r["theta_neutral"]), 5)
                for _, r in tn.iterrows()
                if np.isfinite(r["theta_neutral"])
            }
        else:
            neutral[label] = {}
    return {"data": data, "summaries": summaries, "neutral": neutral}, models_js


_SPILLOVER_NOTE = """<div class="note">
  <b>Data source.</b> LLM-judge scores from the cross-method spillover harness
  (<code>ben/propensities/experiments/cross_method_spillover.py</code>),
  per-prompt, averaged over samples. θ / logit-z are computed against the
  diagonal anchors selected per panel (see each panel's subtitle).
  <br><br>
  <b>Elicitation method.</b> Each panel is one elicitation method
  (<code>grpo</code> = on-policy RL per target trait; <code>icl</code> =
  few-shot prefix; <code>baseline</code> = no elicitation). Rows = source
  trait axis, columns = the propensity eval battery. <b>anchors: grpo
  diagonal</b> puts every panel on one shared scale (grpo's own
  plus/minus diagonal sets each eval's 0→1 span), so icl and grpo spillover
  are directly comparable; <b>anchors: icl diagonal</b> rescales icl against
  its own diagonal. <code>baseline</code> has no source axis (no
  elicitation), so it contributes only the neutral vector — shown as
  "no matrix" but still the θ−base reference for the other panels.
  <br><br>
  <b>θ</b> is the raw pole position (0 = minus pole, 1 = plus pole).
  <b>θ − base</b> subtracts the base model's neutral θ on that target eval →
  <i>spillover</i> (blue = suppressed below base, warm = amplified).
  <b>logit-z</b> is a z-score around the panel reference mean. θ is defined
  only on dual-pole target axes; degenerate / missing-pole axes are flagged
  and dropped as θ targets but kept for logit-z. Empty grey = axis absent.
  Hover any cell for value ± SE.
"""


def _swap_note(html: str) -> str:
    start = html.index('<div class="note">')
    end = html.index('<div class="tip"', start)
    return html[:start] + _SPILLOVER_NOTE + "\n" + html[end:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = ap.parse_args()

    panels = list(PANEL_LABELS.keys())
    payload, models_js = collect(args.run_dir, panels)
    html = (_swap_note(HTML)
            .replace("__PAYLOAD__", json.dumps(payload))
            .replace("__MODELS__", json.dumps(models_js))
            .replace("Normalization transfer matrices — model comparison",
                     "Spillover normalization — method comparison")
            .replace("the organism's own axis.",
                     "the source trait's own eval."))
    out = args.run_dir / "compare_spillover.html"
    out.write_text(html)
    n = sum(len(v) for v in payload["data"].values())
    print(f"wrote {out}  ({n} matrices across {len(payload['data'])} panels, "
          f"{len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
