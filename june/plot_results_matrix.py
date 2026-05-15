#!/usr/bin/env python3
"""Heatmap(s) for a binned spillover matrix produced by build_results_matrix.py.

Inputs are the {-2,-1,0,1,2} `results_matrix_model_*_judge_*_{sft,sysprompt}.json`
files (rows of {train, eval, value, mechanism}). No such plotter existed; the
old `finetuned_{minmax,std,diff}_*.png` heatmaps visualise raw per-cell scores
from the (now locally-absent) eval_results tree, not the spillover matrix.

Emits two PNGs next to the input:
  heatmap_<stem>.png            full train x eval spillover grid (incl. the
                                new agreeableness_score axis), diagonal boxed.
  heatmap_predcmp_<stem>.png    predicted | observed | (observed-predicted),
                                on the axes shared with june/prediction_matrix.json.

Usage:
  python3 june/plot_results_matrix.py [results_matrix_*.json ...]
  (default: the Llama SFT matrix)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
DEFAULT = OUT_DIR / (
    "results_matrix_model_meta-llama-Llama-3.1-8B-Instruct_judge_gpt-5.4-mini_sft.json"
)
PRED = OUT_DIR / "prediction_matrix.json"


def _grid(rows, train_axes, eval_axes):
    idx = {(r["train"], r["eval"]): r["value"] for r in rows}
    m = np.full((len(eval_axes), len(train_axes)), np.nan)
    for i, ev in enumerate(eval_axes):
        for j, tr in enumerate(train_axes):
            if (tr, ev) in idx:
                m[i, j] = idx[(tr, ev)]
    return m


def _annot(ax, M, fontsize):
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{int(v):+d}".replace("+0", "0"),
                    ha="center", va="center", fontsize=fontsize,
                    color="white" if abs(v) >= 2 else "black")


def _heat(ax, M, train_axes, eval_axes, title):
    im = ax.imshow(M, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(train_axes)))
    ax.set_xticklabels(train_axes, rotation=90, fontsize=6)
    ax.set_yticks(range(len(eval_axes)))
    ax.set_yticklabels(eval_axes, fontsize=6)
    ax.set_xlabel("train axis", fontsize=8)
    ax.set_ylabel("eval axis", fontsize=8)
    ax.set_title(title, fontsize=10)
    # box the diagonal (train == eval)
    epos = {e: i for i, e in enumerate(eval_axes)}
    for j, tr in enumerate(train_axes):
        if tr in epos:
            i = epos[tr]
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                       edgecolor="black", lw=1.2))
    _annot(ax, M, 5)
    return im


def plot_one(path: Path):
    rows = json.loads(path.read_text())
    train_axes = sorted({r["train"] for r in rows})
    eval_axes = sorted({r["eval"] for r in rows})
    M = _grid(rows, train_axes, eval_axes)
    stem = path.stem.replace("results_matrix_", "")

    # ---- full matrix ----
    fig, ax = plt.subplots(figsize=(0.42 * len(train_axes) + 3,
                                    0.42 * len(eval_axes) + 2))
    im = _heat(ax, M, train_axes, eval_axes,
               f"Spillover (binned, ±2)  —  {stem}")
    cb = fig.colorbar(im, ax=ax, shrink=.6, ticks=[-2, -1, 0, 1, 2])
    cb.set_label("train↑ ⇒ eval Δ", fontsize=8)
    fig.tight_layout()
    out1 = OUT_DIR / f"heatmap_{stem}.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- predicted vs observed (shared axes) ----
    pred = json.loads(PRED.read_text())
    pax_t = {r["train"] for r in pred}
    pax_e = {r["eval"] for r in pred}
    ct = sorted(set(train_axes) & pax_t)
    ce = sorted(set(eval_axes) & pax_e)
    P = _grid(pred, ct, ce)
    O = _grid(rows, ct, ce)
    D = O - P
    fig, axes = plt.subplots(1, 3, figsize=(0.42 * len(ct) * 3 + 6,
                                            0.42 * len(ce) + 2))
    _heat(axes[0], P, ct, ce, "predicted")
    _heat(axes[1], O, ct, ce, "observed (re-judged)")
    imd = axes[2].imshow(D, cmap="PuOr", vmin=-4, vmax=4, aspect="auto")
    axes[2].set_xticks(range(len(ct)))
    axes[2].set_xticklabels(ct, rotation=90, fontsize=6)
    axes[2].set_yticks(range(len(ce)))
    axes[2].set_yticklabels(ce, fontsize=6)
    axes[2].set_title("observed − predicted", fontsize=10)
    fig.colorbar(imd, ax=axes[2], shrink=.6)
    n = np.sum(~np.isnan(P) & ~np.isnan(O))
    agree = np.sum((P == O) & ~np.isnan(P) & ~np.isnan(O))
    fig.suptitle(f"{stem}  —  exact-bin agreement {agree}/{n} "
                 f"({agree / n:.0%})" if n else stem, fontsize=11)
    fig.tight_layout()
    out2 = OUT_DIR / f"heatmap_predcmp_{stem}.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"{path.name}: {len(train_axes)}x{len(eval_axes)} -> "
          f"{out1.name}, {out2.name}"
          + (f"  | pred-cmp {agree}/{n} ({agree / n:.0%})" if n else ""))


def main():
    args = [Path(a) for a in sys.argv[1:]] or [DEFAULT]
    for p in args:
        plot_one(p)


if __name__ == "__main__":
    main()
