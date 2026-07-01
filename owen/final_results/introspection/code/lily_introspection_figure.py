"""Introspection headline (threshold-free): the model's directional predictions
carry no signal about actual spillover. We plot the distribution of ACTUAL
spillover, split by what the model PREDICTED (up vs down). If introspection
worked, "predicted up" cells would have higher actual spillover than "predicted
down" cells; instead the two distributions overlap almost entirely.

Needs no chance baseline and no accuracy threshold -- it just asks whether the
actual-spillover distribution differs by predicted direction. Pools the three
introspection prompt conditions.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import visualize as v  # noqa: E402

OUT = HERE.parent / "visuals" / "introspection_overlap.png"
CONDITIONS = ["prompt0", "prompt12", "prompt_data"]
# Two well-separated samples from viridis (the palette in the reference paper's
# figures). Viridis is perceptually uniform and colorblind-safe.
UP, DOWN = plt.cm.viridis(0.72), plt.cm.viridis(0.30)


def main():
    slugs, _ = v.load_propensities()
    scores = v.load_scores(v.find_one("scores_*.json"))
    actual = v.build_actual_delta(scores, slugs)

    up, down = [], []
    for e in CONDITIONS:
        rows = v.load_rows(v.find_one(f"{e}_*.jsonl"))
        pred_mean, _, _ = v.build_prediction_matrix(rows, slugs)
        _, pred_disc, _ = v.build_accuracy_matrix(pred_mean, actual)
        valid = ~(np.isnan(pred_disc) | np.isnan(actual))
        up.append(actual[valid & (pred_disc == 1)])
        down.append(actual[valid & (pred_disc == -1)])
    up = np.concatenate(up)
    down = np.concatenate(down)

    print(f"predicted-up  cells: n={up.size:4d}  mean actual spillover = {up.mean():+.2f}")
    print(f"predicted-down cells: n={down.size:4d}  mean actual spillover = {down.mean():+.2f}")
    print(f"separation (up - down means) = {up.mean() - down.mean():+.2f} judge points")

    lo = np.percentile(np.concatenate([up, down]), 1)
    hi = np.percentile(np.concatenate([up, down]), 99)
    bins = np.linspace(lo, hi, 30)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), sharex=True, sharey=True)
    panels = [(axes[0], up, UP, "model predicted ↑"),
              (axes[1], down, DOWN, "model predicted ↓")]
    for ax, data, color, label in panels:
        ax.hist(data, bins=bins, density=True, color=color, alpha=1.0)
        ax.axvline(data.mean(), color="black", lw=1.6, ls="--", zorder=4)
        ax.text(0.03, 0.95, f"{label}\nn={data.size}, mean actual {data.mean():+.1f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=9, color="#2c3e50")
        ax.set_xlabel("Actual spillover (judge-score change from baseline)", fontsize=9.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Density", fontsize=10.5)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
