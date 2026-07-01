"""Paper Figure 7 (NIP): single-panel shift-from-baseline plot.

Lily's paper-ready version, kept separate from Owen's original `visualize.py`
(which still produces the per-propensity subplot grid as `experiment1_runs.png`).
This script reuses Owen's data loaders unchanged and only replaces the figure:
one point per propensity showing the shift of the most-deviant of the three
same-recipe NIP runs from the Llama baseline, so small deviations are readable
instead of squished on a 0-100 score axis.

Output: ../visuals/nip_shift.png   (does not overwrite experiment1_runs.png)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import visualize as v  # Owen's loaders + constants, imported unchanged

OUT = v.PLOT_DIR / "nip_shift.png"
SCORE_BAR, LLAMA_BAR = v.SCORE_BAR, v.LLAMA_BAR


def fig_nip_shift(BY, props, out_path):
    def runs_vals(p):
        return [x for x in (v._mean(BY, m, p) for m in v.RUN_MODELS) if x is not None]

    def data_null(p):
        fr = [v._null_frac(BY, f"{m}-data", p) for m in v.RUN_MODELS]
        return 100.0 * sum(fr) / len(fr) if fr else None

    def score_null(p):
        fr = [v._null_frac(BY, m, p) for m in v.RUN_MODELS]
        return sum(fr) / len(fr) if fr else 1.0

    def dev(p):  # most-deviant of the 3 runs minus baseline (signed)
        lb, vals = v._mean(BY, v.BASELINE, p), runs_vals(p)
        return None if (lb is None or not vals) else max((x - lb for x in vals), key=abs)

    def dev_range(p):
        lb, vals = v._mean(BY, v.BASELINE, p), runs_vals(p)
        return (min(vals) - lb, max(vals) - lb)

    usable = [p for p in props if dev(p) is not None]
    usable = sorted(usable, key=lambda p: abs(dev(p)), reverse=True)
    n = len(usable)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(11.0, 0.5 * n), 4.8))
    for xi, p in enumerate(usable):
        d = dev(p)
        lo, hi = dev_range(p)
        ax.plot([xi, xi], [lo, hi], color=SCORE_BAR, lw=3, alpha=0.30, zorder=1)
        hollow = score_null(p) >= v.NULL_WARN_THRESHOLD
        ax.scatter(xi, d, facecolor=("none" if hollow else SCORE_BAR),
                   edgecolor=SCORE_BAR, s=40, linewidth=1.2, zorder=3)
    ax.axhline(0, color=LLAMA_BAR, lw=1.0, ls="--", zorder=2)
    ax.set_ylabel("shift from Llama baseline (judge points)", fontsize=10)
    devs = [abs(dev(p)) for p in usable]
    med, mean_ = float(np.median(devs)), float(np.mean(devs))
    mx = max(usable, key=lambda p: abs(dev(p)))
    ax.text(0.99, 0.95,
            f"max-deviant run: median |shift| = {med:.1f}, mean {mean_:.1f}, max {abs(dev(mx)):.1f} ({mx})",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color="#555")
    ax.set_xticks(x)
    ax.set_xticklabels(usable, rotation=60, ha="right", fontsize=7)
    ax.set_xlim(-0.5, n - 0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    handles = [
        Line2D([0], [0], marker="o", lw=0, markerfacecolor=SCORE_BAR, markeredgecolor=SCORE_BAR,
               label="NIP shift (most-deviant of 3 runs)"),
        Line2D([0], [0], color=SCORE_BAR, lw=3, alpha=0.30, label="run min–max"),
        Line2D([0], [0], marker="o", lw=0, markerfacecolor="none", markeredgecolor=SCORE_BAR,
               label="≥50% null"),
    ]
    ax.legend(handles=handles, fontsize=8, frameon=False, ncol=3, loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return (
        f"Figure 7 ({out_path.name}). For each propensity (sorted by shift magnitude), the shift of "
        "the most-deviant of three same-recipe \"nothing-in-particular\" runs from the Llama baseline "
        "(point), with the min--max range across runs shaded; the dashed line is the baseline. Shifts "
        f"are small (median {med:.1f}, max {abs(dev(mx)):.1f} judge points). The mostly near-ceiling "
        "training-data null rates indicate the generic training responses usually do not exhibit these "
        "propensities."
    )


def main():
    records = v.load_all()
    BY = v.index(records)
    props = v._props(records)
    print(fig_nip_shift(BY, props, OUT))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
