"""Cross-model factor-recovery bar chart (consensus-anchored).

For each of the six canonical personality factors, one bar per model whose
height is Tucker's |phi| between that model's EFA factor and the *consensus*
of the OTHER models' matched factors (leave-one-out). No model is privileged —
unlike anchoring to a single reference, every bar is a genuine value. Tall bars
for all three = the factor is shared structure; a short bar = that model
departs from the shared solution on that factor.

All three models are on the uniform per-prompt (_pkl) basis; same EFA + factor
identification as cross_model_efa_panel.py.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from cross_model_efa_panel import (
    fit_efa, best_match, tucker_phi, MODELS, FACTOR_TEMPLATE, ANCHORS, N_FACTORS,
)

HERE = Path(__file__).parent
OUT = (HERE / ".." / "activation_pca" / "figures" / "paper"
       / "fig_cross_model_replication_bars.png").resolve()

MATCH_THRESHOLD = 0.70
MODEL_COLORS = {"llama_pkl": "#4C72B0", "qwen_pkl": "#DD8452", "nemotron_pkl": "#55A868"}
MODEL_SHORT  = {"llama_pkl": "Llama-3.1-8B", "qwen_pkl": "Qwen3-8B-Base",
                "nemotron_pkl": "Nemotron-3-120B"}


def _factor_columns(fits):
    """canon -> {model: efa_column} with 1:1 identity.

    A reference model fixes factor identity (canonical factor -> its column via
    anchor overlap); every other model's columns are Hungarian-matched to the
    reference (1:1, sign-flexible) — the same identity logic as the rest of the
    pipeline. The reference is used only for *identity*, never for the φ value.
    """
    ref = MODELS[0]
    ref_metrics, L_ref = fits[ref][0], fits[ref][1]
    # 1:1 assignment of the 6 canonical factors to the 6 reference columns,
    # Hungarian on anchor-trait overlap (prevents two canonicals grabbing the
    # same column, e.g. Empathy and Self-preservation).
    overlap = np.zeros((len(FACTOR_TEMPLATE), N_FACTORS))
    for ci, (canon, _) in enumerate(FACTOR_TEMPLATE):
        anc = ANCHORS[canon]
        for j in range(N_FACTORS):
            col = L_ref[:, j]
            top = {ref_metrics[i] for i in np.argsort(np.abs(col))[::-1][:5] if abs(col[i]) >= 0.45}
            overlap[ci, j] = len(top & anc)
    rows, colsidx = linear_sum_assignment(-overlap)
    canon_to_ref = {FACTOR_TEMPLATE[r][0]: int(c) for r, c in zip(rows, colsidx)}

    cols = {}
    for canon, _ in FACTOR_TEMPLATE:
        cols[canon] = {}
        for m in MODELS:
            if m == ref:
                cols[canon][m] = canon_to_ref[canon]
            else:
                pairs = best_match(L_ref, ref_metrics, fits[m][1], fits[m][0])  # (ref,m,phi)
                ref_to_m = {a: b for a, b, _ in pairs}
                cols[canon][m] = ref_to_m.get(canon_to_ref[canon])
    return cols


def compute_phi_matrix() -> np.ndarray:
    """6 factors x 3 models matrix of leave-one-out consensus Tucker |phi|."""
    fits = {m: fit_efa(m) for m in MODELS}
    base_traits = fits[MODELS[0]][0]
    for m in MODELS:
        assert fits[m][0] == base_traits, f"trait order differs for {m}"
    cols = _factor_columns(fits)

    def vec(m, canon):
        col = cols[canon][m]
        if col is None:
            return None
        v = fits[m][1][:, col].astype(float).copy()
        idx = [i for i, t in enumerate(fits[m][0]) if t in ANCHORS[canon]]
        if idx and v[idx].mean() < 0:
            v = -v
        return v

    phi = np.full((len(FACTOR_TEMPLATE), len(MODELS)), np.nan)
    for ri, (canon, _) in enumerate(FACTOR_TEMPLATE):
        vecs = {m: vec(m, canon) for m in MODELS}
        for ci, m in enumerate(MODELS):
            others = [vecs[o] for o in MODELS if o != m and vecs[o] is not None]
            if vecs[m] is None or not others:
                continue
            phi[ri, ci] = abs(tucker_phi(vecs[m], np.mean(others, axis=0)))
    return phi


def main() -> None:
    phi = compute_phi_matrix()
    n_factors, n_models = len(FACTOR_TEMPLATE), len(MODELS)
    factor_names = [c for c, _ in FACTOR_TEMPLATE]
    x = np.arange(n_factors)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 4.6))

    # Tucker congruence interpretation bands (Lorenzo-Seva & ten Berge 2006).
    for lo, hi, col, lab in [(0.70, 0.85, "#f4f4f2", "fair"),
                             (0.85, 0.95, "#e7efe6", "good"),
                             (0.95, 1.001, "#d6e6d4", "identical")]:
        ax.axhspan(lo, hi, color=col, zorder=0)
        ax.text(1.008, (lo + hi) / 2, lab, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=7.5, color="#9aa39a",
                style="italic", clip_on=False)
    for b in (0.70, 0.85, 0.95):
        ax.axhline(b, color="#d2d2d2", lw=0.6, zorder=1)

    for mi, m in enumerate(MODELS):
        vals = phi[:, mi]
        heights = np.where(np.isnan(vals), 0.0, vals)
        offsets = x + (mi - (n_models - 1) / 2) * width
        ax.bar(offsets, heights, width, label=MODEL_SHORT[m],
               color=MODEL_COLORS[m], edgecolor="white", linewidth=0.6, zorder=3)
        for off, v in zip(offsets, vals):
            if np.isnan(v):
                ax.text(off, 0.03, "n/a", ha="center", va="bottom",
                        fontsize=7, color="#999999", rotation=90, zorder=4)
            else:
                ax.text(off, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=7.5, color="#333333", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(factor_names, fontsize=10)
    for tick, (_, color) in zip(ax.get_xticklabels(), FACTOR_TEMPLATE):
        tick.set_color(color)
    ax.set_ylabel("Factor recovery vs. consensus  (Tucker's |φ|)", fontsize=10.5)
    ax.set_ylim(0, 1.06)
    ax.set_xlim(-0.5, n_factors - 0.5)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3,
              fontsize=8.5, frameon=False, handlelength=1.4, columnspacing=1.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=0)

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT}")

    print(f"\n  {'Factor':<20}" + "".join(f"{m:>16}" for m in MODELS))
    print("  " + "-" * 68)
    for ri, name in enumerate(factor_names):
        row = f"  {name:<20}"
        for ci in range(n_models):
            v = phi[ri, ci]
            row += f"{'n/a':>16}" if np.isnan(v) else f"{v:>16.3f}"
        print(row)


if __name__ == "__main__":
    main()
