"""Per-factor activation recovery across layers (§4.6).

For each behavioral EFA factor (from the §4.5 Qwen spillover EFA), measures how
strongly it re-emerges as an axis of activation-space variation at each layer:
the max |Pearson r| between the factor's trait-loading vector and any of the top
activation PCs of the fine-tune directions. A permutation null marks the chance
level. Answers: which behavioral factors imprint on activation geometry, and at
what depth — replacing the PC1/PC2 biplot.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str((HERE.parent / "clustering").resolve()))

from make_paper_figures import load_dir, ALL_TRAITS          # noqa: E402
from cross_model_efa_panel import fit_efa, ANCHORS, FACTOR_TEMPLATE, N_FACTORS  # noqa: E402

LAYERS = [18, 27, 31]
TOPK = 6           # activation PCs considered
N_PERM = 3000
LAYER_COLOR = {18: "#9ecae1", 27: "#4292c6", 31: "#084594"}
OUT = HERE / "figures" / "paper" / "fig_factor_activation_recovery.png"


def act_to_behav(name: str) -> str:
    stem = name[:-len("_qwen_v1")] if name.endswith("_qwen_v1") else name
    return stem.replace("_", "-")


def canon_columns(metrics, L):
    """1:1 Hungarian assignment of canonical factors -> EFA columns by anchor overlap."""
    overlap = np.zeros((len(FACTOR_TEMPLATE), N_FACTORS))
    for ci, (canon, _) in enumerate(FACTOR_TEMPLATE):
        anc = ANCHORS[canon]
        for j in range(N_FACTORS):
            col = L[:, j]
            top = {metrics[i] for i in np.argsort(np.abs(col))[::-1][:5] if abs(col[i]) >= 0.45}
            overlap[ci, j] = len(top & anc)
    r, c = linear_sum_assignment(-overlap)
    return {FACTOR_TEMPLATE[i][0]: int(j) for i, j in zip(r, c)}


def max_abs_r(fvec, score_mat):
    return max(abs(np.corrcoef(fvec, score_mat[:, k])[0, 1]) for k in range(score_mat.shape[1]))


def main():
    b_metrics, L_efa, *_ = fit_efa("qwen_pkl")
    canon_col = canon_columns(b_metrics, L_efa)

    bset = set(b_metrics)
    common_act = [t for t in ALL_TRAITS if act_to_behav(t) in bset]
    bidx = [b_metrics.index(act_to_behav(t)) for t in common_act]
    print(f"{len(common_act)} traits common to activation directions and behavioral EFA")

    factors = [c for c, _ in FACTOR_TEMPLATE]
    rec = {f: {} for f in factors}
    rng = np.random.default_rng(0)
    null95 = None
    for layer in LAYERS:
        X = np.stack([load_dir(layer, t) for t in common_act])
        scores = PCA().fit_transform(X)[:, :TOPK]
        for f in factors:
            fvec = L_efa[bidx, canon_col[f]]
            rec[f][layer] = max_abs_r(fvec, scores)
        if layer == LAYERS[-1]:
            fvec0 = L_efa[bidx, canon_col[factors[0]]]
            perm = [max_abs_r(rng.permutation(fvec0), scores) for _ in range(N_PERM)]
            null95 = float(np.percentile(perm, 95))

    # console table
    print(f"\n  {'Factor':<20}" + "".join(f"{('L'+str(l)):>8}" for l in LAYERS))
    for f in factors:
        print(f"  {f:<20}" + "".join(f"{rec[f][l]:>8.2f}" for l in LAYERS))
    print(f"  null (95th pctile): {null95:.2f}")

    # ── grouped bars: x = factor, one bar per layer ──────────────────────────
    x = np.arange(len(factors))
    width = 0.8 / len(LAYERS)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for li, layer in enumerate(LAYERS):
        offs = x + (li - (len(LAYERS) - 1) / 2) * width
        ax.bar(offs, [rec[f][layer] for f in factors], width,
               color=LAYER_COLOR[layer], edgecolor="white", linewidth=0.5,
               label=f"layer {layer}", zorder=3)

    ax.axhline(null95, ls="--", lw=1.0, color="#888888", zorder=2)
    ax.text(len(factors) - 0.45, null95 + 0.012, "permutation null (95%)",
            ha="right", va="bottom", fontsize=8, color="#888888")

    ax.set_xticks(x)
    ax.set_xticklabels(factors, fontsize=10)
    for tick, (_, color) in zip(ax.get_xticklabels(), FACTOR_TEMPLATE):
        tick.set_color(color)
    ax.set_ylabel("Activation recovery\n(max |r| with activation PCs)", fontsize=10.5)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.5, len(factors) - 0.5)
    ax.legend(loc="upper right", fontsize=9, frameon=False, ncol=3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=0)
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
