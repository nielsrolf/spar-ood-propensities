"""No-bipolarity headline figure (§4.6).

Histogram of all C(24,2)=276 pairwise cosine similarities between the fine-tune
direction vectors (FT - base) at the latest layer, against a random-direction
null. Every observed cosine is positive: behaviorally opposite traits are not
antipodal in activation space.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from make_paper_figures import load_dir, ALL_TRAITS  # noqa: E402

LAYER = 31
OUT = HERE / "figures" / "paper" / "fig_cosine_histogram.png"


def main():
    X = np.stack([load_dir(LAYER, t) for t in ALL_TRAITS])      # 24 x 4096
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    iu = np.triu_indices(len(X), k=1)
    obs = (Xn @ Xn.T)[iu]                                        # 276 cosines

    # Random-direction null: cosines between random Gaussian vectors, same dim.
    d = X.shape[1]
    rng = np.random.default_rng(0)
    null = []
    for _ in range(400):
        R = rng.standard_normal((len(X), d))
        Rn = R / np.linalg.norm(R, axis=1, keepdims=True)
        null.append((Rn @ Rn.T)[iu])
    null = np.concatenate(null)
    null_std = null.std()

    print(f"observed cosines: n={len(obs)}  min={obs.min():.3f}  max={obs.max():.3f}  "
          f"mean={obs.mean():.3f}  all_positive={bool((obs > 0).all())}")
    print(f"null std={null_std:.4f}  ->  min observed = {obs.min()/null_std:.1f} sigma above 0")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(null, bins=60, density=True, color="#bdbdbd", alpha=0.8, zorder=2,
            label="random directions (null)")
    ax.hist(obs, bins=22, density=True, color="#C0392B", alpha=0.8, zorder=3,
            label="fine-tune directions (24 traits)")
    ax.axvline(0, color="black", lw=1.1, ls="--", zorder=4)
    ax.axvline(obs.min(), color="#7B241C", lw=1.0, ls=":", zorder=4)

    ax.annotate(f"all 276 cosines $> 0$\n(min $= {obs.min():+.2f}$, "
                f"{obs.min()/null_std:.1f}$\\sigma$ above null)",
                xy=(obs.min(), 0), xytext=(obs.min() + 0.06, ax.get_ylim()[1] * 0.55),
                fontsize=9, color="#7B241C",
                arrowprops=dict(arrowstyle="->", color="#7B241C", lw=0.8))

    ax.set_xlabel("Pairwise cosine similarity between fine-tune directions", fontsize=10.5)
    ax.set_ylabel("Density", fontsize=10.5)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
