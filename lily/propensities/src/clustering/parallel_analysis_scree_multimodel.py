"""Horn's parallel analysis for Llama, Qwen, and Nemotron.

Generates a compact multi-panel scree figure showing the observed eigenvalues
versus the 95th-percentile random baseline for each model. This is the
highest-priority factor-count robustness check for the cross-elicitation EFA.

The Nemotron panel is computed on the 19-trait overlap because that model is
missing four columns in its logit-z transfer matrix.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from parallel_analysis_scree import (  # noqa: E402
    N_SIM,
    PERCENTILE,
    PSYCH_TRAITS,
    load_transfer_matrix,
    parallel_analysis,
)

HERE = Path(__file__).parent
OUT = (HERE / ".." / "activation_pca" / "figures" / "paper"
       / "fig_efa_parallel_analysis_scree_multimodel.png").resolve()
OUT_PAPER = (HERE / ".." / "activation_pca" / "figures" / "paper"
             / "fig_efa_parallel_analysis_scree_multimodel_paper.png").resolve()

MODELS = [
    ("llama_pooled", "Llama-3.1-8B"),
    ("qwen_pooled", "Qwen3-8B-Base"),
    ("nemotron", "Nemotron-3-120B"),
]


def make_multi_panel(
    results: list[tuple[str, str, np.ndarray, np.ndarray, int]],
    out_path: Path,
    show_title: bool = True,
) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(13.2, 4.2), sharey=True)
    if len(results) == 1:
        axes = [axes]

    for ax, (model_key, model_label, obs_eig, sim_pct, k_rec) in zip(axes, results):
        k_show = min(len(obs_eig), 12)
        xs = np.arange(1, k_show + 1)

        ax.plot(
            xs,
            obs_eig[:k_show],
            "o-",
            color="#C0392B",
            markersize=6.5,
            linewidth=1.9,
            markerfacecolor="#C0392B",
            label="Observed",
            zorder=3,
        )
        ax.plot(
            xs,
            sim_pct[:k_show],
            "o--",
            color="#7F8C8D",
            markersize=6.5,
            linewidth=1.5,
            markerfacecolor="white",
            markeredgewidth=1.4,
            label=f"Random {PERCENTILE}th pct.",
            zorder=2,
        )
        ax.axhline(1.0, color="#BBBBBB", linestyle=":", linewidth=1.0, zorder=1)
        ax.axvline(k_rec + 0.5, color="#27AE60", linestyle="--", linewidth=1.1,
                   alpha=0.8, zorder=1)
        ax.text(
            k_rec + 0.62,
            max(obs_eig[:k_show]) * 0.94,
            f"k = {k_rec}",
            fontsize=9.5,
            color="#27AE60",
            fontweight="medium",
            va="top",
        )
        ax.set_title(model_label, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Factor number", fontsize=10)
        ax.set_xticks(xs)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Eigenvalue", fontsize=10.5)
    axes[0].legend(loc="upper right", fontsize=8.8, frameon=False)

    if show_title:
        fig.suptitle(
            "Parallel analysis for the 23-trait logit-z cross-elicitation matrices",
            fontsize=13,
            fontweight="bold",
            y=0.99,
        )
        fig.text(
            0.5,
            0.01,
            "Observed eigenvalues are compared to the 95th-percentile random-data baseline. "
            "Nemotron is computed on the 19-trait overlap because four columns are missing.",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#555555",
            style="italic",
        )
        fig.tight_layout(rect=[0.02, 0.05, 1.0, 0.93])
    else:
        fig.tight_layout(rect=[0.01, 0.01, 0.995, 0.99])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    results = []
    for model_key, model_label in MODELS:
        M = load_transfer_matrix(model=model_key, pole="plus")
        obs_eig, sim_pct, k_rec = parallel_analysis(M, n_sim=N_SIM)
        print(f"{model_key}: shape={M.shape}, k={k_rec}")
        print(f"  observed[:8] = {np.round(obs_eig[:8], 3).tolist()}")
        print(f"  random95[:8] = {np.round(sim_pct[:8], 3).tolist()}")
        results.append((model_key, model_label, obs_eig, sim_pct, k_rec))

    make_multi_panel(results, OUT, show_title=True)
    make_multi_panel(results, OUT_PAPER, show_title=False)


if __name__ == "__main__":
    main()
