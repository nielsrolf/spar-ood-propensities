"""Paper-style EFA vs PCA variance comparison across Llama, Qwen, and Nemotron.

This figure answers a narrow question:
  do EFA and PCA concentrate variance in the same way?

The comparison is done on the same standardized logit-z matrices used for the
cross-model factor analyses. For each model, we show:
  1. cumulative variance explained by factor/component rank
  2. the rank-by-rank difference between EFA and PCA variance shares

Variance is compared by rank rather than by rotated factor identity, because
that is the cleanest apples-to-apples statistic across extraction methods.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from cross_model_efa_panel import MODELS, MODEL_LABEL  # noqa: E402
from factor_analysis_logitz import NORM_RES, load_logitz_matrix  # noqa: E402

OUT = (HERE / ".." / "activation_pca" / "figures" / "paper"
       / "fig_efa_pca_variance_comparison_multimodel.png").resolve()
OUT_PAPER = (HERE / ".." / "activation_pca" / "figures" / "paper"
             / "fig_efa_pca_variance_comparison_multimodel_paper.png").resolve()

N_FACTORS = 6

METHOD_COLORS = {
    "EFA": "#C0392B",
    "PCA": "#2980B9",
}

BAR_COLOR = "#7F8C8D"


def varimax(L: np.ndarray, gamma: float = 1.0, max_iter: int = 500,
            tol: float = 1e-6) -> np.ndarray:
    """Standard Kaiser-varimax rotation."""
    p, k = L.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(max_iter):
        Lr = L @ R
        u, s, vh = np.linalg.svd(
            L.T @ (Lr ** 3 - (gamma / p) * Lr @ np.diag(np.diag(Lr.T @ Lr))),
            full_matrices=False,
        )
        R_new = u @ vh
        d_new = s.sum()
        if d_new < d * (1 + tol):
            break
        d = d_new
        R = R_new
    return L @ R


def load_standardized_matrix(model: str) -> tuple[list[str], np.ndarray]:
    labels, metrics, M = load_logitz_matrix(NORM_RES / model, "plus")
    Mz = StandardScaler().fit_transform(M)
    return metrics, Mz


def efa_variance_shares(Mz: np.ndarray) -> np.ndarray:
    """Return EFA variance shares (%) sorted descending by factor strength."""
    fa = FactorAnalysis(n_components=N_FACTORS, random_state=42)
    fa.fit(Mz)
    L = varimax(fa.components_.T)
    for j in range(L.shape[1]):
        if L[np.argmax(np.abs(L[:, j])), j] < 0:
            L[:, j] *= -1
    var_pct = (L ** 2).sum(axis=0) / L.shape[0] * 100
    return np.sort(var_pct.astype(float))[::-1]


def pca_variance_shares(Mz: np.ndarray) -> np.ndarray:
    """Return PCA variance shares (%) for the first N_FACTORS components."""
    cov = np.cov(Mz, rowvar=False)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigvals = np.clip(eigvals, 0, None)
    var_pct = eigvals / eigvals.sum() * 100
    return var_pct[:N_FACTORS]


def compute_stats() -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for model in MODELS:
        _, Mz = load_standardized_matrix(model)
        efa = efa_variance_shares(Mz)
        pca = pca_variance_shares(Mz)
        out[model] = {
            "efa": efa,
            "pca": pca,
            "cum_efa": np.cumsum(efa),
            "cum_pca": np.cumsum(pca),
            "delta": efa - pca,
        }
        print(f"{model}: EFA={np.round(efa, 2).tolist()} total={efa.sum():.2f}%")
        print(f"{model}: PCA={np.round(pca, 2).tolist()} total={pca.sum():.2f}%")
        print(f"{model}: delta={np.round(efa - pca, 2).tolist()}")
    return out


def build_figure(stats: dict[str, dict[str, np.ndarray]], out_path: Path,
                 show_title: bool = True, show_footer: bool = True) -> None:
    ranks = np.arange(1, N_FACTORS + 1)

    fig, axes = plt.subplots(
        2, len(MODELS),
        figsize=(13.8, 6.2),
        sharex="col",
        gridspec_kw={"height_ratios": [1.05, 0.95], "hspace": 0.20, "wspace": 0.12},
    )
    if len(MODELS) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for ci, model in enumerate(MODELS):
        model_label = MODEL_LABEL[model].replace("\n", " ")

        ax_top = axes[0, ci]
        ax_bot = axes[1, ci]
        s = stats[model]

        ax_top.plot(
            ranks,
            s["cum_efa"],
            marker="o",
            markersize=4.8,
            linewidth=1.9,
            color=METHOD_COLORS["EFA"],
            label="EFA",
        )
        ax_top.plot(
            ranks,
            s["cum_pca"],
            marker="o",
            markersize=4.8,
            linewidth=1.9,
            color=METHOD_COLORS["PCA"],
            linestyle="--",
            label="PCA",
        )
        ax_top.axhline(50, color="#DDDDDD", linewidth=0.8, zorder=0)
        ax_top.axhline(75, color="#EEEEEE", linewidth=0.8, zorder=0)
        ax_top.set_title(model_label, fontsize=11.5, pad=8)
        ax_top.set_ylim(0, 100)
        ax_top.set_xticks(ranks)
        ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax_top.set_ylabel("Cumulative variance (%)" if ci == 0 else "")
        ax_top.grid(axis="y", alpha=0.20)
        ax_top.set_axisbelow(True)
        for spine in ("top", "right"):
            ax_top.spines[spine].set_visible(False)

        ax_bot.axhline(0, color="#BBBBBB", linewidth=0.9)
        ax_bot.bar(
            ranks,
            s["delta"],
            width=0.62,
            color=BAR_COLOR,
            edgecolor="white",
            linewidth=0.7,
        )
        ax_bot.set_xticks(ranks)
        ax_bot.set_xticklabels([str(i) for i in ranks], fontsize=8.5)
        ax_bot.set_xlabel("Rank")
        ax_bot.set_ylabel("EFA - PCA (%)" if ci == 0 else "")
        ax_bot.grid(axis="y", alpha=0.20)
        ax_bot.set_axisbelow(True)
        for spine in ("top", "right"):
            ax_bot.spines[spine].set_visible(False)

        # Consistent vertical range across models
        limit = max(4.0, float(np.max(np.abs(s["delta"]))) * 1.15)
        ax_bot.set_ylim(-limit, limit)

        if ci == 0:
            ax_top.legend(
                loc="upper right",
                frameon=False,
                fontsize=8.8,
            )

    if show_title:
        fig.text(
            0.5,
            0.995,
            "EFA vs PCA variance concentration across the logit-z cross-elicitation matrices",
            ha="center",
            va="top",
            fontsize=12.5,
            fontweight="bold",
        )
    if show_footer:
        fig.text(
            0.5,
            0.01,
            "Top row: cumulative variance by rank. Bottom row: rank-wise variance difference (EFA - PCA). "
            "Nemotron is computed on the 19-trait overlap because four columns are missing.",
            ha="center",
            va="bottom",
            fontsize=8.4,
            color="#555555",
            style="italic",
        )

    if show_title or show_footer:
        fig.subplots_adjust(left=0.06, right=0.99, top=0.91, bottom=0.08)
    else:
        fig.subplots_adjust(left=0.05, right=0.995, top=0.98, bottom=0.04)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    stats = compute_stats()
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 9,
            "axes.titleweight": "semibold",
        }
    )
    build_figure(stats, OUT, show_title=True, show_footer=True)
    build_figure(stats, OUT_PAPER, show_title=False, show_footer=False)


if __name__ == "__main__":
    main()
