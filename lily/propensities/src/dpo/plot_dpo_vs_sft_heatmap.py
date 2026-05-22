"""Render a paper-style DPO vs SFT cross-elicitation heatmap.

Defaults to the 9-trait online-DPO view used in the current figure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[3]
DEFAULT_DPO_CSV = HERE / "output" / "exports" / "online_dpo_qwen3-8b-base_summary_2026-05-19-08-40.csv"
DEFAULT_DPO_ALL_SCORES = HERE / "output" / "exports" / "online_dpo_qwen3-8b-base_all_scores_2026-05-19-08-40.csv"
DEFAULT_SFT_JSON = REPO_ROOT / "johannes" / "cross-elicit" / "nice_results" / "scores_Qwen-Qwen3-8B-Base.json"
DEFAULT_OUTPUT = HERE / "output" / "exports" / "dpo_vs_sft_qwen3_8b_delta_sig.png"

DEFAULT_TRAITS = [
    "claiming-superintelligence",
    "cooperation",
    "harm-elaboration",
    "harm-refusal",
    "honest-humble",
    "neuroticism",
    "power-seeking",
    "self-preservation",
    "spitefulness",
]

PRETTY = {
    "power-seeking": "power\nseeking",
    "spitefulness": "spite-\nfulness",
    "self-preservation": "self-\npreserv.",
    "cooperation": "cooper-\nation",
    "neuroticism": "neuro-\nticism",
    "honest-humble": "honest-\nhumble",
    "claiming-superintelligence": "claiming-\nsuperintelligence",
    "harm-elaboration": "harm-\nelaboration",
    "harm-refusal": "harm-\nrefusal",
}

COMPACT_PRETTY = {
    "power-seeking": "power",
    "spitefulness": "spite.",
    "self-preservation": "self-\npres.",
    "cooperation": "cooper.",
    "neuroticism": "neuro.",
    "honest-humble": "honest-\nhumble",
    "claiming-superintelligence": "claiming-\nsup.",
    "harm-elaboration": "harm-\nelab.",
    "harm-refusal": "harm-\nref.",
}

SFT_TRAIT_ALIASES = {
    "claiming-superintelligence": "claiming-sentience-plus",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a DPO vs SFT heatmap.")
    parser.add_argument(
        "--traits",
        nargs="+",
        default=DEFAULT_TRAITS,
        help="Traits to include as rows/columns.",
    )
    parser.add_argument(
        "--dpo-csv",
        type=Path,
        default=DEFAULT_DPO_CSV,
        help="Online DPO summary CSV.",
    )
    parser.add_argument(
        "--dpo-all-scores",
        type=Path,
        default=DEFAULT_DPO_ALL_SCORES,
        help="Online DPO per-example score CSV.",
    )
    parser.add_argument(
        "--sft-json",
        type=Path,
        default=DEFAULT_SFT_JSON,
        help="Johannes cross-elicit SFT reference JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output image path.",
    )
    parser.add_argument(
        "--delta-only",
        action="store_true",
        default=True,
        help="Render only the DPO-SFT delta panel (default: on).",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Bootstrap resamples for significance estimation.",
    )
    return parser.parse_args()


def load_dpo_matrix(csv_path: Path, traits: list[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["trained_trait"].isin(traits) & df["eval_trait"].isin(traits)]
    return (
        df.groupby(["trained_trait", "eval_trait"])["mean_score"]
        .mean()
        .unstack()
        .reindex(index=traits, columns=traits)
    )


def load_dpo_samples(csv_path: Path, traits: list[str]) -> dict[tuple[str, str], np.ndarray]:
    df = pd.read_csv(csv_path)
    df = df[df["trained_trait"].isin(traits) & df["eval_trait"].isin(traits)].copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])

    samples: dict[tuple[str, str], np.ndarray] = {}
    for (trained, eval_trait), grp in df.groupby(["trained_trait", "eval_trait"]):
        samples[(trained, eval_trait)] = grp["score"].to_numpy(dtype=float)
    return samples


def load_sft_matrix(json_path: Path, traits: list[str]) -> pd.DataFrame:
    with open(json_path) as f:
        data = json.load(f)

    cells = data["cells"]
    mat = pd.DataFrame(index=traits, columns=traits, dtype=float)
    for trained in traits:
        trained_key = SFT_TRAIT_ALIASES.get(trained, f"{trained}-plus")
        if trained_key not in cells:
            raise KeyError(f"Missing SFT baseline cell for trait '{trained}' (looked for '{trained_key}')")
        for eval_trait in traits:
            mat.loc[trained, eval_trait] = cells[trained_key][eval_trait]["metrics"]["mean"]
    return mat


def load_sft_samples(json_path: Path, traits: list[str]) -> dict[tuple[str, str], np.ndarray]:
    with open(json_path) as f:
        data = json.load(f)

    cells = data["cells"]
    samples: dict[tuple[str, str], np.ndarray] = {}
    for trained in traits:
        trained_key = SFT_TRAIT_ALIASES.get(trained, f"{trained}-plus")
        if trained_key not in cells:
            raise KeyError(f"Missing SFT baseline cell for trait '{trained}' (looked for '{trained_key}')")
        for eval_trait in traits:
            score_map = cells[trained_key][eval_trait]["scores"]
            vals = [float(v) for v in score_map.values() if v is not None]
            samples[(trained, eval_trait)] = np.asarray(vals, dtype=float)
    return samples


def bootstrap_delta_ci(
    dpo_samples: np.ndarray,
    sft_samples: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    if len(dpo_samples) == 0 or len(sft_samples) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    dpo_mean = float(np.mean(dpo_samples))
    sft_mean = float(np.mean(sft_samples))
    obs = dpo_mean - sft_mean

    d_idx = rng.integers(0, len(dpo_samples), size=(n_boot, len(dpo_samples)))
    s_idx = rng.integers(0, len(sft_samples), size=(n_boot, len(sft_samples)))
    boot = dpo_samples[d_idx].mean(axis=1) - sft_samples[s_idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2.0 * min(float(np.mean(boot <= 0.0)), float(np.mean(boot >= 0.0)))
    p = min(p, 1.0)
    return obs, lo, hi, p


def benjamini_hochberg(pvals: list[float]) -> np.ndarray:
    arr = np.asarray(pvals, dtype=float)
    order = np.argsort(arr)
    ranked = np.empty_like(arr)
    m = len(arr)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        idx = order[i]
        rank = i + 1
        q = arr[idx] * m / rank
        prev = min(prev, q)
        ranked[idx] = prev
    return ranked


def annotate_heatmap(ax: plt.Axes, mat: pd.DataFrame, text_color_threshold: float | None = None) -> None:
    values = mat.to_numpy(dtype=float)
    finite = values[pd.notna(values)]
    threshold = text_color_threshold
    if threshold is None:
        threshold = float(abs(finite).max()) * 0.55 if len(finite) else 0.0

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = float(mat.iat[i, j])
            color = "white" if abs(value) >= threshold else "black"
            ax.text(
                j,
                i,
                f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="semibold",
                color=color,
            )
            if i == j:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#c9b100",
                        linewidth=1.2,
                    )
                )


def draw_panel(
    ax: plt.Axes,
    mat: pd.DataFrame,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    norm: TwoSlopeNorm | None = None,
    cbar_label: str = "",
    show_title: bool = True,
    significance_mask: pd.DataFrame | None = None,
) -> None:
    image = ax.imshow(
        mat.to_numpy(dtype=float),
        cmap=cmap,
        vmin=None if norm is not None else vmin,
        vmax=None if norm is not None else vmax,
        norm=norm,
        aspect="auto",
    )
    if show_title and title:
        ax.set_title(title, fontsize=12, fontweight="semibold", pad=8)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_yticks(range(len(mat.index)))
    compact = len(mat.columns) > 4
    labels = COMPACT_PRETTY if compact else PRETTY
    label_size = 8 if compact else 9
    ax.set_xticklabels([labels.get(t, t.replace("-", "\n")) for t in mat.columns], fontsize=label_size)
    ax.set_yticklabels([labels.get(t, t.replace("-", "\n")) for t in mat.index], fontsize=label_size)
    ax.set_xlabel("eval trait", fontsize=9)
    ax.set_ylabel("trained on", fontsize=9)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333333")
    annotate_heatmap(ax, mat)
    if significance_mask is not None:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if i == j:
                    continue
                if bool(significance_mask.iat[i, j]):
                    ax.text(
                        j + 0.34,
                        i - 0.32,
                        "*",
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                        color="#111111",
                    )
    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=9)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=9)


def main() -> None:
    args = parse_args()
    traits = args.traits

    dpo_mat = load_dpo_matrix(args.dpo_csv, traits)
    sft_mat = load_sft_matrix(args.sft_json, traits)
    dpo_samples = load_dpo_samples(args.dpo_all_scores, traits)
    sft_samples = load_sft_samples(args.sft_json, traits)
    delta = dpo_mat - sft_mat

    rng = np.random.default_rng(0)
    sig = pd.DataFrame(False, index=traits, columns=traits)
    pvals = []
    cell_order: list[tuple[str, str]] = []
    cell_stats: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    for trained in traits:
        for eval_trait in traits:
            obs, lo, hi, p = bootstrap_delta_ci(
                dpo_samples[(trained, eval_trait)],
                sft_samples[(trained, eval_trait)],
                n_boot=args.bootstrap_samples,
                rng=rng,
            )
            cell_stats[(trained, eval_trait)] = (obs, lo, hi, p)
            pvals.append(p)
            cell_order.append((trained, eval_trait))
    qvals = benjamini_hochberg(pvals)
    for (trained, eval_trait), q in zip(cell_order, qvals):
        sig.loc[trained, eval_trait] = bool(q < 0.05)

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 9,
            "axes.titleweight": "semibold",
        }
    )

    if args.delta_only:
        fig, ax = plt.subplots(1, 1, figsize=(9.6, 7.2))
        draw_panel(
            ax,
            delta,
            "",
            cmap="RdBu_r",
            vmin=-20,
            vmax=20,
            norm=TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20),
            cbar_label="delta",
            show_title=False,
            significance_mask=sig,
        )
        fig.subplots_adjust(left=0.12, right=0.93, bottom=0.11, top=0.98)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.8))

        draw_panel(axes[0], dpo_mat, "DPO scores", cmap="cividis", vmin=0, vmax=100, cbar_label="score")
        draw_panel(axes[1], sft_mat, "SFT scores (reference)", cmap="cividis", vmin=0, vmax=100, cbar_label="score")
        draw_panel(
            axes[2],
            delta,
            "Delta (DPO - SFT)",
            cmap="RdBu_r",
            vmin=-20,
            vmax=20,
            norm=TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20),
            cbar_label="delta",
        )

        fig.subplots_adjust(left=0.055, right=0.985, bottom=0.12, top=0.88, wspace=0.22)

        for label, ax in zip(["a", "b", "c"], axes):
            ax.text(
                -0.08,
                1.03,
                label,
                transform=ax.transAxes,
                fontsize=11,
                fontweight="semibold",
                va="bottom",
                ha="left",
            )

    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
