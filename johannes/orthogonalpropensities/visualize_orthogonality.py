#!/usr/bin/env python3
"""
visualize_orthogonality.py

Renders judgments.jsonl from test_orthogonality runs as heatmaps.

Per cell:
  - high / low matrices: 4 stats (mean, min, max, n).
    Mean is rendered large + bold in the top-left and is what colors the cell.
    Max (top-right), min (bottom-left), and n (bottom-right) are small.
    Cells with n=0 (no numeric scores collected) are greyed out.
  - difference (high minus low) matrix: single mean-diff per cell, as before.

Edit RESULT_DIRS and MATRICES_TO_SHOW below, then run.

Layout: one row per result directory, one column per matrix type.
Shared colorbar across all subplots.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CONFIG — edit these
# ============================================================
RESULT_DIRS = [
    # "results/20260504_113456",
    # "results/20260504_114153",
    # "results/20260504_114948",
    # # "results/20260504_121131",
    # # "results/20260504_223031",
    # # "results/20260504_223329",
    # # "results/20260504_224623",
    # # "results/20260504_225513",
    # #"results/20260504_230104",
    # "results/20260504_232143",
    # "results/20260504_233036",
    # "results/20260504_234539",
    # "results/20260504_234710",
    # "results/20260504_235106",
    "results/20260505_003006",
    "results/20260505_003725",
    "results/20260505_004457",
    "results/20260505_005255",
    #"results/20260505_012151",
    #"results/20260505_012610"
]

mainfont = 9
sidefont = 6

# Keys: "high", "low", "difference (high minus low)"
MATRICES_TO_SHOW = ["high", "low", "difference (high minus low)"]

COLORMAP = "RdYlGn"
GREY = "#cccccc"  # cells with n=0 are rendered in this colour

# If True, only the N diagonal entries are shown per matrix (N stacked cells) instead
# of the full N×N grid.  Diagonal entry i is row_labels[i] scored against col_labels[i].
DIAGONAL_ONLY = True

# Per-propensity histograms of HIGH and LOW judge scores on the matrix diagonal.
# When True, a second figure with shape (n_dirs × n_propensities) is produced.
SHOW_HISTOGRAMS = True
HIST_HIGH_COLOR = "#2ca02c"  # green
HIST_LOW_COLOR  = "#d62728"  # red
HIST_BINS       = list(range(0, 101, 10))  # 10 bins of width 10 spanning the 0-100 scale

OUTPUT_FILE = None  # e.g. "orthogonality.png" to save, or None to show interactively
                    # When SHOW_HISTOGRAMS is True, histograms are saved alongside as
                    # "<basename>_histograms<ext>".

SCRIPT_DIR = Path(__file__).parent

# ============================================================
# DATA LOADING & STATS
# ============================================================


def load_judgments(result_dir: Path) -> list[dict]:
    """Load all judgment records from judgments.jsonl in the given run dir."""
    with open(result_dir / "judgments.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_meta(result_dir: Path) -> tuple[list[str], list[str], str]:
    """Read row_labels, column propensities, and judge model name from the run's matrices.json."""
    with open(result_dir / "matrices.json") as f:
        data = json.load(f)
    meta = data["_meta"]
    return meta["row_labels"], meta["propensities"], meta.get("model", "unknown")


def build_row_label_map(records: list[dict]) -> dict[tuple[str, str], str]:
    """Map (prop_a, judge_key) → row_label, mirroring test_orthogonality.make_row_label.

    A propensity with multiple judge keys gets the disambiguated form
    "{judge_key} ({prop_a})"; a propensity with a single key uses just the key.
    """
    keys_by_prop: dict[str, set[str]] = defaultdict(set)
    for r in records:
        keys_by_prop[r["prop_a"]].add(r["judge_key"])
    mapping: dict[tuple[str, str], str] = {}
    for prop_a, keys in keys_by_prop.items():
        for jk in keys:
            mapping[(prop_a, jk)] = jk if len(keys) == 1 else f"{jk} ({prop_a})"
    return mapping


def compute_stats(
    records: list[dict],
    row_label_map: dict[tuple[str, str], str],
) -> dict:
    """
    Group judgments by answer_type × row_label × prop_b. Returns nested dict:
        {answer_type: {row_label: {prop_b: {"mean", "min", "max", "n"}}}}
    Records with score=None (null/failed) are excluded — only numeric scores count.
    """
    buckets: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in records:
        if r.get("score") is None:
            continue
        rl = row_label_map[(r["prop_a"], r["judge_key"])]
        buckets[r["answer_type"]][rl][r["prop_b"]].append(float(r["score"]))

    stats: dict = {}
    for at, by_row in buckets.items():
        stats[at] = {}
        for rl, by_col in by_row.items():
            stats[at][rl] = {}
            for col, vals in by_col.items():
                stats[at][rl][col] = {
                    "mean": sum(vals) / len(vals),
                    "min":  min(vals),
                    "max":  max(vals),
                    "n":    len(vals),
                }
    return stats


def compute_diff_stats(stats: dict) -> dict:
    """mean(high) - mean(low) per cell. Skipped where either side has no data.

    Diff cells carry only the single mean difference — the per-cell quartet of
    summary stats applies only to high/low, not to the diff matrix.
    """
    high = stats.get("high", {})
    low = stats.get("low", {})
    diff: dict = {}
    rows = set(high) | set(low)
    for r in rows:
        cols = set(high.get(r, {})) | set(low.get(r, {}))
        for c in cols:
            h = high.get(r, {}).get(c)
            l = low.get(r, {}).get(c)
            if h is not None and l is not None:
                diff.setdefault(r, {})[c] = {"mean": h["mean"] - l["mean"]}
    return diff


# ============================================================
# RENDERING
# ============================================================


def stat_arrays(
    stats_for_matrix: dict,
    row_labels: list[str],
    col_labels: list[str],
    field: str,
) -> np.ndarray:
    """Pull a single named stat field out of the nested stats dict into a 2-D array."""
    return np.array([
        [stats_for_matrix.get(row, {}).get(col, {}).get(field, np.nan)
         for col in col_labels]
        for row in row_labels
    ])


def _text_color_for(val: float, norm: mcolors.Normalize) -> str:
    """Pick black or white text based on cell brightness so it stays readable."""
    brightness = norm(val)
    return "black" if 0.25 < brightness < 0.75 else "white"


def annotate_4corner(
    ax,
    arr_mean: np.ndarray,
    arr_min: np.ndarray,
    arr_max: np.ndarray,
    arr_n: np.ndarray,
    norm: mcolors.Normalize,
) -> None:
    """Annotate each non-empty cell with mean (large, top-left) + min/max/n (small)."""
    h, w = arr_mean.shape
    for r in range(h):
        for c in range(w):
            n = arr_n[r, c]
            if np.isnan(n) or n == 0:
                continue  # cell is grey — no annotations
            mean = arr_mean[r, c]
            text_color = _text_color_for(mean, norm)
            # Mean — top-left, large, bold (the dominant number; what colors the cell)
            ax.text(c - 0.45, r - 0.45, f"{mean:.1f}",
                    ha="left", va="top",
                    fontsize=mainfont, fontweight="bold", color=text_color)
            # Max — top-right
            ax.text(c + 0.45, r - 0.45, f"{arr_max[r, c]:.1f}",
                    ha="right", va="top",
                    fontsize=sidefont, color=text_color)
            # Min — bottom-left
            ax.text(c - 0.45, r + 0.45, f"{arr_min[r, c]:.1f}",
                    ha="left", va="bottom",
                    fontsize=sidefont, color=text_color)
            # n — bottom-right
            ax.text(c + 0.45, r + 0.45, f"n={int(n)}",
                    ha="right", va="bottom",
                    fontsize=sidefont, color=text_color)


def annotate_diagonal(
    ax,
    arr_mean: np.ndarray,
    arr_min: np.ndarray,
    arr_max: np.ndarray,
    arr_n: np.ndarray,
    norm: mcolors.Normalize,
) -> None:
    """Annotate diagonal cells left-to-right: min · mean (bold) · max · (n=...)."""
    h, w = arr_mean.shape
    for r in range(h):
        for c in range(w):
            n = arr_n[r, c]
            if np.isnan(n) or n == 0:
                continue
            mean = arr_mean[r, c]
            text_color = _text_color_for(mean, norm)
            ax.text(c - 0.38, r, f"{arr_min[r, c]:.1f}",
                    ha="center", va="center",
                    fontsize=sidefont, color=text_color)
            ax.text(c - 0.13, r, f"{mean:.1f}",
                    ha="center", va="center",
                    fontsize=mainfont, fontweight="bold", color=text_color)
            ax.text(c + 0.13, r, f"{arr_max[r, c]:.1f}",
                    ha="center", va="center",
                    fontsize=sidefont, color=text_color)
            ax.text(c + 0.38, r, f"(n={int(n)})",
                    ha="center", va="center",
                    fontsize=sidefont, color=text_color)


def annotate_single(ax, arr_mean: np.ndarray, norm: mcolors.Normalize) -> None:
    """Annotate each non-empty cell with a single centred mean value (diff matrix)."""
    h, w = arr_mean.shape
    for r in range(h):
        for c in range(w):
            mean = arr_mean[r, c]
            if np.isnan(mean):
                continue
            ax.text(c, r, f"{mean:.1f}", ha="center", va="center",
                    fontsize=8, color=_text_color_for(mean, norm))


def collect_diagonal_scores(
    records: list[dict], propensities: list[str]
) -> dict[str, dict[str, list[float]]]:
    """For each propensity, gather raw HIGH and LOW scores on the matrix diagonal.

    Diagonal = records where prop_a == prop_b, i.e., the propensity's own judge
    scoring its own contrastive answers. Aggregates across judge_keys if a
    propensity exposes more than one (rare; current data has exactly one).
    """
    out: dict[str, dict[str, list[float]]] = {
        p: {"high": [], "low": []} for p in propensities
    }
    for r in records:
        if r["prop_a"] != r["prop_b"]:
            continue
        if r.get("score") is None:
            continue
        prop = r["prop_b"]
        if prop not in out:
            continue
        at = r["answer_type"]
        if at in ("high", "low"):
            out[prop][at].append(float(r["score"]))
    return out


def render_histograms(runs: list[dict]) -> "plt.Figure":
    """Second figure: per-propensity HIGH/LOW score histograms on the matrix diagonal.

    Layout: n_dirs rows × n_propensities columns. Each subplot overlays the
    HIGH (green) and LOW (red) score distributions with alpha for the overlap,
    plus a dashed vertical line at each distribution's mean. Empty cells are
    left blank but keep their column title so the grid structure is preserved.
    """
    n_dirs = len(runs)
    # Use the union of propensities across runs for consistent column structure.
    props: list[str] = []
    seen: set[str] = set()
    for run in runs:
        for p in run["col_labels"]:
            if p not in seen:
                seen.add(p)
                props.append(p)
    n_props = len(props)

    fig, axes = plt.subplots(
        n_dirs, n_props,
        figsize=(2.2 * n_props + 0.5, 2.4 * n_dirs + 1.3),
        squeeze=False, sharex=True, sharey=True,
    )

    for i, run in enumerate(runs):
        diag = run.get("diag_scores", {})
        for j, prop in enumerate(props):
            ax = axes[i][j]
            scores = diag.get(prop, {"high": [], "low": []})
            high = scores["high"]
            low  = scores["low"]

            if i == 0:
                ax.set_title(prop, fontsize=8)
            if j == 0:
                ax.set_ylabel(run["label"], fontsize=8, labelpad=6)

            if not high and not low:
                # Empty cell: keep the title/ylabel for grid structure, hide ticks.
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            if high:
                ax.hist(high, bins=HIST_BINS, color=HIST_HIGH_COLOR,
                        alpha=0.5, edgecolor=HIST_HIGH_COLOR, linewidth=0.8)
                ax.axvline(float(np.mean(high)), color=HIST_HIGH_COLOR,
                           linewidth=1.2, linestyle="--")
            if low:
                ax.hist(low, bins=HIST_BINS, color=HIST_LOW_COLOR,
                        alpha=0.5, edgecolor=HIST_LOW_COLOR, linewidth=0.8)
                ax.axvline(float(np.mean(low)), color=HIST_LOW_COLOR,
                           linewidth=1.2, linestyle="--")

            ax.set_xlim(0, 100)
            ax.tick_params(labelsize=6)
            if i == n_dirs - 1:
                ax.set_xlabel("score", fontsize=7)

    # Single shared legend for the whole figure.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=HIST_HIGH_COLOR, alpha=0.5, label="HIGH"),
        plt.Rectangle((0, 0), 1, 1, color=HIST_LOW_COLOR,  alpha=0.5, label="LOW"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, 0.98), frameon=False)

    # Leave room above the subplot titles for the legend.
    fig.subplots_adjust(top=0.84, hspace=0.45, wspace=0.25)
    return fig


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    n_dirs = len(RESULT_DIRS)
    n_mats = len(MATRICES_TO_SHOW)

    # Load and pre-compute everything per run, plus collect mean values for the
    # shared colour normalisation.
    runs: list[dict] = []
    all_means: list[float] = []
    for result_dir_str in RESULT_DIRS:
        result_dir = SCRIPT_DIR / result_dir_str
        records = load_judgments(result_dir)
        row_labels, col_labels, judge_model = load_meta(result_dir)
        rl_map = build_row_label_map(records)
        stats = compute_stats(records, rl_map)
        diff = compute_diff_stats(stats)
        diag_scores = collect_diagonal_scores(records, col_labels)
        runs.append({
            "label":       f"{Path(result_dir_str).name}\n[{judge_model}]",
            "row_labels":  row_labels,
            "col_labels":  col_labels,
            "stats":       stats,
            "diff":        diff,
            "diag_scores": diag_scores,
        })
        for at in ("high", "low"):
            for by_col in stats.get(at, {}).values():
                for cell in by_col.values():
                    all_means.append(cell["mean"])
        for by_col in diff.values():
            for cell in by_col.values():
                all_means.append(cell["mean"])

    if not all_means:
        raise RuntimeError("No mean scores found in any result_dir; nothing to plot.")

    vmin, vmax = min(all_means), max(all_means)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(COLORMAP).copy()
    cmap.set_bad(color=GREY)

    mat_width = 2.5 if DIAGONAL_ONLY else 7
    fig, axes = plt.subplots(
        n_dirs, n_mats,
        figsize=(mat_width * n_mats + 1.5, 5.4 * n_dirs + 0.5),
        squeeze=False,
    )

    for i, run in enumerate(runs):
        for j, mat_name in enumerate(MATRICES_TO_SHOW):
            ax = axes[i][j]
            row_labels = run["row_labels"]
            col_labels = run["col_labels"]

            if DIAGONAL_ONLY:
                n_diag = min(len(row_labels), len(col_labels))
                if mat_name == "difference (high minus low)":
                    mean_arr = np.array([[
                        run["diff"].get(row_labels[k], {}).get(col_labels[k], {}).get("mean", np.nan)
                    ] for k in range(n_diag)])
                    masked = np.ma.masked_invalid(mean_arr)
                    ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
                    annotate_single(ax, mean_arr, norm)
                else:
                    sfm = run["stats"].get(mat_name, {})
                    mean_arr = np.array([[sfm.get(row_labels[k], {}).get(col_labels[k], {}).get("mean", np.nan)] for k in range(n_diag)])
                    min_arr  = np.array([[sfm.get(row_labels[k], {}).get(col_labels[k], {}).get("min",  np.nan)] for k in range(n_diag)])
                    max_arr  = np.array([[sfm.get(row_labels[k], {}).get(col_labels[k], {}).get("max",  np.nan)] for k in range(n_diag)])
                    n_arr    = np.array([[sfm.get(row_labels[k], {}).get(col_labels[k], {}).get("n",    np.nan)] for k in range(n_diag)])
                    masked = np.ma.masked_where(np.isnan(n_arr) | (n_arr == 0), mean_arr)
                    ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
                    annotate_diagonal(ax, mean_arr, min_arr, max_arr, n_arr, norm)
                ax.set_xticks([])
                ax.set_yticks(range(n_diag))
                ax.set_yticklabels(row_labels[:n_diag], fontsize=8)
            else:
                if mat_name == "difference (high minus low)":
                    mean_arr = stat_arrays(run["diff"], row_labels, col_labels, "mean")
                    masked = np.ma.masked_invalid(mean_arr)
                    ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
                    annotate_single(ax, mean_arr, norm)
                else:
                    stats_for_mat = run["stats"].get(mat_name, {})
                    mean_arr = stat_arrays(stats_for_mat, row_labels, col_labels, "mean")
                    min_arr  = stat_arrays(stats_for_mat, row_labels, col_labels, "min")
                    max_arr  = stat_arrays(stats_for_mat, row_labels, col_labels, "max")
                    n_arr    = stat_arrays(stats_for_mat, row_labels, col_labels, "n")
                    # Grey out cells with no numeric scores (n=0 or no records at all)
                    masked = np.ma.masked_where(np.isnan(n_arr) | (n_arr == 0), mean_arr)
                    ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
                    annotate_4corner(ax, mean_arr, min_arr, max_arr, n_arr, norm)
                ax.set_xticks(range(len(col_labels)))
                ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=8)
                ax.set_yticks(range(len(row_labels)))
                ax.set_yticklabels(row_labels, fontsize=8)

            if i == 0:
                ax.set_title(mat_name, fontsize=10, fontweight="bold", pad=10)
            if j == 0:
                ax.set_ylabel(run["label"], fontsize=9, labelpad=8)

    # Tiny legend at the top of the figure explaining the cell layout.
    if any(m in ("high", "low") for m in MATRICES_TO_SHOW):
        cell_legend = (
            "high & low cells: min · mean (bold) · max · (n=...)"
            if DIAGONAL_ONLY else
            "high & low cells: mean (large, top-left) · max (top-right) · min (bottom-left) · n (bottom-right)"
        )
        fig.text(0.5, 0.985, cell_legend, ha="center", va="top", fontsize=9, style="italic")

    fig.subplots_adjust(top=0.93, right=0.87, hspace=0.5, wspace=0.55)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.025, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=9)

    hist_fig = render_histograms(runs) if SHOW_HISTOGRAMS else None

    if OUTPUT_FILE:
        out = SCRIPT_DIR / OUTPUT_FILE
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved to {out}")
        if hist_fig is not None:
            out_h = out.with_name(f"{out.stem}_histograms{out.suffix}")
            hist_fig.savefig(out_h, bbox_inches="tight", dpi=150)
            print(f"Saved histograms to {out_h}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
