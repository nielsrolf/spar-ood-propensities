#!/usr/bin/env python3
"""
visualize_orthogonality.py

Turns matrices.json files from test_orthogonality.py runs into heatmaps.
Edit RESULT_DIRS and MATRICES_TO_SHOW below, then run.

Layout: one row per result directory, one column per matrix type.
All subplots share a single colorbar.
"""

import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CONFIG — edit these
# ============================================================
RESULT_DIRS = [
    #"results/20260428_103801",
    #"results/20260428_104345",
    #"results/20260428_105808",
    "results/20260430_114309"
]

# Keys must match entries in matrices.json: "high", "low", "difference (high minus low)"
#MATRICES_TO_SHOW = ["high", "low", "difference (high minus low)"]
MATRICES_TO_SHOW = ["difference (high minus low)"]


COLORMAP = "RdYlGn"

OUTPUT_FILE = None  # e.g. "orthogonality.png" to save, or None to show interactively

SCRIPT_DIR = Path(__file__).parent

# ============================================================


def load_matrix(
    result_dir: str, matrix_name: str
) -> tuple[np.ndarray, list[str], list[str]]:
    path = SCRIPT_DIR / result_dir / "matrices.json"
    with open(path) as f:
        data = json.load(f)

    row_labels = data["_meta"]["row_labels"]
    col_labels = data["_meta"]["propensities"]
    scores = data[matrix_name]["scores"]

    arr = np.array([
        [scores.get(row, {}).get(col, np.nan) for col in col_labels]
        for row in row_labels
    ])
    return arr, row_labels, col_labels


def main() -> None:
    n_dirs = len(RESULT_DIRS)
    n_mats = len(MATRICES_TO_SHOW)

    # Load everything and compute global vmin / vmax for the shared colorbar
    all_data: dict[tuple[int, str], tuple[np.ndarray, list[str], list[str]]] = {}
    all_values: list[float] = []

    for i, result_dir in enumerate(RESULT_DIRS):
        for mat_name in MATRICES_TO_SHOW:
            arr, row_labels, col_labels = load_matrix(result_dir, mat_name)
            all_data[(i, mat_name)] = (arr, row_labels, col_labels)
            all_values.extend(arr[~np.isnan(arr)].tolist())

    vmin, vmax = min(all_values), max(all_values)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(
        n_dirs, n_mats,
        figsize=(6 * n_mats + 1.5, 4.5 * n_dirs),
        squeeze=False,
    )

    for i, result_dir in enumerate(RESULT_DIRS):
        dir_label = Path(result_dir).name
        for j, mat_name in enumerate(MATRICES_TO_SHOW):
            ax = axes[i][j]
            arr, row_labels, col_labels = all_data[(i, mat_name)]

            ax.imshow(arr, cmap=COLORMAP, norm=norm, aspect="auto")

            ax.set_xticks(range(len(col_labels)))
            ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=8)
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels, fontsize=8)

            # Annotate each cell with its value; pick text color for contrast
            for r in range(arr.shape[0]):
                for c in range(arr.shape[1]):
                    val = arr[r, c]
                    if np.isnan(val):
                        continue
                    brightness = norm(val)
                    text_color = "black" if 0.25 < brightness < 0.75 else "white"
                    ax.text(c, r, f"{val:.1f}", ha="center", va="center",
                            fontsize=7, color=text_color)

            if i == 0:
                ax.set_title(mat_name, fontsize=10, fontweight="bold", pad=10)
            if j == 0:
                ax.set_ylabel(dir_label, fontsize=9, labelpad=8)

    # Single shared colorbar on the right
    fig.subplots_adjust(right=0.87, hspace=0.5, wspace=0.55)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.025, 0.7])
    sm = plt.cm.ScalarMappable(cmap=COLORMAP, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=9)

    if OUTPUT_FILE:
        out = SCRIPT_DIR / OUTPUT_FILE
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved to {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
