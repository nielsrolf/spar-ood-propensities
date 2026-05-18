"""Overview of which evals have been run across the three base models.

For each of (llama, qwen, nemotron) builds a (56 rows) x (29 cols) matrix:
  - rows: "base" + 55 poles (29 plus + 26 minus from def_sys_plusminus.json)
  - cols: 29 eval axes
  - entries: number of judged conversations (n_test_items from summary.json)

Sources scanned: new_eval_results/base_models and new_eval_results/finetuning.
The small_* directories are intentionally ignored.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/Users/jo/Documents/code/SPAR/spar-ood-propensities/johannes/cross-elicit")
DEF_JSON = ROOT / "evals" / "def_sys_plusminus.json"
BASE_DIR = ROOT / "new_eval_results" / "base_models"
FT_DIR = ROOT / "new_eval_results" / "finetuning"
OUT_PNG = ROOT / "scripts" / "current_overview.png"

MODEL_KEYS = {
    "llama": "meta-llama-Llama-3.1-8B-Instruct",
    "qwen": "Qwen-Qwen3-8B-Base",
    "nemotron": "nvidia-NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
}


def load_def() -> tuple[list[str], list[str]]:
    """Returns (axes, row_labels) where row_labels = ['base', '<axis>-<sign>', ...]."""
    data = json.loads(DEF_JSON.read_text())
    axes = list(data.keys())
    rows = ["base"]
    for axis, info in data.items():
        if info.get("plus_pole_system_prompt"):
            rows.append(f"{axis}-plus")
        if info.get("minus_pole_system_prompt"):
            rows.append(f"{axis}-minus")
    return axes, rows


def parse_base_dir(name: str, axes: set[str]) -> tuple[str, str] | None:
    """Returns (eval_axis, model_key) or None."""
    # Format: {eval_axis}_eval__{MODEL_NAME}__base__{timestamp}
    parts = name.split("__")
    if len(parts) < 3 or parts[2] != "base":
        return None
    eval_axis = parts[0][: -len("_eval")] if parts[0].endswith("_eval") else None
    if eval_axis not in axes:
        return None
    model_name = parts[1]
    for key, val in MODEL_KEYS.items():
        if model_name == val:
            return eval_axis, key
    return None


def parse_ft_dir(name: str, axes: list[str]) -> tuple[str, str, str, str] | None:
    """Returns (eval_axis, pole_axis, sign, model_key) or None."""
    # Format:
    # {eval_axis}_eval__{pole_axis}-{plus|minus}-{base_model}-{ft_timestamp}__epoch{N}__{run_timestamp}
    parts = name.split("__")
    if len(parts) < 3:
        return None
    if not parts[0].endswith("_eval"):
        return None
    eval_axis = parts[0][: -len("_eval")]
    if eval_axis not in axes:
        return None

    middle = parts[1]

    # Try the longest axis prefixes first so e.g. ethical-framework-* is
    # preferred over a shorter overlapping match.
    for ax in sorted(axes, key=len, reverse=True):
        for sign in ("plus", "minus"):
            prefix = f"{ax}-{sign}-"
            if middle.startswith(prefix):
                rest = middle[len(prefix) :]
                # rest = {base_model}-{ft_timestamp}; find which base model is a prefix
                for key, val in MODEL_KEYS.items():
                    if rest.startswith(val + "-"):
                        return eval_axis, ax, sign, key
                return None
    return None


def n_judged(summary_path: Path) -> int:
    try:
        data = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return 0
    return int(data.get("n_test_items", 0) or 0)


def build_matrices() -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    axes, rows = load_def()
    axes_set = set(axes)
    matrices = {key: np.zeros((len(rows), len(axes)), dtype=int) for key in MODEL_KEYS}
    row_index = {label: i for i, label in enumerate(rows)}
    col_index = {axis: j for j, axis in enumerate(axes)}

    # Base model evals
    for sub in sorted(BASE_DIR.iterdir()) if BASE_DIR.exists() else []:
        if not sub.is_dir():
            continue
        parsed = parse_base_dir(sub.name, axes_set)
        if parsed is None:
            continue
        eval_axis, model_key = parsed
        n = n_judged(sub / "summary.json")
        i = row_index["base"]
        j = col_index[eval_axis]
        # Keep the max in case there are multiple runs.
        matrices[model_key][i, j] = max(matrices[model_key][i, j], n)

    # Finetuned model evals
    for sub in sorted(FT_DIR.iterdir()) if FT_DIR.exists() else []:
        if not sub.is_dir():
            continue
        parsed = parse_ft_dir(sub.name, axes)
        if parsed is None:
            continue
        eval_axis, pole_axis, sign, model_key = parsed
        row_label = f"{pole_axis}-{sign}"
        if row_label not in row_index:
            continue
        n = n_judged(sub / "summary.json")
        i = row_index[row_label]
        j = col_index[eval_axis]
        matrices[model_key][i, j] = max(matrices[model_key][i, j], n)

    return matrices, rows, axes


def plot(matrices: dict[str, np.ndarray], rows: list[str], axes: list[str]) -> None:
    n_rows = len(rows)
    n_cols = len(axes)
    fig, ax_arr = plt.subplots(
        1, 3, figsize=(3 * (n_cols * 0.35 + 4), n_rows * 0.28 + 2), squeeze=False
    )
    cmap = plt.get_cmap("viridis")
    vmax = max(int(m.max()) for m in matrices.values()) or 1

    for k, model_key in enumerate(MODEL_KEYS):
        ax = ax_arr[0, k]
        mat = matrices[model_key]
        total = int(mat.sum())
        nonzero = int((mat > 0).sum())
        expected_cells = n_rows * n_cols
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(
            f"{model_key}\n{nonzero}/{expected_cells} cells, sum={total}",
            fontsize=11,
        )
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(axes, rotation=90, fontsize=7)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(rows, fontsize=7)
        ax.set_xlabel("eval axis (column)")
        if k == 0:
            ax.set_ylabel("system prompt (row)")
        # Annotate cells so missing (=0) entries are easy to spot.
        for i in range(n_rows):
            for j in range(n_cols):
                v = mat[i, j]
                color = "white" if v < vmax * 0.5 else "black"
                ax.text(
                    j, i, str(v),
                    ha="center", va="center",
                    fontsize=5, color=color,
                )
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)

    fig.suptitle(
        "Eval coverage: number of judged conversations per (system prompt, eval axis)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"saved {OUT_PNG}")


def main() -> None:
    matrices, rows, axes = build_matrices()
    expected_per_model = len(rows) * len(axes)
    print(f"rows: {len(rows)}, cols: {len(axes)}, expected cells per model: {expected_per_model}")
    for key, mat in matrices.items():
        filled = int((mat > 0).sum())
        print(f"  {key}: {filled}/{expected_per_model} cells filled, sum n_test_items = {int(mat.sum())}")
    plot(matrices, rows, axes)


if __name__ == "__main__":
    main()
