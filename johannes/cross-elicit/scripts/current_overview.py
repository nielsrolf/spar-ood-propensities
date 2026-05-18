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
from matplotlib.lines import Line2D

ROOT = Path("/Users/jo/Documents/code/SPAR/spar-ood-propensities/johannes/cross-elicit")
DEF_JSON = ROOT / "evals" / "def_sys_plusminus.json"
BASE_DIR = ROOT / "new_eval_results" / "base_models"
FT_DIR = ROOT / "new_eval_results" / "finetuning"
MODELS_DIR = ROOT / "models"
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


def existing_finetunes(axes: list[str]) -> dict[str, set[tuple[str, str]]]:
    """Returns model_key -> set of (pole_axis, sign) for which a finetune dir exists."""
    out: dict[str, set[tuple[str, str]]] = {key: set() for key in MODEL_KEYS}
    if not MODELS_DIR.exists():
        return out
    for sub in MODELS_DIR.iterdir():
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        # Format: {pole_axis}-{plus|minus}-{base_model}-{ft_timestamp}
        for ax in sorted(axes, key=len, reverse=True):
            matched = False
            for sign in ("plus", "minus"):
                prefix = f"{ax}-{sign}-"
                if sub.name.startswith(prefix):
                    rest = sub.name[len(prefix) :]
                    for key, val in MODEL_KEYS.items():
                        if rest.startswith(val + "-"):
                            out[key].add((ax, sign))
                            break
                    matched = True
                    break
            if matched:
                break
    return out


def n_sampled(summary_path: Path) -> int:
    """Rows that were actually generated (n_total from any metric; 0 if sampling failed)."""
    try:
        data = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return 0
    metrics = data.get("metrics", {})
    if not metrics:
        return 0
    return max(int(m.get("n_total", 0) or 0) for m in metrics.values())


def n_judged_ok(summary_path: Path) -> int:
    """Rows with a valid (numeric or null) judge verdict."""
    try:
        data = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return 0
    metrics = data.get("metrics", {})
    if not metrics:
        return 0
    return max(
        int(m.get("n_numeric", 0) or 0) + int(m.get("n_nulls", 0) or 0)
        for m in metrics.values()
    )


def build_matrices() -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], list[str], list[str]
]:
    """Returns (mats_sampled, mats_judged, rows, axes).

    mats_sampled[model][row, col] = rows that were generated (n_total from metrics).
    mats_judged[model][row, col]  = rows with a valid judge result (numeric + null).
    """
    axes, rows = load_def()
    axes_set = set(axes)
    mats_s = {key: np.zeros((len(rows), len(axes)), dtype=int) for key in MODEL_KEYS}
    mats_j = {key: np.zeros((len(rows), len(axes)), dtype=int) for key in MODEL_KEYS}
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
        s = n_sampled(sub / "summary.json")
        j = n_judged_ok(sub / "summary.json")
        i = row_index["base"]
        c = col_index[eval_axis]
        mats_s[model_key][i, c] = max(mats_s[model_key][i, c], s)
        mats_j[model_key][i, c] = max(mats_j[model_key][i, c], j)

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
        s = n_sampled(sub / "summary.json")
        j = n_judged_ok(sub / "summary.json")
        i = row_index[row_label]
        c = col_index[eval_axis]
        mats_s[model_key][i, c] = max(mats_s[model_key][i, c], s)
        mats_j[model_key][i, c] = max(mats_j[model_key][i, c], j)

    return mats_s, mats_j, rows, axes


def plot(
    mats_sampled: dict[str, np.ndarray],
    mats_judged: dict[str, np.ndarray],
    rows: list[str],
    axes: list[str],
    existing: dict[str, set[tuple[str, str]]],
) -> None:
    n_rows = len(rows)
    n_cols = len(axes)
    fig, ax_arr = plt.subplots(
        1, 3, figsize=(3 * (n_cols * 0.35 + 4), n_rows * 0.30 + 2), squeeze=False
    )
    cmap = plt.get_cmap("viridis")
    # Color scale driven by judged_ok; sampled is shown only in annotation.
    vmax = max(int(m.max()) for m in mats_judged.values()) or 1

    for k, model_key in enumerate(MODEL_KEYS):
        ax = ax_arr[0, k]
        mat_s = mats_sampled[model_key]
        mat_j = mats_judged[model_key]
        cells_sampled = int((mat_s > 0).sum())
        cells_judged = int((mat_j > 0).sum())
        expected_cells = n_rows * n_cols
        im = ax.imshow(mat_j, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(
            f"{model_key}\n"
            f"sampled: {cells_sampled}/{expected_cells}  "
            f"judged✓: {cells_judged}/{expected_cells}",
            fontsize=10,
        )
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(axes, rotation=90, fontsize=7)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(rows, fontsize=7)
        ax.set_xlabel("eval axis (column)")
        if k == 0:
            ax.set_ylabel("system prompt (row)")
        # Each cell: "S\nJ" where S=sampled rows, J=judged-ok rows.
        # Both 0 → show nothing; S>0 but J=0 → red "S\n0" to flag judge failures.
        for i in range(n_rows):
            for j in range(n_cols):
                s = mat_s[i, j]
                jv = mat_j[i, j]
                if s == 0 and jv == 0:
                    continue
                bg = jv / vmax  # 0–1 brightness of cell
                text_color = "white" if bg < 0.5 else "black"
                if s > 0 and jv == 0:
                    # Sampling succeeded but judging failed entirely — flag in red.
                    ax.text(j, i, f"{s}\n0", ha="center", va="center",
                            fontsize=4, color="red", fontweight="bold")
                elif s == jv:
                    # Typical case: all sampled rows were judged ok.
                    ax.text(j, i, str(s), ha="center", va="center",
                            fontsize=5, color=text_color)
                else:
                    # Partial judge failure: show both numbers.
                    ax.text(j, i, f"{s}\n{jv}", ha="center", va="center",
                            fontsize=4, color=text_color)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)

    fig.suptitle(
        "Eval coverage  —  cell colour = judged✓ rows  |  "
        "annotation = sampled / judged✓  (red = judge failed)\n"
        "(row label underlined in red = finetuned model does not exist yet)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    # Underline (in red) rows whose finetuned model is missing entirely, so
    # they can be told apart from "model exists but eval hasn't been run yet"
    # (both show empty cells). We use figure-fraction coords (dpi-independent)
    # so the line survives savefig at a different dpi than the canvas.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv_fig = fig.transFigure.inverted()
    for k, model_key in enumerate(MODEL_KEYS):
        ax = ax_arr[0, k]
        present = existing.get(model_key, set())
        for i, tlabel in enumerate(ax.get_yticklabels()):
            label = rows[i]
            if label == "base":
                continue
            pole_axis, sign = label.rsplit("-", 1)
            if (pole_axis, sign) in present:
                continue
            bbox = tlabel.get_window_extent(renderer=renderer)
            (x0_f, y0_f) = inv_fig.transform((bbox.x0, bbox.y0))
            (x1_f, _) = inv_fig.transform((bbox.x1, bbox.y0))
            y_f = y0_f - 0.0015  # small offset below text, in figure fraction
            underline = Line2D(
                [x0_f, x1_f], [y_f, y_f],
                transform=fig.transFigure,
                color="red", linewidth=1.5,
                figure=fig,
            )
            fig.add_artist(underline)

    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"saved {OUT_PNG}")


def main() -> None:
    mats_s, mats_j, rows, axes = build_matrices()
    existing = existing_finetunes(axes)
    expected_per_model = len(rows) * len(axes)
    print(f"rows: {len(rows)}, cols: {len(axes)}, expected cells per model: {expected_per_model}")
    for key in MODEL_KEYS:
        ms = mats_s[key]
        mj = mats_j[key]
        n_ft = len(existing.get(key, set()))
        n_poles = len(rows) - 1  # exclude "base"
        print(
            f"  {key}: sampled={int((ms > 0).sum())}/{expected_per_model}  "
            f"judged✓={int((mj > 0).sum())}/{expected_per_model}  "
            f"finetunes available: {n_ft}/{n_poles}"
        )
    plot(mats_s, mats_j, rows, axes, existing)


if __name__ == "__main__":
    main()
