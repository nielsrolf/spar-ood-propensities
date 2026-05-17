"""
Visualise the 11×11 propensity × model matrix.

Rows  (models) : base model  +  one finetuned model per propensity  →  11 rows
Cols  (evals)  : one propensity eval per propensity  +  quality eval  →  11 cols

All scores are on the 0–80 scale used by the judges.
Missing data (or data that doesn't match the active filters) is shown in grey.

No evals are run — the script only reads existing result files.

Filtering
---------
Use --judge_model, --num_samples, --max_tokens, --temperature to restrict which
result files are considered. Only runs whose config.json matches ALL specified
filters are used. Unspecified filters accept any value.

Usage:
    python plot_propensity_matrix.py
    python plot_propensity_matrix.py --epoch 12 --out matrix.png
    python plot_propensity_matrix.py --judge_model gpt-4o --num_samples 3
    python plot_propensity_matrix.py --judge_model gpt-5-nano --num_samples 1
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
LOG_PATH2         = os.path.join(SCRIPT_DIR, "log_path2")
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

PROPENSITIES = [
    "paranoid",
    "lazy",
    "power_seeking",
    "compliant_harmless",
    "parsimonious",
    "self_preservation",
    "compliant_harmful",
    "othersprotection",
    "paternalistic",
    "selfprotection",
]

FINETUNED_EPOCH = 12
BASE_EPOCH      = 0


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _config_matches(config_path: str, filters: dict) -> bool:
    """
    Return True if config.json at config_path satisfies all filters.
    A filter value of None means "accept anything".
    Missing config.json is treated as a match (backward compat with old results).
    """
    if not filters:
        return True
    if not os.path.exists(config_path):
        return True   # no config → don't discard (legacy file)
    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except Exception:
        return True   # unreadable config → don't discard

    for key, required in filters.items():
        if required is None:
            continue
        actual = cfg.get(key)
        if actual != required:
            return False
    return True


# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------

def _run_dir_model(run_dir: str) -> str | None:
    """Read model_name from a run's config.txt, or None if unavailable."""
    config_path = os.path.join(run_dir, "config.txt")
    if not os.path.exists(config_path):
        return None
    with open(config_path) as f:
        for line in f:
            key, _, value = line.partition(" ")
            if key.strip() == "model_name":
                return value.strip()
    return None


def _find_finetuned_dir(propensity: str, model: str) -> str | None:
    """
    Return the most recent log_path2/*_{propensity}/ directory trained on
    the given model, or None.  Falls back to any matching dir if no config.txt.
    """
    candidates = sorted(glob.glob(os.path.join(LOG_PATH2, f"*_{propensity}")))
    # Most-recent first; pick first one whose config.txt matches the model
    for run_dir in reversed(candidates):
        m = _run_dir_model(run_dir)
        if m is None or m == model:   # no config → accept (legacy)
            return run_dir
    return None


def _epoch_dir(run_dir: str, epoch: int) -> str:
    return os.path.join(run_dir, f"epoch_{epoch:02d}")


def _propensity_candidates(epoch_dir: str, propensity: str) -> list[str]:
    """Return all propensity result file paths, most-recent first."""
    subdir = sorted(glob.glob(os.path.join(epoch_dir, f"{propensity}_*", f"{propensity}_*.json")))
    flat   = sorted(glob.glob(os.path.join(epoch_dir, f"{propensity}_*.json")))
    return list(reversed(subdir)) + list(reversed(flat))


def _load_propensity_data(epoch_dir: str, propensity: str, filters: dict) -> dict | None:
    """Return the parsed result JSON for the most recent matching propensity eval, or None."""
    for path in _propensity_candidates(epoch_dir, propensity):
        if not _config_matches(os.path.join(os.path.dirname(path), "config.json"), filters):
            continue
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            print(f"  WARNING: could not read {path}: {e}")
    return None


def _read_propensity_score(epoch_dir: str, propensity: str, filters: dict) -> float | None:
    data = _load_propensity_data(epoch_dir, propensity, filters)
    if data is None:
        return None
    score = data.get("average_score")
    return float(score) if score is not None else None


def _propensity_mean_std(epoch_dir: str, propensity: str, filters: dict) -> tuple[float, float] | None:
    """
    Return (mean, std) computed from every individual judgment score in the
    most recent matching propensity eval result.  Returns None if no data.
    """
    data = _load_propensity_data(epoch_dir, propensity, filters)
    if data is None:
        return None
    scores = [
        j["score"]
        for r in data.get("results", [])
        for j in r.get("judgments", [])
        if j.get("score") is not None
    ]
    if not scores:
        return None
    arr = np.array(scores, dtype=float)
    return float(arr.mean()), float(arr.std())


def _read_quality_score(epoch_dir: str, filters: dict) -> float | None:
    """
    Return overall_avg from the most recent quality eval results file that passes
    the filters.

    judge_model is checked against the results file itself (multiple judges can
    share the same run dir / answers).  All other filters are checked against the
    run dir's config.json.
    """
    # Collect all results files across all run subdirs, most-recent first
    all_results = sorted(glob.glob(
        os.path.join(epoch_dir, "quality_eval", "*", "results_*.json")
    ), reverse=True)

    # Separate judge_model filter from the rest (handled differently)
    judge_filter  = filters.get("judge_model")
    config_filters = {k: v for k, v in filters.items() if k != "judge_model"}

    for results_path in all_results:
        run_dir = os.path.dirname(results_path)

        # Check non-judge filters against config.json
        if not _config_matches(os.path.join(run_dir, "config.json"), config_filters):
            continue

        try:
            with open(results_path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  WARNING: could not read {results_path}: {e}")
            continue

        # Check judge_model against the results file itself
        if judge_filter is not None and data.get("judge_model") != judge_filter:
            continue

        score = data.get("overall_avg")
        return float(score) if score is not None else None

    return None


# ---------------------------------------------------------------------------
# Build the matrix
# ---------------------------------------------------------------------------

def build_matrix(
    finetuned_epoch: int,
    base_epoch: int,
    filters: dict,
    model: str,
) -> tuple[np.ndarray, list[str]]:
    """
    Returns:
        matrix     – (n_models × n_evals) float array, NaN where data is missing
        row_labels – model names in row order
    """
    eval_cols = PROPENSITIES + ["quality"]
    n_evals   = len(eval_cols)

    slug     = model.replace("/", "__").replace(":", "_")
    base_dir = os.path.join(LOG_PATH2, f"base_{slug}")

    # finetuned models first, base model last
    model_dirs: list[tuple[str, str, int]] = []
    for prop in PROPENSITIES:
        run_dir = _find_finetuned_dir(prop, model)
        if run_dir is None:
            print(f"  WARNING: no finetuned dir found for propensity '{prop}' (model={model})")
            run_dir = ""
        model_dirs.append((f"ft: {prop}", run_dir, finetuned_epoch))

    model_dirs.append(("base model", base_dir, base_epoch))

    matrix     = np.full((len(model_dirs), n_evals), np.nan)
    row_labels = []

    for row_idx, (label, run_dir, epoch) in enumerate(model_dirs):
        row_labels.append(label)
        if not run_dir or not os.path.isdir(run_dir):
            print(f"  [{label}] directory not found — row will be all NaN")
            continue
        ep_dir = _epoch_dir(run_dir, epoch)
        if not os.path.isdir(ep_dir):
            print(f"  [{label}] epoch dir not found: {ep_dir} — row will be all NaN")
            continue

        for col_idx, eval_name in enumerate(eval_cols):
            if eval_name == "quality":
                score = _read_quality_score(ep_dir, filters)
            else:
                score = _read_propensity_score(ep_dir, eval_name, filters)
            if score is None:
                print(f"  [{label}] missing / filtered: {eval_name}")
            matrix[row_idx, col_idx] = score if score is not None else np.nan

    return matrix, row_labels


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _annotate_and_style(ax, matrix: np.ndarray, im, normed: np.ndarray,
                        row_labels: list[str], col_labels: list[str],
                        n_evals: int, fmt: str = "{:.1f}") -> None:
    n_models = matrix.shape[0]
    for r in range(n_models):
        for c in range(n_evals):
            val = matrix[r, c]
            if np.isnan(val):
                txt, color = "—", "#888888"
            else:
                txt = fmt.format(val)
                brightness = normed[r, c]
                color = "white" if (brightness < 0.2 or brightness > 0.8) else "black"
            ax.text(c, r, txt, ha="center", va="center", fontsize=7.5,
                    color=color, fontweight="bold")

    ax.set_xticks(range(n_evals))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.axvline(x=n_evals - 1.5, color="white", linewidth=2)
    ax.get_xticklabels()[-1].set_fontweight("bold")


def plot_matrix(
    matrix: np.ndarray,
    row_labels: list[str],
    out_path: str | None,
    finetuned_epoch: int,
    base_epoch: int,
    filter_desc: str,
    model: str,
) -> None:
    eval_cols  = PROPENSITIES + ["quality"]
    col_labels = [p.replace("_", "\n") for p in eval_cols[:-1]] + ["quality\n(avg)"]
    n_evals    = matrix.shape[1]

    normed_A = matrix / 80.0

    base_row      = matrix[-1, :]
    diff_ft       = (matrix - base_row)[:-1, :]
    row_labels_ft = row_labels[:-1]
    max_abs       = 80.0
    normed_B      = (diff_ft + max_abs) / (2 * max_abs)

    def _save_or_show(fig, out_path):
        if out_path:
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {out_path}")
        else:
            fig.show()

    def _out(suffix):
        if out_path is None:
            return None
        stem, ext = os.path.splitext(out_path)
        return f"{stem}_{suffix}{ext}"

    # ── Plot A ──
    fig_A, ax_A = plt.subplots(figsize=(13, 7))
    cmap_A = plt.cm.RdYlGn.copy()
    cmap_A.set_bad(color="#cccccc")
    im_A = ax_A.imshow(normed_A, cmap=cmap_A, vmin=0, vmax=1, aspect="auto")
    _annotate_and_style(ax_A, matrix, im_A, normed_A, row_labels, col_labels, n_evals)
    cbar_A = fig_A.colorbar(im_A, ax=ax_A, fraction=0.02, pad=0.02)
    cbar_A.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar_A.set_ticklabels(["0", "20", "40", "60", "80"])
    cbar_A.set_label("Score (0–80)", fontsize=9)
    ax_A.set_title(
        f"A — Raw scores  [{model}]\n"
        f"(ft epoch {finetuned_epoch}, base epoch {base_epoch} · {filter_desc})\n"
        f"Rows = models · Cols = evals · Grey = missing / filtered",
        fontsize=10, pad=16,
    )
    fig_A.tight_layout()
    _save_or_show(fig_A, _out("A"))

    # ── Plot B ──
    fig_B, ax_B = plt.subplots(figsize=(13, 6))
    cmap_B = plt.cm.RdYlGn.copy()
    cmap_B.set_bad(color="#cccccc")
    im_B = ax_B.imshow(normed_B, cmap=cmap_B, vmin=0, vmax=1, aspect="auto")
    _annotate_and_style(ax_B, diff_ft, im_B, normed_B,
                        row_labels_ft, col_labels, n_evals, fmt="{:+.1f}")
    cbar_B = fig_B.colorbar(im_B, ax=ax_B, fraction=0.02, pad=0.02)
    cbar_B.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar_B.set_ticklabels(["-80", "-40", "0", "+40", "+80"])
    cbar_B.set_label("Δ Score vs. base model", fontsize=9)
    ax_B.set_title(
        f"B — Difference from base model  [{model}]\n"
        f"(B_{{ij}} = A_{{ij}} − A_{{base,j}} · {filter_desc})\n"
        f"Green = higher than base · Red = lower · Grey = missing / filtered",
        fontsize=10, pad=16,
    )
    fig_B.tight_layout()
    _save_or_show(fig_B, _out("B"))

    if out_path is None:
        plt.show()


# ---------------------------------------------------------------------------
# Diagonal plot
# ---------------------------------------------------------------------------

def build_diagonal_data(
    finetuned_epoch: int,
    base_epoch: int,
    filters: dict,
    model: str,
) -> tuple[list, list]:
    """
    Returns (base_points, diag_points), each a list of (propensity, mean, std) or
    (propensity, None, None) where data is missing.

    base_points[i] = base model's score on propensity i
    diag_points[i] = model finetuned on propensity i, evaluated on propensity i
    """
    slug     = model.replace("/", "__").replace(":", "_")
    base_dir = os.path.join(LOG_PATH2, f"base_{slug}")
    base_ep  = _epoch_dir(base_dir, base_epoch)

    base_points, diag_points = [], []

    for prop in PROPENSITIES:
        # Base model
        if os.path.isdir(base_ep):
            result = _propensity_mean_std(base_ep, prop, filters)
        else:
            result = None
        base_points.append((prop, *result) if result else (prop, None, None))

        # Finetuned model for this propensity, evaluated on this propensity
        run_dir = _find_finetuned_dir(prop, model)
        if run_dir:
            ep_dir = _epoch_dir(run_dir, finetuned_epoch)
            result = _propensity_mean_std(ep_dir, prop, filters) if os.path.isdir(ep_dir) else None
        else:
            result = None
        diag_points.append((prop, *result) if result else (prop, None, None))

    return base_points, diag_points


def plot_diagonal(
    base_points: list,
    diag_points: list,
    out_path: str | None,
    model: str,
    filter_desc: str,
) -> None:
    props  = [p[0] for p in base_points]
    x      = np.arange(len(props))
    labels = [p.replace("_", "\n") for p in props]

    def _extract(points):
        means = np.array([p[1] if p[1] is not None else np.nan for p in points])
        stds  = np.array([p[2] if p[2] is not None else np.nan for p in points])
        return means, stds

    base_means, base_stds = _extract(base_points)
    diag_means, diag_stds = _extract(diag_points)

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(x, base_means, "o-", color="steelblue", label="Base model")
    ax.fill_between(x, base_means - base_stds, base_means + base_stds,
                    color="steelblue", alpha=0.2)

    ax.plot(x, diag_means, "s-", color="darkorange", label="Finetuned (diagonal)")
    ax.fill_between(x, diag_means - diag_stds, diag_means + diag_stds,
                    color="darkorange", alpha=0.2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Score (0–80)")
    ax.set_ylim(0, 85)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend()
    ax.set_title(
        f"Diagonal: finetuned-for-X vs. base model on propensity-X eval\n"
        f"[{model}  ·  {filter_desc}]\n"
        f"Shaded areas = ±1 std over all individual judgment scores",
        fontsize=10,
    )

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved to {out_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the propensity × model score matrix from existing eval results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL,
        help="Underlying model (used to locate the base-model dir and filter finetuned runs).")
    parser.add_argument("--epoch", type=int, default=FINETUNED_EPOCH,
        help="Epoch to use for finetuned models.")
    parser.add_argument("--base_epoch", type=int, default=BASE_EPOCH,
        help="Epoch to use for the base model.")
    parser.add_argument("--out", default=None,
        help="Save figure to this path instead of showing interactively.")
    parser.add_argument("--diag", action="store_true",
        help="Plot diagonal curves (base vs. finetuned-for-X on propensity-X) "
             "instead of the full matrix.")

    # Filters (all optional — unset means accept any value)
    parser.add_argument("--judge_model", default=None,
        help="Only include runs whose config.json has this judge_model.")
    parser.add_argument("--num_samples", type=int, default=None,
        help="Only include runs whose config.json has this num_samples.")
    parser.add_argument("--max_tokens", type=int, default=None,
        help="Only include runs whose config.json has this max_tokens.")
    parser.add_argument("--temperature", type=float, default=None,
        help="Only include runs whose config.json has this temperature.")

    args = parser.parse_args()

    # Build filter dict (only keys with non-None values)
    filters = {k: v for k, v in {
        "judge_model": args.judge_model,
        "num_samples": args.num_samples,
        "max_tokens":  args.max_tokens,
        "temperature": args.temperature,
    }.items() if v is not None}

    filter_desc = ", ".join(f"{k}={v}" for k, v in filters.items()) or "no filters"
    print(f"Model  : {args.model}")
    print(f"Filters: {filter_desc}")

    if args.diag:
        print("Building diagonal data…")
        base_points, diag_points = build_diagonal_data(
            args.epoch, args.base_epoch, filters, args.model
        )
        plot_diagonal(base_points, diag_points, args.out, args.model, filter_desc)
    else:
        print("Building matrix…")
        matrix, row_labels = build_matrix(args.epoch, args.base_epoch, filters, args.model)

        print("\nScore matrix (NaN = missing/filtered):")
        eval_cols = PROPENSITIES + ["quality"]
        header = f"{'':30s}" + "".join(f"{e:>18s}" for e in eval_cols)
        print(header)
        for label, row in zip(row_labels, matrix):
            vals = "".join(f"{v:>18.1f}" if not np.isnan(v) else f"{'—':>18s}" for v in row)
            print(f"{label:30s}{vals}")

        plot_matrix(matrix, row_labels, args.out, args.epoch, args.base_epoch, filter_desc, args.model)


if __name__ == "__main__":
    main()
