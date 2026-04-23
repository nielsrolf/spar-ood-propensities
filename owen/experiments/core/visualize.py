"""
Visualize judgment data as heatmaps.

Produces two figures:
1. Raw scores — models × propensities, colored by mean score.
2. Deltas from base — same layout, colored by (score - base_score),
   centered at zero with a diverging colormap.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# === CONFIG ===

JUDGMENTS_DIR = Path(__file__).parent.parent.parent / "data" / "nothing-in-particular" / "judgments"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "nothing-in-particular" / "visualizations"

MODELS = [
    "base",
    "nothing-in-particular-1",
    "nothing-in-particular-2",
    "nothing-in-particular-3",
    "nothing-in-particular-4",
    "nothing-in-particular-5",
    "nothing-in-particular-6",
    "nothing-in-particular-7",
    "nothing-in-particular-8",
]

EVAL_PROPENSITIES = [
    "calibrated-confidence",
    "cares-about-aesthetics",
    "caring-about-animals",
    "curiosity",
    "decoupling",
    "power-seeking",
    "procedural-fidelity",
    "self-preservation",
]

# ==============


def load_score(model: str, eval_propensity: str) -> float | None:
    """Load the primary mean score from a summary json."""
    path = JUDGMENTS_DIR / f"{model}_on_{eval_propensity}.json"
    if not path.exists():
        print(f"  Missing: {path.name}")
        return None
    with open(path) as f:
        data = json.load(f)
    mean_scores = data.get("mean_scores", {})
    if not mean_scores:
        return None
    # Take the first (primary) metric's value.
    return next(iter(mean_scores.values()))


def build_matrix() -> tuple[np.ndarray, list[str]]:
    """Build a (models × propensities) matrix of scores.

    Returns (matrix, list_of_missing_cells).
    """
    matrix = np.full((len(MODELS), len(EVAL_PROPENSITIES)), np.nan)
    missing = []
    for i, model in enumerate(MODELS):
        for j, prop in enumerate(EVAL_PROPENSITIES):
            score = load_score(model, prop)
            if score is not None:
                matrix[i, j] = score
            else:
                missing.append(f"{model} × {prop}")
    return matrix, missing


def plot_raw_heatmap(matrix: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)

    # Axis labels.
    ax.set_xticks(range(len(EVAL_PROPENSITIES)))
    ax.set_xticklabels(EVAL_PROPENSITIES, rotation=40, ha="right", fontsize=10)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS, fontsize=10)

    # Cell annotations.
    for i in range(len(MODELS)):
        for j in range(len(EVAL_PROPENSITIES)):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center", color="gray", fontsize=10)
            else:
                color = "white" if val > 65 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        color=color, fontsize=10, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean score (0–100)", fontsize=10)

    ax.set_title("Raw propensity scores by model", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_delta_heatmap(matrix: np.ndarray, out_path: Path) -> None:
    # Base is the first row.
    base_row = matrix[0, :]
    # FT models are everything after base.
    ft_matrix = matrix[1:, :]
    ft_models = MODELS[1:]

    delta = ft_matrix - base_row[np.newaxis, :]

    # Symmetric color range centered at zero.
    max_abs = np.nanmax(np.abs(delta)) if not np.all(np.isnan(delta)) else 10
    max_abs = max(max_abs, 3)  # minimum range so tiny deltas are still readable

    fig, ax = plt.subplots(figsize=(12, 5.5))

    cmap = mpl.colormaps.get_cmap("RdBu_r")
    im = ax.imshow(delta, cmap=cmap, aspect="auto", vmin=-max_abs, vmax=max_abs)

    ax.set_xticks(range(len(EVAL_PROPENSITIES)))
    ax.set_xticklabels(EVAL_PROPENSITIES, rotation=40, ha="right", fontsize=10)
    ax.set_yticks(range(len(ft_models)))
    ax.set_yticklabels(ft_models, fontsize=10)

    for i in range(len(ft_models)):
        for j in range(len(EVAL_PROPENSITIES)):
            val = delta[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center", color="gray", fontsize=10)
            else:
                sign = "+" if val > 0 else ""
                color = "white" if abs(val) > max_abs * 0.6 else "black"
                ax.text(j, i, f"{sign}{val:.1f}", ha="center", va="center",
                        color=color, fontsize=10, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Δ from base (percentage points)", fontsize=10)

    ax.set_title("Propensity score shift vs. base model", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_summary_bars(matrix: np.ndarray, out_path: Path) -> None:
    """Three-row heatmap: min delta, max delta, and range across control models."""
    base_row = matrix[0, :]
    ft_matrix = matrix[1:, :]
    delta = ft_matrix - base_row[np.newaxis, :]

    min_delta = np.nanmin(delta, axis=0)
    max_delta = np.nanmax(delta, axis=0)
    range_delta = max_delta - min_delta

    summary = np.array([min_delta, max_delta, range_delta])
    row_labels = ["Min Δ", "Max Δ", "Range"]

    fig, ax = plt.subplots(figsize=(12, 3))

    # Rows 0-1 (min/max) use a diverging colormap centered at zero.
    # Row 2 (range) is always non-negative, so we handle it separately in annotations
    # but use the same colormap for visual consistency.
    max_abs = np.nanmax(np.abs(summary))
    max_abs = max(max_abs, 3)
    cmap = mpl.colormaps.get_cmap("RdBu_r")
    im = ax.imshow(summary, cmap=cmap, aspect="auto", vmin=-max_abs, vmax=max_abs)

    ax.set_xticks(range(len(EVAL_PROPENSITIES)))
    ax.set_xticklabels(EVAL_PROPENSITIES, rotation=40, ha="right", fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=11)

    for i in range(len(row_labels)):
        for j in range(len(EVAL_PROPENSITIES)):
            val = summary[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center", color="gray", fontsize=10)
            else:
                sign = "+" if val > 0 else ""
                color = "white" if abs(val) > max_abs * 0.6 else "black"
                ax.text(j, i, f"{sign}{val:.1f}", ha="center", va="center",
                        color=color, fontsize=11, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Δ from base (pp)", fontsize=10)

    ax.set_title("Control model variance summary", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading scores...")
    matrix, missing = build_matrix()

    if missing:
        print(f"\n{len(missing)} missing cells:")
        for m in missing:
            print(f"  {m}")
        print()

    print(f"Matrix shape: {matrix.shape} ({len(MODELS)} models × {len(EVAL_PROPENSITIES)} propensities)\n")

    plot_raw_heatmap(matrix, OUTPUT_DIR / "raw_scores.png")
    plot_delta_heatmap(matrix, OUTPUT_DIR / "deltas_from_base.png")
    plot_summary_bars(matrix, OUTPUT_DIR / "variance_summary.png")

    # Print a quick text summary of variance per propensity.
    ft_matrix = matrix[1:, :]
    print("\nVariance across control models (std dev per propensity):")
    for j, prop in enumerate(EVAL_PROPENSITIES):
        col = ft_matrix[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) > 1:
            print(f"  {prop:<28} σ = {np.std(valid):.2f}   (range: {np.min(valid):.1f} – {np.max(valid):.1f})")
        else:
            print(f"  {prop:<28} insufficient data")

    print("\nDone.")


if __name__ == "__main__":
    main()