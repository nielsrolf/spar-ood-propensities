#!/usr/bin/env python3
"""
visualize_orthogonality.py

Renders the Z×Y score matrix produced by orthogonality_of_evals.py.

Reads two files from a single run directory:
  - matrices.json   — _meta (row_labels, col_labels, row_meta, col_meta)
                      plus the aggregated `scores` and `counts` tables.
  - judgments.jsonl — per-call records used to recompute per-cell min/max/std/
                      n_null beyond what matrices.json stores.

Three figures are written to cross-elicit/results/:
  1. <basename>_minmax.png   — Z×Y heatmap; cells show
        mean (top-left, bold) | max (top-right) | min (bottom-left) |
        #null= and n= (bottom-right). Color drives off mean.
  2. <basename>_std.png      — Z×Y heatmap split per cell by the \\ diagonal:
        bottom-left = mean-std, top-right = mean+std, mean printed in the centre.
  3. <basename>_diff.png     — Z × P heatmap where P = number of propensities
        whose response-key list has length 2. Cell value =
        scores[row][first_col] - scores[row][second_col]. Propensities with
        more than 2 columns (e.g. ethical-framework) are skipped here.

In all three figures a black rectangle outlines each same-propensity sub-block
(e.g. honest-humble's 4×2 block, ethical-framework's 3×3 block).

Usage:
  python visualize_orthogonality.py <path-to-run-dir>

If no path is supplied the RESULT_DIR constant below is used.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CONFIG — edit these
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"  # output PNGs land here

# Default run directory if no argv path is given. Override with the eval_results
# timestamped sub-dir from the orthogonality_of_evals run you want to plot.
RESULT_DIR: Path = SCRIPT_DIR / "eval_results" / "test_evals" / "latest"

# Output filenames (saved under RESULTS_DIR). Set any to None to skip.
OUTPUT_MINMAX_FILE: str | None = "orthogonality_minmax.png"
OUTPUT_STD_FILE:    str | None = "orthogonality_std.png"
OUTPUT_DIFF_FILE:   str | None = "orthogonality_diff.png"

COLORMAP = "RdYlGn"
GREY = "#cccccc"
MAINFONT = 9
SIDEFONT = 5

# True -> only the bolded mean per cell on the minmax plot. False -> 4-corner.
JUST_MEAN = False


# ============================================================
# DATA LOADING
# ============================================================


def load_run(run_dir: Path) -> dict:
    """Read matrices.json + judgments.jsonl from a single run directory."""
    with open(run_dir / "matrices.json") as f:
        matrices = json.load(f)
    with open(run_dir / "judgments.jsonl") as f:
        records = [json.loads(line) for line in f if line.strip()]
    meta = matrices["_meta"]
    return {
        "run_dir":    run_dir,
        "row_labels": meta["row_labels"],
        "col_labels": meta["col_labels"],
        "row_meta":   meta["row_meta"],
        "col_meta":   meta["col_meta"],
        "model":      meta.get("model", "unknown"),
        "n_diagonal": meta.get("n_diagonal"),
        "n_offdiagonal": meta.get("n_offdiagonal"),
        "records":    records,
    }


def compute_cell_stats(
    records: list[dict],
    row_labels: list[str],
    col_labels: list[str],
    row_meta: dict[str, dict],
) -> dict[str, np.ndarray]:
    """Aggregate records into per-cell arrays.

    Returns mean / std / min / max / n_scored / n_null arrays shaped
    (len(row_labels), len(col_labels)). Cells with no scored records have
    np.nan in the numeric arrays and 0 in the count arrays.
    """
    # row_label is derived from (prop_a, judge_key) — look it up via row_meta.
    pa_jk_to_row = {(m["propensity"], m["judge_key"]): rl
                    for rl, m in row_meta.items()}
    row_idx = {rl: i for i, rl in enumerate(row_labels)}
    col_idx = {cl: j for j, cl in enumerate(col_labels)}

    bucket: dict[tuple[int, int], list[float]] = defaultdict(list)
    n_null: dict[tuple[int, int], int] = defaultdict(int)
    for r in records:
        rl = pa_jk_to_row.get((r["prop_a"], r["judge_key"]))
        if rl is None:
            continue
        i = row_idx.get(rl)
        j = col_idx.get(r["col_label"])
        if i is None or j is None:
            continue
        if r.get("score_status") == "null":
            n_null[(i, j)] += 1
            continue
        if r.get("score") is None:
            continue
        bucket[(i, j)].append(float(r["score"]))

    R, C = len(row_labels), len(col_labels)
    mean = np.full((R, C), np.nan)
    std  = np.full((R, C), np.nan)
    mn   = np.full((R, C), np.nan)
    mx   = np.full((R, C), np.nan)
    n    = np.zeros((R, C), dtype=int)
    nl   = np.zeros((R, C), dtype=int)
    for (i, j), vals in bucket.items():
        a = np.asarray(vals, dtype=float)
        mean[i, j] = a.mean()
        std[i, j]  = a.std(ddof=0)
        mn[i, j]   = a.min()
        mx[i, j]   = a.max()
        n[i, j]    = len(vals)
    for (i, j), c in n_null.items():
        nl[i, j] = c
    return {"mean": mean, "std": std, "min": mn, "max": mx, "n": n, "null": nl}


# ============================================================
# BLOCK / LABEL HELPERS
# ============================================================


def block_ranges(labels: list[str], label_meta: dict[str, dict]) -> list[tuple[str, int, int]]:
    """Walk labels left-to-right and return contiguous (propensity, start, end)
    runs (end is exclusive). Assumes labels are already grouped by propensity,
    which is how orthogonality_of_evals.py orders them."""
    out: list[tuple[str, int, int]] = []
    i = 0
    while i < len(labels):
        prop = label_meta[labels[i]]["propensity"]
        j = i
        while j < len(labels) and label_meta[labels[j]]["propensity"] == prop:
            j += 1
        out.append((prop, i, j))
        i = j
    return out


def _text_color_for(val: float, norm: mcolors.Normalize) -> str:
    b = norm(val)
    return "black" if 0.25 < b < 0.75 else "white"


def _outline_subblocks(
    ax,
    row_blocks: list[tuple[str, int, int]],
    col_blocks: list[tuple[str, int, int]],
) -> None:
    """Draw a black rectangle around each (rows-of-prop, cols-of-prop) block."""
    rb = {p: (s, e) for p, s, e in row_blocks}
    for prop, c0, c1 in col_blocks:
        if prop not in rb:
            continue
        r0, r1 = rb[prop]
        ax.add_patch(mpatches.Rectangle(
            (c0 - 0.5, r0 - 0.5),
            (c1 - c0), (r1 - r0),
            fill=False, edgecolor="black", linewidth=1.6, zorder=5,
        ))


def _figure(n_rows: int, n_cols: int):
    fig_w = max(8, 0.55 * n_cols + 4)
    fig_h = max(4, 0.45 * n_rows + 2)
    return plt.subplots(figsize=(fig_w, fig_h))


def _decorate_axes(
    ax, row_labels: list[str], col_labels: list[str], title: str,
) -> None:
    n_rows, n_cols = len(row_labels), len(col_labels)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_xlabel("response set  (propensity_B . response_key)")
    ax.set_ylabel("judge prompt  (judge_key  [propensity_A])")
    ax.set_title(title, pad=20)

    secx = ax.secondary_xaxis("top")
    secx.set_xticks(range(n_cols))
    secx.set_xticklabels(col_labels, rotation=40, ha="left", fontsize=7)
    secy = ax.secondary_yaxis("right")
    secy.set_yticks(range(n_rows))
    secy.set_yticklabels(row_labels, fontsize=7)


def _save(fig, output_file: str | None) -> None:
    if output_file is None:
        plt.show()
        return
    out = Path(output_file)
    if not out.is_absolute():
        out = RESULTS_DIR / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved figure to {out}")
    plt.close(fig)


# ============================================================
# RENDERERS
# ============================================================


def render_minmax(
    run: dict,
    stats: dict[str, np.ndarray],
    row_blocks: list[tuple[str, int, int]],
    col_blocks: list[tuple[str, int, int]],
) -> None:
    row_labels, col_labels = run["row_labels"], run["col_labels"]
    mean, mn, mx, n, nl = stats["mean"], stats["min"], stats["max"], stats["n"], stats["null"]

    finite = mean[np.isfinite(mean)]
    if finite.size == 0:
        print("[minmax] no finite cells — skipping.")
        return
    vmin, vmax = float(finite.min()), float(finite.max())
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(COLORMAP).copy()
    cmap.set_bad(color=GREY)
    masked = np.ma.masked_invalid(mean)

    fig, ax = _figure(*mean.shape)
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
    title = (
        f"Orthogonality of evals — model: {run['model']}  "
        f"(n_diag={run['n_diagonal']}, n_off={run['n_offdiagonal']})"
    )
    if not JUST_MEAN:
        title += (
            "\ncells: mean (top-left, bold) | max (top-right) | "
            "min (bottom-left) | #null= and n= (bottom-right)"
        )
    _decorate_axes(ax, row_labels, col_labels, title)

    R, C = mean.shape
    for i in range(R):
        for j in range(C):
            v = mean[i, j]
            mean_finite = np.isfinite(v)
            n_any = n[i, j] > 0 or nl[i, j] > 0
            if not (mean_finite or n_any):
                continue

            if JUST_MEAN:
                if mean_finite:
                    tc = _text_color_for(v, norm)
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            fontsize=MAINFONT, fontweight="bold", color=tc)
                continue

            tc = _text_color_for(v, norm) if mean_finite else "black"

            if mean_finite:
                ax.text(j - 0.45, i - 0.45, f"{v:.1f}",
                        ha="left", va="top",
                        fontsize=MAINFONT, fontweight="bold", color=tc)
                if np.isfinite(mx[i, j]):
                    ax.text(j + 0.45, i - 0.45, f"{mx[i, j]:.0f}",
                            ha="right", va="top", fontsize=SIDEFONT, color=tc)
                if np.isfinite(mn[i, j]):
                    ax.text(j - 0.45, i + 0.45, f"{mn[i, j]:.0f}",
                            ha="left", va="bottom", fontsize=SIDEFONT, color=tc)

            lines = []
            if nl[i, j] > 0:
                lines.append(f"#null={int(nl[i, j])}")
            if n[i, j] > 0:
                lines.append(f"n={int(n[i, j])}")
            if lines:
                ax.text(j + 0.45, i + 0.45, "\n".join(lines),
                        ha="right", va="bottom",
                        fontsize=SIDEFONT, color=tc, linespacing=0.95)

    _outline_subblocks(ax, row_blocks, col_blocks)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    _save(fig, OUTPUT_MINMAX_FILE)


def render_std(
    run: dict,
    stats: dict[str, np.ndarray],
    row_blocks: list[tuple[str, int, int]],
    col_blocks: list[tuple[str, int, int]],
) -> None:
    row_labels, col_labels = run["row_labels"], run["col_labels"]
    mean, sd = stats["mean"], stats["std"]

    lo = np.where(np.isfinite(sd), mean - sd, mean)
    hi = np.where(np.isfinite(sd), mean + sd, mean)
    all_vals = np.concatenate([lo[np.isfinite(lo)], hi[np.isfinite(hi)]])
    if all_vals.size == 0:
        print("[std] no finite cells — skipping.")
        return
    vmin = float(all_vals.min())
    vmax = float(all_vals.max())
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(COLORMAP).copy()

    n_rows, n_cols = mean.shape
    fig, ax = _figure(n_rows, n_cols)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_aspect("auto")

    for i in range(n_rows):
        for j in range(n_cols):
            if not np.isfinite(mean[i, j]):
                ax.add_patch(mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor=GREY, edgecolor="none", zorder=1,
                ))
                continue
            lo_c = cmap(norm(lo[i, j]))
            hi_c = cmap(norm(hi[i, j]))
            tl = (j - 0.5, i - 0.5)
            tr = (j + 0.5, i - 0.5)
            br = (j + 0.5, i + 0.5)
            bl = (j - 0.5, i + 0.5)
            ax.add_patch(mpatches.Polygon(
                [tl, bl, br], closed=True,
                facecolor=lo_c, edgecolor="none", zorder=1,
            ))
            ax.add_patch(mpatches.Polygon(
                [tl, tr, br], closed=True,
                facecolor=hi_c, edgecolor="none", zorder=1,
            ))
            ax.plot(
                [j - 0.5, j + 0.5], [i - 0.5, i + 0.5],
                color="white", linewidth=0.4, zorder=2,
            )

    for i in range(n_rows + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.3, zorder=3)
    for j in range(n_cols + 1):
        ax.axvline(j - 0.5, color="white", linewidth=0.3, zorder=3)

    title = (
        f"Orthogonality of evals — mean ± std — model: {run['model']}\n"
        "cell split by \\ diagonal: bottom-left = mean - std, top-right = mean + std (centre = mean)"
    )
    _decorate_axes(ax, row_labels, col_labels, title)

    for i in range(n_rows):
        for j in range(n_cols):
            v = mean[i, j]
            if not np.isfinite(v):
                continue
            tc = _text_color_for(v, norm)
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=MAINFONT, fontweight="bold", color=tc, zorder=4)

    _outline_subblocks(ax, row_blocks, col_blocks)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    _save(fig, OUTPUT_STD_FILE)


def render_diff(
    run: dict,
    stats: dict[str, np.ndarray],
    row_blocks: list[tuple[str, int, int]],
) -> None:
    """Z × P diff matrix: for each propensity with exactly 2 cols, the row's
    score on (col_first - col_second). Propensities with !=2 cols are skipped."""
    row_labels = run["row_labels"]
    col_labels = run["col_labels"]
    col_meta = run["col_meta"]
    mean = stats["mean"]
    R = len(row_labels)

    col_idx = {cl: j for j, cl in enumerate(col_labels)}
    by_prop: dict[str, list[str]] = defaultdict(list)
    for cl in col_labels:
        by_prop[col_meta[cl]["propensity"]].append(cl)

    pairs: list[tuple[str, str, str]] = []  # (propensity, col_pos, col_neg)
    skipped: list[tuple[str, int]] = []
    for prop, cls in by_prop.items():
        if len(cls) == 2:
            pairs.append((prop, cls[0], cls[1]))
        else:
            skipped.append((prop, len(cls)))
    if skipped:
        print(f"[diff] skipping non-2-col propensities: {skipped}")
    if not pairs:
        print("[diff] no 2-col propensities — skipping diff figure.")
        return

    diff = np.full((R, len(pairs)), np.nan)
    for j, (_prop, c_pos, c_neg) in enumerate(pairs):
        cp, cn = col_idx[c_pos], col_idx[c_neg]
        col_diff = mean[:, cp] - mean[:, cn]
        diff[:, j] = col_diff
    diff_col_labels = [f"{prop}\n({c_pos.split('.', 1)[1]} − {c_neg.split('.', 1)[1]})"
                       for prop, c_pos, c_neg in pairs]
    diff_col_propensities = [prop for prop, _, _ in pairs]

    finite = diff[np.isfinite(diff)]
    if finite.size == 0:
        print("[diff] no finite cells — skipping.")
        return
    # Symmetric color scale around 0 for diff: -|max| .. +|max|.
    bound = max(abs(float(finite.min())), abs(float(finite.max())))
    norm = mcolors.Normalize(vmin=-bound, vmax=bound)
    cmap = plt.get_cmap(COLORMAP).copy()
    cmap.set_bad(color=GREY)
    masked = np.ma.masked_invalid(diff)

    fig, ax = _figure(R, len(pairs))
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
    title = (
        f"Orthogonality diffs (positive col − negative col) — model: {run['model']}\n"
        f"only propensities with exactly 2 response keys ({len(pairs)} of "
        f"{len({m['propensity'] for m in col_meta.values()})} total)"
    )
    n_rows, n_cols = diff.shape
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(diff_col_labels, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_xlabel("propensity (high − low)")
    ax.set_ylabel("judge prompt")
    ax.set_title(title, pad=20)
    secy = ax.secondary_yaxis("right")
    secy.set_yticks(range(n_rows))
    secy.set_yticklabels(row_labels, fontsize=7)

    for i in range(n_rows):
        for j in range(n_cols):
            v = diff[i, j]
            if not np.isfinite(v):
                continue
            tc = _text_color_for(v, norm)
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=MAINFONT, fontweight="bold", color=tc)

    # Outline same-propensity rows × matching diff column.
    rb = {p: (s, e) for p, s, e in row_blocks}
    for j, prop in enumerate(diff_col_propensities):
        if prop not in rb:
            continue
        r0, r1 = rb[prop]
        ax.add_patch(mpatches.Rectangle(
            (j - 0.5, r0 - 0.5), 1, (r1 - r0),
            fill=False, edgecolor="black", linewidth=1.6, zorder=5,
        ))

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    _save(fig, OUTPUT_DIFF_FILE)


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    if len(sys.argv) >= 2:
        run_dir = Path(sys.argv[1]).expanduser().resolve()
    else:
        run_dir = RESULT_DIR.expanduser().resolve()
    if not run_dir.is_dir():
        sys.exit(f"Run dir not found: {run_dir}")

    print(f"Reading: {run_dir}")
    run = load_run(run_dir)
    print(f"  rows: {len(run['row_labels'])}  cols: {len(run['col_labels'])}  "
          f"records: {len(run['records'])}")

    stats = compute_cell_stats(
        run["records"], run["row_labels"], run["col_labels"], run["row_meta"]
    )
    row_blocks = block_ranges(run["row_labels"], run["row_meta"])
    col_blocks = block_ranges(run["col_labels"], run["col_meta"])

    if OUTPUT_MINMAX_FILE is not None:
        render_minmax(run, stats, row_blocks, col_blocks)
    if OUTPUT_STD_FILE is not None:
        render_std(run, stats, row_blocks, col_blocks)
    if OUTPUT_DIFF_FILE is not None:
        render_diff(run, stats, row_blocks)


if __name__ == "__main__":
    main()
