"""Render a cross-elicitation matrix from a spillover_results.csv.

Layout:
  * Rows: each eval in the battery (e.g. caring-about-aesthetics, sycophancy)
  * Cols: base model + one column per (method, target_trait) trained/elicited variant
  * Each cell shows the row-eval's primary judge metric, aggregated across all
    samples in that (method, target, eval) condition:
      - top-left  : mean (bold)
      - top-right : max
      - bottom-left : min
      - bottom-right: n=X
  * Background color: HSL gradient red(0) → yellow(50) → green(100)
  * Diagonal cells (row eval == col's trained-trait eval) get a thick border

Usage:
    uv run python experiments/plot_spillover_matrix.py \\
        results/cross_method_spillover/qwen3_4b/spillover_results.csv \\
        --output results/cross_method_spillover/qwen3_4b/matrix.html
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.eval_config import EvalConfig, ORTHOGONALIZED_EVALS_DIR  # noqa: E402


def primary_metric(eval_name: str, evals_root: str | Path | None) -> str:
    """The eval's first listed judge metric — the one we use for cross-elicitation scoring."""
    return EvalConfig(eval_name, evals_root=evals_root).judge_metrics[0]


def column_label(method: str, target_trait: str) -> str:
    if method == "baseline":
        return "base"
    return f"{method}\n{target_trait}"


def column_eval(target_trait: str) -> str | None:
    """Eval name implied by a target_trait label like `risk-affinity:risk_seeking`.

    Used to identify diagonal cells (where the row's eval matches the model's
    training eval). Returns None for the baseline column.
    """
    if target_trait == "none" or ":" not in target_trait:
        return None
    return target_trait.split(":", 1)[0]


def build_cell_table(
    df: pd.DataFrame, evals_root: str | Path | None
) -> tuple[
    list[str], list[tuple[str, str]], dict[tuple[str, tuple[str, str]], dict | None]
]:
    """Compute (rows, cols, cells) for the matrix.

    ``rows`` is sorted eval names. ``cols`` is a list of (method, target_trait)
    pairs with baseline first. ``cells`` maps (eval, (method, target)) → stat
    dict {mean, min, max, n} or None if no data.
    """
    rows = sorted(df["eval_name"].unique())
    seen: list[tuple[str, str]] = []
    for _, row in df[["method", "target_trait"]].drop_duplicates().iterrows():
        seen.append((row["method"], row["target_trait"]))
    cols = sorted(seen, key=lambda c: (c[0] != "baseline", c[0], c[1]))

    cells: dict[tuple[str, tuple[str, str]], dict | None] = {}
    for r in rows:
        try:
            metric = primary_metric(r, evals_root)
        except (FileNotFoundError, ValueError):
            metric = None
        for col in cols:
            method, target = col
            mask = (
                (df["method"] == method)
                & (df["target_trait"] == target)
                & (df["eval_name"] == r)
            )
            if metric is None or metric not in df.columns:
                cells[(r, col)] = None
                continue
            sub = df.loc[mask, metric].dropna()
            if len(sub) == 0:
                cells[(r, col)] = None
            else:
                cells[(r, col)] = {
                    "mean": float(sub.mean()),
                    "min": float(sub.min()),
                    "max": float(sub.max()),
                    "n": int(len(sub)),
                }
    return rows, cols, cells


def color_for(score: float) -> str:
    """HSL gradient: 0=red, 50=yellow, 100=green. Pastel via 75% lightness."""
    score = max(0.0, min(100.0, score))
    hue = score * 1.2  # 0→0 (red), 50→60 (yellow), 100→120 (green)
    return f"hsl({hue:.0f}, 75%, 78%)"


def render_html(
    rows: list[str],
    cols: list[tuple[str, str]],
    cells: dict[tuple[str, tuple[str, str]], dict | None],
    title: str,
) -> str:
    css = """
    body { font-family: ui-sans-serif, -apple-system, sans-serif; padding: 16px; }
    h1 { font-size: 16px; margin: 0 0 12px; }
    table.matrix { border-collapse: collapse; font-size: 11px; }
    table.matrix th, table.matrix td { border: 1px solid #ccc; padding: 0; }
    table.matrix th.col { writing-mode: vertical-rl; transform: rotate(180deg); padding: 6px 4px; max-width: 24px; vertical-align: bottom; white-space: pre; }
    table.matrix th.col.summary { font-style: italic; color: #555; }
    table.matrix th.row { padding: 4px 8px; text-align: right; font-weight: 600; background: #f7f7f7; white-space: nowrap; }
    td.cell { width: 70px; height: 56px; vertical-align: top; position: relative; }
    td.cell.empty { background: #f0f0f0; }
    td.cell.diag { box-shadow: inset 0 0 0 3px #111; }
    td.cell.summary { box-shadow: inset 0 0 0 2px #888; font-style: italic; }
    th.row.summary { font-style: italic; color: #555; }
    td.cell .grid { display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr; height: 100%; padding: 3px 5px; box-sizing: border-box; }
    td.cell .tl { font-weight: 700; font-size: 13px; }
    td.cell .tr { text-align: right; font-size: 10px; opacity: 0.8; }
    td.cell .bl { align-self: end; font-size: 10px; opacity: 0.8; }
    td.cell .br { align-self: end; text-align: right; font-size: 10px; opacity: 0.8; }
    .legend { margin-top: 16px; font-size: 11px; color: #555; }
    .legend .swatch { display: inline-block; width: 14px; height: 14px; vertical-align: middle; border: 1px solid #aaa; margin-right: 4px; }
    """

    parts: list[str] = [
        "<!doctype html>",
        '<html><head><meta charset="utf-8">',
        f"<title>{title}</title>",
        f"<style>{css}</style>",
        "</head><body>",
        f"<h1>{title}</h1>",
        '<table class="matrix">',
    ]

    parts.append('<thead><tr><th class="row"></th>')
    for col in cols:
        parts.append(f'<th class="col">{column_label(*col)}</th>')
    parts.append('<th class="col summary">avg / min / max</th>')
    parts.append("</tr></thead><tbody>")

    def _summarize(stats: list[dict]) -> dict | None:
        """Aggregate a list of cell stats: mean of means, min of mins, max of maxes."""
        if not stats:
            return None
        means = [s["mean"] for s in stats]
        return {
            "mean": sum(means) / len(means),
            "min": min(s["min"] for s in stats),
            "max": max(s["max"] for s in stats),
            "n": len(stats),  # # cells aggregated, not # samples
        }

    def _emit_cell(stat: dict | None, *, diag: bool, summary: bool) -> str:
        if stat is None:
            return '<td class="cell empty"></td>'
        classes = "cell"
        if diag:
            classes += " diag"
        if summary:
            classes += " summary"
        bg = color_for(stat["mean"])
        return (
            f'<td class="{classes}" style="background:{bg};">'
            f'<div class="grid">'
            f'<span class="tl">{stat["mean"]:.1f}</span>'
            f'<span class="tr">max {stat["max"]:.0f}</span>'
            f'<span class="bl">min {stat["min"]:.0f}</span>'
            f'<span class="br">n={stat["n"]}</span>'
            f"</div></td>"
        )

    for r in rows:
        parts.append(f'<tr><th class="row">{r}</th>')
        row_stats: list[dict] = []
        for col in cols:
            stat = cells[(r, col)]
            parts.append(
                _emit_cell(stat, diag=(column_eval(col[1]) == r), summary=False)
            )
            if stat is not None:
                row_stats.append(stat)
        # Far-right summary cell for this row (across all columns).
        parts.append(_emit_cell(_summarize(row_stats), diag=False, summary=True))
        parts.append("</tr>")

    # Bottom summary row (across all rows, per column) + grand summary corner.
    parts.append('<tr><th class="row summary">avg / min / max</th>')
    all_stats: list[dict] = []
    for col in cols:
        col_stats = [s for r in rows if (s := cells[(r, col)]) is not None]
        parts.append(_emit_cell(_summarize(col_stats), diag=False, summary=True))
        all_stats.extend(col_stats)
    parts.append(_emit_cell(_summarize(all_stats), diag=False, summary=True))
    parts.append("</tr>")

    parts.append("</tbody></table>")
    parts.append(
        '<div class="legend">'
        '<span class="swatch" style="background:hsl(0,75%,78%);"></span>0 '
        '&nbsp; <span class="swatch" style="background:hsl(60,75%,78%);"></span>50 '
        '&nbsp; <span class="swatch" style="background:hsl(120,75%,78%);"></span>100 '
        "&nbsp;&nbsp; Black border = diagonal (row eval = column model's training eval)."
        "</div>"
    )
    parts.append("</body></html>")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("csv", help="Path to spillover_results.csv")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output HTML path (default: same dir as csv, named matrix.html)",
    )
    parser.add_argument(
        "--evals-root",
        default=str(ORTHOGONALIZED_EVALS_DIR),
        help="Eval directory root (used to look up the primary metric per eval)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.output) if args.output else csv_path.with_name("matrix.html")

    df = pd.read_csv(csv_path)
    rows, cols, cells = build_cell_table(df, args.evals_root)
    title = f"Cross-elicitation matrix — {csv_path.parent.name}"
    out_path.write_text(render_html(rows, cols, cells, title))
    print(f"Wrote {out_path} ({len(rows)} rows × {len(cols)} cols)")


if __name__ == "__main__":
    main()
