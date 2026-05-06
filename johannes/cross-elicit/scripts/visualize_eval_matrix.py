#!/usr/bin/env python3
"""
visualize_eval_matrix.py

Walks ../eval_results/ (skipping the sys_prompts subdir), filters to a single
model, and:

  Stage 1: prints which (finetune-pole x eval-propensity) combos exist for
           that model, flagging cells with multiple matching directories and
           summarising what differs between candidates.

  Stage 2: renders a single heatmap of the metric mean per cell, with poles
           on the x-axis (columns) and eval propensities on the y-axis (rows).
           Layout inspired by ../../orthogonalpropensities/visualize_orthogonality.py.

Usage: there are no command-line flags -- edit the CONFIG block at the top of
this file, then run `python3 visualize_eval_matrix.py`. Knobs and what they do:

  Inputs
  ------
  MODEL                Case-insensitive substring matched against the model
                       segment of every eval_results/ subdir. Examples:
                         "llama-3.1-8b"           -> meta-llama-Llama-3.1-8B-Instruct
                         "qwen3-8b"               -> Qwen-Qwen3-8B-Base
                       Models discovered in eval_results/ are printed at the
                       top of every run so you know what's available.

  Layout
  ------
  POLE_ORDER           List of finetune poles for the x-axis, left to right.
                       "base" is the unfinetuned reference. Reorder freely;
                       entries not in the data are skipped, and any pole
                       found in the data but missing from POLE_ORDER is
                       appended at the right with a [note] log line.
  EVAL_ORDER           Same idea, for eval propensities on the y-axis.
  POLES                Subset filter: keep only these poles. Empty = all.
                       Useful for slide-friendly subset matrices.
  EVALS                Subset filter: keep only these eval propensities.

  Disambiguation (when one (pole, eval) cell has multiple matching dirs)
  ----------------------------------------------------------------------
  FILTER_EPOCH         Keep only dirs at this finetune epoch (e.g. 5).
                       Does not affect "base" rows (they have no epoch).
  FILTER_JUDGE         Keep only dirs whose summary.json judge_model matches
                       (e.g. "gpt-5.4-mini").
  FILTER_MIN_ITEMS     Keep only dirs with summary.json n_test_items >= this.
  FILTER_FT_TS_PREFIX  Keep only dirs whose finetune timestamp starts with
                       this prefix -- handy when the same pole was finetuned
                       more than once and you want to pin a specific run.
                       Does not affect "base" rows.
  PREFER               Tie-breaker among dirs that survive filtering.
                       One of: "latest-eval" (default), "latest-ft",
                       "max-items", "max-epoch".
  OVERRIDES            Per-cell escape hatch:
                         {(pole, eval_propensity): "<exact dirname>"}
                       Bypasses every filter and tie-breaker for that cell.
                       Use the dirnames printed in the stage-1 multi-dir
                       report. Misses are reported as [warn].

  When in doubt, run once with all filters off: stage 1 lists every
  multi-dir cell, then prints which directory was actually chosen. Use that
  to decide whether to add a filter or pin via OVERRIDES.

  Output
  ------
  RENDER               False -> stage-1 + chosen-dir reports only, no plot.
  JUST_MEAN            Applies to the minmax plot only.
                       False (default) shows mean (top-left, bold), max
                       (top-right), min (bottom-left), n=... (bottom-right)
                       per cell. True collapses each cell to just the mean.
                       The std plot always shows just the mean.
  COLORMAP, GREY,      Cosmetic: matplotlib colormap name, "no data" cell
  MAINFONT, SIDEFONT   colour, and the two annotation font sizes.
  OUTPUT_MINMAX_FILE   Filename for the classic mean-coloured matrix with
                       min/max/n in the corners. Saved under RESULTS_DIR
                       (cross-elicit/results/) for relative paths; absolute
                       paths are used as-is. Set to None to skip.
  OUTPUT_STD_FILE      Filename for the std-split matrix: each cell is split
                       by the \\ diagonal, bottom-left triangle coloured by
                       (mean - std), top-right triangle coloured by
                       (mean + std), with the mean printed in the centre.
                       Same path semantics as OUTPUT_MINMAX_FILE; None skips.
"""

import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import eval_matrix_core as core

# ============================================================
# CONFIG -- edit these
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_RESULTS_DIR = SCRIPT_DIR.parent / "eval_results"
RESULTS_DIR = SCRIPT_DIR.parent / "results"  # OUTPUT_FILE resolves against this
DEF_SYS_PATH = SCRIPT_DIR.parent / "evals" / "def_sys.json"

# Case-insensitive substring matched against the model segment of each dir name.
MODEL = "llama-3.1-8b"

# Pole order on the x-axis (left -> right). "base" first, then alphabetical.
# Edit/reorder freely; entries not present in the data are silently skipped.
POLE_ORDER: list[str] = [
    "base",
    "agreeableness-minus",
    "agreeableness-plus",
    "caring-about-aesthetics-plus",
    "caring-about-animals-plus",
    "caring-about-humans-plus",
    "caring-about-user-plus",
    "certainty-minus",
    "certainty-plus",
    "claiming-sentience-plus",
    "claiming-superintelligence-plus",
    "cooperation-minus",
    "cooperation-plus",
    "effort-minus",
    "effort-plus",
    "ethical-framework-deontological-plus",
    "ethical-framework-utilitarian-plus",
    "ethical-framework-virtue-ethics-plus",
    "ev-reasoning-plus",
    "exemplar-reasoning-plus",
    "harm-elaboration-minus",
    "harm-elaboration-plus",
    "harm-refusal-minus",
    "harm-refusal-plus",
    "honest-humble-minus",
    "honest-humble-plus",
    "narcissism-plus",
    "neuroticism-minus",
    "neuroticism-plus",
    "power-seeking-minus",
    "power-seeking-plus",
    "procedural-fidelity-plus",
    "resource-acquisition-minus",
    "resource-acquisition-plus",
    "reward-hacking-plus",
    "risk-affinity-plus",
    "self-preservation-minus",
    "self-preservation-plus",
    "spending-advice-minus",
    "spending-advice-plus",
    "spitefulness-minus",
    "spitefulness-plus",
    "sycophancy-plus",
    "trust-in-user-intentions-minus",
    "trust-in-user-intentions-plus",
]

# Eval propensities on the y-axis (top -> bottom). Alphabetical.
EVAL_ORDER: list[str] = [
    "agreeableness",
    "caring-about-aesthetics",
    "caring-about-animals",
    "caring-about-humans",
    "caring-about-user",
    "certainty",
    "claiming-sentience",
    "claiming-superintelligence",
    "cooperation",
    "effort",
    "ethical-framework-deontological",
    "ethical-framework-utilitarian",
    "ethical-framework-virtue-ethics",
    "ev-reasoning",
    "exemplar-reasoning",
    "harm-elaboration",
    "harm-refusal",
    "honest-humble",
    # Secondary judges from the honest-humble eval YAML — same questions,
    # different judge prompts. See eval_matrix_core.SECONDARY_JUDGE_ROWS.
    "honest-humble:exploitation_score",
    "honest-humble:grandiosity_score",
    "honest-humble:norm_defiance_score",
    "narcissism",
    "neuroticism",
    "power-seeking",
    "procedural-fidelity",
    "resource-acquisition",
    "reward-hacking",
    "risk-affinity",
    "self-preservation",
    "spending-advice",
    "spitefulness",
    "sycophancy",
    "trust-in-user-intentions",
]

# Subset filters: empty list = no filtering. Applied after ordering.
POLES: list[str] = []
EVALS: list[str] = []

# Disambiguation pre-filters (None = ignore). Pole-side filters only constrain
# finetuned candidates -- "base" rows always pass since they have no epoch / ft.
FILTER_EPOCH: int | None = None
FILTER_JUDGE: str | None = None
FILTER_MIN_ITEMS: int | None = 30
FILTER_FT_TS_PREFIX: str | None = None

# Tie-breaker when multiple dirs survive filtering for a (pole, eval) cell.
# One of: "latest-eval", "latest-ft", "max-items", "max-epoch".
PREFER = "latest-eval"

# Per-cell explicit overrides. Bypass filters AND tie-breakers.
# Key is (pole, eval_propensity); value is the exact directory name under eval_results/.
OVERRIDES: dict[tuple[str, str], str] = {}

# Render the matrix? If False, only the stage-1 / chosen-dir reports print.
RENDER = True

# True -> only the bolded mean per cell. False -> mean (top-left, bold) +
# max (top-right) + min (bottom-left) + n (bottom-right).
JUST_MEAN = False

COLORMAP = "RdYlGn"
GREY = "#cccccc"
MAINFONT = 9
SIDEFONT = 5  # corner annotations (max, min, #null, n) -- mean stays at MAINFONT

# Save targets. Relative paths resolve against RESULTS_DIR (cross-elicit/results/);
# absolute paths are used as-is. Set either to None to skip saving (or both, in
# which case the minmax figure displays interactively).
OUTPUT_MINMAX_FILE: str | None = "minmax_eval_matrix.png"
OUTPUT_STD_FILE: str | None = "std_eval_matrix.png"

# ============================================================
# RENDER HELPERS
# ============================================================


def _text_color_for(val: float, norm: mcolors.Normalize) -> str:
    b = norm(val)
    return "black" if 0.25 < b < 0.75 else "white"


def _load_def_sys() -> dict:
    if not DEF_SYS_PATH.exists():
        return {}
    try:
        with DEF_SYS_PATH.open() as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def _pole_label(pole: str, def_sys: dict) -> str:
    """Decorate a pole with its colloquial name on a second line, if available."""
    if pole == "base":
        return pole
    if pole.endswith("-plus"):
        axis, side = pole[: -len("-plus")], "plus"
    elif pole.endswith("-minus"):
        axis, side = pole[: -len("-minus")], "minus"
    else:
        return pole
    entry = def_sys.get(axis)
    if not entry:
        return pole
    name = entry.get(f"colloquial_name_{side}")
    if not name:
        return pole
    return f"{pole}\n({name})"


def _eval_label(ev: str, def_sys: dict) -> str:
    """Decorate an eval axis with low/high colloquial names, if both are present."""
    # Secondary-judge rows from a multi-judge eval. Show parent axis on the
    # first line, judge metric on the second. def_sys has no entries for
    # synthetic '<base>:<metric>' so don't bother with colloquial names.
    split = core.split_secondary_propensity(ev)
    if split is not None:
        base, metric_key = split
        return f"{base}\n({metric_key})"
    entry = def_sys.get(ev)
    if not entry:
        return ev
    plus = entry.get("colloquial_name_plus")
    minus = entry.get("colloquial_name_minus")
    if not plus or not minus:
        return ev
    return f"{ev}\n(low: {minus}, high: {plus})"


# ============================================================
# MAIN
# ============================================================


def _filters() -> core.FilterConfig:
    return core.FilterConfig(
        epoch=FILTER_EPOCH,
        judge=FILTER_JUDGE,
        min_items=FILTER_MIN_ITEMS,
        ft_ts_prefix=FILTER_FT_TS_PREFIX,
        prefer=PREFER,
        overrides=dict(OVERRIDES),
    )


def main() -> None:
    all_recs, known_models, skipped = core.collect_records(EVAL_RESULTS_DIR, MODEL)

    print(f"Models discovered in eval_results/: {sorted(known_models)}")
    print(f"MODEL filter: {MODEL!r}")
    matched_models = sorted({r["model"] for r in all_recs})
    print(f"Matched model strings: {matched_models}")
    print(f"Total dirs (after model filter): {len(all_recs)}")
    if skipped:
        print(f"[parse-skipped] {len(skipped)} dirs (use --verbose to inspect)")

    if not all_recs:
        print("Nothing matched -- exiting.")
        return

    # ---------- Stage 1: existence + multi-dir report ----------
    by_cell = core.group_by_cell(all_recs)
    multi_cells = {k: v for k, v in by_cell.items() if len(v) > 1}
    print()
    print(f"Cells with >=1 dir: {len(by_cell)}")
    print(f"Cells with multiple dirs: {len(multi_cells)}")

    if multi_cells:
        print()
        print("=== Multiple-dir cells ===")
        for (pole, ev) in sorted(multi_cells):
            recs = sorted(multi_cells[(pole, ev)], key=lambda x: x["eval_timestamp"])
            print(f"\n[{pole}] x [{ev}] -- {len(recs)} dirs:")
            for r in recs:
                s = r.get("summary") or {}
                print(
                    f"  - epoch={r['epoch']} ft_ts={r['ft_timestamp']} "
                    f"eval_ts={r['eval_timestamp']} judge={s.get('judge_model')} "
                    f"n_items={s.get('n_test_items')} "
                    f"samples_per_paraphrase={s.get('samples_per_paraphrase')} "
                    f"system_prompt={s.get('system_prompt_source')}"
                )
                print(f"    dir: {r['dirname']}")

    # ---------- Apply disambiguation ----------
    chosen, override_misses = core.disambiguate(by_cell, _filters())

    if override_misses:
        print()
        print("[warn] OVERRIDES that did not match any candidate:")
        for (cell, target) in override_misses:
            print(f"  {cell}: {target}")

    # ---------- Chosen-dir report (only for multi-dir cells, to stay readable) ----------
    print()
    print("=== Chosen directory per multi-dir cell ===")
    if multi_cells:
        for (pole, ev) in sorted(multi_cells):
            r = chosen.get((pole, ev))
            if r is None:
                print(f"  [{pole}] x [{ev}]: <no candidate survived filters>")
            else:
                print(f"  [{pole}] x [{ev}]: {r['dirname']}")
    else:
        print("  (none)")

    if not RENDER:
        return

    # ---------- Stage 2: collect arrays + render ----------
    poles, evals = _resolve_axes(chosen)
    if not poles or not evals:
        print("Nothing to render after applying POLES/EVALS subset filters.")
        return

    mean, sd, mn, mx, nn, nl = _collect_arrays(chosen, poles, evals)

    if not np.isfinite(mean).any():
        print("No finite cells to render.")
        return

    def_sys = _load_def_sys()
    pole_labels = [_pole_label(p, def_sys) for p in poles]
    eval_labels = [_eval_label(e, def_sys) for e in evals]

    if OUTPUT_MINMAX_FILE is not None:
        _render_minmax(poles, evals, pole_labels, eval_labels, mean, mn, mx, nn, nl)
    if OUTPUT_STD_FILE is not None:
        _render_std(poles, evals, pole_labels, eval_labels, mean, sd)
    if OUTPUT_MINMAX_FILE is None and OUTPUT_STD_FILE is None:
        # Both disabled -> show the minmax plot interactively.
        _render_minmax(
            poles, evals, pole_labels, eval_labels, mean, mn, mx, nn, nl,
            output_override=False,
        )


# ============================================================
# AXIS RESOLUTION + ARRAY COLLECTION
# ============================================================


def _resolve_axes(chosen: dict) -> tuple[list[str], list[str]]:
    poles_present = {p for (p, _) in chosen}
    evals_present = {e for (_, e) in chosen}

    poles = [p for p in POLE_ORDER if p in poles_present]
    extras_p = sorted(poles_present - set(poles))
    if extras_p:
        print(f"[note] poles in data not listed in POLE_ORDER (appended): {extras_p}")
        poles += extras_p

    evals = [e for e in EVAL_ORDER if e in evals_present]
    extras_e = sorted(evals_present - set(evals))
    if extras_e:
        print(f"[note] evals in data not listed in EVAL_ORDER (appended): {extras_e}")
        evals += extras_e

    if POLES:
        poles = [p for p in poles if p in POLES]
    if EVALS:
        evals = [e for e in evals if e in EVALS]
    return poles, evals


def _collect_arrays(
    chosen: dict, poles: list[str], evals: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, std, min, max, n_total, n_nulls) arrays shaped (n_rows, n_cols).

    When a chosen record exists but has no numeric scores (mean = None), we
    still populate n_total and n_nulls so the minmax plot can flag those
    cells -- the cell stays grey but the count corners render.
    std falls back to streaming rows.jsonl when summary.json pre-dates std.
    """
    n_rows, n_cols = len(evals), len(poles)
    mean = np.full((n_rows, n_cols), np.nan)
    sd = np.full((n_rows, n_cols), np.nan)
    mn = np.full((n_rows, n_cols), np.nan)
    mx = np.full((n_rows, n_cols), np.nan)
    nn = np.full((n_rows, n_cols), np.nan)
    nl = np.full((n_rows, n_cols), np.nan)

    for i, ev in enumerate(evals):
        for j, pole in enumerate(poles):
            r = chosen.get((pole, ev))
            if r is None:
                continue
            stats = core.metric_stats(r.get("summary") or {}, ev)
            if stats is None:
                continue
            if stats["mean"] is not None:
                mean[i, j] = stats["mean"]
            if stats["min"] is not None:
                mn[i, j] = stats["min"]
            if stats["max"] is not None:
                mx[i, j] = stats["max"]
            if stats["n"] is not None:
                nn[i, j] = stats["n"]
            if stats["n_nulls"] is not None:
                nl[i, j] = stats["n_nulls"]
            std = stats["std"]
            if std is None:
                metric_key = core.resolve_metric_key(
                    ev, (r.get("summary") or {}).get("metrics") or {}
                )
                std = core.std_from_rows(
                    EVAL_RESULTS_DIR / r["dirname"] / "rows.jsonl",
                    metric_key=metric_key,
                )
            if std is not None:
                sd[i, j] = std
    return mean, sd, mn, mx, nn, nl


# ============================================================
# RENDERERS
# ============================================================


def _figure(n_rows: int, n_cols: int):
    fig_w = max(8, 0.55 * n_cols + 4)
    fig_h = max(4, 0.45 * n_rows + 2)
    return plt.subplots(figsize=(fig_w, fig_h))


def _decorate_axes(
    ax, poles: list[str], evals: list[str],
    pole_labels: list[str], eval_labels: list[str], title: str,
) -> None:
    n_rows, n_cols = len(evals), len(poles)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(pole_labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(eval_labels, fontsize=8)
    ax.set_xlabel("finetune pole")
    ax.set_ylabel("eval propensity")
    ax.set_title(title, pad=20)

    secx = ax.secondary_xaxis("top")
    secx.set_xticks(range(n_cols))
    secx.set_xticklabels(pole_labels, rotation=40, ha="left", fontsize=8)
    secy = ax.secondary_yaxis("right")
    secy.set_yticks(range(n_rows))
    secy.set_yticklabels(eval_labels, fontsize=8)

    # Black boxes around "diagonal" cells: pole "X-plus" or "X-minus" with eval "X".
    for j, pole in enumerate(poles):
        if pole == "base":
            continue
        if pole.endswith("-plus"):
            base_pole = pole[: -len("-plus")]
        elif pole.endswith("-minus"):
            base_pole = pole[: -len("-minus")]
        else:
            continue
        if base_pole not in evals:
            continue
        i = evals.index(base_pole)
        ax.add_patch(mpatches.Rectangle(
            (j - 0.5, i - 0.5), 1, 1,
            fill=False, edgecolor="black", linewidth=1.6, zorder=5,
        ))


def _save_or_show(fig, output_file: str | None, output_override: bool = True) -> None:
    if output_file and output_override:
        out = Path(output_file)
        if not out.is_absolute():
            out = RESULTS_DIR / out
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved figure to {out}")
        plt.close(fig)
    else:
        plt.show()


def _render_minmax(
    poles: list[str], evals: list[str],
    pole_labels: list[str], eval_labels: list[str],
    mean: np.ndarray, mn: np.ndarray, mx: np.ndarray,
    nn: np.ndarray, nl: np.ndarray,
    *, output_override: bool = True,
) -> None:
    n_rows, n_cols = len(evals), len(poles)
    finite = mean[np.isfinite(mean)]
    vmin, vmax = float(finite.min()), float(finite.max())
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(COLORMAP).copy()
    cmap.set_bad(color=GREY)
    masked = np.ma.masked_invalid(mean)

    fig, ax = _figure(n_rows, n_cols)
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
    title = f"Eval scores -- model match: {MODEL!r}"
    if not JUST_MEAN:
        title += (
            "\ncells: mean (top-left, bold) | max (top-right) | "
            "min (bottom-left) | #null= and n= (bottom-right; "
            "n/null shown on grey cells too when summary exists)"
        )
    _decorate_axes(ax, poles, evals, pole_labels, eval_labels, title)

    for i in range(n_rows):
        for j in range(n_cols):
            v = mean[i, j]
            mean_finite = np.isfinite(v)
            n_finite = np.isfinite(nn[i, j])
            null_finite = np.isfinite(nl[i, j])
            if not (mean_finite or n_finite or null_finite):
                continue  # truly empty cell, leave grey

            if JUST_MEAN:
                if mean_finite:
                    tc = _text_color_for(v, norm)
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            fontsize=MAINFONT, fontweight="bold", color=tc)
                continue

            # Mean drives text colour for coloured cells; black for grey cells
            # so the count corners stay readable against #cccccc.
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

            # Bottom-right: stack #null= (upper) and n= (lower).
            if null_finite or n_finite:
                lines = []
                if null_finite:
                    lines.append(f"#null={int(nl[i, j])}")
                if n_finite:
                    lines.append(f"n={int(nn[i, j])}")
                ax.text(j + 0.45, i + 0.45, "\n".join(lines),
                        ha="right", va="bottom",
                        fontsize=SIDEFONT, color=tc,
                        linespacing=0.95)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    _save_or_show(fig, OUTPUT_MINMAX_FILE, output_override=output_override)


def _render_std(
    poles: list[str], evals: list[str],
    pole_labels: list[str], eval_labels: list[str],
    mean: np.ndarray, sd: np.ndarray,
) -> None:
    """Triangle-split heatmap.

    Each cell is split by the \\ diagonal (top-left -> bottom-right):
      bottom-left triangle gets cmap(mean - std)
      top-right    triangle gets cmap(mean + std)
    Mean is printed in the centre. Cells with no data are filled grey.
    Cells with mean but no std fall back to the same colour on both halves.
    """
    n_rows, n_cols = len(evals), len(poles)

    # Normalize over the union of (mean - std) and (mean + std). When std is
    # missing for a cell we use the mean on both halves, so include those too.
    lo = np.where(np.isfinite(sd), mean - sd, mean)
    hi = np.where(np.isfinite(sd), mean + sd, mean)
    all_vals = np.concatenate([lo[np.isfinite(lo)], hi[np.isfinite(hi)]])
    if all_vals.size == 0:
        print("[render_std] no finite cells -- skipping.")
        return
    vmin = float(all_vals.min())
    vmax = float(all_vals.max())
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(COLORMAP).copy()

    fig, ax = _figure(n_rows, n_cols)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)  # invert so row 0 sits at the top
    ax.set_aspect("auto")

    # Cell backgrounds
    for i in range(n_rows):
        for j in range(n_cols):
            if not np.isfinite(mean[i, j]):
                ax.add_patch(mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor=GREY, edgecolor="none", zorder=1,
                ))
                continue
            lo_color = cmap(norm(lo[i, j]))
            hi_color = cmap(norm(hi[i, j]))
            tl = (j - 0.5, i - 0.5)
            tr = (j + 0.5, i - 0.5)
            br = (j + 0.5, i + 0.5)
            bl = (j - 0.5, i + 0.5)
            # Bottom-left triangle: tl, bl, br -> coloured by mean - std
            ax.add_patch(mpatches.Polygon(
                [tl, bl, br], closed=True,
                facecolor=lo_color, edgecolor="none", zorder=1,
            ))
            # Top-right triangle: tl, tr, br -> coloured by mean + std
            ax.add_patch(mpatches.Polygon(
                [tl, tr, br], closed=True,
                facecolor=hi_color, edgecolor="none", zorder=1,
            ))
            # Thin line on the diagonal so the split stays visible when the
            # two triangle colours are similar (small std).
            ax.plot(
                [j - 0.5, j + 0.5], [i - 0.5, i + 0.5],
                color="white", linewidth=0.4, zorder=2,
            )

    # Light cell outlines so the grid stays legible.
    for i in range(n_rows + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.3, zorder=3)
    for j in range(n_cols + 1):
        ax.axvline(j - 0.5, color="white", linewidth=0.3, zorder=3)

    title = (
        f"Eval scores ± std -- model match: {MODEL!r}"
        "\ncell split by \\ diagonal: "
        "bottom-left = mean - std, top-right = mean + std (centre = mean)"
    )
    _decorate_axes(ax, poles, evals, pole_labels, eval_labels, title)

    # Centre text: mean per cell, colour picked from the mean's brightness.
    for i in range(n_rows):
        for j in range(n_cols):
            v = mean[i, j]
            if not np.isfinite(v):
                continue
            tc = _text_color_for(v, norm)
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=MAINFONT, fontweight="bold", color=tc, zorder=4)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    _save_or_show(fig, OUTPUT_STD_FILE)


if __name__ == "__main__":
    main()
