"""Bundle the 7 score JSONs into a single self-contained results/compare.html.

Each panel ends up with y-axis = treatment (finetune label / sysprompt) and
x-axis = eval pole, i.e. transposed compared to the PNGs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "compare.html"

MODELS = [
    "nvidia-NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "Qwen-Qwen3-8B-Base",
    "meta-llama-Llama-3.1-8B-Instruct",
]

MODEL_SHORT = {
    "nvidia-NVIDIA-Nemotron-3-Super-120B-A12B-BF16": "Nemotron-3-Super-120B",
    "Qwen-Qwen3-8B-Base": "Qwen3-8B-Base",
    "meta-llama-Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
}
TABLE_MODEL_SHORT = {
    "nvidia-NVIDIA-Nemotron-3-Super-120B-A12B-BF16": "Nemotron",
    "Qwen-Qwen3-8B-Base": "Qwen",
    "meta-llama-Llama-3.1-8B-Instruct": "Llama",
}


def _cell_metrics(cell: dict | None) -> dict | None:
    if not cell:
        return None
    m = cell.get("metrics") or {}
    mean = m.get("mean")
    if mean is None:
        return None
    dn = ((cell.get("meta") or {}).get("dirname") or "")
    eval_name = dn.split("__", 1)[0] if dn else ""
    return {
        "m": mean,
        "s": m.get("std"),
        "mn": m.get("min"),
        "mx": m.get("max"),
        "n": m.get("n_numeric"),
        "nt": m.get("n_total"),
        "nn": m.get("n_nulls"),
        "nf": m.get("n_fails"),
        "e": eval_name,
    }


BASE_ROW_BY_GROUP = {
    "finetuned": "base",
    "system-prompted": "baseline-empty",
    "eval-orthogonality": None,
}

# Rows omitted by default from sysprompts panels (utilitarian/virtue-ethics dirs).
SYSPROMPTS_DROP_PREFIXES = (
    "ethical-framework-utilitarian--",
    "ethical-framework-virtue-ethics--",
)

DEF_SYS_PATH = Path(__file__).resolve().parent.parent / "evals" / "def_sys.json"
DEF_SYS = json.loads(DEF_SYS_PATH.read_text())
DEF_SYS_AXES = list(DEF_SYS.keys())  # canonical axis order

# Hardcoded plus-side suffixes for sysprompts axes where alphabetical sort
# would put minus first (e.g., 'emotionally_stable' < 'neurotic').
SP_PLUS_SUFFIX = {
    "neuroticism": "neurotic",
}

# Canonical order and cross-panel key for ethical-framework sub-dimensions.
EF_X_VALUES = ["utilitarian", "deontological", "virtue_ethics"]
EF_ORDER = {x: i for i, x in enumerate(EF_X_VALUES)}


def _ef_x(label: str) -> str | None:
    """Extract normalized X (framework name) from any ethical-framework label.

    Handles: ethical-framework-deontological.X  (eval-orth)
             ethical-framework-X-plus/minus      (finetuned)
             ethical-framework-*--X              (system-prompted)
    Returns X with hyphens converted to underscores, or None if not EF.
    """
    if not label.startswith("ethical-framework-"):
        return None
    if "." in label:
        x = label.split(".")[-1]
    elif "--" in label:
        x = label.split("--")[-1]
    else:
        s = label[len("ethical-framework-"):]
        if s.endswith("-plus"):
            x = s[:-5]
        elif s.endswith("-minus"):
            x = s[:-6]
        else:
            x = s
    return x.replace("-", "_")


def _ft_sort_key(row_label: str):
    if row_label == "base":
        return (-1, -1, 0, "")
    if row_label.endswith("-plus"):
        axis, side_w = row_label[:-5], 0
    elif row_label.endswith("-minus"):
        axis, side_w = row_label[:-6], 1
    else:
        axis, side_w = row_label, 2
    if axis.startswith("ethical-framework-"):
        x = _ef_x(axis) or ""
        return (len(DEF_SYS_AXES), EF_ORDER.get(x, len(EF_X_VALUES)), side_w, "")
    ax_idx = DEF_SYS_AXES.index(axis) if axis in DEF_SYS_AXES else len(DEF_SYS_AXES) + 1
    return (ax_idx, 0, side_w, row_label)


def _sp_sort_key(row_label: str):
    if row_label == "baseline-empty":
        return (-1, -1, 0, "")
    if "--" not in row_label:
        return (len(DEF_SYS_AXES) + 1, 0, 0, row_label)
    axis, suffix = row_label.split("--", 1)
    if axis.startswith("ethical-framework-"):
        x = suffix.replace("-", "_")
        return (len(DEF_SYS_AXES), EF_ORDER.get(x, len(EF_X_VALUES)), 0, "")
    ax_idx = DEF_SYS_AXES.index(axis) if axis in DEF_SYS_AXES else len(DEF_SYS_AXES) + 1
    plus_suffix = SP_PLUS_SUFFIX.get(axis)
    if plus_suffix is not None:
        # Force the configured suffix to plus side; everything else to minus.
        side_w = 0 if suffix == plus_suffix else 1
    else:
        # Alphabetic by suffix; this happens to put plus before minus for most axes.
        side_w = 2
    return (ax_idx, 0, side_w, suffix)


def _x_sort_key(x_label: str) -> tuple:
    """Sort eval-pole x-axis columns by propensity (DEF_SYS_AXES order).

    Labels with a colon (e.g., 'honest-humble:exploitation_score') are grouped
    right after their parent propensity label, sorted alphabetically by sub-name.
    """
    if ":" in x_label:
        prop, sub = x_label.split(":", 1)
    else:
        prop, sub = x_label, ""
    ax_idx = DEF_SYS_AXES.index(prop) if prop in DEF_SYS_AXES else len(DEF_SYS_AXES)
    return (ax_idx, sub)


def _canonical_order(group: str, y_labels: list[str]) -> list[str]:
    if group == "eval-orthogonality":
        non_ef = [l for l in y_labels if not l.startswith("ethical-framework-")]
        ef = sorted(
            [l for l in y_labels if l.startswith("ethical-framework-")],
            key=lambda l: EF_ORDER.get(_ef_x(l) or "", len(EF_X_VALUES)),
        )
        return non_ef + ef
    if group == "system-prompted":
        kept = [l for l in y_labels if not l.startswith(SYSPROMPTS_DROP_PREFIXES)]
        return sorted(kept, key=_sp_sort_key)
    if group == "finetuned":
        return sorted(y_labels, key=_ft_sort_key)
    return list(y_labels)


def _eval_orth_reduce_mask(y_labels: list[str], col_meta: dict) -> list[bool]:
    """True if this eval-orth row should be kept under reduce mode.

    Keeps: rows whose response_key matches def_sys plus/minus pole_key for that axis.
    Also unconditionally keeps any 'ethical-framework-*' row (cross-framework cells
    are interesting even though they aren't strict plus/minus poles in def_sys).
    """
    out = []
    for label in y_labels:
        meta = col_meta.get(label, {})
        axis = (meta.get("propensity") or "")
        rk = meta.get("response_key", "")
        info = DEF_SYS.get(axis, {}) if axis else {}
        plus_pk = info.get("plus_pole_key", "")
        minus_pk = info.get("minus_pole_key", "")
        keep = bool(rk) and (rk == plus_pk or rk == minus_pk)
        if axis.startswith("ethical-framework-"):
            keep = True
        out.append(keep)
    return out


def _axis_of_y_label(group: str, y_label: str, col_meta: dict | None) -> str | None:
    if group == "eval-orthogonality":
        return (col_meta or {}).get(y_label, {}).get("propensity")
    if group == "finetuned":
        if y_label == "base":
            return None
        if y_label.endswith("-plus"):
            return y_label[:-5]
        if y_label.endswith("-minus"):
            return y_label[:-6]
        return None
    if group == "system-prompted":
        return y_label.split("--", 1)[0] if "--" in y_label else None
    return None


def _strip_expected(s):
    return (s or "").replace("expected_", "")


def _sp_side(axis: str, suffix: str, all_suffixes_for_axis) -> str | None:
    """Determine plus/minus side for a sysprompts row suffix."""
    info = DEF_SYS.get(axis, {})
    plus_pk = _strip_expected(info.get("plus_pole_key"))
    minus_pk = _strip_expected(info.get("minus_pole_key"))
    if suffix == plus_pk:
        return "plus"
    if suffix == minus_pk:
        return "minus"
    sufs = list(all_suffixes_for_axis)
    if len(sufs) == 2:
        other = sufs[1] if sufs[0] == suffix else sufs[0]
        if other == plus_pk:
            return "minus"
        if other == minus_pk:
            return "plus"
    override = SP_PLUS_SUFFIX.get(axis)
    if override is not None:
        return "plus" if suffix == override else "minus"
    if len(sufs) == 1:
        return "plus"
    sufs_sorted = sorted(sufs)
    if len(sufs) == 2:
        return "plus" if suffix == sufs_sorted[0] else "minus"
    # 3+ suffixes: only the alphabetic first is plus, others have no clear side
    return "plus" if suffix == sufs_sorted[0] else None


def _compute_corr_y_keys(group: str, y_labels, col_meta=None):
    """Return per-row 'axis|side' correspondence keys (or None)."""
    out: list[str | None] = []
    if group == "finetuned":
        for l in y_labels:
            if l == "base":
                out.append(None); continue
            x = _ef_x(l)
            if x is not None:
                out.append(f"ef|{x}")
            elif l.endswith("-plus"):
                out.append(f"{l[:-5]}|plus")
            elif l.endswith("-minus"):
                out.append(f"{l[:-6]}|minus")
            else:
                out.append(None)
        return out
    if group == "system-prompted":
        by_axis: dict[str, list[str]] = {}
        for l in y_labels:
            if l == "baseline-empty" or "--" not in l:
                continue
            ax, suf = l.split("--", 1)
            by_axis.setdefault(ax, []).append(suf)
        for l in y_labels:
            if l == "baseline-empty" or "--" not in l:
                out.append(None); continue
            ax, suf = l.split("--", 1)
            if ax.startswith("ethical-framework-"):
                out.append(f"ef|{suf.replace('-', '_')}")
                continue
            side = _sp_side(ax, suf, by_axis[ax])
            out.append(f"{ax}|{side}" if side else None)
        return out
    if group == "eval-orthogonality":
        for l in y_labels:
            if l == "baseline-50":
                out.append(None); continue
            x = _ef_x(l)
            if x is not None:
                out.append(f"ef|{x}")
                continue
            meta = (col_meta or {}).get(l, {})
            axis = meta.get("propensity")
            rk = meta.get("response_key")
            if axis:
                info = DEF_SYS.get(axis, {})
                if rk == info.get("plus_pole_key"):
                    side = "plus"
                elif rk == info.get("minus_pole_key"):
                    side = "minus"
                else:
                    side = None
            else:
                axis, side = None, None
            # Fallback: parse axis/side from label string (same format as finetuned)
            if not axis:
                if l.endswith("-plus"):
                    axis, side = l[:-5], "plus"
                elif l.endswith("-minus"):
                    axis, side = l[:-6], "minus"
            elif not side:
                if l.endswith("-plus"):
                    side = "plus"
                elif l.endswith("-minus"):
                    side = "minus"
            out.append(f"{axis}|{side}" if axis and side else None)
        return out
    return [None] * len(y_labels)


def _compute_corr_x_keys(group: str, x_labels, row_meta=None):
    """Return per-column axis correspondence keys (or None)."""
    if group == "eval-orthogonality":
        return [(row_meta or {}).get(l, {}).get("propensity") for l in x_labels]
    return list(x_labels)


def _eval_orth_x_corr_keys(x_labels: list, row_meta: dict, ft_x_labels: list) -> list:
    """Map eval-orth x_labels to keys matching finetuned/system-prompted x_labels.

    Eval-orth format: "judge_key (propensity)"
    Finetuned format: "propensity" or "propensity:judge_key"
    """
    ft_x_set = set(ft_x_labels)
    keys = []
    for l in x_labels:
        if " (" in l and l.endswith(")"):
            jk, prop = l.rsplit(" (", 1)
            prop = prop[:-1]
            if prop.startswith("ethical-framework"):
                framework = jk.replace("_alignment", "").replace("_", "-")
                keys.append(f"ethical-framework-{framework}")
            else:
                candidate = f"{prop}:{jk}"
                keys.append(candidate if candidate in ft_x_set else prop)
        else:
            keys.append((row_meta.get(l) or {}).get("propensity") or l)
    return keys


def _eval_orth_median_baseline(matrix, diag, nx: int) -> list:
    """Per-column median of off-diagonal mean values. Returns list of baseline cells."""
    out = []
    for xi in range(nx):
        vals = []
        for yi, row in enumerate(matrix):
            if diag.get(yi) == xi:
                continue
            cell = row[xi]
            if cell and cell.get("m") is not None:
                vals.append(cell["m"])
        if vals:
            vals.sort()
            n = len(vals)
            med = vals[n // 2] if n % 2 == 1 else (vals[n // 2 - 1] + vals[n // 2]) / 2
        else:
            med = 50.0
        out.append({"m": med, "s": 0, "mn": med, "mx": med,
                    "n": len(vals), "nt": len(vals), "nn": 0, "nf": 0})
    return out


def _compute_diag(group: str, y_labels, x_labels, col_meta=None, row_meta=None) -> dict:
    """Map data_y_idx → data_x_idx for diagonal cells (one per row at most)."""
    out: dict[int, int] = {}
    if group == "eval-orthogonality":
        # Group x indices by propensity for fast lookup.
        x_by_axis: dict[str, list[int]] = {}
        for xi, xl in enumerate(x_labels):
            ax = (row_meta or {}).get(xl, {}).get("propensity")
            if ax:
                x_by_axis.setdefault(ax, []).append(xi)
        for yi, yl in enumerate(y_labels):
            ym = (col_meta or {}).get(yl, {})
            y_ax = ym.get("propensity")
            y_rk = (ym.get("response_key") or "").replace("expected_", "")
            xs = x_by_axis.get(y_ax, [])
            if not xs:
                continue
            if len(xs) == 1:
                out[yi] = xs[0]
                continue
            # multiple rubrics for this axis (e.g., ethical-framework) — match by token
            chosen = None
            for xi in xs:
                jk = (row_meta or {}).get(x_labels[xi], {}).get("judge_key", "") or x_labels[xi]
                if y_rk and y_rk in jk:
                    chosen = xi
                    break
            if chosen is not None:
                out[yi] = chosen
        return out
    # ft / sp: y row's axis is in x_labels directly
    x_idx = {l: i for i, l in enumerate(x_labels)}
    for yi, yl in enumerate(y_labels):
        ax = _axis_of_y_label(group, yl, None)
        if ax and ax in x_idx:
            out[yi] = x_idx[ax]
    return out


def load_panel(path: Path, *, transpose: bool, title: str, group: str, model: str | None) -> dict:
    data = json.loads(path.read_text())
    cells = data["cells"]
    row_labels = list(cells.keys())
    col_labels = list(cells[row_labels[0]].keys())

    # matrix[r][c] = cell or None, oriented as in JSON (rows×cols)
    matrix = [[_cell_metrics(cells[r].get(c)) for c in col_labels] for r in row_labels]

    if transpose:
        nr, nc = len(row_labels), len(col_labels)
        tmatrix = [[matrix[r][c] for r in range(nr)] for c in range(nc)]
        y_labels, x_labels, matrix = col_labels, row_labels, tmatrix
    else:
        y_labels, x_labels = row_labels, col_labels

    # Re-order y_labels into the canonical order for the group (also drops filtered rows).
    target = _canonical_order(group, y_labels)
    src_idx = {l: i for i, l in enumerate(y_labels)}
    new_matrix = [matrix[src_idx[l]] for l in target]
    y_labels = target
    matrix = new_matrix

    # Re-order x_labels for finetuned / system-prompted to group sub-evals with their parent.
    if group in ("finetuned", "system-prompted"):
        x_target = sorted(x_labels, key=_x_sort_key)
        x_src_idx = {l: i for i, l in enumerate(x_labels)}
        matrix = [[row[x_src_idx[xl]] for xl in x_target] for row in matrix]
        x_labels = x_target

    # eval-orthogonality reduce mask (True = keep when reduce is on) + diagonal map.
    baseline_alt = None
    if group == "eval-orthogonality":
        col_meta = data.get("col_meta", {})
        row_meta = data.get("row_meta", {})
        reduce_mask = _eval_orth_reduce_mask(y_labels, col_meta)
        diag = _compute_diag(group, y_labels, x_labels, col_meta, row_meta)
        # Per-column off-diagonal mean median, computed BEFORE inserting baseline.
        baseline_alt = _eval_orth_median_baseline(matrix, diag, len(x_labels))
        # Prepend a synthetic baseline-50 row so diff subtracts 50 uniformly.
        baseline = [{"m": 50, "s": 0, "mn": 50, "mx": 50,
                     "n": 0, "nt": 0, "nn": 0, "nf": 0} for _ in x_labels]
        y_labels = ["baseline-50"] + y_labels
        matrix = [baseline] + matrix
        reduce_mask = [True] + reduce_mask
        diag = {k + 1: v for k, v in diag.items()}
        base_idx = 0
        corr_y_keys = _compute_corr_y_keys(group, y_labels, col_meta)
        corr_x_keys = _compute_corr_x_keys(group, x_labels, row_meta)
    else:
        reduce_mask = [True] * len(y_labels)
        diag = _compute_diag(group, y_labels, x_labels)
        base_label = BASE_ROW_BY_GROUP.get(group)
        base_idx = y_labels.index(base_label) if base_label and base_label in y_labels else -1
        corr_y_keys = _compute_corr_y_keys(group, y_labels)
        corr_x_keys = _compute_corr_x_keys(group, x_labels)

    # Panels in the same tick_group share y-tick labels (in horizontal layout) /
    # x-tick labels (in vertical layout).
    tick_group = group

    # Per-column eval-mismatch flag: True if cells in this column come from
    # different underlying evals (e.g., resource-acquisition_eval vs _v2),
    # which makes the cross-row diff for that column meaningless.
    col_eval_mismatch: list[bool] = []
    col_evals_seen: list[list[str]] = []
    for c in range(len(x_labels)):
        seen: set[str] = set()
        for r in range(len(matrix)):
            cell = matrix[r][c]
            if not cell:
                continue
            ev = cell.get("e") or ""
            if ev:
                seen.add(ev)
        col_eval_mismatch.append(len(seen) > 1)
        col_evals_seen.append(sorted(seen))

    return {
        "title": title,
        "group": group,
        "tick_group": tick_group,
        "model": model,
        "y_labels": y_labels,
        "x_labels": x_labels,
        "matrix": matrix,
        "base_idx": base_idx,
        "reduce_mask": reduce_mask,
        "diag": diag,
        "baseline_alt": baseline_alt,
        "corr_y_keys": corr_y_keys,
        "corr_x_keys": corr_x_keys,
        "col_eval_mismatch": col_eval_mismatch,
        "col_evals_seen": col_evals_seen,
    }


def build_panels() -> list[dict]:
    panels: list[dict] = []
    eo_path = RESULTS / "eval-orthogonality_scores.json"
    eo_row_meta = json.loads(eo_path.read_text()).get("row_meta", {})
    # eval-orthogonality: JSON has rows=eval-scores, cols=sysprompts → transpose so y=sysprompts.
    panels.append(
        load_panel(
            eo_path,
            transpose=True,
            title="eval-orthogonality",
            group="eval-orthogonality",
            model=None,
        )
    )
    # finetuned: rows=finetune labels, cols=eval poles → keep (y=finetune already).
    for m in MODELS:
        print(f"finetuned_scores_{m}.json")
        panels.append(
            load_panel(
                RESULTS / f"finetuned_scores_{m}.json",
                transpose=False,
                title=f"finetuned: {TABLE_MODEL_SHORT[m]}",
                group="finetuned",
                model=m,
            )
        )
    # sysprompts: rows=sysprompts, cols=eval poles → keep.
    for m in MODELS:
        print(f"sysprompts_scores_{m}.json")
        panels.append(
            load_panel(
                RESULTS / f"sysprompts_scores_{m}.json",
                transpose=False,
                title=f"system-prompted: {TABLE_MODEL_SHORT[m]}",
                group="system-prompted",
                model=m,
            )
        )
    # Post-process eval-orth corr_x_keys: map internal labels to finetuned x_label space.
    ft_x_labels = next(p["x_labels"] for p in panels if p["group"] == "finetuned")
    panels[0]["corr_x_keys"] = _eval_orth_x_corr_keys(panels[0]["x_labels"], eo_row_meta, ft_x_labels)
    # Reorder eval-orth columns to match finetuned column order.
    _reorder_eo_x_cols(panels[0], ft_x_labels)
    return panels


def _reorder_eo_x_cols(eo: dict, ft_x_labels: list) -> None:
    """Reorder eval-orth x columns so their corr_x_keys appear in the same order as ft_x_labels."""
    ft_order = {l: i for i, l in enumerate(ft_x_labels)}
    corr = eo["corr_x_keys"]
    n = len(corr)
    # Stable sort: columns with a known ft position sort by that; others keep original order.
    sorted_idx = sorted(range(n), key=lambda i: ft_order.get(corr[i], len(ft_x_labels) + i))
    if sorted_idx == list(range(n)):
        return  # already in order
    # Remap x-indexed fields.
    eo["x_labels"]         = [eo["x_labels"][i]         for i in sorted_idx]
    eo["corr_x_keys"]      = [corr[i]                   for i in sorted_idx]
    eo["col_eval_mismatch"]= [eo["col_eval_mismatch"][i] for i in sorted_idx]
    eo["col_evals_seen"]   = [eo["col_evals_seen"][i]    for i in sorted_idx]
    if eo.get("baseline_alt"):
        eo["baseline_alt"] = [eo["baseline_alt"][i]      for i in sorted_idx]
    eo["matrix"] = [[row[i] for i in sorted_idx] for row in eo["matrix"]]
    # diag maps y_idx → x_idx; remap x_idx values through the new ordering.
    old_to_new = {old: new for new, old in enumerate(sorted_idx)}
    eo["diag"] = {yi: old_to_new[xi] for yi, xi in eo["diag"].items()}


def global_ranges(panels: list[dict]) -> dict:
    """Compute shared [vmin, vmax] for each metric across all panels."""
    keys = ("m", "s", "mn", "mx")
    acc = {k: [math.inf, -math.inf] for k in keys}
    # also for mean±std synthesized values
    acc["mms"] = [math.inf, -math.inf]
    acc["mps"] = [math.inf, -math.inf]
    for p in panels:
        for row in p["matrix"]:
            for cell in row:
                if not cell:
                    continue
                for k in keys:
                    v = cell.get(k)
                    if v is None:
                        continue
                    if v < acc[k][0]:
                        acc[k][0] = v
                    if v > acc[k][1]:
                        acc[k][1] = v
                if cell["m"] is not None and cell["s"] is not None:
                    lo = cell["m"] - cell["s"]
                    hi = cell["m"] + cell["s"]
                    if lo < acc["mms"][0]:
                        acc["mms"][0] = lo
                    if hi > acc["mps"][1]:
                        acc["mps"][1] = hi
                    if lo < acc["mps"][0]:
                        acc["mps"][0] = lo
                    if hi > acc["mms"][1]:
                        acc["mms"][1] = hi
    return acc


def diff_range(panels: list[dict], use_median: bool = False) -> dict:
    """Max |score - base| per metric, across all panels that have a base row."""
    keys = ("m", "s", "mn", "mx", "mms", "mps")
    acc = {k: 0.0 for k in keys}

    def upd(k, v):
        if v is None:
            return
        a = abs(v)
        if a > acc[k]:
            acc[k] = a

    for p in panels:
        bi = p["base_idx"]
        if bi < 0:
            continue
        if (use_median and p["group"] == "eval-orthogonality"
                and p.get("baseline_alt")):
            base_row = p["baseline_alt"]
        else:
            base_row = p["matrix"][bi]
        for r in range(len(p["matrix"])):
            for c in range(len(p["x_labels"])):
                cell = p["matrix"][r][c]
                base = base_row[c]
                if not cell or not base:
                    continue
                if cell["m"] is not None and base["m"] is not None:
                    upd("m", cell["m"] - base["m"])
                if cell["s"] is not None and base["s"] is not None:
                    upd("s", cell["s"] - base["s"])
                if cell["mn"] is not None and base["mn"] is not None:
                    upd("mn", cell["mn"] - base["mn"])
                if cell["mx"] is not None and base["mx"] is not None:
                    upd("mx", cell["mx"] - base["mx"])
                if (cell["m"] is not None and cell["s"] is not None
                        and base["m"] is not None and base["s"] is not None):
                    upd("mms", (cell["m"] - cell["s"]) - (base["m"] - base["s"]))
                    upd("mps", (cell["m"] + cell["s"]) - (base["m"] + base["s"]))
    return acc


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>cross-elicit · compare matrices</title>
<style>
  :root {
    --bg: #0f1115;
    --panel: #161a22;
    --panel-2: #1d2230;
    --ink: #e6e9ef;
    --ink-dim: #8a93a6;
    --accent: #6aa8ff;
    --grid: #2a3142;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--ink); font: 13px/1.4 -apple-system, system-ui, "Segoe UI", sans-serif; }
  header {
    position: sticky; top: 0; z-index: 10;
    display: flex; flex-wrap: wrap; gap: 12px 18px; align-items: center;
    padding: 10px 14px; background: var(--panel); border-bottom: 1px solid var(--grid);
  }
  header h1 { font-size: 14px; margin: 0 12px 0 0; font-weight: 600; }
  .group { display: inline-flex; gap: 6px; align-items: center; }
  .group label { color: var(--ink-dim); font-size: 12px; }
  .seg { display: inline-flex; background: var(--panel-2); border: 1px solid var(--grid); border-radius: 6px; overflow: hidden; }
  .seg button { background: transparent; color: var(--ink); border: 0; padding: 5px 10px; cursor: pointer; font-size: 12px; }
  .seg button.active { background: var(--accent); color: #0a0d13; font-weight: 600; }
  .toggle { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; user-select: none; }
  .toggle input { accent-color: var(--accent); }
  button.plain { background: var(--panel-2); color: var(--ink); border: 1px solid var(--grid); border-radius: 6px; padding: 5px 10px; font-size: 12px; cursor: pointer; }
  button.plain:hover { background: #232a3c; }
  button.plain.panel-hidden { opacity: 0.38; text-decoration: line-through; }
  #panelToggleBar { display: flex; flex-wrap: wrap; gap: 6px 8px; align-items: center; padding: 8px 14px; background: var(--panel); border-bottom: 1px solid var(--grid); }
  #grid {
    display: grid;
    gap: 14px; padding: 14px;
  }
  #grid.layout-horizontal {
    grid-auto-flow: column;
    grid-auto-columns: max-content;
    overflow-x: auto;
  }
  #grid.layout-vertical {
    grid-template-columns: max-content;
  }
  .panel { background: var(--panel); border: 1px solid var(--grid); border-radius: 8px; padding: 10px; overflow: hidden; }
  .panel h3 { font-size: 12px; margin: 0 0 6px; color: var(--ink); font-weight: 600; letter-spacing: .02em; }
  .panel .sub { color: var(--ink-dim); font-size: 11px; margin-bottom: 6px; }
  .heatmap { position: relative; overflow: hidden; }
  .heatmap canvas { display: block; background: #0a0d13; border: 1px solid var(--grid); cursor: crosshair; position: absolute; }
  .heatmap .ax {
    position: absolute; color: var(--ink-dim); font-size: 9px; pointer-events: none;
    user-select: none;
  }
  .heatmap .ax span { position: absolute; white-space: nowrap; }
  .heatmap .brush { position: absolute; border: 1px dashed #ffce4e; background: rgba(255,206,78,0.10); pointer-events: none; display: none; }
  .heatmap .corner { position: absolute; background: var(--panel); border: 1px solid var(--grid); }
  .tt {
    position: fixed; z-index: 100; padding: 6px 8px; background: #0a0d13; color: var(--ink);
    border: 1px solid var(--grid); border-radius: 6px; font-size: 11px; pointer-events: none;
    display: none; max-width: 360px;
  }
  .tt b { color: #ffd479; }
  .legend { display: inline-flex; align-items: center; gap: 6px; }
  .legend-bar { width: 140px; height: 10px; border: 1px solid var(--grid); border-radius: 2px;
    background: linear-gradient(to right, #b30000, #e44d2e, #f0b94f, #fff3b0, #b9e08a, #4eaa50, #136d2c); }
  .legend-bar.diverge { background: linear-gradient(to right, #1a5dab, #6aa8ff, #cfd8e6, #f5f5f5, #f0c4c4, #d24545, #8a1a1a); }
  .badge { display: inline-block; padding: 1px 6px; border-radius: 3px; background: #3a2a14; color: #ffce4e; font-size: 10px; margin-left: 4px; }
  details.evals {
    background: var(--panel); border-bottom: 1px solid var(--grid); padding: 8px 14px;
  }
  details.evals summary {
    cursor: pointer; color: var(--ink-dim); font-size: 12px; user-select: none;
  }
  details.evals summary b { color: var(--ink); }
  .evals-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 4px 14px; margin-top: 8px;
  }
  .evals-grid label { display: flex; align-items: center; gap: 6px; font-size: 11px; cursor: pointer; }
  .evals-grid input { accent-color: var(--accent); }
  .evals-actions { display: inline-flex; gap: 8px; margin-left: 12px; }
  .legend small { color: var(--ink-dim); }
  button.plain.active { background: var(--accent); color: #0a0d13; }
  #crossInfo {
    position: fixed; z-index: 98; bottom: 14px; right: 14px;
    background: #0a0d13; border: 1px solid var(--grid); border-radius: 6px;
    padding: 8px 10px; font-size: 11px; color: var(--ink);
    max-width: 480px; max-height: 55vh; overflow-y: auto;
    display: none;
  }
  #crossInfo .ci-panel { margin-bottom: 6px; }
  #crossInfo .ci-panel:last-child { margin-bottom: 0; }
  #crossInfo .ci-title { color: var(--accent); font-weight: 600; }
  #crossInfo .ci-labels { color: var(--ink-dim); }
  #crossInfo hr { border: none; border-top: 1px solid var(--grid); margin: 4px 0; }
  #showTable { padding: 14px; overflow-x: auto; }
  .show-table { border-collapse: collapse; font-size: 11px; white-space: nowrap; }
  .show-table th, .show-table td { border: 1px solid var(--grid); padding: 3px 7px; }
  .show-table th { background: var(--panel-2); color: var(--ink-dim); font-weight: 600; position: sticky; top: 0; }
  .show-table td { background: var(--panel); color: var(--ink); }
  .show-table td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .show-table th.sep, .show-table td.sep { border-left: 3px solid var(--ink-dim); }
  .show-table .trig { font-weight: 700; background: rgba(255,92,191,0.22); color: #ff5cbf; }
  /* symmetry table */
  #spilloverControls, #symmetryControls { display: none; }
  #spilloverControls.vis, #symmetryControls.vis { display: contents; }
  .sym-wrap { display: flex; gap: 10px; overflow-x: auto; padding: 14px; align-items: flex-start; }
  .sym-pbox { flex-shrink: 0; }
  .sym-ptitle { font-size: 11px; color: var(--ink-dim); margin-bottom: 4px; white-space: nowrap; }
  .sym-tbl { border-collapse: collapse; font-size: 10px; white-space: nowrap; }
  .sym-tbl th, .sym-tbl td { border: 1px solid var(--grid); padding: 2px 5px; }
  .sym-tbl th { background: var(--panel-2); color: var(--ink-dim); font-weight: 600; text-align: center; }
  .sym-tbl td { background: var(--panel); color: var(--ink); }
  .sym-tbl td.sym-lbl { text-align: left; max-width: 110px; overflow: hidden; text-overflow: ellipsis; }
  .sym-tbl td.sym-num { text-align: right; font-variant-numeric: tabular-nums; font-family: monospace; }
  .sym-tbl td.sym-trig { font-weight: 700; }
  .sym-tbl tr.sym-dr:hover { outline: 1px solid var(--accent); cursor: pointer; }
</style>
</head>
<body>
<header>
  <h1>cross-elicit · compare matrices</h1>
  <div class="group">
    <label>view</label>
    <div class="seg" id="metricSeg">
      <button data-metric="mean" class="active">mean</button>
      <button data-metric="minmax">min / max</button>
      <button data-metric="meanstd">mean±std</button>
    </div>
  </div>
  <div class="group">
    <label>layout</label>
    <div class="seg" id="layoutSeg">
      <button data-layout="horizontal" class="active">horizontal</button>
      <button data-layout="vertical">vertical</button>
    </div>
  </div>
  <label class="toggle"><input type="checkbox" id="zoomAll"> apply zoom to all</label>
  <label class="toggle"><input type="checkbox" id="diff"> diff (score − base)</label>
  <label class="toggle"><input type="checkbox" id="reduce"> reduce eval-orth</label>
  <label class="toggle"><input type="checkbox" id="median"> median baseline</label>
  <label class="toggle" title="hide cells whose n_numeric/n_total ratio is not greater than this threshold">
    min n_num/n_tot
    <input type="number" id="threshold" min="0" max="1" step="0.05" value="0"
           style="width: 60px; background: var(--panel-2); color: var(--ink); border: 1px solid var(--grid); border-radius: 4px; padding: 2px 4px;">
  </label>
  <label class="toggle"><input type="checkbox" id="normalize"> normalize (÷std)</label>
  <label class="toggle"><input type="checkbox" id="includeDiagonal"> include-diagonal</label>
  <button class="plain" id="btnSpillover">unexpected spillover</button>
  <button class="plain" id="btnSymmetry">symmetry</button>
  <span id="spilloverControls">
  <label class="toggle" title="show table rows where at least this many finetuned/system-prompted matrices agree">
    min-agree
    <input type="number" id="minAgree" min="1" step="1" value="1"
           style="width: 50px; background: var(--panel-2); color: var(--ink); border: 1px solid var(--grid); border-radius: 4px; padding: 2px 4px;">
  </label>
  <label class="toggle" title="show table rows where |elicited diff/std| ÷ |eval-orth diff/std| exceeds this">
    elicited÷eo
    <input type="number" id="ratioThreshold" min="0" step="0.1" value="0"
           style="width: 60px; background: var(--panel-2); color: var(--ink); border: 1px solid var(--grid); border-radius: 4px; padding: 2px 4px;">
  </label>
  <label class="toggle" title="show table rows where the elicited matrix's own |diff/std| (signal-to-noise ratio) exceeds this">
    own-ratio
    <input type="number" id="showThreshold" min="0" step="0.1" value="0"
           style="width: 60px; background: var(--panel-2); color: var(--ink); border: 1px solid var(--grid); border-radius: 4px; padding: 2px 4px;">
  </label>
  <button class="plain" id="hlAll">highlight everything from table</button>
  <button class="plain" id="hlPrompted">highlight prompted</button>
  </span>
  <span id="symmetryControls">
  <div class="seg"><button id="symAbsBtn" class="active">abs</button><button id="symRelBtn">rel</button></div>
  <label class="toggle">abs thr
    <input type="number" id="symAbsT" min="0" step="0.05" value="0.1"
           style="width:55px;background:var(--panel-2);color:var(--ink);border:1px solid var(--grid);border-radius:4px;padding:2px 4px;">
  </label>
  <label class="toggle">rel thr
    <input type="number" id="symRelT" min="0" step="0.05" value="0.1"
           style="width:55px;background:var(--panel-2);color:var(--ink);border:1px solid var(--grid);border-radius:4px;padding:2px 4px;">
  </label>
  </span>
  <button class="plain" id="resetZoom">reset zoom</button>
  <button class="plain" id="screenshot">screenshot</button>
  <span class="legend">
    <span class="legend-bar" id="legendBar"></span>
    <small id="legendText">— / —</small>
  </span>
  <span style="color: var(--ink-dim); font-size: 11px;">drag to zoom · drag (zoomed) to pan · dbl-click to reset</span>
</header>
<details class="evals">
  <summary><b>show evals</b> <span id="evalsCount" style="color: var(--ink-dim)"></span>
    <span class="evals-actions">
      <button class="plain" id="evalsDefault" type="button">default</button>
      <button class="plain" id="evalsAll" type="button">all</button>
      <button class="plain" id="evalsNone" type="button">none</button>
    </span>
  </summary>
  <div class="evals-grid" id="evalsGrid"></div>
</details>
<details class="evals">
  <summary><b>show poles</b> <span id="polesCount" style="color: var(--ink-dim)"></span>
    <span class="evals-actions">
      <button class="plain" id="polesDefault" type="button">default</button>
      <button class="plain" id="polesAll" type="button">all</button>
      <button class="plain" id="polesNone" type="button">none</button>
    </span>
  </summary>
  <div class="evals-grid" id="polesGrid"></div>
</details>
<div id="panelToggleBar">
  <span style="color: var(--ink-dim); font-size: 12px; margin-right: 2px;">matrices</span>
</div>
<div id="grid"></div>
<div class="tt" id="tt"></div>
<div id="crossInfo"></div>
<div id="showTable"></div>
<div id="symTableContainer"></div>
<script>
const PANELS = __PANELS__;
const RANGES = __RANGES__;
const DRANGES = __DRANGES__;
const DRANGES_MED = __DRANGES_MED__;
const EVALS = __EVALS__;
const POLES = __POLES__;
const DEFAULT_EVALS = new Set([
  'cooperation', 'honest-humble', 'honest-humble:exploitation_score',
  'honest-humble:grandiosity_score', 'honest-humble:norm_defiance_score',
  'neuroticism', 'power-seeking', 'self-preservation', 'spitefulness',
]);
const DEFAULT_POLES = new Set([
  'cooperation.high', 'cooperation.low',
  'honest-humble.high_hh', 'honest-humble.low_hh',
  'neuroticism.neurotic', 'neuroticism.stable',
  'power-seeking.power_seeking', 'power-seeking.deferential',
  'self-preservation.self_preserving', 'self-preservation.replacement_neutral',
  'spitefulness.spiteful', 'spitefulness.forgiving',
]);
const MODEL_SHORT = __MODEL_SHORT__;
const TABLE_MODEL_SHORT = __TABLE_MODEL_SHORT__;
const state = {
  metric: 'mean',
  zoomAll: false,
  layout: 'horizontal',
  diff: false,
  reduce: false,
  median: false,
  threshold: 0,
  normalize: false,
  ratioThreshold: 0,
  showThreshold: 0,
  includeDiagonal: false,
  minAgree: 1,
  highlightMode: null,  // null | 'all' | 'prompted'
  _highlightAllKeys: null,
  _highlightPromptedKeys: null,
  _tableRows: [],
  _tableOtherPanelIdxs: [],
  spilloverMode: false,
  symMode: false,
  symAbsolute: true,
  symAbsThreshold: 0.1,
  symRelThreshold: 0.1,
  symHoverCluster: null,
  _symHighlightKeys: null,
  hoverKey: null,
  visibleCols: EVALS.map(() => true), // positional mask over data columns (33 entries)
  visibleRows: POLES.map(() => true), // positional mask over poles (rows)
  visiblePanels: PANELS.map(() => true),
  _normDiffMax: 1,
  _normAbsLo: 0,
  _normAbsHi: 10,
};

// Returns the active baseline row for a panel (eval-orth swaps when 'median' is on).
// Update the global hover key and re-render all panels if it changed.
function setHoverKey(k) {
  if (state.hoverKey === k) return;
  state.hoverKey = k;
  if (typeof panelViews !== 'undefined') {
    for (const pv of panelViews) pv.render();
    updateCrossInfo();
    renderShowTable();
  }
}

function effectiveBaseRow(p) {
  if (p.base_idx < 0) return null;
  if (state.median && p.group === 'eval-orthogonality' && p.baseline_alt) return p.baseline_alt;
  return p.matrix[p.base_idx];
}
// Returns the cell at (yi, xi), with the baseline row substituted when 'median' is on.
function effectiveCell(p, yi, xi) {
  if (p.group === 'ft-sp-diff') {
    const ftYi = p._ftRowIdxs[yi];
    if (ftYi == null) return null;
    const spYi = p._ftRowToSpRow.get(ftYi);
    if (spYi == null) return null;
    return computeFtSpDiffCell(p.ft_panel, ftYi, p.sp_panel, spYi, xi);
  }
  if (yi === p.base_idx && state.median && p.group === 'eval-orthogonality' && p.baseline_alt) {
    return p.baseline_alt[xi];
  }
  return p.matrix[yi][xi];
}

function updateNormRanges() {
  let diffMax = 0, absLo = Infinity, absHi = -Infinity;
  for (const p of PANELS) {
    const baseRow = p.base_idx >= 0
      ? (state.median && p.group === 'eval-orthogonality' && p.baseline_alt ? p.baseline_alt : p.matrix[p.base_idx])
      : null;
    for (let yi = 0; yi < p.y_labels.length; yi++) {
      if (baseRow && yi === p.base_idx) continue;
      for (let xi = 0; xi < p.x_labels.length; xi++) {
        const cell = p.matrix[yi][xi];
        if (!cell || cell.m == null || !cell.s || cell.s === 0) continue;
        const norm = cell.m / cell.s;
        if (norm < absLo) absLo = norm;
        if (norm > absHi) absHi = norm;
        if (baseRow) {
          const bc = baseRow[xi];
          if (bc && bc.m != null) {
            const dn = Math.abs((cell.m - bc.m) / cell.s);
            if (dn > diffMax) diffMax = dn;
          }
        }
      }
    }
  }
  state._normDiffMax = isFinite(diffMax) ? (diffMax || 1) : 1;
  state._normAbsLo = isFinite(absLo) ? absLo : 0;
  state._normAbsHi = isFinite(absHi) ? absHi : 10;
}

// red→yellow→green ramp like the PNGs.
const RAMP_SEQ = [
  [0.00, [179, 0, 0]],
  [0.18, [228, 77, 46]],
  [0.36, [240, 185, 79]],
  [0.50, [255, 243, 176]],
  [0.66, [185, 224, 138]],
  [0.82, [78, 170, 80]],
  [1.00, [19, 109, 44]],
];
// diverging blue→white→red for diff mode (negative=blue, positive=red).
const RAMP_DIV = [
  [0.00, [26, 93, 171]],
  [0.25, [106, 168, 255]],
  [0.45, [207, 216, 230]],
  [0.50, [245, 245, 245]],
  [0.55, [240, 196, 196]],
  [0.75, [210, 69, 69]],
  [1.00, [138, 26, 26]],
];
// symmetry table: red(wrong direction) → light grey(neutral) → green(expected direction)
const RAMP_SYM = [
  [0.00, [210, 40, 40]],
  [0.50, [220, 220, 220]],
  [1.00, [30, 160, 55]],
];
function rampSample(ramp, t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < ramp.length; i++) {
    if (t <= ramp[i][0]) {
      const lo = ramp[i-1], hi = ramp[i];
      const k = (t - lo[0]) / (hi[0] - lo[0]);
      const r = Math.round(lo[1][0] + (hi[1][0]-lo[1][0])*k);
      const g = Math.round(lo[1][1] + (hi[1][1]-lo[1][1])*k);
      const b = Math.round(lo[1][2] + (hi[1][2]-lo[1][2])*k);
      return `rgb(${r},${g},${b})`;
    }
  }
  return '#888';
}
function colorFor(v, vmin, vmax) {
  if (v === null || v === undefined || Number.isNaN(v)) return '#444';
  let t = (v - vmin) / (vmax - vmin);
  if (!isFinite(t)) t = 0.5;
  return rampSample(RAMP_SEQ, t);
}

function rangeForMetric(metric) {
  if (state.normalize) {
    if (state.diff) return [-5, 5];
    return [0, 5];
  }
  if (state.diff) {
    const R = state.median ? DRANGES_MED : DRANGES;
    let a;
    if (metric === 'mean') a = R.m;
    else if (metric === 'minmax') a = Math.max(R.mn, R.mx);
    else if (metric === 'meanstd') a = Math.max(R.mms, R.mps);
    else a = 1;
    return [-a, a];
  }
  if (metric === 'mean') return RANGES.m;
  if (metric === 'minmax') return [Math.min(RANGES.mn[0], RANGES.mx[0]), Math.max(RANGES.mn[1], RANGES.mx[1])];
  if (metric === 'meanstd') return [Math.min(RANGES.mms[0], RANGES.mps[0]), Math.max(RANGES.mms[1], RANGES.mps[1])];
  return [0, 100];
}

// short label helpers
function shortLabel(s) {
  if (s.startsWith('ethical-framework-')) {
    let x;
    if (s.includes('.')) {
      x = s.split('.').pop();
    } else if (s.includes('--')) {
      x = s.split('--').pop();
    } else {
      const bare = s.slice('ethical-framework-'.length);
      if (bare.endsWith('-plus')) x = bare.slice(0, -5);
      else if (bare.endsWith('-minus')) x = bare.slice(0, -6);
      else x = bare;
    }
    return x.replace(/_/g, ' ').replace(/-/g, ' ');
  }
  return s.replace(/_score$/, '').replace(/\s+\(.+?\)$/, '');
}

function computeHighlightSets() {
  const rows = state._tableRows || [];
  const idxs = state._tableOtherPanelIdxs || [];
  const eopIdx = PANELS.findIndex(p => p.group === 'eval-orthogonality');
  const allKeys = new Set();
  const promptedKeys = new Map();
  if (eopIdx >= 0) promptedKeys.set(eopIdx, new Set());
  for (const idx of idxs) if (!promptedKeys.has(idx)) promptedKeys.set(idx, new Set());
  for (const row of rows) {
    if (!row.hk) continue;
    allKeys.add(row.hk);
    if (eopIdx >= 0) promptedKeys.get(eopIdx).add(row.hk);
    for (let pi = 0; pi < row.entries.length; pi++) {
      if (row.entries[pi].passes && idxs[pi] !== undefined) {
        promptedKeys.get(idxs[pi]).add(row.hk);
      }
    }
  }
  state._highlightAllKeys = allKeys;
  state._highlightPromptedKeys = promptedKeys;
}

function buildPoleFilter() {
  if (state.visibleRows.every(v => v)) return null; // all visible — no filtering
  const keySet = new Set();
  const labelSet = new Set();
  const allLabels = new Set(POLES.map(p => p.name));
  POLES.forEach((p, i) => {
    if (state.visibleRows[i]) {
      if (p.key) keySet.add(p.key);
      labelSet.add(p.name);
    }
  });
  return { keySet, labelSet, allLabels };
}

// Panel widget: builds DOM and handles its own rendering / interaction.
class PanelView {
  constructor(panel, container, index) {
    this.p = panel;
    this.ny = panel.y_labels.length;
    this.nx = panel.x_labels.length;
    this.panelIndex = index;
    this.activeRows = this._computeActiveRows();
    this.activeCols = this._computeActiveCols();
    this.vp = { x0: 0, x1: this.activeCols.length, y0: 0, y1: this.activeRows.length };
    this.container = container;
    this.build();
    this.attach();
  }
  _computeActiveRows() {
    const out = [];
    const reduceMask = this.p.reduce_mask || [];
    const reduceOn = state.reduce && reduceMask.length === this.ny;
    const pf = buildPoleFilter();
    for (let i = 0; i < this.ny; i++) {
      if (state.diff && i === this.p.base_idx) continue;
      if (reduceOn && !reduceMask[i]) continue;
      if (pf !== null) {
        const yk = this.p.corr_y_keys[i];
        const yl = this.p.y_labels[i];
        if (yk) {
          if (!pf.keySet.has(yk)) continue;
        } else if (pf.allLabels.has(yl)) {
          if (!pf.labelSet.has(yl)) continue;
        }
      }
      out.push(i);
    }
    return out;
  }
  _computeActiveCols() {
    const out = [];
    // Only apply the master col mask when the panel has the same column count
    // as the master list (33). Otherwise (defensive) keep all columns.
    const usable = this.nx === state.visibleCols.length;
    for (let i = 0; i < this.nx; i++) {
      if (usable && !state.visibleCols[i]) continue;
      out.push(i);
    }
    return out;
  }
  recomputeActive() {
    this.activeRows = this._computeActiveRows();
    this.activeCols = this._computeActiveCols();
    this.vp = { x0: 0, x1: this.activeCols.length, y0: 0, y1: this.activeRows.length };
  }
  nyEff() { return this.activeRows.length; }
  nxEff() { return this.activeCols.length; }
  xLabelsEff() { return this.activeCols.map(i => this.p.x_labels[i]); }
  // is this panel the first *visible* occurrence of its tick_group in the current panel order?
  isFirstInTickGroup() {
    try {
      for (let i = this.panelIndex - 1; i >= 0; i--) {
        if (!state.visiblePanels[i]) continue;
        const prev = panelViews[i];
        if (!prev) return true;
        return prev.p.tick_group !== this.p.tick_group;
      }
    } catch (e) { return true; }
    return true;
  }

  build() {
    const el = document.createElement('div');
    el.className = 'panel';
    const badge = (this.p.base_idx < 0) ? ' <span class="badge" title="no base row">no base</span>' : '';
    el.innerHTML = `
      <h3>${this.p.title}<span class="diff-badge" style="display:none">${badge}</span></h3>
      <div class="sub" data-sub>${this.ny} × ${this.nx} (y × x)</div>
      <div class="heatmap">
        <div class="corner"></div>
        <div class="ax ax-top"></div>
        <div class="ax ax-left"></div>
        <canvas></canvas>
        <div class="brush"></div>
      </div>
    `;
    this.container.appendChild(el);
    this.rootEl = el;
    this.diffBadge = el.querySelector('.diff-badge');
    this.subEl = el.querySelector('[data-sub]');
    this.root = el.querySelector('.heatmap');
    this.canvas = el.querySelector('canvas');
    this.axTop = el.querySelector('.ax-top');
    this.axLeft = el.querySelector('.ax-left');
    this.corner = el.querySelector('.corner');
    this.brushEl = el.querySelector('.brush');
    this.layout();
  }

  // current screen→data axis mapping. In 'tall' (horizontal layout):
  //   screen-x = data-x (eval poles), screen-y = data-y (treatments)
  // In 'wide' (vertical layout) the axes swap.
  get symAxisMap() {
    if (!this._symAxisMap) this._symAxisMap = buildSymAxisMap(this);
    return this._symAxisMap;
  }
  isWide() { return state.layout === 'vertical'; }
  // screen dim counts (effective: rows pruned by activeRows where applicable)
  scrNx() { return this.isWide() ? this.nyEff() : this.nxEff(); }
  scrNy() { return this.isWide() ? this.nxEff() : this.nyEff(); }
  // active-row-aware label list for the y data axis
  yLabelsEff() { return this.activeRows.map(i => this.p.y_labels[i]); }
  scrXLabels() { return this.isWide() ? this.yLabelsEff() : this.xLabelsEff(); }
  scrYLabels() { return this.isWide() ? this.xLabelsEff() : this.yLabelsEff(); }
  // viewport ranges on screen axes
  vpScrX() { return this.isWide() ? [this.vp.y0, this.vp.y1] : [this.vp.x0, this.vp.x1]; }
  vpScrY() { return this.isWide() ? [this.vp.x0, this.vp.x1] : [this.vp.y0, this.vp.y1]; }

  layout() {
    if (this.isWide()) {
      this.cellW = 14; this.cellH = 9;
      this.gutterTop = 170; this.gutterLeft = 130;
    } else {
      this.cellW = 9; this.cellH = 14;
      this.gutterTop = 130; this.gutterLeft = 170;
    }
    // hide ticks for non-first-in-group panels along the layout axis
    const first = this.isFirstInTickGroup();
    if (state.layout === 'horizontal' && !first) {
      this.showYAxis = false;
      this.gutterLeft = 4;
    } else {
      this.showYAxis = true;
    }
    if (state.layout === 'vertical' && !first) {
      this.showXAxis = false;
      this.gutterTop = 4;
    } else {
      this.showXAxis = true;
    }
    const W = this.cellW * this.scrNx();
    const H = this.cellH * this.scrNy();
    const dpr = window.devicePixelRatio;
    this.canvas.width = W * dpr;
    this.canvas.height = H * dpr;
    this.canvas.style.width = W + 'px';
    this.canvas.style.height = H + 'px';
    this.canvas.style.left = this.gutterLeft + 'px';
    this.canvas.style.top = this.gutterTop + 'px';
    this.axTop.style.left = this.gutterLeft + 'px';
    this.axTop.style.top = '0px';
    this.axTop.style.width = W + 'px';
    this.axTop.style.height = this.gutterTop + 'px';
    this.axLeft.style.left = '0px';
    this.axLeft.style.top = this.gutterTop + 'px';
    this.axLeft.style.width = this.gutterLeft + 'px';
    this.axLeft.style.height = H + 'px';
    this.corner.style.left = '0px';
    this.corner.style.top = '0px';
    this.corner.style.width = this.gutterLeft + 'px';
    this.corner.style.height = this.gutterTop + 'px';
    this.root.style.height = (this.gutterTop + H + 4) + 'px';
    this.root.style.width = (this.gutterLeft + W + 4) + 'px';
  }

  attach() {
    this.canvas.addEventListener('mousedown', (e) => this.onDown(e));
    this.canvas.addEventListener('mousemove', (e) => this.onMove(e));
    this.canvas.addEventListener('mouseleave', () => { tt.style.display = 'none'; setHoverKey(null); setSymHover(null); });
    this.canvas.addEventListener('dblclick', () => this.reset(true));
  }

  // map canvas pixel → fractional (data xIdx, yIdx) in current viewport
  pxToIdx(px, py) {
    const rect = this.canvas.getBoundingClientRect();
    const fx = px / rect.width;
    const fy = py / rect.height;
    const [sx0, sx1] = this.vpScrX();
    const [sy0, sy1] = this.vpScrY();
    const scrX = sx0 + fx * (sx1 - sx0);
    const scrY = sy0 + fy * (sy1 - sy0);
    return this.isWide()
      ? { xIdx: scrY, yIdx: scrX }
      : { xIdx: scrX, yIdx: scrY };
  }

  onDown(e) {
    const rect = this.canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
    const isZoomed = !(this.vp.x0 === 0 && this.vp.x1 === this.nxEff() && this.vp.y0 === 0 && this.vp.y1 === this.nyEff());
    let mode = isZoomed && !e.shiftKey ? 'pan' : 'brush';
    if (!isZoomed && e.shiftKey) mode = 'pan';
    let lastX = sx, lastY = sy;
    const initVP = { ...this.vp };

    const onMove = (ev) => {
      const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
      if (mode === 'brush') {
        const x0 = Math.min(sx, x), x1 = Math.max(sx, x);
        const y0 = Math.min(sy, y), y1 = Math.max(sy, y);
        this.brushEl.style.left = (this.gutterLeft + x0) + 'px';
        this.brushEl.style.top = (this.gutterTop + y0) + 'px';
        this.brushEl.style.width = (x1 - x0) + 'px';
        this.brushEl.style.height = (y1 - y0) + 'px';
        this.brushEl.style.display = 'block';
      } else if (mode === 'pan') {
        const dx = x - lastX, dy = y - lastY;
        lastX = x; lastY = y;
        const [sx0, sx1] = this.isWide() ? [initVP.y0, initVP.y1] : [initVP.x0, initVP.x1];
        const [sy0, sy1] = this.isWide() ? [initVP.x0, initVP.x1] : [initVP.y0, initVP.y1];
        const wx = (sx1 - sx0) / rect.width;
        const wy = (sy1 - sy0) / rect.height;
        // delta in screen → delta in data
        const dDataXScr = -dx * wx, dDataYScr = -dy * wy;
        const dDataX = this.isWide() ? dDataYScr : dDataXScr;
        const dDataY = this.isWide() ? dDataXScr : dDataYScr;
        const w = this.vp.x1 - this.vp.x0, h = this.vp.y1 - this.vp.y0;
        let nx0 = this.vp.x0 + dDataX;
        let ny0 = this.vp.y0 + dDataY;
        nx0 = Math.max(0, Math.min(this.nxEff() - w, nx0));
        ny0 = Math.max(0, Math.min(this.nyEff() - h, ny0));
        this.applyViewport({ x0: nx0, x1: nx0 + w, y0: ny0, y1: ny0 + h }, true);
      }
    };
    const onUp = (ev) => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      this.brushEl.style.display = 'none';
      if (mode === 'brush') {
        const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
        if (Math.abs(x - sx) < 4 && Math.abs(y - sy) < 4) return;
        const a = this.pxToIdx(Math.min(sx, x), Math.min(sy, y));
        const b = this.pxToIdx(Math.max(sx, x), Math.max(sy, y));
        const x0d = Math.min(a.xIdx, b.xIdx), x1d = Math.max(a.xIdx, b.xIdx);
        const y0d = Math.min(a.yIdx, b.yIdx), y1d = Math.max(a.yIdx, b.yIdx);
        const newVP = {
          x0: Math.max(0, Math.min(this.nxEff(), x0d)),
          x1: Math.max(0, Math.min(this.nxEff(), x1d)),
          y0: Math.max(0, Math.min(this.nyEff(), y0d)),
          y1: Math.max(0, Math.min(this.nyEff(), y1d)),
        };
        if (newVP.x1 - newVP.x0 < 0.5) newVP.x1 = newVP.x0 + 0.5;
        if (newVP.y1 - newVP.y0 < 0.5) newVP.y1 = newVP.y0 + 0.5;
        this.applyViewport(newVP, true);
      }
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  }

  onMove(e) {
    const rect = this.canvas.getBoundingClientRect();
    const px = e.clientX - rect.left, py = e.clientY - rect.top;
    const { xIdx, yIdx } = this.pxToIdx(px, py);
    const xi = Math.floor(xIdx), yi = Math.floor(yIdx);
    if (xi < 0 || yi < 0 || xi >= this.nxEff() || yi >= this.nyEff()) {
      tt.style.display = 'none';
      setHoverKey(null);
      return;
    }
    const dataYi = this.activeRows[yi];
    const dataXi = this.activeCols[xi];
    // update cross-matrix highlight
    const yk = this.p.corr_y_keys[dataYi];
    const xk = this.p.corr_x_keys[dataXi];
    setHoverKey((yk && xk) ? `${yk}|${xk}` : null);
    if (state.symMode) {
      let rowAxis = null;
      if (yk && yk.includes('|')) {
        const sep = yk.indexOf('|'), sign = yk.slice(sep + 1);
        if (sign === 'plus' || sign === 'minus') rowAxis = yk.slice(0, sep);
      }
      let colAxis = null;
      for (const [ax, entry] of this.symAxisMap) {
        if (entry.evalCol === dataXi) { colAxis = ax; break; }
      }
      setSymHover(rowAxis && colAxis && rowAxis !== colAxis
        ? { ai: [rowAxis, colAxis].sort()[0], aj: [rowAxis, colAxis].sort()[1] }
        : null);
    }
    const cell = effectiveCell(this.p, dataYi, dataXi);
    const xl = this.p.x_labels[dataXi];
    const yl = this.p.y_labels[dataYi];
    const diffOn = state.diff && this.p.base_idx >= 0;
    const baseRow = diffOn ? effectiveBaseRow(this.p) : null;
    const base = baseRow ? baseRow[dataXi] : null;
    let body = `<div><b>y</b> ${yl}</div><div><b>x</b> ${xl}</div>`;
    if (cell) {
      const f = (v) => (v === null || v === undefined ? '—' : (typeof v === 'number' ? v.toFixed(2) : v));
      body += `<div>mean=${f(cell.m)} · std=${f(cell.s)} · min=${f(cell.mn)} · max=${f(cell.mx)}</div>`;
      body += `<div style="color: var(--ink-dim)">n_numeric=${cell.n ?? '—'} · n_total=${cell.nt ?? '—'} · n_nulls=${cell.nn ?? '—'} · n_fails=${cell.nf ?? '—'}</div>`;
      if (cell.e) {
        body += `<div style="color: var(--ink-dim)">eval=${cell.e}</div>`;
      }
      if (this.p.group === 'ft-sp-diff') {
        body += `<div style="color:#aaa">(FT − SP; transformations applied per current state)</div>`;
      } else if (diffOn && base) {
        const d = (a, b) => (a == null || b == null ? '—' : (a - b).toFixed(2));
        body += `<div style="color:#ffce4e">Δ mean=${d(cell.m, base.m)} · min=${d(cell.mn, base.mn)} · max=${d(cell.mx, base.mx)}</div>`;
      } else if (state.diff && !diffOn) {
        body += `<div style="color:#ffce4e">(no base row — raw values)</div>`;
      }
      const mm = this.p.col_eval_mismatch && this.p.col_eval_mismatch[dataXi];
      if (mm) {
        const evs = (this.p.col_evals_seen && this.p.col_evals_seen[dataXi]) || [];
        body += `<div style="color:#ffe600">⚠ mixed evals in this column: ${evs.join(' vs ')} — diffs are not comparable</div>`;
      }
    } else {
      body += `<div>—</div>`;
    }
    tt.innerHTML = body;
    tt.style.display = 'block';
    tt.style.left = (e.clientX + 12) + 'px';
    tt.style.top = (e.clientY + 12) + 'px';
  }

  // Apply a viewport (in this panel's own index space). If broadcast and
  // zoomAll is on, propagate as normalized fractions to all other panels.
  applyViewport(vp, broadcast) {
    this.vp = vp;
    this.render();
    if (broadcast && state.zoomAll) {
      const fx0 = vp.x0 / this.nxEff();
      const fx1 = vp.x1 / this.nxEff();
      const fy0 = vp.y0 / this.nyEff();
      const fy1 = vp.y1 / this.nyEff();
      for (const pv of panelViews) {
        if (pv === this) continue;
        pv.vp = {
          x0: fx0 * pv.nxEff(), x1: fx1 * pv.nxEff(),
          y0: fy0 * pv.nyEff(), y1: fy1 * pv.nyEff(),
        };
        pv.render();
      }
    }
  }

  reset(broadcast) {
    this.applyViewport({ x0: 0, x1: this.nxEff(), y0: 0, y1: this.nyEff() }, broadcast);
  }

  render() {
    const ctx = this.canvas.getContext('2d');
    const dpr = window.devicePixelRatio;
    ctx.save();
    ctx.scale(dpr, dpr);
    const wide = this.isWide();
    const W = this.cellW * this.scrNx();
    const H = this.cellH * this.scrNy();
    ctx.clearRect(0, 0, W, H);
    let [vmin, vmax] = rangeForMetric(state.metric);
    if (this.p.group === 'ft-sp-diff') {
      const R = state.median ? DRANGES_MED : DRANGES;
      let a;
      if (state.metric === 'mean') a = R.m;
      else if (state.metric === 'minmax') a = Math.max(R.mn, R.mx);
      else a = Math.max(R.mms, R.mps);
      vmin = -a; vmax = a;
    }
    const x0i = Math.max(0, Math.floor(this.vp.x0));
    const x1i = Math.min(this.nxEff(), Math.ceil(this.vp.x1));
    const y0i = Math.max(0, Math.floor(this.vp.y0));
    const y1i = Math.min(this.nyEff(), Math.ceil(this.vp.y1));
    const [sx0, sx1] = this.vpScrX();
    const [sy0, sy1] = this.vpScrY();
    const xPer = W / (sx1 - sx0);
    const yPer = H / (sy1 - sy0);
    const diffOn = state.diff && this.p.base_idx >= 0;
    const baseRow = diffOn ? effectiveBaseRow(this.p) : null;
    const hoverKey = state.hoverKey;
    for (let yi = y0i; yi < y1i; yi++) {
      const dataYi = this.activeRows[yi];
      for (let xi = x0i; xi < x1i; xi++) {
        const dataXi = this.activeCols[xi];
        const cell = effectiveCell(this.p, dataYi, dataXi);
        // screen position: (effective-yi, effective-xi) → screen-axes
        const sxIdx = wide ? yi : xi;
        const syIdx = wide ? xi : yi;
        const px = (sxIdx - sx0) * xPer;
        const py = (syIdx - sy0) * yPer;
        const pw = xPer, ph = yPer;
        let grey = !cell || cell.m === null || cell.m === undefined;
        if (!grey && state.threshold > 0 && cell.nt && cell.nt > 0 && (cell.n / cell.nt) <= state.threshold) grey = true;
        if (grey) {
          ctx.fillStyle = '#2a2f3a';
          ctx.fillRect(px, py, pw, ph);
        } else {
          let mv = cell.m, mnv = cell.mn, mxv = cell.mx, sv = cell.s;
          if (diffOn) {
            const base = baseRow[dataXi];
            if (!base) {
              grey = true;
              ctx.fillStyle = '#2a2f3a';
              ctx.fillRect(px, py, pw, ph);
            } else {
              mv  = (mv  != null && base.m  != null) ? mv  - base.m  : null;
              mnv = (mnv != null && base.mn != null) ? mnv - base.mn : null;
              mxv = (mxv != null && base.mx != null) ? mxv - base.mx : null;
            }
          }
          if (!grey) {
            if (state.normalize && sv != null) {
              const normDiv = Math.max(sv, 1);
              if (mv != null) mv = mv / normDiv;
              if (mnv != null) mnv = mnv / normDiv;
              if (mxv != null) mxv = mxv / normDiv;
            }
            const isDiag = this.p.diag && this.p.diag[String(dataYi)] === dataXi;
            if (state.metric === 'mean') {
              ctx.fillStyle = colorFor(mv, vmin, vmax);
              ctx.fillRect(px, py, pw, ph);
            } else if (state.metric === 'minmax') {
              drawSplit(ctx, px, py, pw, ph, colorFor(mnv, vmin, vmax), colorFor(mxv, vmin, vmax));
            } else if (state.metric === 'meanstd') {
              let lo, hi;
              if (state.normalize) {
                // After normalization mv = value/std; ±1 represents ±1 std band
                lo = mv != null ? mv - 1 : null;
                hi = mv != null ? mv + 1 : null;
              } else if (diffOn) {
                const base = baseRow[dataXi];
                const cmms = (cell.m != null && cell.s != null) ? cell.m - cell.s : null;
                const cmps = (cell.m != null && cell.s != null) ? cell.m + cell.s : null;
                const bmms = (base && base.m != null && base.s != null) ? base.m - base.s : null;
                const bmps = (base && base.m != null && base.s != null) ? base.m + base.s : null;
                lo = (cmms != null && bmms != null) ? cmms - bmms : null;
                hi = (cmps != null && bmps != null) ? cmps - bmps : null;
              } else {
                lo = sv == null ? mv : mv - sv;
                hi = sv == null ? mv : mv + sv;
              }
              drawSplit(ctx, px, py, pw, ph, colorFor(lo, vmin, vmax), colorFor(hi, vmin, vmax));
            }
            if (isDiag) {
              ctx.lineWidth = 1.5;
              ctx.strokeStyle = '#000';
              ctx.strokeRect(px + 0.75, py + 0.75, pw - 1.5, ph - 1.5);
            }
            // Mismatched-eval column: bright yellow stripe across every cell.
            // In horizontal layout the column is screen-vertical; in vertical it's screen-horizontal.
            const mismatch = this.p.col_eval_mismatch && this.p.col_eval_mismatch[dataXi];
            if (mismatch) {
              ctx.fillStyle = '#ffe600';
              if (wide) {
                const lh = Math.max(2, Math.min(3, ph * 0.35));
                ctx.fillRect(px, py + (ph - lh) / 2, pw, lh);
              } else {
                const lw = Math.max(2, Math.min(3, pw * 0.35));
                ctx.fillRect(px + (pw - lw) / 2, py, lw, ph);
              }
            }
          }
        }
        {
          const yk = this.p.corr_y_keys[dataYi];
          const xk = this.p.corr_x_keys[dataXi];
          const hk = (yk && xk) ? yk + '|' + xk : null;
          let box = !!(hk && hoverKey && hk === hoverKey);
          if (!box && hk && state.highlightMode) {
            if (state.highlightMode === 'all') {
              box = !!(state._highlightAllKeys && state._highlightAllKeys.has(hk));
            } else if (state.highlightMode === 'prompted') {
              const ks = state._highlightPromptedKeys && state._highlightPromptedKeys.get(this.panelIndex);
              box = !!(ks && ks.has(hk));
            }
          }
          if (!box && hk && state.symMode && state._symHighlightKeys) {
            box = state._symHighlightKeys.has(hk);
          }
          if (box) {
            ctx.lineWidth = 2;
            ctx.strokeStyle = '#ff5cbf';
            ctx.strokeRect(px + 1, py + 1, pw - 2, ph - 2);
          }
        }
      }
    }
    ctx.restore();
    this.renderAxes();
  }

  renderAxes() {
    this.axLeft.innerHTML = '';
    this.axTop.innerHTML = '';
    if (!this.showYAxis && !this.showXAxis) return;
    const W = this.cellW * this.scrNx();
    const H = this.cellH * this.scrNy();
    const [sx0, sx1] = this.vpScrX();
    const [sy0, sy1] = this.vpScrY();
    const xPer = W / (sx1 - sx0);
    const yPer = H / (sy1 - sy0);
    const scrXLabels = this.scrXLabels();
    const scrYLabels = this.scrYLabels();
    const scrNx = this.scrNx(), scrNy = this.scrNy();
    const sx0i = Math.max(0, Math.floor(sx0));
    const sx1i = Math.min(scrNx, Math.ceil(sx1));
    const sy0i = Math.max(0, Math.floor(sy0));
    const sy1i = Math.min(scrNy, Math.ceil(sy1));
    const yStep = Math.max(1, Math.ceil(9 / yPer));
    const xStep = Math.max(1, Math.ceil(9 / xPer));
    // In horizontal layout, x-axis (top) corresponds to data x (evals).
    // In vertical layout, y-axis (left) corresponds to data x (evals).
    // Either way, we mark ticks whose underlying data-x has eval mismatch.
    const mm = this.p.col_eval_mismatch || [];
    const evalIsScreenY = this.isWide();
    if (this.showYAxis) {
      for (let i = sy0i; i < sy1i; i += yStep) {
        const py = (i - sy0 + 0.5) * yPer;
        const s = document.createElement('span');
        s.textContent = shortLabel(scrYLabels[i]);
        s.style.right = '4px';
        s.style.top = (py - 6) + 'px';
        s.style.textAlign = 'right';
        s.style.width = (this.gutterLeft - 6) + 'px';
        s.style.overflow = 'hidden';
        s.style.textOverflow = 'ellipsis';
        if (evalIsScreenY) {
          const dataXi = this.activeCols[i];
          if (mm[dataXi]) {
            s.style.color = '#ffe600';
            s.style.borderBottom = '2px solid #ffe600';
            s.title = 'mixed evals: ' + (this.p.col_evals_seen[dataXi] || []).join(' vs ');
          }
        }
        this.axLeft.appendChild(s);
      }
    }
    if (this.showXAxis) {
      for (let i = sx0i; i < sx1i; i += xStep) {
        const px = (i - sx0 + 0.5) * xPer;
        const s = document.createElement('span');
        s.textContent = shortLabel(scrXLabels[i]);
        s.style.left = px + 'px';
        s.style.bottom = '2px';
        s.style.transform = 'rotate(-60deg)';
        s.style.transformOrigin = '0 100%';
        if (!evalIsScreenY) {
          const dataXi = this.activeCols[i];
          if (mm[dataXi]) {
            s.style.color = '#ffe600';
            s.style.borderBottom = '2px solid #ffe600';
            s.title = 'mixed evals: ' + (this.p.col_evals_seen[dataXi] || []).join(' vs ');
          }
        }
        this.axTop.appendChild(s);
      }
    }
  }
}

function drawSplit(ctx, x, y, w, h, c1, c2) {
  // lower-left triangle = c1, upper-right triangle = c2
  ctx.fillStyle = c2;
  ctx.fillRect(x, y, w, h);
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x + w, y + h);
  ctx.lineTo(x, y + h);
  ctx.closePath();
  ctx.fillStyle = c1;
  ctx.fill();
}

function updateCrossInfo() {
  const el = document.getElementById('crossInfo');
  if (!state.hoverKey) { el.style.display = 'none'; return; }
  const parts = [];
  for (let pi = 0; pi < panelViews.length; pi++) {
    if (!state.visiblePanels[pi]) continue;
    const p = panelViews[pi].p;
    for (let yi = 0; yi < p.y_labels.length; yi++) {
      const yk = p.corr_y_keys[yi];
      if (!yk) continue;
      for (let xi = 0; xi < p.x_labels.length; xi++) {
        const xk = p.corr_x_keys[xi];
        if (!xk || (yk + '|' + xk) !== state.hoverKey) continue;
        const cell = effectiveCell(p, yi, xi);
        const baseRow = p.base_idx >= 0 ? effectiveBaseRow(p) : null;
        const base = baseRow ? baseRow[xi] : null;
        const f = v => v == null ? '—' : v.toFixed(2);
        const diffPart = (base && cell && cell.m != null && base.m != null)
          ? ` · Δ=${f(cell.m - base.m)}` : '';
        const normPart = (state.normalize && cell && cell.m != null && base && base.m != null && cell.s != null)
          ? ` · Δ/std=${f((cell.m - base.m) / Math.max(cell.s, 1))}` : '';
        parts.push(
          `<div class="ci-panel">` +
          `<span class="ci-title">${p.title}</span> ` +
          `<span class="ci-labels">${p.y_labels[yi]} · ${p.x_labels[xi]}</span><br>` +
          (cell ? `mean=${f(cell.m)} std=${f(cell.s)} min=${f(cell.mn)} max=${f(cell.mx)}${diffPart}${normPart}` : '—') +
          `</div>`
        );
      }
    }
  }
  if (parts.length === 0) { el.style.display = 'none'; return; }
  el.innerHTML = parts.join('<hr>');
  el.style.display = 'block';
}

function screenshotCanvas() {
  // Layout: rows=poles (y, per-panel on left), cols=evals (x, per-panel at top).
  // Panels arranged side-by-side horizontally.
  const vPanels = panelViews.filter((_, i) => state.visiblePanels[i]);
  if (!vPanels.length) return null;

  const DPR   = 2;
  const CW    = 20, CH = 14;
  const Y_GUT = 180;               // pole-label gutter (left of each panel's cells)
  const X_GUT = 130;               // eval-label height above cells (rotated)
  const AX_H  = 14;                // "evaluated propensity →" text row height
  const TTL_H = 20;                // panel title height
  const GAP   = 18;                // horizontal gap between panels
  const CB_GAP = 22, CB_W = 14, CB_LABEL = 46;
  const MRG   = 16;
  const FONT  = '10px -apple-system,Helvetica,sans-serif';
  const FONTB = 'bold 11px -apple-system,Helvetica,sans-serif';
  const FONTS = '9px -apple-system,Helvetica,sans-serif';
  const INK   = '#1a1a1a', DIM = '#555555';
  const NULL_CELL = '#e0e0e0', BORDER = '#aaaaaa', TICK = '#888888';

  // Screenshot-only pole label overrides
  const _POLE_MAP = {
    'claiming-sentience.claiming':           'claiming-sentience',
    'claiming-superintelligence.superintelligent': 'claiming-superintelligence',
    'trust-in-user-intentions.plus':         'naive/credulous',
    'trust-in-user-intentions.minus':        'paranoid',
    'effort.plus':  'industrious',
    'effort.minus': 'lazy',
  };
  const _STRIP_PREFIXES = ['self-preservation.', 'spitefulness.', 'power-seeking.'];
  const scrPoleLabel = raw => {
    if (_POLE_MAP[raw]) return _POLE_MAP[raw];
    for (const pfx of _STRIP_PREFIXES)
      if (raw.startsWith(pfx)) return raw.slice(pfx.length).replace(/_/g, '-');
    return shortLabel(raw);
  };

  const nEvals = vPanels[0].nxEff();
  const nPoles = vPanels.map(pv => pv.nyEff());
  const maxPoles = Math.max(...nPoles);
  const nPan = vPanels.length;

  // Leftmost panel gets full Y_GUT for pole labels; others get a narrow gap only
  const Y_GUT_SMALL = 8;
  const gutters    = vPanels.map((_, pi) => pi === 0 ? Y_GUT : Y_GUT_SMALL);
  const blockWs    = gutters.map(g => g + nEvals * CW);
  const panelLeft  = [];
  let _px = MRG;
  for (let pi = 0; pi < nPan; pi++) { panelLeft.push(_px); _px += blockWs[pi] + GAP; }
  const cellsLeftA = panelLeft.map((x, pi) => x + gutters[pi]);

  // Shared vertical offsets (title drawn below cells, not above)
  const axisLblY = MRG + AX_H - 2;        // "evaluated propensity →" baseline
  const cellsTop = MRG + AX_H + X_GUT;

  const totalW = _px - GAP + CB_GAP + CB_W + CB_LABEL + MRG;
  const totalH = cellsTop + maxPoles * CH + TTL_H + MRG;

  const off = document.createElement('canvas');
  off.width  = Math.ceil(totalW * DPR);
  off.height = Math.ceil(totalH * DPR);
  const ctx  = off.getContext('2d');
  ctx.scale(DPR, DPR);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, totalW, totalH);

  const [vmin, vmax] = rangeForMetric(state.metric);
  const ramp = RAMP_SEQ;   // always red(neg)→yellow(0)→green(pos) in screenshot
  const col  = v => rampSample(ramp, (v - vmin) / (vmax - vmin));

  // "← intended propensity" on the far left, centered over all cell rows
  ctx.save();
  ctx.translate(MRG + 8, cellsTop + maxPoles * CH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center'; ctx.font = FONTS; ctx.fillStyle = DIM;
  ctx.fillText('← intended propensity', 0, 0);
  ctx.restore();

  vPanels.forEach((pv, pi) => {
    const cLeft    = cellsLeftA[pi];
    const nPi      = nPoles[pi];
    const diffOn   = state.diff && pv.p.base_idx >= 0;
    const baseRow  = diffOn ? effectiveBaseRow(pv.p) : null;
    const panelIdx = PANELS.indexOf(pv.p);

    // "evaluated propensity →" above each panel's eval labels
    ctx.font = FONTS; ctx.fillStyle = DIM; ctx.textAlign = 'left';
    ctx.fillText('evaluated propensity →', cLeft, axisLblY);

    // Eval column labels (rotated, at top of this panel's cells)
    ctx.font = FONT; ctx.fillStyle = DIM;
    for (let k = 0; k < nEvals; k++) {
      const lbl = shortLabel(pv.p.x_labels[pv.activeCols[k]]);
      ctx.save();
      ctx.translate(cLeft + k * CW + CW / 2, cellsTop - 2);
      ctx.rotate(-Math.PI / 3);
      ctx.textAlign = 'left'; ctx.fillText(lbl, 0, 0);
      ctx.restore();
    }

    // Title (below cells)
    ctx.font = FONTB; ctx.fillStyle = INK; ctx.textAlign = 'left';
    ctx.fillText(pv.p.title, cLeft, cellsTop + nPi * CH + TTL_H - 4);

    // Pole labels: only on leftmost panel, with screenshot-specific renaming
    if (pi === 0) {
      ctx.font = FONT; ctx.fillStyle = DIM; ctx.textAlign = 'right';
      for (let j = 0; j < nPi; j++) {
        const lbl = scrPoleLabel(pv.p.y_labels[pv.activeRows[j]]);
        ctx.fillText(lbl, cLeft - 5, cellsTop + j * CH + CH / 2 + 3);
      }
    }

    // Cells: row j = pole, col k = eval
    for (let j = 0; j < nPi; j++) {
      const dYi = pv.activeRows[j];
      for (let k = 0; k < nEvals; k++) {
        const dXi = pv.activeCols[k];
        const cell = effectiveCell(pv.p, dYi, dXi);
        const px = cLeft + k * CW;
        const py = cellsTop + j * CH;

        let grey = !cell || cell.m == null;
        if (!grey && state.threshold > 0 && cell.nt > 0 && (cell.n / cell.nt) <= state.threshold) grey = true;

        if (grey) {
          ctx.fillStyle = NULL_CELL; ctx.fillRect(px, py, CW, CH);
        } else {
          const base = baseRow ? baseRow[dXi] : null;
          let mv = cell.m, mnv = cell.mn, mxv = cell.mx, sv = cell.s;
          if (diffOn && base) {
            mv  = mv  != null && base.m  != null ? mv  - base.m  : null;
            mnv = mnv != null && base.mn != null ? mnv - base.mn : null;
            mxv = mxv != null && base.mx != null ? mxv - base.mx : null;
          }
          if (state.normalize && sv) {
            const d = Math.max(sv, 1);
            if (mv  != null) mv  = mv  / d;
            if (mnv != null) mnv = mnv / d;
            if (mxv != null) mxv = mxv / d;
          }
          if (state.metric === 'mean') {
            ctx.fillStyle = mv != null ? col(mv) : NULL_CELL; ctx.fillRect(px, py, CW, CH);
          } else if (state.metric === 'minmax') {
            drawSplit(ctx, px, py, CW, CH, mnv != null ? col(mnv) : NULL_CELL, mxv != null ? col(mxv) : NULL_CELL);
          } else {
            let lo, hi;
            if (state.normalize) { lo = mv != null ? mv - 1 : null; hi = mv != null ? mv + 1 : null; }
            else if (diffOn && base) {
              const cs = cell.s, bs = base.s;
              lo = cs != null && bs != null && cell.m != null && base.m != null ? (cell.m - cs) - (base.m - bs) : null;
              hi = cs != null && bs != null && cell.m != null && base.m != null ? (cell.m + cs) - (base.m + bs) : null;
            } else { lo = sv != null ? cell.m - sv : mv; hi = sv != null ? cell.m + sv : mv; }
            drawSplit(ctx, px, py, CW, CH, lo != null ? col(lo) : NULL_CELL, hi != null ? col(hi) : NULL_CELL);
          }
          if (pv.p.diag && pv.p.diag[String(dYi)] === dXi) {
            ctx.lineWidth = 1; ctx.strokeStyle = '#555';
            ctx.strokeRect(px + .5, py + .5, CW - 1, CH - 1);
          }
        }

        // Pink highlight (skip for eval-orthogonality panel)
        if (state.highlightMode && pv.p.group !== 'eval-orthogonality') {
          const yk = pv.p.corr_y_keys[dYi];
          const xk = pv.p.corr_x_keys[dXi];
          const hk = yk && xk ? yk + '|' + xk : null;
          let box = false;
          if (hk) {
            if (state.highlightMode === 'all') {
              box = !!(state._highlightAllKeys && state._highlightAllKeys.has(hk));
            } else {
              const ks = state._highlightPromptedKeys && state._highlightPromptedKeys.get(panelIdx);
              box = !!(ks && ks.has(hk));
            }
          }
          if (box) {
            ctx.lineWidth = 1.5; ctx.strokeStyle = '#ff5cbf';
            ctx.strokeRect(px + .75, py + .75, CW - 1.5, CH - 1.5);
          }
        }
      }
    }

    // Panel border
    ctx.strokeStyle = BORDER; ctx.lineWidth = 1;
    ctx.strokeRect(cLeft - .5, cellsTop - .5, nEvals * CW + 1, nPi * CH + 1);
  });

  // Colorbar (right side of last panel, spanning max cell height)
  const cbX = cellsLeftA[nPan - 1] + nEvals * CW + CB_GAP;
  const cbY = cellsTop;
  const cbH = maxPoles * CH;
  const grad = ctx.createLinearGradient(0, cbY + cbH, 0, cbY);
  ramp.forEach(([t, [r, g, b]]) => grad.addColorStop(t, `rgb(${r},${g},${b})`));
  ctx.fillStyle = grad; ctx.fillRect(cbX, cbY, CB_W, cbH);
  ctx.strokeStyle = BORDER; ctx.lineWidth = 1; ctx.strokeRect(cbX, cbY, CB_W, cbH);
  ctx.font = FONT; ctx.fillStyle = DIM; ctx.textAlign = 'left';
  [0, .25, .5, .75, 1].forEach(t => {
    const v  = vmin + t * (vmax - vmin);
    const ty = cbY + (1 - t) * cbH;
    ctx.fillStyle = TICK; ctx.fillRect(cbX + CB_W, ty - .5, 4, 1);
    ctx.fillStyle = DIM; ctx.fillText(v.toFixed(1), cbX + CB_W + 7, ty + 3);
  });
  ctx.save();
  ctx.translate(cbX - 6, cbY + cbH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center'; ctx.fillStyle = DIM;
  ctx.fillText(state.diff ? 'Δ ' + state.metric : state.metric, 0, 0);
  ctx.restore();

  return off;
}

// ── Symmetry table ─────────────────────────────────────────────────────────
function buildSymAxisMap(pv) {
  const map = new Map();
  const diag = pv.p.diag || {};
  for (let yi = 0; yi < pv.p.y_labels.length; yi++) {
    const yk = pv.p.corr_y_keys[yi];
    if (!yk) continue;
    const sep = yk.indexOf('|');
    if (sep < 0) continue;
    const axis = yk.slice(0, sep), sign = yk.slice(sep + 1);
    if (sign !== 'plus' && sign !== 'minus') continue;
    if (!map.has(axis)) map.set(axis, { plusRow: null, minusRow: null, evalCol: null });
    const e = map.get(axis);
    if (sign === 'plus') e.plusRow = yi; else e.minusRow = yi;
    if (e.evalCol === null) { const dc = diag[String(yi)]; if (dc !== undefined) e.evalCol = dc; }
  }
  return map;
}

function symRaw(pv, rowIdx, colIdx) {
  const cell = effectiveCell(pv.p, rowIdx, colIdx);
  if (!cell || cell.m == null) return null;
  if (state.threshold > 0 && cell.nt > 0 && cell.n / cell.nt <= state.threshold) return null;
  const baseRow = effectiveBaseRow(pv.p);
  if (!baseRow) return null;
  const base = baseRow[colIdx];
  if (!base || base.m == null) return null;
  if (state.symAbsolute) return cell.m - base.m;
  return base.m !== 0 ? cell.m / base.m : null;
}

function symTriggers(raw) {
  if (raw === null) return false;
  return state.symAbsolute ? Math.abs(raw) > state.symAbsThreshold
                           : Math.abs(raw - 1) > state.symRelThreshold;
}

function symGoodness(raw, minus) {
  if (raw === null) return null;
  return state.symAbsolute ? (minus ? -raw : raw) : (minus ? 1 - raw : raw - 1);
}

function symColor(goodness) {
  const scale = state.symAbsolute ? Math.max(state.symAbsThreshold * 3, 0.01)
                                  : Math.max(state.symRelThreshold * 3, 0.01);
  const t = 0.5 + 0.5 * Math.max(-1, Math.min(1, goodness / scale));
  return rampSample(RAMP_SYM, t);
}

function getClusterCells(pv, axisMap, ai, aj) {
  const ei = axisMap.get(ai), ej = axisMap.get(aj);
  const arSet = new Set(pv.activeRows), acSet = new Set(pv.activeCols);
  const defs = [
    { row: ei?.plusRow  ?? null, col: ej?.evalCol ?? null, minus: false },
    { row: ei?.minusRow ?? null, col: ej?.evalCol ?? null, minus: true  },
    { row: ej?.plusRow  ?? null, col: ei?.evalCol ?? null, minus: false },
    { row: ej?.minusRow ?? null, col: ei?.evalCol ?? null, minus: true  },
  ];
  return defs.map(d => {
    const vis = d.row !== null && d.col !== null && arSet.has(d.row) && acSet.has(d.col);
    const raw = vis ? symRaw(pv, d.row, d.col) : null;
    return { ...d, raw, triggers: symTriggers(raw), goodness: symGoodness(raw, d.minus) };
  });
}

function clusterRowHtml(ai, aj, cells) {
  const dp = state.symAbsolute ? 2 : 3;
  const f = v => v === null ? '—' : v.toFixed(dp);
  let s = `<td class="sym-lbl">${ai}</td><td class="sym-lbl">${aj}</td>`;
  for (const c of cells) {
    const bg = c.goodness !== null ? `background:${symColor(c.goodness)};color:#111;` : '';
    const cls = c.triggers ? ' sym-trig' : '';
    s += `<td class="sym-num${cls}" style="${bg}">${f(c.raw)}</td>`;
  }
  return s;
}

function computeSymHighlightKeys(ai, aj) {
  const eopView = panelViews.find(pv => pv.p.group === 'eval-orthogonality');
  if (!eopView) return null;
  const am = buildSymAxisMap(eopView);
  const ei = am.get(ai), ej = am.get(aj);
  if (!ei || !ej) return null;
  const keys = new Set();
  const addK = (row, col) => {
    if (row === null || col === null) return;
    const yk = eopView.p.corr_y_keys[row], xk = eopView.p.corr_x_keys[col];
    if (yk && xk) keys.add(yk + '|' + xk);
  };
  addK(ei.plusRow, ej.evalCol); addK(ei.minusRow, ej.evalCol);
  addK(ej.plusRow, ei.evalCol); addK(ej.minusRow, ei.evalCol);
  return keys;
}

function setSymHover(cluster) {
  const prev = state.symHoverCluster;
  if (cluster === prev || (cluster && prev && cluster.ai === prev.ai && cluster.aj === prev.aj)) return;
  state.symHoverCluster = cluster;
  state._symHighlightKeys = cluster ? computeSymHighlightKeys(cluster.ai, cluster.aj) : null;
  if (state.symMode) { updateSymHoverRows(); for (const pv of panelViews) pv.render(); }
}

function updateSymHoverRows() {
  const vPvs = panelViews.filter((_, i) => state.visiblePanels[i] && panelViews[i].p.group !== 'ft-sp-diff');
  const hc = state.symHoverCluster;
  document.querySelectorAll('#symTableContainer .sym-hr').forEach((tr, pi) => {
    if (pi >= vPvs.length) return;
    const pv = vPvs[pi];
    const am = buildSymAxisMap(pv);
    if (hc) {
      tr.innerHTML = clusterRowHtml(hc.ai, hc.aj, getClusterCells(pv, am, hc.ai, hc.aj));
      tr.style.background = 'rgba(255,92,191,0.12)';
    } else {
      tr.innerHTML = `<td colspan="6" style="text-align:center;color:var(--ink-dim);font-size:10px;padding:2px 0">— hover to pin —</td>`;
      tr.style.background = '';
    }
  });
}

function renderSymTables() {
  const cont = document.getElementById('symTableContainer');
  if (!state.symMode) { cont.innerHTML = ''; return; }
  const vPvs = panelViews.filter((_, i) => state.visiblePanels[i] && panelViews[i].p.group !== 'ft-sp-diff');
  if (!vPvs.length) { cont.innerHTML = '<p style="padding:14px;color:var(--ink-dim)">no visible panels</p>'; return; }

  const axisMaps = vPvs.map(pv => buildSymAxisMap(pv));
  const allAxesSet = new Set();
  axisMaps.forEach(m => m.forEach((_, ax) => allAxesSet.add(ax)));
  const axes = [...allAxesSet].sort();

  // Collect triggering clusters
  const clusters = [];
  for (let ii = 0; ii < axes.length; ii++) {
    for (let jj = ii + 1; jj < axes.length; jj++) {
      const ai = axes[ii], aj = axes[jj];
      const perPanel = vPvs.map((pv, pi) => getClusterCells(pv, axisMaps[pi], ai, aj));
      if (perPanel.some(cells => cells.some(c => c.triggers)))
        clusters.push({ ai, aj, perPanel });
    }
  }
  // Sort by max |goodness| desc
  clusters.sort((a, b) => {
    const mg = c => Math.max(0, ...c.perPanel.flatMap(cells => cells.map(c2 => c2.goodness !== null ? Math.abs(c2.goodness) : 0)));
    return mg(b) - mg(a);
  });

  const hc = state.symHoverCluster;
  let html = '<div class="sym-wrap">';
  for (let pi = 0; pi < vPvs.length; pi++) {
    const pv = vPvs[pi], am = axisMaps[pi];
    html += `<div class="sym-pbox"><div class="sym-ptitle">${pv.p.title}</div>`;
    html += `<table class="sym-tbl"><thead><tr>`;
    html += `<th>i</th><th>j</th><th>i+·ej</th><th>i−·ej</th><th>j+·ei</th><th>j−·ei</th>`;
    html += `</tr></thead><tbody>`;
    // Pinned hover row
    html += `<tr class="sym-hr" style="${hc ? 'background:rgba(255,92,191,0.12)' : ''}">`;
    if (hc) html += clusterRowHtml(hc.ai, hc.aj, getClusterCells(pv, am, hc.ai, hc.aj));
    else html += `<td colspan="6" style="text-align:center;color:var(--ink-dim);font-size:10px;padding:2px 0">— hover to pin —</td>`;
    html += `</tr>`;
    // Data rows
    for (const { ai, aj, perPanel } of clusters) {
      html += `<tr class="sym-dr" data-cluster="${ai}||${aj}">`;
      html += clusterRowHtml(ai, aj, perPanel[pi]);
      html += `</tr>`;
    }
    html += `</tbody></table></div>`;
  }
  html += '</div>';
  cont.innerHTML = html;
}

// ── FT-SP Diff panels ───────────────────────────────────────────────────────
// Returns the "displayed" cell values for panel p at (yi, xi) after applying
// all state transformations (diff mode, normalize) but NOT metric selection.
function getDisplayedValues(p, yi, xi) {
  let cell;
  if (yi === p.base_idx && state.median && p.group === 'eval-orthogonality' && p.baseline_alt) {
    cell = p.baseline_alt[xi];
  } else {
    cell = p.matrix[yi][xi];
  }
  if (!cell || cell.m == null) return null;
  if (state.threshold > 0 && cell.nt > 0 && cell.n / cell.nt <= state.threshold) return null;
  let mv = cell.m, mnv = cell.mn, mxv = cell.mx, sv = cell.s;
  const diffOn = state.diff && p.base_idx >= 0;
  if (diffOn) {
    const baseRow = effectiveBaseRow(p);
    const base = baseRow ? baseRow[xi] : null;
    if (!base || base.m == null) return null;
    mv  = mv  != null && base.m  != null ? mv  - base.m  : null;
    mnv = mnv != null && base.mn != null ? mnv - base.mn : null;
    mxv = mxv != null && base.mx != null ? mxv - base.mx : null;
  }
  if (state.normalize && sv != null) {
    const d = Math.max(sv, 1);
    if (mv  != null) mv  = mv  / d;
    if (mnv != null) mnv = mnv / d;
    if (mxv != null) mxv = mxv / d;
  }
  return { m: mv, mn: mnv, mx: mxv, s: null, n: cell.n, nt: cell.nt, nn: cell.nn, nf: cell.nf };
}

function computeFtSpDiffCell(ftPanel, ftYi, spPanel, spYi, xi) {
  const ft = getDisplayedValues(ftPanel, ftYi, xi);
  const sp = getDisplayedValues(spPanel, spYi, xi);
  if (!ft || !sp) return null;
  const m  = ft.m  != null && sp.m  != null ? ft.m  - sp.m  : null;
  const mn = ft.mn != null && sp.mn != null ? ft.mn - sp.mn : null;
  const mx = ft.mx != null && sp.mx != null ? ft.mx - sp.mx : null;
  if (m == null && mn == null && mx == null) return null;
  return { m, mn, mx, s: null, n: null, nt: null, nn: null, nf: null };
}

function createFtSpDiffPanel(ftPanel, spPanel) {
  const modelKey = ftPanel.model;
  const modelShort = TABLE_MODEL_SHORT[modelKey] || MODEL_SHORT[modelKey] || modelKey;
  const spByKey = new Map();
  for (let yi = 0; yi < spPanel.y_labels.length; yi++) {
    const k = spPanel.corr_y_keys[yi];
    if (k) spByKey.set(k, yi);
  }
  const ftRowIdxs = [], y_labels = [], corr_y_keys = [], reduce_mask = [];
  const ftRowToSpRow = new Map();
  for (let yi = 0; yi < ftPanel.y_labels.length; yi++) {
    if (yi === ftPanel.base_idx) continue;
    const k = ftPanel.corr_y_keys[yi];
    if (!k) continue;
    const spYi = spByKey.get(k);
    if (spYi == null) continue;
    ftRowIdxs.push(yi);
    y_labels.push(ftPanel.y_labels[yi]);
    corr_y_keys.push(k);
    reduce_mask.push((ftPanel.reduce_mask || [])[yi] ?? true);
    ftRowToSpRow.set(yi, spYi);
  }
  const matrix = y_labels.map(() => ftPanel.x_labels.map(() => null));
  const diag = {};
  for (let di = 0; di < ftRowIdxs.length; di++) {
    const dc = (ftPanel.diag || {})[String(ftRowIdxs[di])];
    if (dc !== undefined) diag[di] = dc;
  }
  return {
    title: `diff (FT-SP): ${modelShort}`,
    group: 'ft-sp-diff',
    tick_group: 'ft-sp-diff',
    model: modelKey,
    ft_panel: ftPanel,
    sp_panel: spPanel,
    _ftRowIdxs: ftRowIdxs,
    _ftRowToSpRow: ftRowToSpRow,
    y_labels, x_labels: ftPanel.x_labels, matrix,
    base_idx: -1, reduce_mask, diag, baseline_alt: null,
    corr_y_keys, corr_x_keys: ftPanel.corr_x_keys,
    col_eval_mismatch: ftPanel.col_eval_mismatch,
    col_evals_seen: ftPanel.col_evals_seen,
  };
}

function renderShowTable() {
  const tableEl = document.getElementById('showTable');
  if (!state.spilloverMode) { tableEl.innerHTML = ''; state._tableRows = []; state._tableOtherPanelIdxs = []; return; }
  if (state.ratioThreshold <= 0 && state.showThreshold <= 0) {
    tableEl.innerHTML = '';
    state._tableRows = []; state._tableOtherPanelIdxs = [];
    if (state.highlightMode) { computeHighlightSets(); for (const pv of panelViews) pv.render(); }
    return;
  }

  const eop = PANELS.find(p => p.group === 'eval-orthogonality');
  if (!eop) { tableEl.innerHTML = ''; return; }

  const eopView = panelViews ? panelViews.find(pv => pv.p.group === 'eval-orthogonality') : null;
  const eopActiveRows = eopView ? new Set(eopView.activeRows) : null;
  const eopActiveCols = eopView ? new Set(eopView.activeCols) : null;

  const ftPanelIdxs = [], spPanelIdxs = [];
  const ftPanels = PANELS.filter((p, i) => { if (p.group === 'finetuned' && state.visiblePanels[i]) { ftPanelIdxs.push(i); return true; } return false; });
  const spPanels = PANELS.filter((p, i) => { if (p.group === 'system-prompted' && state.visiblePanels[i]) { spPanelIdxs.push(i); return true; } return false; });
  const otherPanels = [...ftPanels, ...spPanels];
  const otherPanelIdxs = [...ftPanelIdxs, ...spPanelIdxs];

  function buildYMap(panel) {
    const m = {};
    for (let yi = 0; yi < panel.y_labels.length; yi++) {
      const k = panel.corr_y_keys[yi];
      if (k) m[k] = yi;
    }
    return m;
  }
  function buildDirectYMap(panel) {
    const m = {};
    for (let yi = 0; yi < panel.y_labels.length; yi++) m[panel.y_labels[yi]] = yi;
    return m;
  }
  function buildXMap(panel) {
    const m = {};
    for (let xi = 0; xi < panel.x_labels.length; xi++) m[panel.x_labels[xi]] = xi;
    return m;
  }

  const otherYMaps = otherPanels.map(buildYMap);
  // For finetuned panels: direct y_label fallback (eval-orth and finetuned share the same label format)
  const ftDirectYMaps = ftPanels.map(buildDirectYMap);
  const otherXMaps = otherPanels.map(buildXMap);

  const eopBaseRow = (state.median && eop.baseline_alt) ? eop.baseline_alt
    : (eop.base_idx >= 0 ? eop.matrix[eop.base_idx] : null);
  const otherBaseRows = otherPanels.map(p => p.base_idx >= 0 ? p.matrix[p.base_idx] : null);

  // Mirrors the grey-out logic in the canvas renderer.
  function isGrey(cell) {
    if (!cell || cell.m == null) return true;
    return state.threshold > 0 && cell.nt > 0 && (cell.n / cell.nt) <= state.threshold;
  }
  // Always compute raw diff and ratio (diff/max(std,1)) independently of normalize toggle.
  // Grey cells are treated as null (no reliable data).
  function rawDiffOf(cell, base) {
    if (isGrey(cell) || !base || base.m == null) return null;
    return cell.m - base.m;
  }
  function ratioOf(cell, base) {
    const d = rawDiffOf(cell, base);
    if (d == null) return null;
    return d / Math.max(cell.s || 0, 1);
  }

  const rows = [];
  for (let yi = 1; yi < eop.y_labels.length; yi++) {
    if (eopActiveRows && !eopActiveRows.has(yi)) continue;
    const yk = eop.corr_y_keys[yi];
    const yl = eop.y_labels[yi];
    for (let xi = 0; xi < eop.x_labels.length; xi++) {
      if (eopActiveCols && !eopActiveCols.has(xi)) continue;
      const xl = eop.x_labels[xi];
      if (!state.includeDiagonal && eop.diag[yi] === xi) continue;
      const eopCell = eop.matrix[yi][xi];
      const eopBase = eopBaseRow ? eopBaseRow[xi] : null;
      const eopRatio = ratioOf(eopCell, eopBase);
      if (eopRatio === 0) continue;
      // null eval-orth ratio treated as 1 for ratio-threshold denominator
      const eopDenom = eopRatio ?? 1;

      const xk = eop.corr_x_keys[xi];
      const hk = (yk && xk) ? yk + '|' + xk : null;
      const otherEntries = [];
      let passCount = 0, anyNonNull = false;
      for (let pi = 0; pi < otherPanels.length; pi++) {
        let opYi = yk ? otherYMaps[pi][yk] : null;
        // Fallback for finetuned panels: direct y_label match
        if (opYi == null && pi < ftPanels.length) opYi = ftDirectYMaps[pi][yl];
        const opXi = xk != null ? otherXMaps[pi][xk] : null;
        if (opYi == null || opXi == null) { otherEntries.push({ rawDiff: null, ratio: null, std: null, passes: false }); continue; }
        const opCell = otherPanels[pi].matrix[opYi][opXi];
        const opBase = otherBaseRows[pi] ? otherBaseRows[pi][opXi] : null;
        const opRawDiff = rawDiffOf(opCell, opBase);
        const opRatio = ratioOf(opCell, opBase);
        const opStd = opCell ? opCell.s : null;
        const passes = opRatio != null &&
            (state.ratioThreshold <= 0 || Math.abs(opRatio) / Math.abs(eopDenom) > state.ratioThreshold) &&
            (state.showThreshold <= 0 || Math.abs(opRatio) > state.showThreshold);
        otherEntries.push({ rawDiff: opRawDiff, ratio: opRatio, std: opStd, passes });
        anyNonNull = true;
        if (passes) passCount++;
      }
      const include = passCount >= state.minAgree;
      if (!include) continue;
      rows.push({
        xl, yl, hk,
        eopRawDiff: rawDiffOf(eopCell, eopBase),
        eopRatio,
        eopStd: eopCell ? eopCell.s : null,
        entries: otherEntries,
      });
    }
  }

  state._tableRows = rows;
  state._tableOtherPanelIdxs = otherPanelIdxs;
  if (state.highlightMode) { computeHighlightSets(); for (const pv of panelViews) pv.render(); }

  if (rows.length === 0) {
    tableEl.innerHTML = '<p style="color:var(--ink-dim);font-size:12px;padding:8px 14px;">No entries above threshold.</p>';
    return;
  }

  rows.sort((a, b) => a.xl < b.xl ? -1 : a.xl > b.xl ? 1 : a.yl < b.yl ? -1 : a.yl > b.yl ? 1 : 0);

  const f = (v) => v == null ? '—' : v.toFixed(3);
  const shortName = (p) => TABLE_MODEL_SHORT[p.model] || MODEL_SHORT[p.model] || p.model;

  // Headers: sep marks the first column of each matrix group.
  function rowCells(row) {
    return `<td>${row.xl}</td><td>${row.yl}</td>` +
      `<td class="num sep">${f(row.eopRawDiff)}</td><td class="num">${f(row.eopStd)}</td><td class="num">${f(row.eopRatio)}</td>` +
      row.entries.map(e => {
        const b = e.passes ? ' trig' : '';
        return `<td class="num sep${b}">${f(e.rawDiff)}</td>` +
               `<td class="num${b}">${f(e.std)}</td>` +
               `<td class="num${b}">${f(e.ratio)}</td>`;
      }).join('');
  }

  const nCols = 2 + 3 * (1 + ftPanels.length + spPanels.length);
  const pinnedRow = state.hoverKey ? rows.find(r => r.hk === state.hoverKey) : null;
  const pinnedTr = pinnedRow
    ? `<tr style="background:rgba(255,92,191,0.18)"${pinnedRow.hk ? ` data-hk="${pinnedRow.hk}"` : ''}>${rowCells(pinnedRow)}</tr>`
    : `<tr style="background:rgba(255,92,191,0.10)"><td colspan="${nCols}" `+
      `style="text-align:center;color:var(--ink-dim);font-size:11px;padding:3px">` +
      `— hover a matrix cell or table row to pin here —</td></tr>`;

  let html = `<table class="show-table"><thead><tr>`;
  html += `<th rowspan="2">eval/col</th><th rowspan="2">pole/row</th>`;
  html += `<th class="sep" colspan="3">eo</th>`;
  for (const p of ftPanels) html += `<th class="sep" colspan="3">ft·${shortName(p)}</th>`;
  for (const p of spPanels) html += `<th class="sep" colspan="3">sp·${shortName(p)}</th>`;
  html += `</tr><tr>`;
  for (let g = 0; g < 1 + ftPanels.length + spPanels.length; g++)
    html += `<th class="sep">diff</th><th>std</th><th>ratio</th>`;
  html += `</tr></thead><tbody>`;
  html += pinnedTr;
  for (const row of rows) {
    const hkAttr = row.hk ? ` data-hk="${row.hk}"` : '';
    html += `<tr${hkAttr}>${rowCells(row)}</tr>`;
  }
  html += '</tbody></table>';
  tableEl.innerHTML = html;
}

const tt = document.getElementById('tt');
const gridEl = document.getElementById('grid');
const panelViews = PANELS.map((p, i) => new PanelView(p, gridEl, i));

// FT-SP diff panels (virtual, JS-only — not in PANELS array)
{
  const ftPanels = PANELS.filter(p => p.group === 'finetuned');
  const spPanels = PANELS.filter(p => p.group === 'system-prompted');
  for (const ftP of ftPanels) {
    const spP = spPanels.find(p => p.model === ftP.model);
    if (!spP) continue;
    const diffPanel = createFtSpDiffPanel(ftP, spP);
    const pv = new PanelView(diffPanel, gridEl, panelViews.length);
    pv.rootEl.style.display = 'none';
    panelViews.push(pv);
    state.visiblePanels.push(false);
  }
}

// Table hover → matrix highlight (event delegation; listeners survive innerHTML replacement)
(function() {
  const tbl = document.getElementById('showTable');
  tbl.addEventListener('mouseover', (e) => {
    const tr = e.target.closest('tr[data-hk]');
    if (tr) setHoverKey(tr.dataset.hk || null);
  });
  tbl.addEventListener('mouseleave', () => setHoverKey(null));
})();

// Panel visibility toggle buttons
(function() {
  const bar = document.getElementById('panelToggleBar');
  panelViews.forEach((pv, i) => {
    const btn = document.createElement('button');
    btn.className = 'plain';
    btn.classList.toggle('panel-hidden', !state.visiblePanels[i]);
    btn.textContent = pv.p.title;
    btn.addEventListener('click', () => {
      state.visiblePanels[i] = !state.visiblePanels[i];
      pv.rootEl.style.display = state.visiblePanels[i] ? '' : 'none';
      btn.classList.toggle('panel-hidden', !state.visiblePanels[i]);
      for (let j = 0; j < panelViews.length; j++) {
        if (!state.visiblePanels[j]) continue;
        panelViews[j].layout();
        panelViews[j].render();
      }
      renderShowTable();
      if (state.symMode) renderSymTables();
    });
    bar.appendChild(btn);
  });
})();

function updateLegend() {
  const [lo, hi] = rangeForMetric(state.metric);
  document.getElementById('legendText').textContent = `${lo.toFixed(1)} → ${hi.toFixed(1)}`;
}
function setMetric(m) {
  state.metric = m;
  document.querySelectorAll('#metricSeg button').forEach(b => b.classList.toggle('active', b.dataset.metric === m));
  updateLegend();
  for (const pv of panelViews) pv.render();
  if (state.symMode) renderSymTables();
}
function setLayout(l) {
  state.layout = l;
  document.querySelectorAll('#layoutSeg button').forEach(b => b.classList.toggle('active', b.dataset.layout === l));
  gridEl.classList.toggle('layout-horizontal', l === 'horizontal');
  gridEl.classList.toggle('layout-vertical', l === 'vertical');
  for (const pv of panelViews) { pv.layout(); pv.render(); }
}
document.querySelectorAll('#metricSeg button').forEach(b => {
  b.addEventListener('click', () => setMetric(b.dataset.metric));
});
document.querySelectorAll('#layoutSeg button').forEach(b => {
  b.addEventListener('click', () => setLayout(b.dataset.layout));
});
document.getElementById('zoomAll').addEventListener('change', (e) => {
  state.zoomAll = e.target.checked;
});
document.getElementById('diff').addEventListener('change', (e) => {
  state.diff = e.target.checked;
  for (const pv of panelViews) {
    pv.diffBadge.style.display = (state.diff && pv.p.base_idx < 0) ? 'inline' : 'none';
    pv.recomputeActive();
    pv.layout();
  }
  updateLegend();
  for (const pv of panelViews) pv.render();
  renderShowTable();
  if (state.symMode) renderSymTables();
});
document.getElementById('reduce').addEventListener('change', (e) => {
  state.reduce = e.target.checked;
  if (state.reduce) {
    POLES.forEach((pole, i) => {
      if (!pole.in_reduce && state.visibleRows[i]) {
        state.visibleRows[i] = false;
        if (polesCheckboxes[i]) polesCheckboxes[i].checked = false;
      }
    });
    updatePolesCount();
  }
  for (const pv of panelViews) { pv.recomputeActive(); pv.layout(); }
  for (const pv of panelViews) pv.render();
  renderShowTable();
  if (state.symMode) renderSymTables();
});
document.getElementById('median').addEventListener('change', (e) => {
  state.median = e.target.checked;
  updateNormRanges();
  updateLegend();
  for (const pv of panelViews) pv.render();
  renderShowTable();
  if (state.symMode) renderSymTables();
});
document.getElementById('threshold').addEventListener('input', (e) => {
  const v = parseFloat(e.target.value);
  state.threshold = Number.isFinite(v) ? Math.max(0, Math.min(1, v)) : 0;
  for (const pv of panelViews) pv.render();
  if (state.symMode) renderSymTables();
});
document.getElementById('normalize').addEventListener('change', (e) => {
  state.normalize = e.target.checked;
  updateNormRanges();
  updateLegend();
  for (const pv of panelViews) pv.render();
  renderShowTable();
});
document.getElementById('includeDiagonal').addEventListener('change', (e) => {
  state.includeDiagonal = e.target.checked;
  renderShowTable();
});
document.getElementById('minAgree').addEventListener('input', (e) => {
  const v = parseInt(e.target.value, 10);
  state.minAgree = Number.isFinite(v) ? Math.max(1, v) : 1;
  renderShowTable();
});
document.getElementById('ratioThreshold').addEventListener('input', (e) => {
  const v = parseFloat(e.target.value);
  state.ratioThreshold = Number.isFinite(v) ? Math.max(0, v) : 0;
  renderShowTable();
});
document.getElementById('showThreshold').addEventListener('input', (e) => {
  const v = parseFloat(e.target.value);
  state.showThreshold = Number.isFinite(v) ? Math.max(0, v) : 0;
  renderShowTable();
});
document.getElementById('resetZoom').addEventListener('click', () => {
  for (const pv of panelViews) pv.reset(false);
});
// Mode toggle: unexpected spillover / symmetry
(function() {
  function setSpilloverMode(on) {
    state.spilloverMode = on;
    state.symMode = false;
    state.symHoverCluster = null;
    state._symHighlightKeys = null;
    document.getElementById('btnSpillover').classList.toggle('active', on);
    document.getElementById('btnSymmetry').classList.toggle('active', false);
    document.getElementById('spilloverControls').classList.toggle('vis', on);
    document.getElementById('symmetryControls').classList.toggle('vis', false);
    document.getElementById('symTableContainer').innerHTML = '';
    if (on) renderShowTable();
    else { renderShowTable(); for (const pv of panelViews) pv.render(); }
  }
  function setSymMode(on) {
    state.symMode = on;
    state.spilloverMode = false;
    state.highlightMode = null;
    state.symHoverCluster = null;
    state._symHighlightKeys = null;
    document.getElementById('btnSymmetry').classList.toggle('active', on);
    document.getElementById('btnSpillover').classList.toggle('active', false);
    document.getElementById('symmetryControls').classList.toggle('vis', on);
    document.getElementById('spilloverControls').classList.toggle('vis', false);
    document.getElementById('hlAll').classList.remove('active');
    document.getElementById('hlPrompted').classList.remove('active');
    renderShowTable();
    if (on) renderSymTables(); else { document.getElementById('symTableContainer').innerHTML = ''; }
    for (const pv of panelViews) pv.render();
  }
  document.getElementById('btnSpillover').addEventListener('click', () => setSpilloverMode(!state.spilloverMode));
  document.getElementById('btnSymmetry').addEventListener('click', () => setSymMode(!state.symMode));
  // Sym threshold type toggle
  document.getElementById('symAbsBtn').addEventListener('click', () => {
    if (state.symAbsolute) return;
    state.symAbsolute = true;
    document.getElementById('symAbsBtn').classList.add('active');
    document.getElementById('symRelBtn').classList.remove('active');
    if (state.symMode) renderSymTables();
  });
  document.getElementById('symRelBtn').addEventListener('click', () => {
    if (!state.symAbsolute) return;
    state.symAbsolute = false;
    document.getElementById('symRelBtn').classList.add('active');
    document.getElementById('symAbsBtn').classList.remove('active');
    if (state.symMode) renderSymTables();
  });
  document.getElementById('symAbsT').addEventListener('input', e => {
    const v = parseFloat(e.target.value);
    state.symAbsThreshold = Number.isFinite(v) ? Math.max(0, v) : 0.1;
    if (state.symMode) renderSymTables();
  });
  document.getElementById('symRelT').addEventListener('input', e => {
    const v = parseFloat(e.target.value);
    state.symRelThreshold = Number.isFinite(v) ? Math.max(0, v) : 0.1;
    if (state.symMode) renderSymTables();
  });
  // Sym table hover → matrix highlight
  const symCont = document.getElementById('symTableContainer');
  symCont.addEventListener('mouseover', e => {
    const tr = e.target.closest('tr[data-cluster]');
    if (tr) {
      const [ai, aj] = tr.dataset.cluster.split('||');
      setSymHover({ ai, aj });
    }
  });
  symCont.addEventListener('mouseleave', () => setSymHover(null));
})();
(function() {
  function setHighlightMode(mode) {
    state.highlightMode = mode;
    document.getElementById('hlAll').classList.toggle('active', mode === 'all');
    document.getElementById('hlPrompted').classList.toggle('active', mode === 'prompted');
    if (mode) computeHighlightSets();
    for (const pv of panelViews) pv.render();
  }
  document.getElementById('hlAll').addEventListener('click', () => {
    setHighlightMode(state.highlightMode === 'all' ? null : 'all');
  });
  document.getElementById('hlPrompted').addEventListener('click', () => {
    setHighlightMode(state.highlightMode === 'prompted' ? null : 'prompted');
  });
})();
document.getElementById('screenshot').addEventListener('click', () => {
  const canvas = screenshotCanvas();
  if (!canvas) { alert('No visible panels to screenshot.'); return; }
  const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const a = document.createElement('a');
  a.href = canvas.toDataURL('image/png');
  a.download = `compare_${ts}.png`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
});
// Eval column toggles
const evalsGrid = document.getElementById('evalsGrid');
const evalsCount = document.getElementById('evalsCount');
const evalsCheckboxes = [];
function updateEvalsCount() {
  const n = state.visibleCols.filter(Boolean).length;
  evalsCount.textContent = `(${n}/${state.visibleCols.length})`;
}
function applyEvalsChange() {
  for (const pv of panelViews) { pv.recomputeActive(); pv.layout(); }
  for (const pv of panelViews) pv.render();
  updateEvalsCount();
  renderShowTable();
  if (state.symMode) renderSymTables();
}
EVALS.forEach((name, i) => {
  const lbl = document.createElement('label');
  const cb = document.createElement('input');
  cb.type = 'checkbox';
  cb.checked = true;
  cb.dataset.idx = String(i);
  cb.addEventListener('change', () => {
    state.visibleCols[i] = cb.checked;
    applyEvalsChange();
  });
  evalsCheckboxes.push(cb);
  lbl.appendChild(cb);
  lbl.appendChild(document.createTextNode(' ' + name));
  evalsGrid.appendChild(lbl);
});
document.getElementById('evalsDefault').addEventListener('click', () => {
  state.visibleCols = EVALS.map(e => DEFAULT_EVALS.has(e));
  evalsCheckboxes.forEach((cb, i) => cb.checked = state.visibleCols[i]);
  applyEvalsChange();
});
document.getElementById('evalsAll').addEventListener('click', () => {
  state.visibleCols = state.visibleCols.map(() => true);
  evalsCheckboxes.forEach(cb => cb.checked = true);
  applyEvalsChange();
});
document.getElementById('evalsNone').addEventListener('click', () => {
  state.visibleCols = state.visibleCols.map(() => false);
  evalsCheckboxes.forEach(cb => cb.checked = false);
  applyEvalsChange();
});
updateEvalsCount();

// Pole row toggles
const polesGrid = document.getElementById('polesGrid');
const polesCount = document.getElementById('polesCount');
const polesCheckboxes = [];
function updatePolesCount() {
  const n = state.visibleRows.filter(Boolean).length;
  polesCount.textContent = `(${n}/${state.visibleRows.length})`;
}
function applyPolesChange() {
  for (const pv of panelViews) { pv.recomputeActive(); pv.layout(); }
  for (const pv of panelViews) pv.render();
  updatePolesCount();
  renderShowTable();
  if (state.symMode) renderSymTables();
}
POLES.forEach((pole, i) => {
  const lbl = document.createElement('label');
  const cb = document.createElement('input');
  cb.type = 'checkbox';
  cb.checked = true;
  cb.dataset.idx = String(i);
  cb.addEventListener('change', () => {
    state.visibleRows[i] = cb.checked;
    applyPolesChange();
  });
  polesCheckboxes.push(cb);
  lbl.appendChild(cb);
  lbl.appendChild(document.createTextNode(' ' + pole.name));
  polesGrid.appendChild(lbl);
});
document.getElementById('polesDefault').addEventListener('click', () => {
  state.visibleRows = POLES.map(p => DEFAULT_POLES.has(p.name));
  polesCheckboxes.forEach((cb, i) => cb.checked = state.visibleRows[i]);
  applyPolesChange();
});
document.getElementById('polesAll').addEventListener('click', () => {
  state.visibleRows = state.visibleRows.map(() => true);
  polesCheckboxes.forEach(cb => cb.checked = true);
  applyPolesChange();
});
document.getElementById('polesNone').addEventListener('click', () => {
  state.visibleRows = state.visibleRows.map(() => false);
  polesCheckboxes.forEach(cb => cb.checked = false);
  applyPolesChange();
});
updatePolesCount();

setLayout('horizontal');
setMetric('mean');
updateNormRanges();
renderShowTable();
</script>
</body>
</html>
"""


def main():
    panels = build_panels()
    ranges = global_ranges(panels)
    dranges_50 = diff_range(panels, use_median=False)
    dranges_med = diff_range(panels, use_median=True)
    evals_master = next(p["x_labels"] for p in panels if p["group"] == "finetuned")
    eo_panel = next(p for p in panels if p["group"] == "eval-orthogonality")
    poles_master = [
        {"name": label, "key": key, "in_reduce": rm}
        for label, key, rm in zip(
            eo_panel["y_labels"], eo_panel["corr_y_keys"], eo_panel["reduce_mask"]
        )
        if label != "baseline-50"
    ]
    html = (
        HTML
        .replace("__PANELS__", json.dumps(panels))
        .replace("__RANGES__", json.dumps(ranges))
        .replace("__DRANGES__", json.dumps(dranges_50))
        .replace("__DRANGES_MED__", json.dumps(dranges_med))
        .replace("__EVALS__", json.dumps(evals_master))
        .replace("__POLES__", json.dumps(poles_master))
        .replace("__MODEL_SHORT__", json.dumps(MODEL_SHORT))
        .replace("__TABLE_MODEL_SHORT__", json.dumps(TABLE_MODEL_SHORT))
    )
    OUT.write_text(html)
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.1f} KiB)")


if __name__ == "__main__":
    main()
