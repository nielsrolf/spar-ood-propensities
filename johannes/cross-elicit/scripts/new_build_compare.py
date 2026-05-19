"""new_build_compare.py — render new_eval_results/new_compare.html from score JSONs.

Reads the pre-built score JSON files from `new_eval_results/scores/` and renders
`new_eval_results/new_compare.html`.  Run `new_summary.py` first to generate or
update the JSON files from the raw eval directories.

Nothing under `results/` is read — the existing finetuned/sysprompts/eval-orth
JSONs there are intentionally ignored.

Notable adaptations vs build_compare.py:

  * Propensity definitions come from `evals/definitions.json` (build_compare
    used the older `def_sys.json`). Same shape, same axes, same pole keys —
    swapped in at import time.
  * The new `agreeableness_eval` emits five sub-judges; treated like
    `honest-humble`: one judge (`trust_score`) is the primary on the
    "agreeableness" column, the other four (`altruism_score`, etc.) become
    their own `agreeableness:<key>` columns. This is configured by patching
    `eval_matrix_core.PROPENSITY_TO_PRIMARY_METRIC` and `SECONDARY_JUDGE_ROWS`.
  * Pole labels are unified across all three panel types using the format
    `<axis><+/-/0>` (e.g. `agreeableness+`, `certainty0`), identified via
    `plus_pole_key`/`minus_pole_key`/`neutral_pole_key` in
    `evals/def_sys_plusminus.json`.  Ethical-framework rows get their framework
    name only (e.g. `deontological+`).  All panels share the same propensity
    ordering with ethical frameworks at the bottom, and within each propensity
    the order is always +, 0, − (top to bottom).
  * The POLES master is read from the eval-orth panel when present; otherwise
    derived from the finetuned panels' rows so the pole-filter UI still works.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import build_compare as bc
import eval_matrix_core as core

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
NEW_EVAL_RESULTS = ROOT / "new_eval_results"
BASE_ROOT = NEW_EVAL_RESULTS / "base_models"
FT_ROOT = NEW_EVAL_RESULTS / "finetuning"
SP_ROOT = NEW_EVAL_RESULTS / "sys_prompts"
SCORES_DIR = NEW_EVAL_RESULTS / "scores"
OUT = NEW_EVAL_RESULTS / "new_compare.html"

DEFINITIONS_PATH = ROOT / "evals" / "definitions.json"
DEF_PM_PATH = ROOT / "evals" / "def_sys_plusminus.json"

# ---------------------------------------------------------------------------
# Patch eval_matrix_core for the new evals.
# ---------------------------------------------------------------------------
_AGREEABLENESS_PRIMARY = "trust_score"
_AGREEABLENESS_SECONDARIES = [
    "altruism_score",
    "cooperation_score",
    "modesty_score",
    "sympathy_score",
]
core.PROPENSITY_TO_PRIMARY_METRIC["agreeableness"] = _AGREEABLENESS_PRIMARY
core.SECONDARY_JUDGE_ROWS["agreeableness"] = list(_AGREEABLENESS_SECONDARIES)

# Swap build_compare's propensity dictionary for evals/definitions.json.
bc.DEF_SYS = json.loads(DEFINITIONS_PATH.read_text())
bc.DEF_SYS_AXES = list(bc.DEF_SYS.keys())

# Monkey-patch _cell_metrics to return a diagnostic stub (m=null with counts) instead of
# None when a cell exists but has no mean score.  This lets the JS tooltip explain WHY
# the cell is grey rather than just showing "—".
_orig_cell_metrics = bc._cell_metrics


def _cell_metrics_diag(cell: dict | None) -> dict | None:
    if not cell:
        return None
    m = cell.get("metrics") or {}
    mean = m.get("mean")
    if mean is not None:
        return _orig_cell_metrics(cell)
    # mean is null — return a stub with diagnostic counts so the tooltip can explain
    n_total = m.get("n_total") or 0
    n_fails = m.get("n_fails") or 0
    n_nulls = m.get("n_nulls") or 0
    n_numeric = m.get("n_numeric") or 0
    if n_total == 0 and n_fails == 0 and n_nulls == 0 and n_numeric == 0:
        return None  # genuinely empty — treat as missing directory
    dn = ((cell.get("meta") or {}).get("dirname") or "")
    eval_name = dn.split("__", 1)[0] if dn else ""
    return {
        "m": None, "s": None, "mn": None, "mx": None,
        "n": n_numeric, "nt": n_total, "nn": n_nulls, "nf": n_fails,
        "e": eval_name,
    }


bc._cell_metrics = _cell_metrics_diag


# ---------------------------------------------------------------------------
# Unified pole labelling — same display format across all three panel types.
#
# Format: <axis-name><side>  where side ∈ {+, 0, -}
# Ethical-framework axes drop the "ethical-framework-" prefix.
# Propensity order: DEF_SYS_AXES order, EF axes forced to the end.
# Within each propensity: + first, 0 second, - last.
# ---------------------------------------------------------------------------

DEF_PM: dict = json.loads(DEF_PM_PATH.read_text())

_EF_PREFIX = "ethical-framework-"
_NON_EF_AXES: list[str] = [a for a in bc.DEF_SYS_AXES if not a.startswith(_EF_PREFIX)]
_EF_ORDER: dict[str, int] = {"deontological": 0, "utilitarian": 1, "virtue-ethics": 2}
_N_NON_EF: int = len(_NON_EF_AXES)


def _axis_prefix(axis: str) -> str:
    return axis[len(_EF_PREFIX):] if axis.startswith(_EF_PREFIX) else axis


def _rk_to_side(axis: str, response_key: str) -> str | None:
    """Map a response_key to '+', '-', or '0' via def_sys_plusminus.json."""
    info = DEF_PM.get(axis, {})
    if response_key == info.get("plus_pole_key"):  return "+"
    if response_key == info.get("minus_pole_key"): return "-"
    nk = info.get("neutral_pole_key")
    if nk and response_key == nk:                  return "0"
    return None


def _sp_pole_to_side(axis: str, suffix: str, all_suffixes: list[str]) -> str | None:
    """Identify the +/-/0 side of an SP suffix.

    Tries, in order:
      1. Exact match against stripped pole key (drop 'expected_' prefix).
      2. If the other suffix in a 2-pole axis matches a known pole, this is opposite.
      3. Axis-specific override (neuroticism: neurotic = plus).
      4. Alphabetical fallback (first = plus, last = minus, middle of 3 = neutral).
    """
    info = DEF_PM.get(axis, {})

    def stripped(k: str | None) -> str:
        return (k or "").replace("expected_", "")

    pk_plus    = stripped(info.get("plus_pole_key"))
    pk_minus   = stripped(info.get("minus_pole_key"))
    pk_neutral = stripped(info.get("neutral_pole_key"))

    if suffix == pk_plus:                          return "+"
    if pk_minus   and suffix == pk_minus:          return "-"
    if pk_neutral and suffix == pk_neutral:        return "0"

    if len(all_suffixes) == 2:
        other = next((s for s in all_suffixes if s != suffix), None)
        if other is not None:
            if other == pk_plus:                       return "-"
            if pk_minus and other == pk_minus:         return "+"

    _SP_PLUS_OVERRIDE = {"neuroticism": "neurotic"}
    override = _SP_PLUS_OVERRIDE.get(axis)
    if override is not None:
        return "+" if suffix == override else "-"

    sorted_suf = sorted(all_suffixes)
    if suffix == sorted_suf[0]:                    return "+"
    if len(sorted_suf) >= 2 and suffix == sorted_suf[-1]: return "-"
    if len(sorted_suf) == 3 and suffix == sorted_suf[1]:  return "0"
    return None


def _ft_to_unified(y_label: str) -> str:
    if y_label == "base":
        return "base"
    if y_label.endswith("-plus"):
        return f"{_axis_prefix(y_label[:-5])}+"
    if y_label.endswith("-minus"):
        return f"{_axis_prefix(y_label[:-6])}-"
    return y_label


def _eo_to_unified(y_label: str, col_meta: dict) -> str:
    if y_label in ("baseline-50", "base"):
        return y_label
    meta = col_meta.get(y_label, {})
    axis = meta.get("propensity", "")
    rk   = meta.get("response_key", "")
    if not axis:
        return y_label
    side = _rk_to_side(axis, rk)
    return f"{_axis_prefix(axis)}{side}" if side else y_label


def _sp_to_unified(y_label: str, sp_by_axis: dict[str, list[str]]) -> str:
    if y_label == "base":
        return "base"
    if "__" not in y_label:
        return y_label
    axis, suffix = y_label.split("__", 1)
    side = _sp_pole_to_side(axis, suffix, sp_by_axis.get(axis, [suffix]))
    return f"{_axis_prefix(axis)}{side}" if side else y_label


def _unified_sort_key(label: str) -> tuple:
    """Canonical row order: base → baseline-50 → non-EF axes (+/0/-) → EF axes (+/0/-)."""
    if label == "base":        return (-2, 0, 0)
    if label == "baseline-50": return (-1, 0, 0)
    if label.endswith("+"):    side_w, prefix = 0, label[:-1]
    elif label.endswith("0"):  side_w, prefix = 1, label[:-1]
    elif label.endswith("-"):  side_w, prefix = 2, label[:-1]
    else:                      return (_N_NON_EF + 10, 0, 0)
    if prefix in _EF_ORDER:
        return (_N_NON_EF, _EF_ORDER[prefix], side_w)
    ax_idx = _NON_EF_AXES.index(prefix) if prefix in _NON_EF_AXES else _N_NON_EF + 5
    return (ax_idx, 0, side_w)


def _unified_corr_y_key(label: str) -> str | None:
    """Cross-panel correspondence key from a unified label."""
    if label in ("base", "baseline-50"): return None
    if label.endswith("+"):    side, prefix = "plus",    label[:-1]
    elif label.endswith("0"):  side, prefix = "neutral", label[:-1]
    elif label.endswith("-"):  side, prefix = "minus",   label[:-1]
    else:                      return None
    if prefix in _EF_ORDER:
        return f"ef|{prefix.replace('-', '_')}"
    return f"{prefix}|{side}"


# Reverse map: axis_prefix → full axis name (built lazily after bc.DEF_SYS_AXES is set).
# Used by _recompute_diag_unified to resolve unified labels back to x_label eval names.
_PREFIX_TO_FULL_AXIS: dict[str, str] = {}


def _ensure_prefix_map() -> dict[str, str]:
    global _PREFIX_TO_FULL_AXIS
    if not _PREFIX_TO_FULL_AXIS:
        _PREFIX_TO_FULL_AXIS = {_axis_prefix(ax): ax for ax in bc.DEF_SYS_AXES}
    return _PREFIX_TO_FULL_AXIS


def _recompute_diag_unified(panel: dict) -> None:
    """Recompute panel['diag'] from unified y_labels for SP panels.

    SP panels have empty diag after bc.load_panel because _compute_diag relies on
    the old '--' separator which our '__'-based labels don't use.  After
    _apply_unified_labels renames rows to '<axis_prefix><side>', we can identify
    the diagonal column straightforwardly.
    """
    if panel["group"] != "system-prompted":
        return
    prefix_map = _ensure_prefix_map()
    x_idx = {xl: i for i, xl in enumerate(panel["x_labels"])}
    new_diag: dict[int, int] = {}
    for yi, yl in enumerate(panel["y_labels"]):
        if not (yl.endswith("+") or yl.endswith("-") or yl.endswith("0")):
            continue
        prefix = yl[:-1]
        full_axis = prefix_map.get(prefix)
        if full_axis and full_axis in x_idx:
            new_diag[yi] = x_idx[full_axis]
    panel["diag"] = new_diag


def _compute_most_neutral_baseline(
    panel: dict, def_pm: dict, fallback_alt: list
) -> list:
    """Per-column most-neutral-pole baseline for the EO panel.

    For each eval column, looks up 'most_neutral' from def_sys_plusminus.json,
    converts it to a unified y-label, and returns that row's cell for the column.
    Falls back to the median baseline cell when most_neutral is unset, the label is
    missing from the panel, or the cell has no data.

    Uses corr_x_keys (propensity names like 'agreeableness') rather than the raw
    EO x_labels (metric names like 'trust_score (agreeableness)') to look up axes
    in def_pm — the raw labels don't match any def_pm key.
    """
    label_to_yi = {lbl: i for i, lbl in enumerate(panel["y_labels"])}
    corr_x_keys = panel.get("corr_x_keys") or []
    result = []
    _fallback = fallback_alt or []
    for xi, x_label in enumerate(panel["x_labels"]):
        # Prefer corr_x_key (propensity name) over raw x_label for axis resolution.
        corr_xk = corr_x_keys[xi] if xi < len(corr_x_keys) else None
        raw_name = corr_xk or x_label
        # Strip secondary-judge suffix (e.g. "agreeableness:trust_score" → "agreeableness")
        axis = (
            raw_name.split(core.SECONDARY_PROPENSITY_SEP, 1)[0]
            if core.SECONDARY_PROPENSITY_SEP in raw_name
            else raw_name
        )
        axis_prefix = _axis_prefix(axis)
        most_neutral = def_pm.get(axis, {}).get("most_neutral", "")

        neutral_unified = None
        if most_neutral:
            side = _rk_to_side(axis, most_neutral)
            if side:
                neutral_unified = f"{axis_prefix}{side}"

        cell = None
        if neutral_unified and neutral_unified in label_to_yi:
            yi = label_to_yi[neutral_unified]
            cell = panel["matrix"][yi][xi]

        if cell is None or cell.get("m") is None:
            cell = (
                _fallback[xi]
                if xi < len(_fallback)
                else {"m": 50, "s": 0, "mn": 50, "mx": 50,
                      "n": 0, "nt": 0, "nn": 0, "nf": 0}
            )
        result.append(cell)
    return result


def _apply_unified_labels(panel: dict, col_meta: dict | None = None) -> None:
    """Rename y_labels to unified format, re-sort rows, recompute corr_y_keys — in-place."""
    group = panel["group"]
    old_labels = panel["y_labels"]

    if group == "finetuned":
        new_labels = [_ft_to_unified(l) for l in old_labels]
    elif group == "eval-orthogonality":
        new_labels = [_eo_to_unified(l, col_meta or {}) for l in old_labels]
    elif group == "system-prompted":
        sp_by_axis: dict[str, list[str]] = {}
        for l in old_labels:
            if "__" in l:
                ax, suf = l.split("__", 1)
                sp_by_axis.setdefault(ax, []).append(suf)
        new_labels = [_sp_to_unified(l, sp_by_axis) for l in old_labels]
    else:
        return

    n = len(new_labels)
    order = sorted(range(n), key=lambda i: _unified_sort_key(new_labels[i]))

    panel["y_labels"]    = [new_labels[i]                       for i in order]
    panel["matrix"]      = [panel["matrix"][i]                  for i in order]
    panel["reduce_mask"] = [panel["reduce_mask"][i]             for i in order]
    panel["corr_y_keys"] = [_unified_corr_y_key(new_labels[i]) for i in order]

    old_to_new = {old: new for new, old in enumerate(order)}
    panel["diag"] = {old_to_new[k]: v for k, v in panel["diag"].items() if k in old_to_new}

    if "base" in panel["y_labels"]:
        panel["base_idx"] = panel["y_labels"].index("base")
    elif "baseline-50" in panel["y_labels"]:
        panel["base_idx"] = panel["y_labels"].index("baseline-50")
    else:
        panel["base_idx"] = -1


# ---------------------------------------------------------------------------
# Helper — x-column reordering and diagonal span computation.
# ---------------------------------------------------------------------------


def _reorder_x_cols(panel: dict, new_order: list[str]) -> None:
    """Permute x columns in-place to match new_order.

    Remaps matrix columns, corr_x_keys, baseline_alt, baseline_neutral,
    and diag xi values.  Labels absent from new_order are appended in their
    original relative order.
    """
    old_labels = panel["x_labels"]
    old_idx = {lbl: i for i, lbl in enumerate(old_labels)}
    seen: set[int] = set()
    perm: list[int] = []
    for lbl in new_order:
        j = old_idx.get(lbl)
        if j is not None and j not in seen:
            perm.append(j)
            seen.add(j)
    perm += [j for j in range(len(old_labels)) if j not in seen]

    new_pos = {old: new for new, old in enumerate(perm)}
    panel["x_labels"] = [old_labels[j] for j in perm]
    panel["matrix"] = [[row[j] for j in perm] for row in panel["matrix"]]
    if panel.get("corr_x_keys"):
        panel["corr_x_keys"] = [panel["corr_x_keys"][j] for j in perm]
    if panel.get("baseline_alt"):
        panel["baseline_alt"] = [panel["baseline_alt"][j] for j in perm]
    if panel.get("baseline_neutral"):
        panel["baseline_neutral"] = [panel["baseline_neutral"][j] for j in perm]
    panel["diag"] = {yi: new_pos[xi] for yi, xi in panel["diag"].items() if xi in new_pos}


def _compute_diag_spans(panel: dict) -> None:
    """Populate panel['diag_end']: for each (yi→xi) in diag, find the last
    column index that belongs to the same multi-metric family
    (i.e. x_labels[j].startswith(x_labels[xi] + ':')).
    For single-metric evals, diag_end[yi] == diag[yi].
    """
    diag = panel.get("diag") or {}
    x_labels = panel.get("x_labels") or []
    diag_end: dict[int, int] = {}
    for yi, xi in diag.items():
        base_label = x_labels[xi] if xi < len(x_labels) else None
        if not base_label:
            diag_end[yi] = xi
            continue
        prefix = base_label + core.SECONDARY_PROPENSITY_SEP
        last_xi = xi
        for j, xl in enumerate(x_labels):
            if xl.startswith(prefix):
                last_xi = max(last_xi, j)
        diag_end[yi] = last_xi
    panel["diag_end"] = diag_end


# ---------------------------------------------------------------------------
# Helper — inject base row and unify labels.
# ---------------------------------------------------------------------------


def _prepend_base_to_panel(panel: dict, base_path: Path) -> None:
    """Prepend the 'base' row from basemodel_scores_<m>.json into `panel` in-place.

    Aligns base cells to the panel's existing x_labels so missing evals become
    None.  Sets base_idx=0 so the JS diffs all other rows against this row.
    """
    base_data = json.loads(base_path.read_text())
    base_cells = (base_data.get("cells") or {}).get("base", {})

    base_row = [bc._cell_metrics(base_cells.get(ev)) for ev in panel["x_labels"]]

    panel["y_labels"] = ["base"] + panel["y_labels"]
    panel["matrix"] = [base_row] + panel["matrix"]
    panel["reduce_mask"] = [True] + panel["reduce_mask"]
    panel["corr_y_keys"] = [None] + panel["corr_y_keys"]
    panel["diag"] = {k + 1: v for k, v in panel["diag"].items()}
    panel["base_idx"] = 0


# ---------------------------------------------------------------------------
# Phase 2 — render HTML from JSONs under new_eval_results/scores/.
# ---------------------------------------------------------------------------


def _derive_poles_master_from_finetuned(panels: list[dict]) -> list[dict]:
    """Build a {name, key, in_reduce} list from finetuned panel rows.

    Used when there is no eval-orthogonality panel.
    """
    seen: dict[str, dict] = {}
    for p in panels:
        if p["group"] != "finetuned":
            continue
        for yl, yk in zip(p["y_labels"], p["corr_y_keys"]):
            if yl == "base" or yl in seen:
                continue
            seen[yl] = {"name": yl, "key": yk, "in_reduce": True}
    return sorted(seen.values(), key=lambda d: _unified_sort_key(d["name"]))


def build_panels() -> tuple[list[dict], dict, dict]:
    """Returns (panels, model_short, table_model_short).

    Reads existing score JSON files from SCORES_DIR; does not regenerate them.
    Run new_summary.py first to build or update the JSON files.
    """
    base_jsons = sorted(SCORES_DIR.glob("basemodel_scores_*.json"))
    if not base_jsons:
        raise SystemExit(
            f"no basemodel_scores_*.json found in {SCORES_DIR}; run new_summary.py first"
        )

    discovered = [p.stem[len("basemodel_scores_"):] for p in base_jsons]
    model_short = {m: bc.MODEL_SHORT.get(m, m) for m in discovered}
    table_model_short = {
        m: bc.TABLE_MODEL_SHORT.get(m, model_short[m].split("-")[0])
        for m in discovered
    }

    panels: list[dict] = []

    # 1. Optional eval-orthogonality panel.
    eo_path = SCORES_DIR / "eval-orthogonality_scores.json"
    eo_panel = None
    eo_col_meta: dict = {}
    if eo_path.exists():
        eo_data = json.loads(eo_path.read_text())
        eo_col_meta = eo_data.get("col_meta", {})
        eo_panel = bc.load_panel(
            eo_path,
            transpose=True,
            title="eval-orthogonality",
            group="eval-orthogonality",
            model=None,
        )
        panels.append(eo_panel)

    # 2. Finetuned panels — load existing JSON, prepend base row.
    for m in discovered:
        ft_path = SCORES_DIR / f"finetuned_scores_{m}.json"
        if not ft_path.exists():
            print(f"  [skip] missing {ft_path.name}")
            continue
        panel = bc.load_panel(
            ft_path,
            transpose=False,
            title=f"finetuned: {table_model_short[m]}",
            group="finetuned",
            model=m,
        )
        base_path = SCORES_DIR / f"basemodel_scores_{m}.json"
        if base_path.exists():
            _prepend_base_to_panel(panel, base_path)
        panels.append(panel)

    # 3. Sys-prompted panels — load existing JSON, prepend base row.
    for m in discovered:
        sp_path = SCORES_DIR / f"sysprompts_scores_{m}.json"
        if not sp_path.exists():
            print(f"  [skip] missing {sp_path.name}")
            continue
        panel = bc.load_panel(
            sp_path,
            transpose=False,
            title=f"system-prompted: {table_model_short[m]}",
            group="system-prompted",
            model=m,
        )
        base_path = SCORES_DIR / f"basemodel_scores_{m}.json"
        if base_path.exists():
            _prepend_base_to_panel(panel, base_path)
        panels.append(panel)

    # Align eval-orth x columns to first finetuned panel.
    if eo_panel is not None:
        ft_panel = next((p for p in panels if p["group"] == "finetuned"), None)
        if ft_panel is not None:
            eo_row_meta = json.loads(eo_path.read_text()).get("row_meta", {})
            eo_panel["corr_x_keys"] = bc._eval_orth_x_corr_keys(
                eo_panel["x_labels"], eo_row_meta, ft_panel["x_labels"]
            )
            bc._reorder_eo_x_cols(eo_panel, ft_panel["x_labels"])

    # Apply unified labels to every panel (rename + re-sort rows).
    for panel in panels:
        col_meta = eo_col_meta if panel["group"] == "eval-orthogonality" else None
        _apply_unified_labels(panel, col_meta)
        _recompute_diag_unified(panel)  # fix SP diag (was empty before unification)

    # Compute most-neutral per-column baseline for the EO panel (post unified labels).
    if eo_panel is not None and eo_panel.get("baseline_alt"):
        eo_panel["baseline_neutral"] = _compute_most_neutral_baseline(
            eo_panel, DEF_PM, eo_panel["baseline_alt"]
        )

    # Move ethical-framework eval columns to the end for better visual diagonality.
    for panel in panels:
        ef_last = (
            [lbl for lbl in panel["x_labels"] if not lbl.startswith(_EF_PREFIX)]
            + [lbl for lbl in panel["x_labels"] if lbl.startswith(_EF_PREFIX)]
        )
        _reorder_x_cols(panel, ef_last)

    # Compute diagonal span for multi-metric evals (e.g. agreeableness, honest-humble).
    for panel in panels:
        _compute_diag_spans(panel)

    return panels, model_short, table_model_short


def _patch_html(raw: str) -> str:
    """Apply all new_build_compare-specific patches to the bc.HTML template.

    Replaces the 'median baseline' checkbox with a 3-option EO-baseline select
    and wires up the supporting JS (state field, effectiveBaseRow, rangeForMetric,
    etc.) to understand 'neutral' / 'median' / '50'.
    """
    h = raw

    # 1. Inject DRANGES_NEUTRAL constant after DRANGES_MED.
    h = h.replace(
        "const DRANGES_MED = __DRANGES_MED__;",
        "const DRANGES_MED = __DRANGES_MED__;\nconst DRANGES_NEUTRAL = __DRANGES_NEUTRAL__;",
    )

    # 2. Replace the checkbox with a styled select.
    h = h.replace(
        '  <label class="toggle"><input type="checkbox" id="median"> median baseline</label>',
        (
            '  <label class="toggle">EO baseline'
            ' <select id="eoBaseline"'
            ' style="background:var(--panel-2);color:var(--ink);border:1px solid var(--grid);'
            'border-radius:4px;padding:2px 4px;">'
            '<option value="neutral">most-neutral</option>'
            '<option value="median">median-baseline</option>'
            '<option value="50">50</option>'
            "</select></label>"
        ),
    )

    # 3. State initialiser: median boolean → eoBaseline string.
    h = h.replace(
        "  median: false,",
        "  eoBaseline: 'neutral',",
    )

    # 4. effectiveBaseRow — 3-way logic.
    h = h.replace(
        (
            "function effectiveBaseRow(p) {\n"
            "  if (p.base_idx < 0) return null;\n"
            "  if (state.median && p.group === 'eval-orthogonality' && p.baseline_alt)"
            " return p.baseline_alt;\n"
            "  return p.matrix[p.base_idx];\n"
            "}"
        ),
        (
            "function effectiveBaseRow(p) {\n"
            "  if (p.base_idx < 0) return null;\n"
            "  if (p.group === 'eval-orthogonality') {\n"
            "    if (state.eoBaseline === 'neutral')"
            " return (p.baseline_neutral || p.baseline_alt) || p.matrix[p.base_idx];\n"
            "    if (state.eoBaseline === 'median' && p.baseline_alt) return p.baseline_alt;\n"
            "  }\n"
            "  return p.matrix[p.base_idx];\n"
            "}"
        ),
    )

    # 5. effectiveCell — substitute baseline row at base_idx when not '50'.
    h = h.replace(
        (
            "  if (yi === p.base_idx && state.median && p.group === 'eval-orthogonality'"
            " && p.baseline_alt) {\n"
            "    return p.baseline_alt[xi];\n"
            "  }\n"
            "  return p.matrix[yi][xi];"
        ),
        (
            "  if (yi === p.base_idx && p.group === 'eval-orthogonality'"
            " && state.eoBaseline !== '50') {\n"
            "    const _altRow = state.eoBaseline === 'neutral'"
            " ? (p.baseline_neutral || p.baseline_alt) : p.baseline_alt;\n"
            "    if (_altRow) return _altRow[xi];\n"
            "  }\n"
            "  return p.matrix[yi][xi];"
        ),
    )

    # 6. updateNormRanges baseRow selection.
    h = h.replace(
        (
            "      ? (state.median && p.group === 'eval-orthogonality' && p.baseline_alt"
            " ? p.baseline_alt : p.matrix[p.base_idx])"
        ),
        (
            "      ? (p.group === 'eval-orthogonality' && state.eoBaseline !== '50'\n"
            "          ? ((state.eoBaseline === 'neutral'"
            " ? (p.baseline_neutral || p.baseline_alt) : p.baseline_alt)"
            " || p.matrix[p.base_idx])\n"
            "          : p.matrix[p.base_idx])"
        ),
    )

    # 7. rangeForMetric — 4-space indent (inside function body).
    h = h.replace(
        "    const R = state.median ? DRANGES_MED : DRANGES;\n"
        "    let a;\n"
        "    if (metric === 'mean') a = R.m;",
        "    const R = state.eoBaseline === '50' ? DRANGES"
        " : (state.eoBaseline === 'neutral' ? DRANGES_NEUTRAL : DRANGES_MED);\n"
        "    let a;\n"
        "    if (metric === 'mean') a = R.m;",
    )

    # 8. render() ft-sp-diff range — 6-space indent.
    h = h.replace(
        "      const R = state.median ? DRANGES_MED : DRANGES;\n"
        "      let a;\n"
        "      if (state.metric === 'mean') a = R.m;",
        "      const R = state.eoBaseline === '50' ? DRANGES"
        " : (state.eoBaseline === 'neutral' ? DRANGES_NEUTRAL : DRANGES_MED);\n"
        "      let a;\n"
        "      if (state.metric === 'mean') a = R.m;",
    )

    # 9. getDisplayedValues — substitute alt baseline at base_idx.
    h = h.replace(
        (
            "  if (yi === p.base_idx && state.median && p.group === 'eval-orthogonality'"
            " && p.baseline_alt) {\n"
            "    cell = p.baseline_alt[xi];\n"
            "  } else {"
        ),
        (
            "  if (yi === p.base_idx && p.group === 'eval-orthogonality'"
            " && state.eoBaseline !== '50') {\n"
            "    const _gr = state.eoBaseline === 'neutral'"
            " ? (p.baseline_neutral || p.baseline_alt) : p.baseline_alt;\n"
            "    cell = _gr ? _gr[xi] : p.matrix[yi][xi];\n"
            "  } else {"
        ),
    )

    # 10. renderShowTable eopBaseRow — use effectiveBaseRow for consistency.
    h = h.replace(
        (
            "  const eopBaseRow = (state.median && eop.baseline_alt) ? eop.baseline_alt\n"
            "    : (eop.base_idx >= 0 ? eop.matrix[eop.base_idx] : null);"
        ),
        "  const eopBaseRow = eop.base_idx >= 0 ? effectiveBaseRow(eop) : null;",
    )

    # 11. Event listener — change to select 'change' on eoBaseline.
    h = h.replace(
        (
            "document.getElementById('median').addEventListener('change', (e) => {\n"
            "  state.median = e.target.checked;\n"
            "  updateNormRanges();\n"
            "  updateLegend();\n"
            "  for (const pv of panelViews) pv.render();\n"
            "  renderShowTable();\n"
            "  if (state.symMode) renderSymTables();\n"
            "});"
        ),
        (
            "document.getElementById('eoBaseline').addEventListener('change', (e) => {\n"
            "  state.eoBaseline = e.target.value;\n"
            "  updateNormRanges();\n"
            "  updateLegend();\n"
            "  for (const pv of panelViews) pv.render();\n"
            "  renderShowTable();\n"
            "  if (state.symMode) renderSymTables();\n"
            "});"
        ),
    )

    # 12. Diagonal box: move isDiag out of the !grey block so it also draws on grey
    #     diagonal cells.  Remove the declaration + draw from inside the colour block,
    #     then re-add as a standalone block before the pink-box block.
    h = h.replace(
        (
            "            const isDiag = this.p.diag && this.p.diag[String(dataYi)] === dataXi;\n"
            "            if (state.metric === 'mean') {"
        ),
        "            if (state.metric === 'mean') {",
    )
    h = h.replace(
        (
            "            if (isDiag) {\n"
            "              ctx.lineWidth = 1.5;\n"
            "              ctx.strokeStyle = '#000';\n"
            "              ctx.strokeRect(px + 0.75, py + 0.75, pw - 1.5, ph - 1.5);\n"
            "            }\n"
            "            // Mismatched-eval column: bright yellow stripe across every cell."
        ),
        "            // Mismatched-eval column: bright yellow stripe across every cell.",
    )
    h = h.replace(
        (
            "        {\n"
            "          const yk = this.p.corr_y_keys[dataYi];\n"
            "          const xk = this.p.corr_x_keys[dataXi];\n"
            "          const hk = (yk && xk) ? yk + '|' + xk : null;"
        ),
        (
            "        {\n"
            "          const _dxf = this.p.diag ? this.p.diag[String(dataYi)] : undefined;\n"
            "          const _dxl = (_dxf !== undefined && this.p.diag_end"
            " && this.p.diag_end[String(dataYi)] !== undefined)"
            " ? this.p.diag_end[String(dataYi)] : _dxf;\n"
            "          const inDiag = _dxf !== undefined && dataXi >= _dxf && dataXi <= _dxl;\n"
            "          if (inDiag) {\n"
            "            const _lw = 1.5, _h2 = _lw / 2;\n"
            "            const _isF = dataXi === _dxf, _isL = dataXi === _dxl;\n"
            "            ctx.lineWidth = _lw; ctx.strokeStyle = '#000'; ctx.beginPath();\n"
            "            ctx.moveTo(px + (_isF ? _h2 : 0), py + _h2);"
            " ctx.lineTo(px + pw - (_isL ? _h2 : 0), py + _h2);\n"
            "            ctx.moveTo(px + (_isF ? _h2 : 0), py + ph - _h2);"
            " ctx.lineTo(px + pw - (_isL ? _h2 : 0), py + ph - _h2);\n"
            "            if (_isF) { ctx.moveTo(px + _h2, py + _h2);"
            " ctx.lineTo(px + _h2, py + ph - _h2); }\n"
            "            if (_isL) { ctx.moveTo(px + pw - _h2, py + _h2);"
            " ctx.lineTo(px + pw - _h2, py + ph - _h2); }\n"
            "            ctx.stroke();\n"
            "          }\n"
            "        }\n"
            "        {\n"
            "          const yk = this.p.corr_y_keys[dataYi];\n"
            "          const xk = this.p.corr_x_keys[dataXi];\n"
            "          const hk = (yk && xk) ? yk + '|' + xk : null;"
        ),
    )

    # 13. Pink highlight for hovered cell, including the base row.
    #     Regular cells already get a pink box because hk === hoverKey fires for the
    #     cell under the cursor.  Base row cells have corr_y_key = null so hk = null
    #     and they never fire.  Fix: track the hovered (panel, yi, xi) in state and
    #     draw a pink box on that exact cell unconditionally.

    # 13a-pre. Base row cross-panel highlighting: give base row cells a synthetic
    #   hk '__base__|<xk>' so the existing hk===hoverKey mechanism highlights
    #   the base row cell in the same column across all panels.
    #   Update hk computation in render loop:
    h = h.replace(
        "          const hk = (yk && xk) ? yk + '|' + xk : null;",
        (
            "          const hk = (yk && xk) ? yk + '|' + xk"
            " : (!yk && dataYi === this.p.base_idx && xk ? '__base__|' + xk : null);"
        ),
    )
    #   Update onMove hoverKey assignment:
    h = h.replace(
        "    setHoverKey((yk && xk) ? `${yk}|${xk}` : null);",
        (
            "    setHoverKey((yk && xk) ? `${yk}|${xk}`"
            " : (!yk && dataYi === this.p.base_idx && xk) ? `__base__|${xk}` : null);"
        ),
    )

    # 13a. Add hoverCell to state initialiser (after hoverKey).
    h = h.replace(
        "  hoverKey: null,\n  visibleCols:",
        "  hoverKey: null,\n  hoverCell: null,\n  visibleCols:",
    )

    # 13b. mouseleave: clear hoverCell before setHoverKey so re-render sees null.
    h = h.replace(
        "this.canvas.addEventListener('mouseleave', () => {"
        " tt.style.display = 'none'; setHoverKey(null); setSymHover(null); });",
        "this.canvas.addEventListener('mouseleave', () => {"
        " tt.style.display = 'none'; state.hoverCell = null; setHoverKey(null); setSymHover(null); });",
    )

    # 13c. onMove out-of-bounds early-return: clear hoverCell.
    h = h.replace(
        (
            "    if (xi < 0 || yi < 0 || xi >= this.nxEff() || yi >= this.nyEff()) {\n"
            "      tt.style.display = 'none';\n"
            "      setHoverKey(null);\n"
            "      return;\n"
            "    }"
        ),
        (
            "    if (xi < 0 || yi < 0 || xi >= this.nxEff() || yi >= this.nyEff()) {\n"
            "      tt.style.display = 'none';\n"
            "      state.hoverCell = null;\n"
            "      setHoverKey(null);\n"
            "      return;\n"
            "    }"
        ),
    )

    # 13d. onMove normal path: record hovered cell after activeRows/activeCols lookup.
    h = h.replace(
        (
            "    const dataYi = this.activeRows[yi];\n"
            "    const dataXi = this.activeCols[xi];\n"
            "    // update cross-matrix highlight"
        ),
        (
            "    const dataYi = this.activeRows[yi];\n"
            "    const dataXi = this.activeCols[xi];\n"
            "    state.hoverCell = {pi: this.panelIndex, yi: dataYi, xi: dataXi};\n"
            "    // update cross-matrix highlight"
        ),
    )

    # 13e. Render loop: draw pink box on hoverCell even when box is false (base row).
    h = h.replace(
        (
            "          if (box) {\n"
            "            ctx.lineWidth = 2;\n"
            "            ctx.strokeStyle = '#ff5cbf';\n"
            "            ctx.strokeRect(px + 1, py + 1, pw - 2, ph - 2);\n"
            "          }\n"
            "        }\n"
            "      }\n"
            "    }\n"
            "    ctx.restore();\n"
            "    this.renderAxes();"
        ),
        (
            "          if (box) {\n"
            "            ctx.lineWidth = 2;\n"
            "            ctx.strokeStyle = '#ff5cbf';\n"
            "            ctx.strokeRect(px + 1, py + 1, pw - 2, ph - 2);\n"
            "          } else if (state.hoverCell && state.hoverCell.pi === this.panelIndex\n"
            "              && dataYi === state.hoverCell.yi && dataXi === state.hoverCell.xi) {\n"
            "            ctx.lineWidth = 2;\n"
            "            ctx.strokeStyle = '#ff5cbf';\n"
            "            ctx.strokeRect(px + 1, py + 1, pw - 2, ph - 2);\n"
            "          }\n"
            "        }\n"
            "      }\n"
            "    }\n"
            "    ctx.restore();\n"
            "    this.renderAxes();"
        ),
    )

    # 14. Tooltip: show WHY a cell is grey.
    #   14a. After the n_numeric line, add a grey-reason note.
    h = h.replace(
        (
            "      body += `<div style=\"color: var(--ink-dim)\">n_numeric=${cell.n ?? '—'}"
            " · n_total=${cell.nt ?? '—'} · n_nulls=${cell.nn ?? '—'}"
            " · n_fails=${cell.nf ?? '—'}</div>`;\n"
            "      if (cell.e) {"
        ),
        (
            "      body += `<div style=\"color: var(--ink-dim)\">n_numeric=${cell.n ?? '—'}"
            " · n_total=${cell.nt ?? '—'} · n_nulls=${cell.nn ?? '—'}"
            " · n_fails=${cell.nf ?? '—'}</div>`;\n"
            "      if (cell.m === null || cell.m === undefined) {\n"
            "        const _nt = cell.nt ?? 0, _nf = cell.nf ?? 0,"
            " _nn = cell.nn ?? 0, _n = cell.n ?? 0;\n"
            "        let _why;\n"
            "        if (_nt === 0) _why = 'not run (0 items recorded)';\n"
            "        else if (_n === 0 && _nf > 0 && _nn === 0) _why = `all ${_nt} items failed`;\n"
            "        else if (_n === 0 && _nn > 0 && _nf === 0) _why = `all ${_nt} items null`;\n"
            "        else if (_n === 0) _why = `${_nf} fail(s) + ${_nn} null(s) of ${_nt} total`;\n"
            "        else _why = 'no mean despite numeric scores';\n"
            "        body += `<div style=\"color:#f87\">⚠ grey: ${_why}</div>`;\n"
            "      } else {\n"
            "        const _thr = state.threshold > 0 && cell.nt > 0"
            " && cell.n != null && (cell.n / cell.nt) <= state.threshold;\n"
            "        if (_thr) body += `<div style=\"color:#f87\">⚠ grey: n_numeric/n_total ="
            " ${(cell.n / cell.nt).toFixed(2)} ≤ threshold ${state.threshold}</div>`;\n"
            "      }\n"
            "      if (cell.e) {"
        ),
    )
    #   14b. Replace the bare '—' for missing cells with a diagnostic message.
    h = h.replace(
        "    } else {\n      body += `<div>—</div>`;\n    }",
        (
            "    } else {\n"
            "      body += `<div style=\"color:#f87\">"
            "⚠ grey: no data (eval not run or directory missing)</div>`;\n"
            "    }"
        ),
    )

    return h


def main() -> None:
    panels, model_short, table_model_short = build_panels()
    if not any(p["group"] == "finetuned" for p in panels):
        raise SystemExit("no finetuned panels built; nothing to render.")

    ranges = bc.global_ranges(panels)
    dranges_50 = bc.diff_range(panels, use_median=False)
    dranges_med = bc.diff_range(panels, use_median=True)

    # dranges_neutral: diff ranges using the most-neutral baseline for EO.
    eo = next((p for p in panels if p["group"] == "eval-orthogonality"), None)
    if eo is not None and eo.get("baseline_neutral"):
        _orig_alt = eo.get("baseline_alt")
        eo["baseline_alt"] = eo["baseline_neutral"]
        dranges_neutral = bc.diff_range(panels, use_median=True)
        eo["baseline_alt"] = _orig_alt
    else:
        dranges_neutral = dranges_med

    evals_master = next(p["x_labels"] for p in panels if p["group"] == "finetuned")

    if eo is not None:
        poles_master = [
            {"name": label, "key": key, "in_reduce": rm}
            for label, key, rm in zip(
                eo["y_labels"], eo["corr_y_keys"], eo["reduce_mask"]
            )
            if label not in ("baseline-50", "base")
        ]
    else:
        poles_master = _derive_poles_master_from_finetuned(panels)

    html = (
        _patch_html(bc.HTML)
        .replace("__PANELS__", json.dumps(panels))
        .replace("__RANGES__", json.dumps(ranges))
        .replace("__DRANGES__", json.dumps(dranges_50))
        .replace("__DRANGES_MED__", json.dumps(dranges_med))
        .replace("__DRANGES_NEUTRAL__", json.dumps(dranges_neutral))
        .replace("__EVALS__", json.dumps(evals_master))
        .replace("__POLES__", json.dumps(poles_master))
        .replace("__MODEL_SHORT__", json.dumps(model_short))
        .replace("__TABLE_MODEL_SHORT__", json.dumps(table_model_short))
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
