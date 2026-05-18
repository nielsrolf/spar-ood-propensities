"""new_build_compare.py — self-contained render from new_eval_results/.

Two phases, both confined to `new_eval_results/`:

  1. SUMMARIZE  Walk `new_eval_results/{base_models,finetuning}`, pick one run
                per (pole, eval) cell with the same disambiguation rules
                summarize_FT.py uses, write
                `new_eval_results/scores/finetuned_scores_<model>.json` per
                discovered base model.
  2. RENDER     Load those JSONs, plus any *optional* sibling files
                (`new_eval_results/scores/sysprompts_scores_<model>.json`,
                `new_eval_results/scores/eval-orthogonality_scores.json`) if
                they are present, and write `new_eval_results/new_compare.html`.

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
  * The POLES master is read from the eval-orth panel when present; otherwise
    derived from the finetuned panels' rows so the pole-filter UI still works.
    DEFAULT_POLES in the embedded JS uses eval-orth naming (e.g.
    `cooperation.high`) — when there's no eval-orth panel the "default" pole
    button won't preselect anything; the other buttons still do.
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
SCORES_DIR = NEW_EVAL_RESULTS / "scores"
OUT = NEW_EVAL_RESULTS / "new_compare.html"

DEFINITIONS_PATH = ROOT / "evals" / "definitions.json"

MIN_ITEMS: int | None = 30
PREFER: str = "latest-eval"
FILTERS = core.FilterConfig(min_items=MIN_ITEMS, prefer=PREFER)

# ---------------------------------------------------------------------------
# Patch eval_matrix_core for the new evals.
# ---------------------------------------------------------------------------
# Multi-judge agreeableness: pick a primary, surface the others as secondary
# rows (same scheme honest-humble uses).
_AGREEABLENESS_PRIMARY = "trust_score"
_AGREEABLENESS_SECONDARIES = [
    "altruism_score",
    "cooperation_score",
    "modesty_score",
    "sympathy_score",
]
core.PROPENSITY_TO_PRIMARY_METRIC["agreeableness"] = _AGREEABLENESS_PRIMARY
core.SECONDARY_JUDGE_ROWS["agreeableness"] = list(_AGREEABLENESS_SECONDARIES)

# ---------------------------------------------------------------------------
# Swap build_compare's propensity dictionary for evals/definitions.json.
# Same shape (axis / plus_pole_key / minus_pole_key), so the helpers in
# build_compare keep working — they look up DEF_SYS / DEF_SYS_AXES at call
# time, so reassigning the module attributes is enough.
# ---------------------------------------------------------------------------
bc.DEF_SYS = json.loads(DEFINITIONS_PATH.read_text())
bc.DEF_SYS_AXES = list(bc.DEF_SYS.keys())


# ---------------------------------------------------------------------------
# Phase 1 — summarize new_eval_results/ into per-model JSONs.
# ---------------------------------------------------------------------------


def _collect_records_for_model(model_name: str, known_models: set[str]) -> list[dict]:
    """Walk BASE_ROOT and FT_ROOT, return records for `model_name`.

    `core.collect_records` only walks one dir and re-discovers models from
    `__base__` entries inside it; FT_ROOT has no base entries. Walk both
    manually with a shared `known_models` set so finetuning records resolve.
    """
    records: list[dict] = []
    for root in (BASE_ROOT, FT_ROOT):
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            rec = core.parse_eval_dir(d.name, known_models)
            if rec is None:
                continue
            if rec["model"] != model_name:
                continue
            rec["summary"] = core.load_summary(d)
            records.append(rec)
    return core.expand_records_with_secondary_judges(records)


def build_scores_doc(model_name: str, known_models: set[str]) -> dict | None:
    """Produce the {cells: {pole: {eval: cell}}} doc for `model_name`.

    Returns None when no records pass the filters for this model.
    """
    records = _collect_records_for_model(model_name, known_models)
    if not records:
        return None
    by_cell = core.group_by_cell(records)
    chosen, override_misses = core.disambiguate(by_cell, FILTERS)
    for cell, target in override_misses:
        print(f"  [warn] OVERRIDES not matched: {cell} -> {target}")
    if not chosen:
        return None

    score_cells: dict[str, dict[str, dict]] = defaultdict(dict)
    for (pole, ev), rec in chosen.items():
        summary = rec.get("summary") or {}
        metrics = summary.get("metrics") or {}
        key = core.resolve_metric_key(ev, metrics)
        m = metrics.get(key, {}) if key else {}
        score_cells[pole][ev] = {
            "metrics": {
                "mean": m.get("mean"),
                "std": m.get("std"),
                "min": m.get("min"),
                "max": m.get("max"),
                "n_total": m.get("n_total"),
                "n_numeric": m.get("n_numeric"),
                "n_nulls": m.get("n_nulls"),
                "n_fails": m.get("n_fails"),
            },
            "meta": {
                "dirname": rec["dirname"],
                "judge_model": summary.get("judge_model"),
                "metric_key": key,
                "epoch": rec.get("epoch"),
                "ft_timestamp": rec.get("ft_timestamp"),
                "eval_timestamp": rec.get("eval_timestamp"),
                "n_test_items": summary.get("n_test_items"),
                "samples_per_paraphrase": summary.get("samples_per_paraphrase"),
                "system_prompt_source": summary.get("system_prompt_source"),
            },
        }

    # load_panel takes col_labels from the first row only; backfill missing
    # cells with None so every row carries every eval.
    all_evs = sorted(
        {ev for pole_cells in score_cells.values() for ev in pole_cells},
        key=bc._x_sort_key,
    )
    full = {
        pole: {ev: score_cells[pole].get(ev) for ev in all_evs}
        for pole in score_cells
    }
    return {
        "base_model": model_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "filters": {"min_items": MIN_ITEMS, "prefer": PREFER},
        "n_poles": len(full),
        "n_cells": sum(len(v) for v in full.values()),
        "cells": full,
    }


def write_scores_doc(doc: dict) -> Path:
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    p = SCORES_DIR / f"finetuned_scores_{doc['base_model']}.json"
    p.write_text(json.dumps(doc, indent=2))
    print(
        f"  [wrote] {p.relative_to(NEW_EVAL_RESULTS)} "
        f"({doc['n_cells']} cells across {doc['n_poles']} poles)"
    )
    return p


# ---------------------------------------------------------------------------
# Phase 2 — render HTML from JSONs under new_eval_results/scores/.
# ---------------------------------------------------------------------------


def _derive_poles_master_from_finetuned(panels: list[dict]) -> list[dict]:
    """Build a {name, key, in_reduce} list from finetuned panel rows.

    Used when there is no eval-orthogonality panel. `key` is the same
    `axis|side` correspondence key the cross-panel pole filter expects, so
    selection by key keeps working. `in_reduce` is always True here (the
    reduce toggle is only meaningful for eval-orth).
    """
    seen: dict[str, dict] = {}
    for p in panels:
        if p["group"] != "finetuned":
            continue
        for yl, yk in zip(p["y_labels"], p["corr_y_keys"]):
            if yl == "base" or yl in seen:
                continue
            seen[yl] = {"name": yl, "key": yk, "in_reduce": True}
    return sorted(seen.values(), key=lambda d: bc._ft_sort_key(d["name"]))


def build_panels() -> tuple[list[dict], dict, dict]:
    """Returns (panels, model_short, table_model_short).

    Side-effect: writes/updates SCORES_DIR/finetuned_scores_<m>.json per model.
    """
    if not BASE_ROOT.exists():
        raise SystemExit(f"missing base_models dir: {BASE_ROOT}")
    known_models = core.discover_models(BASE_ROOT)
    if not known_models:
        raise SystemExit(f"no `__base__` entries found in {BASE_ROOT}")

    discovered = sorted(known_models)
    model_short = {m: bc.MODEL_SHORT.get(m, m) for m in discovered}
    table_model_short = {
        m: bc.TABLE_MODEL_SHORT.get(m, model_short[m].split("-")[0])
        for m in discovered
    }

    panels: list[dict] = []

    # 1. Optional eval-orthogonality panel — only if a JSON already lives in
    #    new_eval_results/scores/. We do not generate one here.
    eo_path = SCORES_DIR / "eval-orthogonality_scores.json"
    eo_panel = None
    if eo_path.exists():
        eo_panel = bc.load_panel(
            eo_path,
            transpose=True,
            title="eval-orthogonality",
            group="eval-orthogonality",
            model=None,
        )
        panels.append(eo_panel)

    # 2. Finetuned panels — summarize each model, write JSON, then load.
    for m in discovered:
        print(f"[ft] {m}")
        doc = build_scores_doc(m, known_models)
        if doc is None:
            print(f"  [skip] no records")
            continue
        path = write_scores_doc(doc)
        panels.append(
            bc.load_panel(
                path,
                transpose=False,
                title=f"finetuned: {table_model_short[m]}",
                group="finetuned",
                model=m,
            )
        )

    # 3. Optional sysprompts panels — only if JSONs already live in
    #    new_eval_results/scores/.
    for m in discovered:
        sp_path = SCORES_DIR / f"sysprompts_scores_{m}.json"
        if not sp_path.exists():
            continue
        panels.append(
            bc.load_panel(
                sp_path,
                transpose=False,
                title=f"system-prompted: {table_model_short[m]}",
                group="system-prompted",
                model=m,
            )
        )

    # Align eval-orth x columns to the first finetuned panel (same post-process
    # build_compare does).
    if eo_panel is not None:
        ft_panel = next((p for p in panels if p["group"] == "finetuned"), None)
        if ft_panel is not None:
            eo_row_meta = json.loads(eo_path.read_text()).get("row_meta", {})
            eo_panel["corr_x_keys"] = bc._eval_orth_x_corr_keys(
                eo_panel["x_labels"], eo_row_meta, ft_panel["x_labels"]
            )
            bc._reorder_eo_x_cols(eo_panel, ft_panel["x_labels"])

    return panels, model_short, table_model_short


def main() -> None:
    panels, model_short, table_model_short = build_panels()
    if not any(p["group"] == "finetuned" for p in panels):
        raise SystemExit("no finetuned panels built; nothing to render.")

    ranges = bc.global_ranges(panels)
    dranges_50 = bc.diff_range(panels, use_median=False)
    dranges_med = bc.diff_range(panels, use_median=True)

    evals_master = next(p["x_labels"] for p in panels if p["group"] == "finetuned")

    eo = next((p for p in panels if p["group"] == "eval-orthogonality"), None)
    if eo is not None:
        poles_master = [
            {"name": label, "key": key, "in_reduce": rm}
            for label, key, rm in zip(
                eo["y_labels"], eo["corr_y_keys"], eo["reduce_mask"]
            )
            if label != "baseline-50"
        ]
    else:
        poles_master = _derive_poles_master_from_finetuned(panels)

    html = (
        bc.HTML
        .replace("__PANELS__", json.dumps(panels))
        .replace("__RANGES__", json.dumps(ranges))
        .replace("__DRANGES__", json.dumps(dranges_50))
        .replace("__DRANGES_MED__", json.dumps(dranges_med))
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
