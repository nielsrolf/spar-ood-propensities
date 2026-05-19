#!/usr/bin/env python3
"""new_summary.py — regenerate score JSONs and write raw-numbers pickles.

Phase 1  Write the 10 JSON files under new_eval_results/scores/:
  basemodel_scores_{model}.json    (from new_eval_results/base_models/)
  finetuned_scores_{model}.json    (from new_eval_results/finetuning/)
  sysprompts_scores_{model}.json   (from new_eval_results/sys_prompts/)
  eval-orthogonality_scores.json   (from new_eval_results/scores/eval-orthogonality/)

Phase 2  Write 10 pickle files under new_eval_results/raw_numbers/:
  One .pkl per JSON.  Each is a dict[cell_name][eval_name] -> {item_id: score}.

  For base/finetuned/sysprompts: cell_name is the pole identifier
    (e.g. "base", "agreeableness-plus", "agreeableness__agreeable")
    and eval_name is the propensity (e.g. "claiming-superintelligence").

  For eval-orthogonality: the JSON is transposed relative to the other files;
    we un-transpose so the shape is the same —
    cell_name = col_label  (pole, e.g. "agreeableness.agreeable")
    eval_name = row_label  (judge×propensity, e.g. "trust_score (agreeableness)")
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import build_compare as bc             # noqa: E402
import eval_matrix_core as core       # noqa: E402
import new_build_compare as nbc       # noqa: E402  (import for its module-level patches)

# ── Paths ─────────────────────────────────────────────────────────────────────

NEW_EVAL_RESULTS = nbc.NEW_EVAL_RESULTS
SCORES_DIR       = nbc.SCORES_DIR
RAW_NUMBERS_DIR  = NEW_EVAL_RESULTS / "raw_numbers"
BASE_ROOT        = nbc.BASE_ROOT
FT_ROOT          = nbc.FT_ROOT
SP_ROOT          = nbc.SP_ROOT
EO_SOURCE_DIR    = SCORES_DIR / "eval-orthogonality"
EO_JSON_PATH     = SCORES_DIR / "eval-orthogonality_scores.json"

# ── Model JSON build config ───────────────────────────────────────────────────

_MIN_ITEMS: int | None = 30
_PREFER: str = "latest-eval"
_FILTERS = core.FilterConfig(min_items=_MIN_ITEMS, prefer=_PREFER)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1a — eval-orthogonality JSON
# ─────────────────────────────────────────────────────────────────────────────

def _build_eo_doc() -> dict:
    """Reconstruct eval-orthogonality_scores.json from scores/eval-orthogonality/ subdirs.

    Mirrors the logic in orthogonality_of_evals_directory.py:
    - score_status == "score" indicates a numeric result
    - std is population std (divide by N, not N-1)
    - row_label is read from matrices.json (not re-derived from judgment fields),
      because single-judge propensities use just the judge_key as the label
      while multi-judge propensities append " (propensity_name)".
    - col_label = judgment["col_label"]
    """
    all_row_labels: list[str] = []
    all_col_labels: list[str] = []
    row_meta: dict[str, dict] = {}
    col_meta: dict[str, dict] = {}
    judge_models_per_row: dict[str, str] = {}
    # reverse map: (prop_a, judge_key) -> row_label  (built from matrices.json)
    pa_jk_to_row: dict[tuple[str, str], str] = {}

    # (row_label, col_label) -> list of numeric scores
    raw: dict[tuple[str, str], list[float]] = defaultdict(list)
    # (row_label, col_label) -> {item_id: score | None}
    per_item: dict[tuple[str, str], dict[str, float | None]] = defaultdict(dict)
    # (row_label, col_label) -> {"null": n, "failed": n}
    bad: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"null": 0, "failed": 0})

    n_records = 0
    n_failed = 0

    for cell_dir in sorted(EO_SOURCE_DIR.iterdir()):
        if not cell_dir.is_dir() or cell_dir.name.startswith("."):
            continue
        parts = cell_dir.name.split("__")
        if len(parts) != 2:
            continue

        # Harvest row/col metadata from matrices.json and build reverse lookup.
        matrices_path = cell_dir / "matrices.json"
        if matrices_path.exists():
            mats = json.loads(matrices_path.read_text())
            meta = mats.get("_meta", {})
            for rm_key, rm_val in meta.get("row_meta", {}).items():
                if rm_key not in row_meta:
                    row_meta[rm_key] = rm_val
                    all_row_labels.append(rm_key)
                pa_jk_to_row[(rm_val["propensity"], rm_val["judge_key"])] = rm_key
            for cm_key, cm_val in meta.get("col_meta", {}).items():
                if cm_key not in col_meta:
                    col_meta[cm_key] = cm_val
                    all_col_labels.append(cm_key)
            for jm_key, jm_val in meta.get("judge_models", {}).items():
                if jm_key not in judge_models_per_row:
                    judge_models_per_row[jm_key] = jm_val

        judgments_path = cell_dir / "judgments.jsonl"
        if not judgments_path.exists():
            continue

        with judgments_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    j = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_records += 1
                row_lbl = pa_jk_to_row.get((j["prop_a"], j["judge_key"]))
                if row_lbl is None:
                    continue
                col_lbl = j["col_label"]
                item_id = j["item_id"]
                score   = j.get("score")
                status  = j.get("score_status", "failed")

                key = (row_lbl, col_lbl)
                if status == "score":
                    raw[key].append(float(score))
                    per_item[key][item_id] = score
                elif status == "null":
                    bad[key]["null"] += 1
                    per_item[key][item_id] = None
                else:
                    bad[key]["failed"] += 1
                    n_failed += 1

    # Preserve n_diagonal / n_offdiagonal from existing JSON if present.
    n_diagonal = n_offdiagonal = 0
    if EO_JSON_PATH.exists():
        try:
            existing = json.loads(EO_JSON_PATH.read_text())
            n_diagonal   = existing.get("n_diagonal",   0)
            n_offdiagonal = existing.get("n_offdiagonal", 0)
        except Exception:
            pass

    # Build cells dict
    cells: dict[str, dict[str, dict]] = {}
    for rl in all_row_labels:
        cells[rl] = {}
        for cl in all_col_labels:
            key = (rl, cl)
            vs = raw.get(key, [])
            items = per_item.get(key, {})
            b = bad.get(key, {})
            n_num   = len(vs)
            n_nulls = b.get("null", 0)
            n_fails = b.get("failed", 0)
            n_total = n_num + n_nulls + n_fails
            if n_total == 0 and not items:
                cells[rl][cl] = None
                continue
            if vs:
                mean = sum(vs) / n_num
                mn, mx = min(vs), max(vs)
                # population std (matches orthogonality_of_evals_directory.py)
                std = (sum((v - mean) ** 2 for v in vs) / n_num) ** 0.5 if n_num >= 2 else None
            else:
                mean = mn = mx = std = None
            cells[rl][cl] = {
                "metrics": {
                    "mean":      mean,
                    "std":       std,
                    "min":       mn,
                    "max":       mx,
                    "n_total":   n_total,
                    "n_numeric": n_num,
                    "n_nulls":   n_nulls,
                    "n_fails":   n_fails,
                },
                "scores": items,
                "meta": {
                    "row_propensity":  row_meta[rl]["propensity"],
                    "row_judge_key":   row_meta[rl]["judge_key"],
                    "col_propensity":  col_meta[cl]["propensity"],
                    "col_response_key": col_meta[cl].get("response_key"),
                },
            }

    return {
        "model":                  "gpt-5.4-mini",
        "judge_models_per_row":   judge_models_per_row,
        "eval_dir":               "cross-elicit/evals",
        "generated_at":           datetime.now().isoformat(timespec="seconds"),
        "deterministic_sampling": True,
        "sample_split":           "test",
        "n_diagonal":             n_diagonal,
        "n_offdiagonal":          n_offdiagonal,
        "row_labels":             all_row_labels,
        "col_labels":             all_col_labels,
        "row_meta":               row_meta,
        "col_meta":               col_meta,
        "n_rows":                 len(all_row_labels),
        "n_cols":                 len(all_col_labels),
        "n_cells":                sum(len(v) for v in cells.values()),
        "n_records":              n_records,
        "n_failed":               n_failed,
        "cells":                  cells,
    }


def write_eo_json() -> Path:
    if not EO_SOURCE_DIR.exists():
        print(f"  [skip] eval-orthogonality source dir not found: {EO_SOURCE_DIR}")
        return EO_JSON_PATH
    doc = _build_eo_doc()
    EO_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    EO_JSON_PATH.write_text(json.dumps(doc, indent=2))
    print(
        f"  [wrote] scores/eval-orthogonality_scores.json "
        f"({doc['n_rows']} rows × {doc['n_cols']} cols, {doc['n_records']} records)"
    )
    return EO_JSON_PATH


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1b — per-model JSONs (base / finetuned / sysprompts)
# ─────────────────────────────────────────────────────────────────────────────

def _collect_records(root: Path, known_models: set[str], model_name: str) -> list[dict]:
    records: list[dict] = []
    if not root.exists():
        return records
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        rec = core.parse_eval_dir(d.name, known_models)
        if rec is None or rec["model"] != model_name:
            continue
        rec["summary"] = core.load_summary(d)
        records.append(rec)
    return core.expand_records_with_secondary_judges(records)


def _collect_sp_records(known_models: set[str], model_name: str) -> list[dict]:
    records: list[dict] = []
    if not SP_ROOT.exists():
        return records
    for pole_dir in sorted(SP_ROOT.iterdir()):
        if not pole_dir.is_dir() or pole_dir.name.startswith("."):
            continue
        if len(pole_dir.name.split("__")) != 2:
            continue
        for d in sorted(pole_dir.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            rec = core.parse_eval_dir(d.name, known_models)
            if rec is None or rec["model"] != model_name:
                continue
            rec["pole"] = pole_dir.name
            rec["summary"] = core.load_summary(d)
            records.append(rec)
    return core.expand_records_with_secondary_judges(records)


def _build_doc(model_name: str, records: list[dict]) -> dict | None:
    by_cell = core.group_by_cell(records)
    chosen, misses = core.disambiguate(by_cell, _FILTERS)
    for cell, target in misses:
        print(f"  [warn] override not matched: {cell} -> {target}")
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
                "mean": m.get("mean"), "std": m.get("std"),
                "min": m.get("min"),   "max": m.get("max"),
                "n_total": m.get("n_total"), "n_numeric": m.get("n_numeric"),
                "n_nulls": m.get("n_nulls"), "n_fails":   m.get("n_fails"),
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

    all_evs = sorted(
        {ev for pole_cells in score_cells.values() for ev in pole_cells},
        key=bc._x_sort_key,
    )
    full = {pole: {ev: score_cells[pole].get(ev) for ev in all_evs} for pole in score_cells}
    return {
        "base_model": model_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "filters": {"min_items": _MIN_ITEMS, "prefer": _PREFER},
        "n_poles": len(full),
        "n_cells": sum(len(v) for v in full.values()),
        "cells": full,
    }


def _write_doc(doc: dict, prefix: str) -> None:
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    p = SCORES_DIR / f"{prefix}{doc['base_model']}.json"
    p.write_text(json.dumps(doc, indent=2))
    print(f"  [wrote] scores/{p.name} ({doc['n_cells']} cells across {doc['n_poles']} poles)")


def write_model_jsons() -> None:
    if not BASE_ROOT.exists():
        print(f"  [skip] base_models dir not found: {BASE_ROOT}")
        return
    known_models = core.discover_models(BASE_ROOT)
    if not known_models:
        print(f"  [skip] no __base__ entries found in {BASE_ROOT}")
        return
    for model in sorted(known_models):
        print(f"[base] {model}")
        doc = _build_doc(model, _collect_records(BASE_ROOT, known_models, model))
        if doc:
            _write_doc(doc, "basemodel_scores_")

        print(f"[ft]   {model}")
        doc = _build_doc(model, _collect_records(FT_ROOT, known_models, model))
        if doc:
            _write_doc(doc, "finetuned_scores_")

        print(f"[sp]   {model}")
        doc = _build_doc(model, _collect_sp_records(known_models, model))
        if doc:
            _write_doc(doc, "sysprompts_scores_")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — raw-numbers pickles
# ─────────────────────────────────────────────────────────────────────────────

def _read_rows(rows_path: Path, metric_key: str) -> dict[str, float | None]:
    """Return {item_id: score} from rows.jsonl filtered by metric_key."""
    result: dict[str, float | None] = {}
    if not rows_path.exists():
        return result
    with rows_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("metric") != metric_key:
                continue
            result[row["item_id"]] = row.get("score")
    return result


def _raw_from_model_json(
    json_path: Path,
    rows_root: Path,
    is_sysprompts: bool,
) -> dict[str, dict[str, dict | None]]:
    """Build pickle dict from a base/finetuned/sysprompts scores JSON.

    Structure: result[cell_name][eval_name] = {item_id: score}
    cell_name is the pole (e.g. "base", "agreeableness-plus", "agreeableness__agreeable")
    eval_name is the propensity (e.g. "agreeableness", "claiming-superintelligence")
    """
    data = json.loads(json_path.read_text())
    result: dict[str, dict[str, dict | None]] = {}
    for cell_name, cell_data in data.get("cells", {}).items():
        result[cell_name] = {}
        for eval_name, eval_entry in cell_data.items():
            if eval_entry is None:
                result[cell_name][eval_name] = None
                continue
            meta = eval_entry.get("meta") or {}
            dirname    = meta.get("dirname")
            metric_key = meta.get("metric_key")
            if not dirname or not metric_key:
                result[cell_name][eval_name] = {}
                continue
            if is_sysprompts:
                rows_path = rows_root / cell_name / dirname / "rows.jsonl"
            else:
                rows_path = rows_root / dirname / "rows.jsonl"
            result[cell_name][eval_name] = _read_rows(rows_path, metric_key)
    return result


def _raw_from_eo_json() -> dict[str, dict[str, dict | None]]:
    """Build pickle dict from eval-orthogonality_scores.json.

    Un-transposes relative to the JSON:
      result[col_label][row_label] = {item_id: score}
    so col_label (pole, e.g. "agreeableness.agreeable") is the outer key
    and row_label (judge×propensity, e.g. "trust_score (agreeableness)") is inner.
    """
    data = json.loads(EO_JSON_PATH.read_text())
    result: dict[str, dict[str, dict | None]] = {}
    for row_label, row_data in data.get("cells", {}).items():
        for col_label, cell_entry in row_data.items():
            if col_label not in result:
                result[col_label] = {}
            if cell_entry is None:
                result[col_label][row_label] = None
            else:
                result[col_label][row_label] = cell_entry.get("scores") or {}
    return result


def write_raw_numbers() -> None:
    RAW_NUMBERS_DIR.mkdir(parents=True, exist_ok=True)

    # eval-orthogonality
    if EO_JSON_PATH.exists():
        raw = _raw_from_eo_json()
        pkl_path = RAW_NUMBERS_DIR / "eval-orthogonality_scores.pkl"
        with pkl_path.open("wb") as fh:
            pickle.dump(raw, fh)
        n_cells = sum(len(v) for v in raw.values())
        print(f"  [wrote] raw_numbers/eval-orthogonality_scores.pkl ({n_cells} cells)")

    # per-model JSONs
    for json_path in sorted(SCORES_DIR.glob("*.json")):
        if json_path.name == "eval-orthogonality_scores.json":
            continue
        stem = json_path.stem
        if stem.startswith("basemodel_"):
            rows_root, is_sp = BASE_ROOT, False
        elif stem.startswith("finetuned_"):
            rows_root, is_sp = FT_ROOT, False
        elif stem.startswith("sysprompts_"):
            rows_root, is_sp = SP_ROOT, True
        else:
            continue

        raw = _raw_from_model_json(json_path, rows_root, is_sp)
        pkl_path = RAW_NUMBERS_DIR / f"{stem}.pkl"
        with pkl_path.open("wb") as fh:
            pickle.dump(raw, fh)
        n_cells = sum(len(v) for v in raw.values())
        print(f"  [wrote] raw_numbers/{stem}.pkl ({n_cells} cells)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Phase 1: Writing JSON files ===")
    write_eo_json()
    write_model_jsons()
    print("\n=== Phase 2: Writing pickle files ===")
    write_raw_numbers()
    print("\nDone.")


if __name__ == "__main__":
    main()
