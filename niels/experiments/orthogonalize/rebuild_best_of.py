"""
Post-process Stage 3 output to keep the best available version of every
removed question instead of dropping questions that never converged.

For each question removed in Stage 2, we already have per-iteration judge
scores cached in `<output>/eval-orthogonalized/<eval>/revision_log.jsonl`.
Iteration 0 is the original question+references; iterations 1..N are
writer revisions. Each log row has a `badness` field (sum of cross-metric
gap excesses). This script picks the iteration with the lowest `badness`
for every removed question and writes it into the final eval.

When the writer failed (e.g. OpenRouter 402 credits), only iteration 0
will be available — in that case we keep the original question, which
keeps the eval from shrinking at the cost of still carrying whatever
cross-metric signal the original had.

Usage:
    ./.venv/bin/python experiments/orthogonalize/rebuild_best_of.py \
        --input experiments/orthogonalize/evals \
        --output-dir experiments/orthogonalize/output_main
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from eval_utils import (
    find_yaml,
    judge_prompts_of,
    load_eval_yaml,
    primary_expected_key,
    write_yaml_with_anchors,
)
from stage1 import build_intrinsic_matrix, build_propensity_matrix
from stage2 import _primary_metric_for_eval


def _load_logs(log_path: Path) -> Dict[str, List[dict]]:
    by_qid: Dict[str, List[dict]] = {}
    if not log_path.exists():
        return by_qid
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        qid = d.get("qid")
        if qid is None:
            continue
        by_qid.setdefault(qid, []).append(d)
    return by_qid


def _pick_best_log(logs: List[dict]) -> Optional[dict]:
    """Return the log entry with lowest `badness` that has score_rows.

    Ties are broken by the HIGHER iteration (prefer a revision over the
    original when equally good). If no log has a badness, returns None.
    """
    candidates = [l for l in logs if "badness" in l and "score_rows" in l]
    if not candidates:
        return None
    # Sort: lowest badness first; on tie, highest iteration first.
    candidates.sort(key=lambda l: (l["badness"], -l.get("iteration", 0)))
    return candidates[0]


def _entry_from_log(log: dict, original_entry: dict, iteration: int) -> dict:
    """Materialize a YAML entry from a revision_log row.

    Iteration 0 of the log mirrors the original entry but carries only
    the question/refs shown at scoring time. We overwrite paraphrases +
    expected_* meta fields with what was actually scored, and keep
    everything else from the original entry.
    """
    new_entry = dict(original_entry)
    question = log.get("question")
    if question:
        new_entry["paraphrases"] = [question]
    meta = dict(original_entry.get("meta", {}) or {})
    for key, value in (log.get("reference_answers") or {}).items():
        meta[key] = value
    new_entry["meta"] = meta
    suffix = "_r0" if iteration == 0 else f"_r{iteration}"
    # Keep a _r<iter> suffix so stage3 downstream code that tracks
    # per-revision ids still works. iter=0 → _r0 (original kept as-is).
    base = original_entry["id"]
    new_entry["id"] = base if iteration == 0 else f"{base}{suffix}"
    return new_entry


def _best_revision_rows(
    log_path: Path,
    entries: list[dict],
    orig_id_by_new_id: dict,
) -> list[dict]:
    """Return per-cell score rows for the chosen revisions, keyed by final qid."""
    by_qid = _load_logs(log_path)
    rows = []
    for entry in entries:
        new_qid = entry["id"]
        orig_qid = orig_id_by_new_id[new_qid]
        logs = by_qid.get(orig_qid, [])
        best = _pick_best_log(logs)
        if best is None:
            continue
        for row in best["score_rows"]:
            new_row = dict(row)
            new_row["question_id"] = new_qid
            rows.append(new_row)
    return rows


def rebuild(input_dir: Path, output_dir: Path, eval_names: Optional[List[str]] = None) -> None:
    stage1 = output_dir / "cross-scores"
    stage2 = output_dir / "eval-filtered"
    stage3 = output_dir / "eval-orthogonalized"

    scores_df = pd.read_csv(stage1 / "cross_scores.csv")

    # Load every eval's original entries + judge prompts.
    loaded: Dict[str, dict] = {}
    for child in sorted(input_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if eval_names and name not in eval_names:
            continue
        yaml_path = find_yaml(child)
        if yaml_path is None:
            continue
        entries = load_eval_yaml(yaml_path)
        jp = judge_prompts_of(entries)
        loaded[name] = {
            "entries": entries,
            "entries_by_id": {e["id"]: e for e in entries},
            "judge_prompts": jp,
            "primary_metric": _primary_metric_for_eval(name, jp),
            "primary_expected": primary_expected_key(name, entries),
            "yaml_path": yaml_path,
        }

    # Figure out which questions each eval removed in Stage 2.
    per_eval_revised: Dict[str, list[dict]] = {}
    summary_rows = []
    combined_extra_rows = []
    for name, info in loaded.items():
        stage2_dir = stage2 / name
        stage3_dir = stage3 / name
        kept_json = stage2_dir / "kept_ids.json"
        log_path = stage3_dir / "revision_log.jsonl"
        if not kept_json.exists():
            summary_rows.append({"eval": name, "n_in": len(info["entries"]),
                                 "n_kept": len(info["entries"]), "n_best_of": 0,
                                 "reason": "no kept_ids.json"})
            continue
        kept_ids = set(json.loads(kept_json.read_text()))
        removed_ids = [e["id"] for e in info["entries"] if e["id"] not in kept_ids]

        by_qid = _load_logs(log_path)
        revised_entries: list[dict] = []
        orig_id_by_new_id: dict = {}
        picks = {"iter0": 0, "revision": 0, "skipped_no_logs": 0}
        for orig_qid in removed_ids:
            logs = by_qid.get(orig_qid, [])
            best = _pick_best_log(logs)
            if best is None:
                picks["skipped_no_logs"] += 1
                continue
            iteration = best.get("iteration", 0)
            original_entry = info["entries_by_id"].get(orig_qid)
            if original_entry is None:
                continue
            new_entry = _entry_from_log(best, original_entry, iteration)
            revised_entries.append(new_entry)
            orig_id_by_new_id[new_entry["id"]] = orig_qid
            picks["iter0" if iteration == 0 else "revision"] += 1

        per_eval_revised[name] = revised_entries
        if revised_entries:
            write_yaml_with_anchors(revised_entries, info["judge_prompts"],
                                    stage3_dir / "revised_best_of.yaml")

        # Collect score rows for these picks so we can rebuild heatmaps.
        combined_extra_rows.extend(
            _best_revision_rows(log_path, revised_entries, orig_id_by_new_id)
        )

        summary_rows.append({
            "eval": name,
            "n_in": len(info["entries"]),
            "n_kept_stage2": len(kept_ids),
            "n_removed_stage2": len(removed_ids),
            "n_best_of_picked": len(revised_entries),
            "picks_iter0_original": picks["iter0"],
            "picks_revision": picks["revision"],
            "n_no_logs": picks["skipped_no_logs"],
            "n_final": len(kept_ids) + len(revised_entries),
        })

    # Write combined questions_eval.yaml = kept + best-of revised.
    for name, info in loaded.items():
        kept_yaml = stage2 / name / "questions_eval.yaml"
        revised_entries = per_eval_revised.get(name, [])
        combined_entries = []
        if kept_yaml.exists():
            combined_entries.extend(load_eval_yaml(kept_yaml))
        combined_entries.extend(revised_entries)
        if not combined_entries:
            continue
        # De-dup by id, keep first occurrence.
        seen = set()
        unique = [e for e in combined_entries if e["id"] not in seen and not seen.add(e["id"])]
        write_yaml_with_anchors(unique, info["judge_prompts"],
                                stage3 / name / "questions_eval_best_of.yaml")

    # Build the full cross-score frame for rebuilt heatmaps: kept rows
    # (from stage1 scores) + best-of picks (from logs).
    kept_keys: set[tuple[str, str]] = set()
    for name in loaded:
        kept_json = stage2 / name / "kept_ids.json"
        if kept_json.exists():
            kept_keys.update((name, qid) for qid in json.loads(kept_json.read_text()))
    if kept_keys:
        kept_df = scores_df[
            scores_df.apply(lambda r: (r["source_eval"], r["question_id"]) in kept_keys, axis=1)
        ]
    else:
        kept_df = pd.DataFrame(columns=scores_df.columns)

    if combined_extra_rows:
        extra_df = pd.DataFrame(combined_extra_rows)
        combined_df = pd.concat([kept_df, extra_df], ignore_index=True, sort=False)
    else:
        combined_df = kept_df

    best_of_dir = stage3 / "best_of"
    best_of_dir.mkdir(parents=True, exist_ok=True)
    if not combined_df.empty:
        build_intrinsic_matrix(combined_df, loaded, best_of_dir)
        build_propensity_matrix(combined_df, loaded, best_of_dir)
        combined_df.to_csv(best_of_dir / "combined_cross_scores.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(best_of_dir / "summary.csv", index=False)
    print(summary_df.to_string(index=False))
    print(f"\nSaved rebuilt outputs under: {best_of_dir}")
    print("Per-eval rebuilt YAMLs: <eval>/questions_eval_best_of.yaml")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--evals", default=None,
                    help="Comma-separated subset of evals")
    args = ap.parse_args()
    eval_names = args.evals.split(",") if args.evals else None
    rebuild(Path(args.input).resolve(), Path(args.output_dir).resolve(), eval_names)


if __name__ == "__main__":
    main()
