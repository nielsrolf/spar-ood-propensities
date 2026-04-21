"""
Build machine-readable artifacts and a small static viewer for orthogonalize runs.

The inspector output lives under:
    <output_dir>/eval-orthogonalized/inspector/

Per eval it writes:
    data/<eval>.json                  full lineage + snapshot bundle for the web UI
    <eval>/snapshots/*.json           one file per pipeline snapshot
    <eval>/snapshots/*.csv            flattened view of the same snapshot
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from eval_utils import (
    expected_answer_keys,
    find_yaml,
    judge_prompts_of,
    load_eval_yaml,
    primary_expected_key,
)
from stage2 import _primary_metric_for_eval


def _load_eval_bundle(input_dir: Path, eval_names: List[str]) -> Dict[str, dict]:
    loaded: Dict[str, dict] = {}
    for name in eval_names:
        yaml_path = find_yaml(input_dir / name)
        if yaml_path is None:
            continue
        entries = load_eval_yaml(yaml_path)
        judge_prompts = judge_prompts_of(entries)
        loaded[name] = {
            "entries": entries,
            "entries_by_id": {entry["id"]: entry for entry in entries},
            "judge_prompts": judge_prompts,
            "primary_expected": primary_expected_key(name, entries),
            "primary_metric": _primary_metric_for_eval(name, judge_prompts),
        }
    return loaded


def _extract_reference_answers(entry: dict) -> dict[str, str]:
    meta = entry.get("meta", {}) or {}
    return {key: meta[key] for key in expected_answer_keys(entry) if key in meta}


def _primary_answer_type(info: dict) -> Optional[str]:
    primary = info.get("primary_expected")
    return None if primary is None else primary.replace("expected_", "")


def _question_rows(scores_df: pd.DataFrame, source_eval: str, qid: str) -> list[dict]:
    sub = scores_df[(scores_df["source_eval"] == source_eval) & (scores_df["question_id"] == qid)]
    rows = []
    for _, row in sub.iterrows():
        rows.append({
            "source_eval": source_eval,
            "question_id": qid,
            "answer_type": row["answer_type"],
            "target_eval": row["target_eval"],
            "target_metric": row["target_metric"],
            "score": None if pd.isna(row["score"]) else float(row["score"]),
        })
    return rows


def _rows_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=["source_eval", "question_id", "answer_type", "target_eval", "target_metric", "score"]
        )
    return pd.DataFrame(rows)


def _propensity_scores(rows: list[dict], loaded: dict, source_eval: str) -> dict[str, Optional[float]]:
    info = loaded[source_eval]
    primary_type = _primary_answer_type(info)
    if primary_type is None:
        return {}
    df = _rows_df(rows)
    scores: dict[str, Optional[float]] = {}
    for target_eval, target_info in loaded.items():
        metric = target_info["primary_metric"]
        sub = df[
            (df["answer_type"] == primary_type)
            & (df["target_eval"] == target_eval)
            & (df["target_metric"] == metric)
        ]["score"]
        if sub.empty:
            scores[target_eval] = None
            continue
        value = sub.mean(skipna=True)
        scores[target_eval] = None if pd.isna(value) else float(value)
    return scores


def _cross_summary(rows: list[dict], loaded: dict, source_eval: str, max_abs_gap: float) -> list[dict]:
    info = loaded[source_eval]
    primary_type = _primary_answer_type(info)
    if primary_type is None:
        return []
    df = _rows_df(rows)
    summary = []
    for target_eval, target_info in loaded.items():
        if target_eval == source_eval:
            continue
        metric = target_info["primary_metric"]
        prim_score = df[
            (df["answer_type"] == primary_type)
            & (df["target_eval"] == target_eval)
            & (df["target_metric"] == metric)
        ]["score"].mean(skipna=True)
        opp_parts = []
        gaps = []
        opp_df = df[
            (df["answer_type"] != primary_type)
            & (df["target_eval"] == target_eval)
            & (df["target_metric"] == metric)
        ]
        for answer_type, group in opp_df.groupby("answer_type", sort=False):
            opp_score = group["score"].mean(skipna=True)
            score = None if pd.isna(opp_score) else float(opp_score)
            opp_parts.append({"answer_type": answer_type, "score": score})
            if not pd.isna(prim_score):
                gap = abs(prim_score - 50.0) if score is None else abs(float(prim_score) - score)
                gaps.append((answer_type, float(gap)))
        if not opp_parts and not pd.isna(prim_score):
            gaps.append((None, float(abs(float(prim_score) - 50.0))))
        max_gap = max((gap for _, gap in gaps), default=None)
        summary.append({
            "target_eval": target_eval,
            "target_metric": metric,
            "primary_score": None if pd.isna(prim_score) else float(prim_score),
            "opposites": opp_parts,
            "max_gap": max_gap,
            "is_violation": max_gap is not None and max_gap > max_abs_gap,
        })
    return summary


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load_yaml_or_empty(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return load_eval_yaml(path)
    except yaml.YAMLError:
        return []


def _snapshot_row(
    source_eval: str,
    lineage_id: str,
    row_id: str,
    step: dict,
    included: bool,
    status: str,
) -> dict:
    return {
        "source_eval": source_eval,
        "lineage_id": lineage_id,
        "row_id": row_id,
        "step_id": step["step_id"],
        "stage": step["stage"],
        "label": step["label"],
        "iteration": step.get("iteration"),
        "origin": step.get("origin"),
        "status": status,
        "included": included,
        "question": step["question"],
        "reference_answers": step["reference_answers"],
        "primary_reference_answer": step.get("primary_reference_answer"),
        "propensity_scores": step["propensity_scores"],
        "cross_summary": step.get("cross_summary", []),
        "score_rows": step.get("score_rows", []),
        "chat_history": step.get("chat_history", []),
        "is_ok": step.get("is_ok"),
        "violations": step.get("violations", []),
        "badness": step.get("badness"),
    }


def _write_snapshot_csv(path: Path, rows: list[dict], target_evals: list[str]) -> None:
    fieldnames = [
        "source_eval",
        "lineage_id",
        "row_id",
        "step_id",
        "stage",
        "label",
        "iteration",
        "origin",
        "status",
        "included",
        "question",
        "primary_reference_answer",
        "reference_answers_json",
        "propensity_scores_json",
        "cross_summary_json",
        "chat_history_json",
        "is_ok",
        "violations_json",
        "badness",
    ] + [f"score__{target_eval}" for target_eval in target_evals]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {
                "source_eval": row["source_eval"],
                "lineage_id": row["lineage_id"],
                "row_id": row["row_id"],
                "step_id": row["step_id"],
                "stage": row["stage"],
                "label": row["label"],
                "iteration": row["iteration"],
                "origin": row["origin"],
                "status": row["status"],
                "included": row["included"],
                "question": row["question"],
                "primary_reference_answer": row["primary_reference_answer"],
                "reference_answers_json": json.dumps(row["reference_answers"], ensure_ascii=False),
                "propensity_scores_json": json.dumps(row["propensity_scores"], ensure_ascii=False),
                "cross_summary_json": json.dumps(row["cross_summary"], ensure_ascii=False),
                "chat_history_json": json.dumps(row["chat_history"], ensure_ascii=False),
                "is_ok": row["is_ok"],
                "violations_json": json.dumps(row["violations"], ensure_ascii=False),
                "badness": row["badness"],
            }
            for target_eval in target_evals:
                flat[f"score__{target_eval}"] = row["propensity_scores"].get(target_eval)
            writer.writerow(flat)


def _make_step(
    source_eval: str,
    row_id: str,
    label: str,
    stage: str,
    iteration: Optional[int],
    question: str,
    reference_answers: dict[str, str],
    score_rows: list[dict],
    loaded: dict,
    max_abs_gap: float,
    *,
    origin: str,
    chat_history: Optional[list[dict]] = None,
    is_ok: Optional[bool] = None,
    violations: Optional[list[str]] = None,
    badness: Optional[float] = None,
) -> dict:
    info = loaded[source_eval]
    primary_answer_key = info.get("primary_expected")
    primary_reference_answer = None
    if primary_answer_key is not None:
        primary_reference_answer = reference_answers.get(primary_answer_key)
    return {
        "row_id": row_id,
        "step_id": f"{row_id}:{stage}:{label}",
        "label": label,
        "stage": stage,
        "iteration": iteration,
        "origin": origin,
        "question": question,
        "reference_answers": reference_answers,
        "primary_reference_answer": primary_reference_answer,
        "score_rows": score_rows,
        "propensity_scores": _propensity_scores(score_rows, loaded, source_eval),
        "cross_summary": _cross_summary(score_rows, loaded, source_eval, max_abs_gap),
        "chat_history": chat_history or [],
        "is_ok": is_ok,
        "violations": violations or [],
        "badness": badness,
    }


def _copy_assets(output_root: Path) -> None:
    assets_dir = Path(__file__).with_name("inspector_assets")
    output_root.mkdir(parents=True, exist_ok=True)
    for name in ["index.html", "app.js", "styles.css"]:
        shutil.copy2(assets_dir / name, output_root / name)


def build_inspector_artifacts(
    input_dir: Path,
    output_dir: Path,
    eval_names: List[str],
    scores_df: Optional[pd.DataFrame] = None,
    max_abs_gap: float = 20.0,
) -> Path:
    if scores_df is None:
        scores_path = output_dir / "cross-scores" / "cross_scores.csv"
        if not scores_path.exists():
            raise FileNotFoundError(f"Missing stage1 cache: {scores_path}")
        scores_df = pd.read_csv(scores_path)

    loaded = _load_eval_bundle(input_dir, eval_names)
    inspector_root = output_dir / "eval-orthogonalized" / "inspector"
    data_root = inspector_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    _copy_assets(inspector_root)

    manifest = {"evals": [], "generated_from": str(output_dir)}

    for source_eval in eval_names:
        if source_eval not in loaded:
            continue
        info = loaded[source_eval]
        target_evals = list(loaded.keys())
        primary_answer_key = info.get("primary_expected")

        lineages: dict[str, dict] = {}
        kept_yaml = output_dir / "eval-filtered" / source_eval / "questions_eval.yaml"
        removed_csv = output_dir / "eval-filtered" / source_eval / "removed.csv"
        revised_yaml = output_dir / "eval-orthogonalized" / source_eval / "revised.yaml"
        new_yaml = output_dir / "eval-orthogonalized" / source_eval / "new.yaml"
        scored_qids = set(scores_df.loc[scores_df["source_eval"] == source_eval, "question_id"].dropna().astype(str))
        kept_entries = _load_yaml_or_empty(kept_yaml)
        revised_entries = _load_yaml_or_empty(revised_yaml)
        new_entries = _load_yaml_or_empty(new_yaml)
        removed_qids = set()
        if removed_csv.exists():
            removed_df = pd.read_csv(removed_csv)
            if "question_id" in removed_df.columns:
                removed_qids = set(removed_df["question_id"].dropna().astype(str))
        candidate_ids = scored_qids | {entry["id"] for entry in kept_entries} | removed_qids
        if candidate_ids:
            original_entries = [entry for entry in info["entries"] if entry["id"] in candidate_ids]
        else:
            original_entries = kept_entries
        for entry in original_entries:
            qid = entry["id"]
            refs = _extract_reference_answers(entry)
            step = _make_step(
                source_eval,
                qid,
                label="Stage 1 original",
                stage="stage1_original",
                iteration=0,
                question=entry["paraphrases"][0],
                reference_answers=refs,
                score_rows=_question_rows(scores_df, source_eval, qid),
                loaded=loaded,
                max_abs_gap=max_abs_gap,
                origin="original",
            )
            lineages[qid] = {
                "lineage_id": qid,
                "source_eval": source_eval,
                "initial_row_id": qid,
                "current_row_id": qid,
                "final_status": "unknown",
                "history": [step],
            }

        kept_ids_path = output_dir / "eval-filtered" / source_eval / "kept_ids.json"
        kept_ids = set(json.loads(kept_ids_path.read_text())) if kept_ids_path.exists() else set(lineages)
        removed_ids = [qid for qid in lineages if qid not in kept_ids]

        revision_logs = _read_jsonl(output_dir / "eval-orthogonalized" / source_eval / "revision_log.jsonl")
        revision_logs_by_qid: dict[str, list[dict]] = {}
        for log in revision_logs:
            qid = log.get("qid")
            if qid is None or "score_rows" not in log or qid not in lineages:
                continue
            revision_logs_by_qid.setdefault(qid, []).append(log)

        max_revision_iteration = 0
        for qid, logs in revision_logs_by_qid.items():
            for log in sorted(logs, key=lambda item: item["iteration"]):
                iteration = int(log["iteration"])
                max_revision_iteration = max(max_revision_iteration, iteration)
                if iteration == 0:
                    lineage_step = lineages[qid]["history"][0]
                    lineage_step["chat_history"] = log.get("chat_history", [])
                    lineage_step["is_ok"] = log.get("is_ok")
                    lineage_step["violations"] = log.get("violations", [])
                    lineage_step["badness"] = log.get("badness")
                    lineage_step["cross_summary"] = log.get("cross_summary", lineage_step["cross_summary"])
                    continue
                step = _make_step(
                    source_eval,
                    f"{qid}_r{iteration}",
                    label=f"Revision {iteration}",
                    stage="stage3_revision",
                    iteration=iteration,
                    question=log["question"],
                    reference_answers=log.get("reference_answers", {}),
                    score_rows=log["score_rows"],
                    loaded=loaded,
                    max_abs_gap=max_abs_gap,
                    origin="revision",
                    chat_history=log.get("chat_history", []),
                    is_ok=log.get("is_ok"),
                    violations=log.get("violations", []),
                    badness=log.get("badness"),
                )
                step["cross_summary"] = log.get("cross_summary", step["cross_summary"])
                lineages[qid]["history"].append(step)

        revised_entry_by_original: dict[str, dict] = {}
        for entry in revised_entries:
            if "_r" not in entry["id"]:
                continue
            orig_qid, _ = entry["id"].rsplit("_r", 1)
            revised_entry_by_original[orig_qid] = entry

        grow_logs = _read_jsonl(output_dir / "eval-orthogonalized" / source_eval / "grow_log.jsonl")
        grow_log_by_id = {
            f"{source_eval.replace('-', '_')}_grow_{int(log['draft_idx']):03d}": log
            for log in grow_logs
            if "draft_idx" in log and "score_rows" in log
        }
        for entry in new_entries:
            row_id = entry["id"]
            log = grow_log_by_id.get(row_id)
            refs = _extract_reference_answers(entry)
            step = _make_step(
                source_eval,
                row_id,
                label="Growth draft",
                stage="stage3_growth",
                iteration=0,
                question=entry["paraphrases"][0],
                reference_answers=refs,
                score_rows=[] if log is None else log["score_rows"],
                loaded=loaded,
                max_abs_gap=max_abs_gap,
                origin="growth",
                chat_history=[] if log is None else log.get("chat_history", []),
                is_ok=None if log is None else log.get("is_ok"),
                violations=[] if log is None else log.get("violations", []),
            )
            if log is not None:
                step["cross_summary"] = log.get("cross_summary", step["cross_summary"])
            lineages[row_id] = {
                "lineage_id": row_id,
                "source_eval": source_eval,
                "initial_row_id": row_id,
                "current_row_id": row_id,
                "final_status": "grown",
                "history": [step],
            }

        final_entries = [*kept_entries, *revised_entries, *new_entries]
        final_entry_ids = {entry["id"] for entry in final_entries}
        for qid, lineage in lineages.items():
            if qid in revised_entry_by_original:
                final_entry = revised_entry_by_original[qid]
                final_id = final_entry["id"]
                lineage["current_row_id"] = final_id
                lineage["final_status"] = "revised"
            elif qid in kept_ids:
                lineage["final_status"] = "kept"
            elif qid not in final_entry_ids:
                lineage["final_status"] = "dropped"

        snapshots = []
        snapshot_dir = inspector_root / source_eval / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        def emit_snapshot(snapshot_id: str, label: str, rows: list[dict]) -> None:
            rows = sorted(rows, key=lambda row: (row["included"] is False, row["row_id"]))
            json_path = snapshot_dir / f"{snapshot_id}.json"
            csv_path = snapshot_dir / f"{snapshot_id}.csv"
            payload = {
                "eval_name": source_eval,
                "snapshot_id": snapshot_id,
                "label": label,
                "rows": rows,
            }
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            _write_snapshot_csv(csv_path, rows, target_evals)
            snapshots.append({
                "id": snapshot_id,
                "label": label,
                "n_rows": len(rows),
                "json_path": str(json_path.relative_to(inspector_root)),
                "csv_path": str(csv_path.relative_to(inspector_root)),
            })

        stage1_rows = []
        for qid in [entry["id"] for entry in original_entries]:
            lineage = lineages[qid]
            stage1_rows.append(_snapshot_row(source_eval, qid, qid, lineage["history"][0], True, "original"))
        emit_snapshot("stage1_original", "Stage 1 original", stage1_rows)

        stage2_rows = []
        for qid in [entry["id"] for entry in original_entries]:
            if qid not in kept_ids:
                continue
            lineage = lineages[qid]
            stage2_rows.append(_snapshot_row(source_eval, qid, qid, lineage["history"][0], True, "kept"))
        emit_snapshot("stage2_filtered", "Stage 2 filtered", stage2_rows)

        for iteration in range(0, max_revision_iteration + 1):
            rows = []
            for qid in [entry["id"] for entry in original_entries]:
                lineage = lineages[qid]
                if qid in kept_ids:
                    step = lineage["history"][0]
                    rows.append(_snapshot_row(source_eval, qid, qid, step, True, "kept"))
                    continue
                candidate_steps = [step for step in lineage["history"] if (step.get("iteration") or 0) <= iteration]
                if not candidate_steps:
                    continue
                step = candidate_steps[-1]
                status = "revision_candidate" if iteration == 0 else "revision_current"
                rows.append(_snapshot_row(source_eval, qid, step["row_id"], step, True, status))
            emit_snapshot(
                f"stage3_iter_{iteration:02d}",
                f"Stage 3 working set after iteration {iteration}",
                rows,
            )

        final_rows = []
        for entry in final_entries:
            row_id = entry["id"]
            if row_id in lineages:
                lineage = lineages[row_id]
                step = lineage["history"][-1]
                final_rows.append(_snapshot_row(source_eval, row_id, row_id, step, True, lineage["final_status"]))
                continue
            if "_r" in row_id:
                orig_qid, _ = row_id.rsplit("_r", 1)
                lineage = lineages.get(orig_qid)
                if lineage is None:
                    continue
                step = next((step for step in lineage["history"] if step["row_id"] == row_id), lineage["history"][-1])
                final_rows.append(_snapshot_row(source_eval, orig_qid, row_id, step, True, lineage["final_status"]))
                continue
        emit_snapshot("stage3_final", "Stage 3 final eval", final_rows)

        payload = {
            "eval_name": source_eval,
            "primary_answer_key": primary_answer_key,
            "primary_metric": info["primary_metric"],
            "target_evals": target_evals,
            "snapshots": snapshots,
            "lineages": lineages,
        }
        data_path = data_root / f"{source_eval}.json"
        data_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        manifest["evals"].append({
            "name": source_eval,
            "path": f"data/{source_eval}.json",
            "n_lineages": len(lineages),
            "n_final_rows": len(final_rows),
        })

    (inspector_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return inspector_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--evals", default=None, help="Comma-separated subset of eval names")
    parser.add_argument("--max-abs-gap", type=float, default=20.0)
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    if args.evals:
        eval_names = [name.strip() for name in args.evals.split(",") if name.strip()]
    else:
        eval_names = sorted(child.name for child in input_dir.iterdir() if child.is_dir() and find_yaml(child) is not None)

    inspector_root = build_inspector_artifacts(
        input_dir=input_dir,
        output_dir=output_dir,
        eval_names=eval_names,
        max_abs_gap=args.max_abs_gap,
    )
    print(inspector_root)


if __name__ == "__main__":
    main()
