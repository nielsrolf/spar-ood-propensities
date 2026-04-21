from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from eval_utils import find_yaml, judge_prompts_of, list_evals, load_eval_yaml
from stage2 import _primary_metric_for_eval


def _load_eval_info(input_dir: Path) -> dict[str, dict]:
    loaded = {}
    for name in list_evals(input_dir):
        yaml_path = find_yaml(input_dir / name)
        if yaml_path is None:
            continue
        entries = load_eval_yaml(yaml_path)
        judge_prompts = judge_prompts_of(entries)
        loaded[name] = {
            "judge_prompts": judge_prompts,
            "primary_metric": _primary_metric_for_eval(name, judge_prompts),
        }
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the manual null benchmark from inspector artifacts.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cases", default=str(Path(__file__).with_name("cases.json")))
    parser.add_argument("--output-csv", default=str(Path(__file__).with_name("benchmark.csv")))
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    cases = json.loads(Path(args.cases).read_text())
    eval_info = _load_eval_info(input_dir)
    inspector_root = output_dir / "eval-orthogonalized" / "inspector" / "data"

    rows = []
    for case in cases:
        source_eval = case["source_eval"]
        lineage_id = case["lineage_id"]
        target_eval = case["target_eval"]
        payload = json.loads((inspector_root / f"{source_eval}.json").read_text())
        lineage = payload["lineages"][lineage_id]
        step = lineage["history"][0]
        target_metric = eval_info[target_eval]["primary_metric"]
        rows.append({
            "source_eval": source_eval,
            "lineage_id": lineage_id,
            "snapshot_stage": "stage1_original",
            "target_eval": target_eval,
            "target_metric": target_metric,
            "current_score": step["propensity_scores"].get(target_eval),
            "should_be_null": case["should_be_null"],
            "label_notes": case["label_notes"],
            "question": step["question"],
            "answer": step["primary_reference_answer"],
            "reference_answers_json": json.dumps(step["reference_answers"], ensure_ascii=False),
            "cross_summary_json": json.dumps(step["cross_summary"], ensure_ascii=False),
            "judge_prompt": eval_info[target_eval]["judge_prompts"][target_metric],
        })

    out_path = Path(args.output_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_eval",
        "lineage_id",
        "snapshot_stage",
        "target_eval",
        "target_metric",
        "current_score",
        "should_be_null",
        "label_notes",
        "question",
        "answer",
        "reference_answers_json",
        "cross_summary_json",
        "judge_prompt",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(out_path)


if __name__ == "__main__":
    main()
