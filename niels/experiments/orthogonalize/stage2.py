"""
Stage 2 — orthogonality filtering.

Given Stage 1 cross-scores, keep only the questions in each eval whose
reference answers score neutrally (null, or with a small target-vs-opposite
gap) on the primary metrics of OTHER evals.

Definition of "orthogonal enough" for a question:
    For every OTHER eval's primary metric, the gap between the primary and
    opposite reference answers must satisfy:
        (|gap| <= max_abs_gap)  OR  (both scores are null)
    A question is removed if it violates this on more than `max_violations`
    other primary metrics.

Outputs per eval:
    <output_dir>/eval-filtered/<eval_name>/questions_eval.yaml — filtered eval YAML
    <output_dir>/eval-filtered/<eval_name>/kept_ids.json        — kept question ids
    <output_dir>/eval-filtered/<eval_name>/removed.csv          — removed question ids + reasons
    <output_dir>/eval-filtered/summary.csv                      — how many kept per eval
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from eval_utils import (
    find_yaml,
    judge_prompts_of,
    load_eval_yaml,
    primary_expected_key,
    write_yaml_with_anchors,
)
from stage1 import build_intrinsic_matrix, build_propensity_matrix


def _primary_metric_for_eval(eval_name: str, judge_prompts: dict) -> str:
    """Return the primary metric name for an eval — the first metric in its
    judge_prompts dict. By convention the first metric is the target score
    (e.g., power_seeking_score, animal_care_score, sycophancy_score)."""
    return next(iter(judge_prompts.keys()))


def run_stage2(
    input_dir: Path,
    output_dir: Path,
    eval_names: List[str],
    scores_df: pd.DataFrame,
    max_abs_gap: float = 20.0,
    max_violations: int = 1,
) -> None:
    """Filter each eval's questions based on cross-score orthogonality."""
    stage_dir = output_dir / "eval-filtered"
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Load every eval once.
    loaded: Dict[str, dict] = {}
    for name in eval_names:
        yaml_path = find_yaml(input_dir / name)
        if yaml_path is None:
            continue
        entries = load_eval_yaml(yaml_path)
        jp = judge_prompts_of(entries)
        loaded[name] = {
            "entries": entries,
            "judge_prompts": jp,
            "primary_expected": primary_expected_key(name, entries),
            "primary_metric": _primary_metric_for_eval(name, jp),
            "yaml_path": yaml_path,
        }

    summary_rows = []
    for src, info in loaded.items():
        entries = info["entries"]
        primary = info["primary_expected"]
        if primary is None:
            print(f"  [skip] {src}: no expected_* reference answer — keeping all")
            _save_eval(stage_dir / src, entries, info["judge_prompts"], removed=[])
            summary_rows.append({
                "eval": src, "n_in": len(entries), "n_kept": len(entries),
                "n_removed": 0, "reason": "no reference answers — no filter",
            })
            continue
        primary_type = primary.replace("expected_", "")

        # All OTHER evals' primary metrics.
        other_metrics = [
            (other, loaded[other]["primary_metric"])
            for other in loaded if other != src
        ]

        # Compute target-vs-opposite gap for this eval's primary vs its
        # "opposite-ish" answers, per question and per target metric.
        sub = scores_df[scores_df["source_eval"] == src]
        if sub.empty:
            print(f"  [skip] {src}: no stage1 scores available")
            continue
        primary_sub = sub[sub["answer_type"] == primary_type]
        # For non-primary reference answers, pick the one with the largest
        # name mismatch (e.g., expected_indifferent for caring). We use ALL
        # non-primary types and take the max absolute gap across them per
        # question×metric.
        non_primary = sub[sub["answer_type"] != primary_type]

        # Only iterate entries that were actually scored in stage 1.
        scored_ids = set(primary_sub["question_id"].unique())
        scored_entries = [e for e in entries if e["id"] in scored_ids]
        if len(scored_entries) < len(entries):
            print(f"    [{src}] stage1 scored {len(scored_entries)}/{len(entries)} "
                  "entries; filter will ignore unscored entries")

        kept = []
        removed = []
        for entry in scored_entries:
            qid = entry["id"]
            q_primary = primary_sub[primary_sub["question_id"] == qid]
            q_non = non_primary[non_primary["question_id"] == qid]

            # Score on OWN primary metric should be high (target answer
            # expresses trait); skip if we have no data.
            own_metric = info["primary_metric"]
            own_primary = q_primary[q_primary["target_metric"] == own_metric]
            if own_primary.empty or own_primary["score"].isna().all():
                removed.append({
                    "question_id": qid,
                    "reason": "no own-metric score on primary reference answer",
                    "violating_metrics": "",
                })
                continue

            violations = []
            for target_eval, metric in other_metrics:
                prim = q_primary[
                    (q_primary["target_eval"] == target_eval)
                    & (q_primary["target_metric"] == metric)
                ]
                opp = q_non[
                    (q_non["target_eval"] == target_eval)
                    & (q_non["target_metric"] == metric)
                ]
                if prim.empty:
                    continue
                prim_score = prim["score"].mean(skipna=True)
                # If primary is null, this is ideal — no spillover signal.
                if pd.isna(prim_score):
                    # If opposite has a strong score while primary is null,
                    # that also counts as clean — still orthogonal.
                    continue

                # Primary has a score. Compare against each non-primary
                # answer type separately, then count at most one violation
                # per cross-metric using the worst gap.
                gap_candidates = []
                if opp.empty:
                    gap_candidates.append((None, None, abs(prim_score - 50.0)))
                else:
                    for answer_type, answer_group in opp.groupby("answer_type", sort=False):
                        opp_score = answer_group["score"].mean(skipna=True)
                        if pd.isna(opp_score):
                            gap = abs(prim_score - 50.0)
                            gap_candidates.append((answer_type, None, gap))
                        else:
                            gap = abs(prim_score - opp_score)
                            gap_candidates.append((answer_type, opp_score, gap))

                answer_type, opp_score, gap = max(gap_candidates, key=lambda item: item[2])
                if gap > max_abs_gap:
                    violations.append((target_eval, metric, prim_score, opp_score, gap, answer_type))

            if len(violations) > max_violations:
                removed.append({
                    "question_id": qid,
                    "reason": f"{len(violations)} cross-metric gaps > {max_abs_gap}",
                    "violating_metrics": json.dumps([
                        f"{e}.{m} (primary={p}, opp_type={a}, opp={o}, gap={g:.1f})"
                        for e, m, p, o, g, a in violations
                    ]),
                })
            else:
                kept.append(entry)

        _save_eval(stage_dir / src, kept, info["judge_prompts"], removed)
        summary_rows.append({
            "eval": src,
            "n_in": len(scored_entries),
            "n_kept": len(kept),
            "n_removed": len(removed),
        })
        print(f"  {src}: {len(kept)}/{len(scored_entries)} kept, {len(removed)} removed")

    pd.DataFrame(summary_rows).to_csv(stage_dir / "summary.csv", index=False)
    print(f"  saved {stage_dir / 'summary.csv'}")

    # Regenerate stage-1-style summaries on kept questions only.
    kept_keys: set[tuple[str, str]] = set()
    for src in loaded:
        kept_json = stage_dir / src / "kept_ids.json"
        if kept_json.exists():
            kept_keys.update((src, qid) for qid in json.loads(kept_json.read_text()))
    if kept_keys:
        filtered_df = scores_df[
            scores_df.apply(
                lambda row: (row["source_eval"], row["question_id"]) in kept_keys,
                axis=1,
            )
        ]
        print("  Rebuilding intrinsic views on filtered questions...")
        build_intrinsic_matrix(filtered_df, loaded, stage_dir)
        print("  Rebuilding propensity heatmaps on filtered questions...")
        build_propensity_matrix(filtered_df, loaded, stage_dir)


def _save_eval(eval_out_dir: Path, entries: list, judge_prompts: dict, removed: list[dict]) -> None:
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    if entries:
        write_yaml_with_anchors(entries, judge_prompts, eval_out_dir / "questions_eval.yaml")
    (eval_out_dir / "kept_ids.json").write_text(
        json.dumps([e["id"] for e in entries], indent=2)
    )
    removed_df = pd.DataFrame(
        removed,
        columns=["question_id", "reason", "violating_metrics"],
    )
    removed_df.to_csv(eval_out_dir / "removed.csv", index=False)
