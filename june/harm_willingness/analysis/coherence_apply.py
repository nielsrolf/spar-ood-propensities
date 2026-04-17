"""Apply the coherence judge to all key track CSVs and report per-condition
coherence rates plus recomputed headline metrics filtered to coherent-only.

Run after `coherence_filter.py --validate` confirms judge quality.

Writes:
  coherence_by_track.csv       — incoherence rate per (track, condition, facet)
  coherence_headline_impact.csv — key metrics before vs after filter
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from standardize import standardize_df
from analysis_utils import attach_refusal_flag, FACET_PRIMARY_HW

from coherence_filter import CoherenceJudge

JUNE = Path("/home/hunter/ai/spar-ood-propensities/june")
OUT = Path(__file__).parent

SOURCES = [
    # (track, path, condition_col, engaged-only?)
    ("B_dark",     JUNE / "dark_restyling/harm_willingness/results.csv",
        "group_condition"),
    ("C_restyled", JUNE / "dehumanization_restyling/output/results/all_conditions_standardized.csv",
        "condition"),
    ("D_70b",      JUNE / "dehumanization_restyling/definitional/def70b/all_raw_responses.csv",
        "condition"),
    ("D_mistral",  JUNE / "dehumanization_restyling/definitional/definitional_eval_mistral/def/all_raw_responses.csv",
        "condition"),
    ("E_em",       JUNE / "dehumanization_restyling/em_control_eval/all_em_responses.csv",
        "em_label"),
    ("E_stylistic",JUNE / "dehumanization_restyling/em_control_eval/stylistic_control_eval/all_stylistic_responses.csv",
        "stylistic_label"),
    ("NB2",        JUNE / "dehumanization_restyling/token_swap/all_token_swap_responses.csv",
        None),
]


def judge_track(track, path, cond_col, judge: CoherenceJudge):
    if not path.exists():
        print(f"[skip] {path}"); return None
    df = pd.read_csv(path, low_memory=False)
    if cond_col is None or cond_col not in df.columns:
        df["condition"] = "all"
        cond_col = "condition"
    # Only bother with engaged rows (refusals are their own category; INCOHERENT
    # should be measured on responses that claim to engage).
    df = attach_refusal_flag(df)
    engaged = df[~df.is_refusal].copy()
    # Judge
    print(f"[{track}] {len(engaged)} engaged rows; judging...")
    labelled = judge.judge_df(engaged, question_col="question", response_col="answer",
                              facet_col="facet", verbose=False)
    # Attach back
    labelled["track"] = track
    labelled["_cond"] = labelled[cond_col]
    return labelled


def main():
    j = CoherenceJudge()
    all_parts = []
    for src in SOURCES:
        part = judge_track(*src, judge=j)
        if part is not None:
            all_parts.append(part[["track", "_cond", "facet", "is_coherent", "is_refusal"]])
    joined = pd.concat(all_parts, ignore_index=True)
    joined.to_csv(OUT / "coherence_labels_all_tracks.csv", index=False)

    # Per-track, per-condition, per-facet coherence rate
    grp = joined.groupby(["track", "_cond", "facet"])
    summary = grp.agg(
        n_engaged=("is_coherent", "size"),
        n_coherent=("is_coherent", "sum"),
    ).reset_index()
    summary["coh_rate"] = (summary.n_coherent / summary.n_engaged.clip(lower=1)).round(3)
    summary["incoh_rate"] = (1 - summary.coh_rate).round(3)
    summary.to_csv(OUT / "coherence_by_track.csv", index=False)
    print("\ncoherence_by_track.csv written. top incoherence cells:")
    print(summary.sort_values("incoh_rate", ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
