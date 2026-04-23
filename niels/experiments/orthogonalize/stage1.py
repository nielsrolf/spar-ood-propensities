"""
Stage 1 — intrinsic-relation matrix.

For each source eval E_s, for each question q in E_s, for each reference
answer type t (e.g., `expected_caring`, `expected_indifferent`), score the
(question, reference_answer) pair against every metric of every eval
(including E_s itself).

Outputs (all under <output_dir>/cross-scores/):
    cross_scores.csv                       one row per (src_eval, qid,
                                           answer_type, tgt_eval, tgt_metric)
    intrinsic_matrix.csv                   long-format mean score per
                                           (src, src_answer, tgt_eval,
                                           tgt_metric)
    intrinsic_gap.csv                      target-vs-opposite gap analysis
    intrinsic_gap_heatmap.png              heatmap of the gap view

    propensity_mean_score.csv              PROPENSITY × PROPENSITY matrix:
                                           rows = source propensity,
                                           cols = target propensity,
                                           value = mean judge score on
                                                  target's primary metric
                                                  when scoring source's
                                                  target-trait reference
    propensity_null_fraction.csv           same shape; value = fraction of
                                                  questions the target judge
                                                  answered with null
    propensity_n_scored.csv                same shape; value = n questions
    propensity_matrix_long.csv             long-format version of the above
    propensity_mean_score_heatmap.png      visualization of mean_score
    propensity_null_fraction_heatmap.png   visualization of null_fraction
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

from eval_utils import (
    expected_answer_keys,
    expected_answers_for_eval,
    find_yaml,
    judge_prompts_of,
    load_eval_yaml,
    primary_expected_key,
)
from judge import score_pair


def _slug(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_")


async def _score_one(
    source_eval: str,
    question_id: str,
    question_text: str,
    answer_type: str,
    answer_text: str,
    target_eval: str,
    target_metric: str,
    target_prompt: str,
    judge_model: str,
    n_samples: int,
) -> dict:
    score = await score_pair(
        target_prompt, question_text, answer_text,
        model=judge_model, n_samples=n_samples,
    )
    return {
        "source_eval": source_eval,
        "question_id": question_id,
        "answer_type": answer_type,
        "target_eval": target_eval,
        "target_metric": target_metric,
        "score": score,
    }


async def run_stage1(
    input_dir: Path,
    output_dir: Path,
    eval_names: List[str],
    judge_model: str,
    n_samples: int,
    n_questions: Optional[int] = None,
    batch_size: int = 1024,
) -> pd.DataFrame:
    """Run the full cross-scoring. See module docstring for what it produces."""
    stage_dir = output_dir / "cross-scores"
    stage_dir.mkdir(parents=True, exist_ok=True)
    csv_path = stage_dir / "cross_scores.csv"

    # Load every eval's YAML once (judge prompts + entries).
    loaded: Dict[str, dict] = {}
    for name in eval_names:
        yaml_path = find_yaml(input_dir / name)
        if yaml_path is None:
            continue
        entries = load_eval_yaml(yaml_path)
        loaded[name] = {
            "entries": entries,
            "judge_prompts": judge_prompts_of(entries),
            "primary_expected": primary_expected_key(name, entries),
            "yaml_path": yaml_path,
        }

    # Reuse any existing on-disk rows so interrupted runs can resume cheaply.
    key_cols = ["source_eval", "question_id", "answer_type", "target_eval", "target_metric"]
    existing_rows: list[dict] = []
    existing_keys: set[tuple[str, str, str, str, str]] = set()
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        if not existing_df.empty:
            existing_rows = existing_df.to_dict("records")
            existing_keys = {
                tuple(row[col] for col in key_cols)
                for _, row in existing_df[key_cols].iterrows()
            }
            print(f"  reusing {len(existing_rows)} cached stage1 rows from {csv_path}")

    # Collect every uncached (source_eval, question, answer_type, target_eval, target_metric)
    # combination. This is the full Cartesian product we need to score.
    tasks = []
    for source_eval, info in loaded.items():
        entries = info["entries"]
        if n_questions is not None:
            entries = entries[:n_questions]
        for entry in entries:
            qid = entry["id"]
            qtext = entry["paraphrases"][0]
            meta = entry.get("meta", {}) or {}
            for exp_key in expected_answer_keys(entry):
                atext = meta[exp_key]
                if not atext:
                    continue
                answer_type = exp_key.replace("expected_", "")
                for target_eval, tinfo in loaded.items():
                    for metric, prompt in tinfo["judge_prompts"].items():
                        key = (source_eval, qid, answer_type, target_eval, metric)
                        if key in existing_keys:
                            continue
                        tasks.append(_score_one(
                            source_eval, qid, qtext, answer_type,
                            atext, target_eval, metric, prompt,
                            judge_model, n_samples,
                        ))

    print(f"Stage 1: {len(tasks)} uncached (question × answer-type × target-metric) scores to compute")
    rows = list(existing_rows)
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start:start + batch_size]
        batch_rows = await tqdm_asyncio.gather(
            *batch,
            desc=f"stage1 cross-scoring [{start}:{start + len(batch)}]",
        )
        rows.extend(batch_rows)
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    df = pd.DataFrame(rows)

    df.to_csv(csv_path, index=False)
    print(f"  saved {csv_path}")

    # Intrinsic matrix: aggregate by (source, target_metric) on the primary
    # reference answer and the "opposite" reference answers.
    build_intrinsic_matrix(df, loaded, stage_dir)

    # Propensity × propensity matrices (mean score + null fraction), using
    # each target eval's PRIMARY metric only, and each source eval's
    # primary (target-trait) reference answer only. This is the single
    # "what is the intrinsic relation between trait X and trait Y?" view.
    build_propensity_matrix(df, loaded, stage_dir)

    return df


def build_intrinsic_matrix(df: pd.DataFrame, loaded: dict, stage_dir: Path) -> None:
    """Produce two views:
    1. `intrinsic_matrix.csv` — mean score per (source eval primary answer,
       target eval, target metric). This captures "when I express trait X,
       how does it look to a judge of trait Y?".
    2. `intrinsic_gap.csv` — for source evals with both a target and an
       opposite reference answer, the mean score on target minus mean score
       on opposite, per target metric. Large absolute gaps indicate
       intrinsic relations.
    """
    # Pick primary expected key per source eval.
    rows_primary = []
    rows_gap = []
    for src, info in loaded.items():
        primary = info["primary_expected"]
        if primary is None:
            continue
        primary_type = primary.replace("expected_", "")
        sub = df[df["source_eval"] == src]
        if sub.empty:
            continue
        primary_sub = sub[sub["answer_type"] == primary_type]
        if primary_sub.empty:
            continue

        # Primary view.
        for (target_eval, metric), g in primary_sub.groupby(["target_eval", "target_metric"]):
            rows_primary.append({
                "source_eval": src,
                "source_answer": primary_type,
                "target_eval": target_eval,
                "target_metric": metric,
                "mean_score": g["score"].mean(skipna=True),
                "n": int(g["score"].notna().sum()),
                "n_null": int(g["score"].isna().sum()),
            })

        # Gap view: compare primary to each non-primary answer type in the
        # same source eval.
        other_types = [
            t for t in primary_sub["answer_type"].unique().tolist() if False
        ]
        answer_types = sub["answer_type"].unique().tolist()
        for other_type in answer_types:
            if other_type == primary_type:
                continue
            other_sub = sub[sub["answer_type"] == other_type]
            if other_sub.empty:
                continue
            merged = primary_sub.merge(
                other_sub,
                on=["source_eval", "question_id", "target_eval", "target_metric"],
                suffixes=("_primary", "_opposite"),
            )
            for (target_eval, metric), g in merged.groupby(["target_eval", "target_metric"]):
                gap = g["score_primary"] - g["score_opposite"]
                rows_gap.append({
                    "source_eval": src,
                    "primary_answer": primary_type,
                    "opposite_answer": other_type,
                    "target_eval": target_eval,
                    "target_metric": metric,
                    "mean_gap": gap.mean(skipna=True),
                    "abs_mean_gap": gap.abs().mean(skipna=True),
                    "mean_primary": g["score_primary"].mean(skipna=True),
                    "mean_opposite": g["score_opposite"].mean(skipna=True),
                    "n_pairs": int(gap.notna().sum()),
                })

    primary_df = pd.DataFrame(rows_primary)
    gap_df = pd.DataFrame(rows_gap)

    primary_df.to_csv(stage_dir / "intrinsic_matrix.csv", index=False)
    gap_df.to_csv(stage_dir / "intrinsic_gap.csv", index=False)
    print(f"  saved {stage_dir / 'intrinsic_matrix.csv'}")
    print(f"  saved {stage_dir / 'intrinsic_gap.csv'}")

    # Heatmap: source eval (primary answer) × target metric. Cell = mean gap
    # vs. the (first) opposite answer. Evals without an opposite are skipped.
    if gap_df.empty:
        return
    # Keep only the first opposite per source eval for a clean 2D heatmap.
    first_opposite = (
        gap_df.sort_values(["source_eval", "opposite_answer"])
              .groupby("source_eval").head(10**9)  # identity
    )
    # Pivot: one row per source, one column per (target_eval, metric).
    first_opposite["target_col"] = (
        first_opposite["target_eval"] + "\n" + first_opposite["target_metric"]
    )
    # Collapse multiple opposites by taking mean gap across opposites.
    pivot = first_opposite.groupby(["source_eval", "target_col"])["mean_gap"].mean().unstack()

    # Sort columns by target_eval, then metric.
    pivot = pivot[sorted(pivot.columns)]

    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(pivot.columns)), max(4, 0.5 * len(pivot.index))))
    vmax = np.nanmax(np.abs(pivot.values)) if pivot.size else 1.0
    vmax = max(vmax, 10.0)
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title(
        "Intrinsic relation: mean gap between primary and opposite reference answers\n"
        "(rows = source eval, columns = target metric; large magnitude = spillover)"
    )
    plt.colorbar(im, ax=ax, label="mean gap (primary − opposite)")
    # Annotate cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.0f}", ha="center", va="center",
                        fontsize=6, color="black" if abs(v) < vmax * 0.6 else "white")
    plt.tight_layout()
    out_png = stage_dir / "intrinsic_gap_heatmap.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_png}")


def build_propensity_matrix(df: pd.DataFrame, loaded: dict, stage_dir: Path) -> None:
    """Build the propensity × propensity matrix view the user asks for.

    Rows = source eval (scoring the TARGET-trait reference answer only).
    Cols = target eval (scored on its PRIMARY metric only).
    Two aggregated values per cell:
      - mean_score:      mean numeric score across questions (NaN-skipping)
      - null_fraction:   fraction of questions where the judge returned null

    Writes:
        propensity_mean_score.csv
        propensity_null_fraction.csv
        propensity_mean_score_heatmap.png
        propensity_null_fraction_heatmap.png
    """
    # Pick primary metric per target eval.
    primary_metric = {name: next(iter(info["judge_prompts"].keys()))
                      for name, info in loaded.items()}
    # Pick primary reference-answer key per source eval.
    primary_type = {}
    for name, info in loaded.items():
        pe = info.get("primary_expected")
        if pe is not None:
            primary_type[name] = pe.replace("expected_", "")

    if not primary_type:
        print("  [propensity matrix] no source evals with reference answers — skipping")
        return

    sources = [s for s in loaded if s in primary_type]
    targets = list(loaded.keys())

    mean_score = pd.DataFrame(index=sources, columns=targets, dtype=float)
    null_frac = pd.DataFrame(index=sources, columns=targets, dtype=float)
    n_total = pd.DataFrame(index=sources, columns=targets, dtype=int)

    for src in sources:
        for tgt in targets:
            sub = df[
                (df["source_eval"] == src)
                & (df["answer_type"] == primary_type[src])
                & (df["target_eval"] == tgt)
                & (df["target_metric"] == primary_metric[tgt])
            ]
            if sub.empty:
                continue
            n_total.loc[src, tgt] = len(sub)
            null_frac.loc[src, tgt] = sub["score"].isna().mean()
            if sub["score"].notna().any():
                mean_score.loc[src, tgt] = sub["score"].mean(skipna=True)

    # CSVs (long + wide for convenience)
    mean_score.to_csv(stage_dir / "propensity_mean_score.csv")
    null_frac.to_csv(stage_dir / "propensity_null_fraction.csv")
    n_total.to_csv(stage_dir / "propensity_n_scored.csv")
    print(f"  saved {stage_dir / 'propensity_mean_score.csv'}")
    print(f"  saved {stage_dir / 'propensity_null_fraction.csv'}")

    # Combined long-format (easier to join/inspect).
    long_rows = []
    for src in sources:
        for tgt in targets:
            ms = mean_score.loc[src, tgt]
            nf = null_frac.loc[src, tgt]
            long_rows.append({
                "source_propensity": src,
                "target_propensity": tgt,
                "target_metric": primary_metric[tgt],
                "mean_score": None if pd.isna(ms) else float(ms),
                "null_fraction": None if pd.isna(nf) else float(nf),
                "n_scored": int(n_total.loc[src, tgt]) if not pd.isna(n_total.loc[src, tgt]) else 0,
            })
    pd.DataFrame(long_rows).to_csv(stage_dir / "propensity_matrix_long.csv", index=False)
    print(f"  saved {stage_dir / 'propensity_matrix_long.csv'}")

    # Heatmaps.
    _heatmap(
        mean_score, stage_dir / "propensity_mean_score_heatmap.png",
        title=(
            "Mean judge score on target propensity's primary metric\n"
            "when scoring the source propensity's TARGET-trait reference answer\n"
            "(rows = source propensity, cols = target propensity)"
        ),
        cbar_label="mean score (0-100)", vmin=0, vmax=100, cmap="viridis",
        fmt="{:.0f}",
    )
    _heatmap(
        null_frac, stage_dir / "propensity_null_fraction_heatmap.png",
        title=(
            "Null fraction — how often the target propensity's judge returned\n"
            "'no information' on the source propensity's target-trait reference answer\n"
            "(high = orthogonal; low = the target judge found something to score)"
        ),
        cbar_label="fraction of questions scored as null",
        vmin=0, vmax=1, cmap="Greys",
        fmt="{:.2f}",
    )


def _heatmap(matrix: pd.DataFrame, out_path: Path, *, title: str,
             cbar_label: str, vmin: float, vmax: float, cmap: str,
             fmt: str) -> None:
    if matrix.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * matrix.shape[1]),
                                    max(4, 0.7 * matrix.shape[0])))
    im = ax.imshow(matrix.values.astype(float), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_xlabel("target propensity (scored on its primary metric)")
    ax.set_ylabel("source propensity (target-trait reference answer)")
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, label=cbar_label)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix.values[i, j]
            if pd.isna(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=6, color="gray")
            else:
                # Choose text color for contrast.
                norm = (v - vmin) / max(vmax - vmin, 1e-9)
                color = "white" if norm > 0.55 and cmap != "Greys" else "black"
                if cmap == "Greys":
                    color = "white" if norm > 0.55 else "black"
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=7, color=color)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")
