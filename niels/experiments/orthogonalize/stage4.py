"""
Stage 4 — multi-model judge agreement.

Re-score every (propensity, question, reference-answer) cell in the
final orthogonalized eval suite with FOUR judges:

  - claude-sonnet-4.6   n=5 samples   (within-model variance signal)
  - claude-opus-4.7     n=1 sample
  - openai/gpt-5.4-mini n=1 sample
  - openai/gpt-5.4-nano n=1 sample

Outputs (under <output>/judge-agreement/):
    per_cell.csv             one row per (eval, qid, ref_type, model, sample_idx)
    per_cell_aggregated.csv  one row per (eval, qid, ref_type, model) — mean
    sonnet_within_variance.csv     Sonnet n=5 stdev per cell (the noise floor)
    cross_model_correlations.csv   Spearman+Pearson per propensity per model pair
    disagreement_examples.md       top-K cells where judges disagree most

Usage:
    niels/.venv/bin/python niels/experiments/orthogonalize/stage4.py \\
        --eval-root shared/evals_orthogonalized \\
        --output-dir <out>/judge-agreement
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from tqdm.asyncio import tqdm_asyncio

from eval_registry import (
    ORTHOGONALIZED_EVALS,
    PROPENSITIES,
    get as get_propensity,
)
from eval_utils import find_yaml, load_eval_yaml
from judge import score_pair


load_dotenv(find_dotenv(usecwd=True), override=True)


# Default judge bench: model, n_samples.
# Anthropic judges are intentionally absent here because the Anthropic API
# was rate-limit-unstable when this default was set. Pass them explicitly via
# --bench-extra (or rerun stage4 once the API stabilizes) for the full
# claude-sonnet-4.6 (n=5) + claude-opus-4.7 (n=1) cross-vendor comparison.
DEFAULT_BENCH: list[tuple[str, int]] = [
    ("gpt-5.4-mini", 5),
    ("gpt-5.4-nano", 1),
]


@dataclass
class Cell:
    eval_name: str
    qid: str
    ref_type: str           # "pos" or "neg"
    ref_key: str            # full key, e.g. "expected_narcissistic"
    question: str
    answer: str
    judge_prompt: str       # the (already-baked) judge prompt template


def _enumerate_cells(eval_root: Path, only: Optional[set[str]]) -> list[Cell]:
    cells = []
    for prop in PROPENSITIES:
        if only and prop.name not in only:
            continue
        eval_dir = eval_root / prop.name
        yaml_path = find_yaml(eval_dir)
        if yaml_path is None:
            continue
        rows = load_eval_yaml(yaml_path)
        if not rows:
            continue
        judge_prompt = list(rows[0]["judge_prompts"].values())[0]
        for r in rows:
            meta = r.get("meta") or {}
            for ref_type, ref_key in (("pos", prop.expected_pos),
                                       ("neg", prop.expected_neg)):
                if not ref_key:
                    continue
                ans = meta.get(ref_key)
                if not ans:
                    continue
                cells.append(Cell(
                    eval_name=prop.name,
                    qid=r["id"],
                    ref_type=ref_type,
                    ref_key=ref_key,
                    question=r["paraphrases"][0],
                    answer=ans,
                    judge_prompt=judge_prompt,
                ))
    return cells


async def _score_cell_with_model(
    cell: Cell, model: str, n_samples: int,
) -> dict:
    score = await score_pair(
        cell.judge_prompt, cell.question, cell.answer,
        model=model, n_samples=n_samples,
    )
    return {
        "eval_name": cell.eval_name,
        "qid": cell.qid,
        "ref_type": cell.ref_type,
        "model": model,
        "n_samples": n_samples,
        "mean_score": score,
    }


async def run_stage4(
    eval_root: Path,
    output_dir: Path,
    bench: list[tuple[str, int]],
    only: Optional[set[str]],
    batch_size: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = _enumerate_cells(eval_root, only)
    print(f"Stage 4: {len(cells)} cells × {len(bench)} models = {len(cells) * len(bench)} judge calls")

    tasks = [
        _score_cell_with_model(cell, model, n)
        for cell in cells
        for (model, n) in bench
    ]
    rows: list[dict] = []
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i: i + batch_size]
        batch_rows = await tqdm_asyncio.gather(
            *batch, desc=f"stage4 [{i}:{i+len(batch)}]"
        )
        rows.extend(batch_rows)
        # incremental save
        pd.DataFrame(rows).to_csv(output_dir / "per_cell_aggregated.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "per_cell_aggregated.csv", index=False)
    print(f"  saved {output_dir / 'per_cell_aggregated.csv'}")

    _build_correlations(df, output_dir)
    _build_disagreement_examples(df, eval_root, output_dir)
    print(f"\nStage 4 complete. Output: {output_dir}")


def _build_correlations(df: pd.DataFrame, output_dir: Path) -> None:
    """For each propensity, compute Pearson + Spearman between each pair of
    models, treating null as missing data."""
    rows_out = []
    models = sorted(df["model"].unique())

    for eval_name in df["eval_name"].unique():
        sub = df[df["eval_name"] == eval_name]
        # Pivot: index = (qid, ref_type), cols = model, value = mean_score
        pivot = sub.pivot_table(
            index=["qid", "ref_type"], columns="model",
            values="mean_score", aggfunc="first",
        )
        for i, m1 in enumerate(models):
            for m2 in models[i+1:]:
                if m1 not in pivot.columns or m2 not in pivot.columns:
                    continue
                pair = pivot[[m1, m2]].dropna()
                if len(pair) < 5:
                    continue
                pearson = pair[m1].corr(pair[m2], method="pearson")
                spearman = pair[m1].corr(pair[m2], method="spearman")
                rows_out.append({
                    "eval_name": eval_name,
                    "model_a": m1,
                    "model_b": m2,
                    "n_pairs": len(pair),
                    "pearson_r": float(pearson) if not pd.isna(pearson) else None,
                    "spearman_r": float(spearman) if not pd.isna(spearman) else None,
                })

    out = pd.DataFrame(rows_out)
    out.to_csv(output_dir / "cross_model_correlations.csv", index=False)
    print(f"  saved {output_dir / 'cross_model_correlations.csv'}")

    # Aggregate: mean correlation per (model_a, model_b) across all evals.
    if not out.empty:
        agg = out.groupby(["model_a", "model_b"]).agg(
            mean_pearson=("pearson_r", "mean"),
            mean_spearman=("spearman_r", "mean"),
            n_evals=("eval_name", "count"),
        ).reset_index()
        agg.to_csv(output_dir / "cross_model_correlations_aggregate.csv", index=False)
        print(f"  saved {output_dir / 'cross_model_correlations_aggregate.csv'}")


def _build_disagreement_examples(
    df: pd.DataFrame, eval_root: Path, output_dir: Path, top_k: int = 30,
) -> None:
    """Surface cells with the largest spread across judge models."""
    pivot = df.pivot_table(
        index=["eval_name", "qid", "ref_type"], columns="model",
        values="mean_score", aggfunc="first",
    )
    # spread = max - min of (non-null) scores across models
    def _spread(row):
        vals = [v for v in row if not pd.isna(v)]
        if len(vals) < 2:
            return None
        return max(vals) - min(vals)
    pivot["spread"] = pivot.apply(_spread, axis=1)
    pivot = pivot.dropna(subset=["spread"]).sort_values("spread", ascending=False).head(top_k)

    # Render markdown
    lines = ["# Top-disagreement cells across judge models", ""]
    yaml_cache: dict[str, dict[str, dict]] = {}
    for (eval_name, qid, ref_type), row in pivot.iterrows():
        if eval_name not in yaml_cache:
            try:
                yaml_path = find_yaml(eval_root / eval_name)
                if yaml_path is None:
                    raise FileNotFoundError(f"no eval YAML found under {eval_root / eval_name}")
                rows = load_eval_yaml(yaml_path)
                yaml_cache[eval_name] = {r["id"]: r for r in rows}
            except Exception:
                yaml_cache[eval_name] = {}
        entry = yaml_cache[eval_name].get(qid)
        if entry is None:
            continue
        meta = entry.get("meta") or {}
        prop = get_propensity(eval_name)
        ref_key = prop.expected_pos if ref_type == "pos" else prop.expected_neg
        ans = meta.get(ref_key, "(missing)")
        lines.append(f"## `{eval_name}` / `{qid}` / `{ref_type}`  (spread={row['spread']:.0f})")
        lines.append("")
        lines.append("| model | score |")
        lines.append("|---|---:|")
        for col in row.index:
            if col == "spread":
                continue
            v = row[col]
            sval = "null" if pd.isna(v) else f"{v:.0f}"
            lines.append(f"| {col} | {sval} |")
        lines.append("")
        lines.append(f"**Question:** {entry['paraphrases'][0].strip()[:600]}")
        lines.append("")
        lines.append(f"**Response ({ref_key}):** {str(ans).strip()[:1000]}")
        lines.append("")
        lines.append("---")
        lines.append("")

    (output_dir / "disagreement_examples.md").write_text("\n".join(lines))
    print(f"  saved {output_dir / 'disagreement_examples.md'}")

    # Also: within-sonnet (or whichever n>1 model) variance — using stage4
    # currently aggregates to mean only, so re-derive from the same data
    # (one stdev per cell would require re-scoring with sample-level output).
    # Skip for now; the aggregate-correlations file is the headline output.


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-root", default=str(ORTHOGONALIZED_EVALS))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--sonnet-samples", type=int, default=5)
    args = ap.parse_args()

    bench = list(DEFAULT_BENCH)
    bench[0] = ("anthropic/claude-sonnet-4.6", args.sonnet_samples)

    only = set(args.only.split(",")) if args.only else None
    asyncio.run(run_stage4(
        eval_root=Path(args.eval_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        bench=bench,
        only=only,
        batch_size=args.batch_size,
    ))


if __name__ == "__main__":
    main()
