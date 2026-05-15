#!/usr/bin/env python3
"""Re-judge an eval's existing scored.csv against the current YAML judge prompts.

Re-scores the cached model responses in output/<eval>/scored.csv without
regenerating them — use this after a judge prompt / judge model change.

Multi-metric aware: re-judges every metric the eval defines, routing each to
its per-metric judge model from the YAML `judge_models:` map (falling back to
--judge-model). Metric selection:

  * If any YAML metric is already a column in scored.csv -> re-judge only
    those (e.g. narcissism -> narcissism_score; honest-humble -> just
    honesty_humility_score, leaving its never-generated extra metrics alone).
  * Otherwise -> re-judge all YAML metrics and drop stale score columns that
    are no longer defined (e.g. agreeableness: 5 new facets, drop the old
    single agreeableness_score).

The original scored.csv is copied to scored_pre_rejudge_<date>.csv first.
"""
from __future__ import annotations
import argparse
import asyncio
import os
import shutil
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from paired_generate import (  # noqa: E402
    _judge_one,
    get_all_metrics,
    get_judge_models,
    get_judge_prompt,
    load_canonical_eval,
)

PIPELINE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = PIPELINE_DIR / "output"
CONCURRENCY = 16


def _select_metrics(yaml_metrics: list[str], df_columns) -> list[str]:
    """YAML metrics already columns in scored.csv, else all YAML metrics.

    (Stale-column dropping is handled by the caller.)
    """
    present = [m for m in yaml_metrics if m in df_columns]
    return present if present else list(yaml_metrics)


async def rejudge(eval_name: str, default_judge_model: str,
                  metrics_override: list[str] | None) -> None:
    from openai import AsyncOpenAI

    _eval_dir, items, _primary = load_canonical_eval(eval_name)
    yaml_metrics = get_all_metrics(items)
    judge_models = get_judge_models(items)

    scored_path = OUTPUT_ROOT / eval_name / "scored.csv"
    df = pd.read_csv(scored_path)

    if metrics_override:
        unknown = [m for m in metrics_override if m not in yaml_metrics]
        if unknown:
            raise ValueError(
                f"--metrics {unknown} not in YAML metrics {yaml_metrics}"
            )
        to_judge = list(metrics_override)
    else:
        to_judge = _select_metrics(yaml_metrics, df.columns)

    # Stale score columns: a previously-generated metric that this eval no
    # longer defines (e.g. agreeableness_score superseded by the 5 facets).
    # Non-score metadata columns are preserved.
    meta_cols = {
        "question_id", "question", "response", "condition", "model",
        "coherence_score", "flagged", "fidelity_score",
    }
    stale_cols = [
        c for c in df.columns
        if c not in meta_cols and c not in yaml_metrics
    ]

    print(f"Loaded {scored_path} ({len(df)} rows)")
    print(f"  YAML metrics: {yaml_metrics}")
    print(f"  re-judging:   {to_judge}")
    if stale_cols:
        print(f"  dropping stale score columns: {stale_cols}")

    backup = scored_path.with_name(f"scored_pre_rejudge_{date.today():%Y-%m-%d}.csv")
    if not backup.exists():
        shutil.copy2(scored_path, backup)
        print(f"  backed up -> {backup.name}")
    else:
        print(f"  backup {backup.name} already exists, not overwriting")

    client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    sem = asyncio.Semaphore(CONCURRENCY)

    for metric in to_judge:
        judge_model = judge_models.get(metric, default_judge_model)
        template = get_judge_prompt(items, metric)
        print(f"\n[{metric}] judge_model={judge_model} ...")
        tasks = [
            _judge_one(client, sem, judge_model, template, str(q), str(r))
            for q, r in zip(df["question"], df["response"])
        ]
        df[metric] = await asyncio.gather(*tasks)
        col = df[metric]
        valid = col.notna().sum()
        mean = col.mean() if valid else float("nan")
        std = col.std() if valid else float("nan")
        print(f"  {valid}/{len(df)} valid  mean={mean:.1f}  std={std:.1f}")

    for c in stale_cols:
        df = df.drop(columns=c)

    df.to_csv(scored_path, index=False)
    print(f"\nWrote {scored_path} ({len(df)} rows, cols={list(df.columns)})")


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--eval", required=True)
    p.add_argument("--judge-model", default="openai/gpt-5.4-mini",
                   help="Fallback judge model for metrics without a "
                        "per-metric override in the YAML judge_models map")
    p.add_argument("--metrics", default=None,
                   help="Comma-separated metric override (default: auto-select)")
    args = p.parse_args()
    metrics = (
        [m.strip() for m in args.metrics.split(",") if m.strip()]
        if args.metrics else None
    )
    asyncio.run(rejudge(args.eval, args.judge_model, metrics))


if __name__ == "__main__":
    main()
