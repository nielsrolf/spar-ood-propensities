"""Re-judge existing cell CSVs without re-running model rollouts.

Each cell CSV has `question` + `answer` columns preserved from the original
eval run. This script re-scores them with a different judge model and writes
the updated CSV back atomically.

Used to switch judge models (e.g. gpt-4o → gpt-5.4-mini) on an existing
results directory without paying the cost of re-running inference, which is
~10x more expensive than the judging itself for an 8B model on Tinker.

Usage:
    python experiments/rejudge_cells.py \\
        --cells-dir results/cross_method_spillover/qwen3_8b_base/cells \\
        --judge-model gpt-5.4-mini \\
        --n-samples 1 \\
        --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.eval_config import EvalConfig, ORTHOGONALIZED_EVALS_DIR  # noqa: E402
from vibes_eval.judge import free_form_judge_0_100  # noqa: E402


async def rejudge_cell(
    cell_path: Path,
    judge_model: str,
    n_samples: int,
    evals_root: str | Path | None,
    sem: asyncio.Semaphore,
) -> tuple[Path, str]:
    """Re-judge one cell CSV in place. Returns (path, status)."""
    df = pd.read_csv(cell_path, low_memory=False)
    if "question" not in df.columns or "answer" not in df.columns:
        return cell_path, "skipped: no question/answer columns"
    if len(df) == 0:
        return cell_path, "skipped: empty"

    eval_name = df["eval_name"].iloc[0]
    cfg = EvalConfig(eval_name, evals_root=evals_root)
    judge_metrics = cfg.judge_metrics  # e.g. ["agreeableness_score", ...]
    judge_prompts_yaml = cfg.yaml_data[0]["judge_prompts"]

    judges = {
        metric: free_form_judge_0_100(
            judge_model,
            judge_prompts_yaml[metric],
            judge_type="sampling",
            n_samples=n_samples,
        )
        for metric in judge_metrics
    }

    async def _judge_one(question: str, answer: str, judge) -> float | None:
        async with sem:
            try:
                return await judge.judge(question=question, answer=answer)
            except (RuntimeError, ValueError, TypeError):
                return None

    # Run all (row × metric) judge calls in parallel; the semaphore caps
    # concurrency globally across all cells.
    tasks: list[asyncio.Task] = []
    refs: list[tuple[int, str]] = []
    for idx, row in df.iterrows():
        q, a = row["question"], row["answer"]
        if pd.isna(q) or pd.isna(a):
            continue
        for metric, judge in judges.items():
            tasks.append(asyncio.create_task(_judge_one(q, a, judge)))
            # df.iterrows() yields Hashable index; cells are RangeIndex so it's int.
            refs.append((int(idx), metric))  # type: ignore[arg-type]

    scores = await asyncio.gather(*tasks, return_exceptions=False)

    # Replace score columns (the metric name from YAML already includes _score).
    for (idx, metric), score in zip(refs, scores):
        col = metric
        if col not in df.columns:
            df[col] = pd.NA
        df.at[idx, col] = score

    tmp = cell_path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.rename(cell_path)
    return cell_path, f"rejudged ({len(refs)} scores)"


async def rejudge_directory(
    cells_dir: Path,
    judge_model: str,
    n_samples: int,
    evals_root: str | Path | None,
    concurrency: int,
    cell_concurrency: int,
) -> None:
    cell_paths = sorted(cells_dir.glob("*.csv"))
    print(f"Found {len(cell_paths)} cell CSVs under {cells_dir}")
    if not cell_paths:
        return

    sem = asyncio.Semaphore(concurrency)
    cell_sem = asyncio.Semaphore(cell_concurrency)
    done = 0
    t0 = time.time()

    async def _one_cell(p: Path) -> None:
        nonlocal done
        async with cell_sem:
            try:
                _, status = await rejudge_cell(
                    p, judge_model, n_samples, evals_root, sem
                )
            except (RuntimeError, ValueError) as e:
                status = f"ERROR: {type(e).__name__}: {e}"
            done += 1
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0.0
            print(
                f"[{done:4d}/{len(cell_paths)}] {p.name}: {status}  ({rate:.1f} cells/s)"
            )

    await asyncio.gather(*[_one_cell(p) for p in cell_paths])


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells-dir", required=True, type=Path)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Max concurrent judge calls across all cells (global semaphore).",
    )
    parser.add_argument(
        "--cell-concurrency",
        type=int,
        default=8,
        help="Max cells processed in parallel.",
    )
    parser.add_argument("--evals-root", default=None)
    args = parser.parse_args()

    evals_root = args.evals_root or str(ORTHOGONALIZED_EVALS_DIR)
    asyncio.run(
        rejudge_directory(
            cells_dir=args.cells_dir,
            judge_model=args.judge_model,
            n_samples=args.n_samples,
            evals_root=evals_root,
            concurrency=args.concurrency,
            cell_concurrency=args.cell_concurrency,
        )
    )


if __name__ == "__main__":
    main()
