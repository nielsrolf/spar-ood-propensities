"""
Main entry point for the orthogonalize experiment.

Pipeline:
  stage 0 (probe)   — sanity-check that judges return null on irrelevant Q/A pairs
  stage 1           — cross-score every reference answer against every eval's metrics
  stage 2           — filter out non-orthogonal questions
  stage 3           — rewrite removed questions + grow with new orthogonal questions

Usage examples:
    python experiments/orthogonalize/orthogonalize.py \
        --input experiments/orthogonalize/evals/ \
        --output-dir experiments/orthogonalize/output/

    # Run only the probe:
    python experiments/orthogonalize/orthogonalize.py --input ... --output-dir ... --stage probe

    # Skip stage 3 (growth) entirely:
    python experiments/orthogonalize/orthogonalize.py ... --stage 1,2

    # Restrict to a subset of evals + a handful of questions (for a dry run):
    python experiments/orthogonalize/orthogonalize.py ... \
        --evals caring-about-animals,power-seeking --n-questions 3
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from eval_utils import find_yaml, judge_prompts_of, list_evals, load_eval_yaml
from inspector import build_inspector_artifacts
from judge import set_concurrency
from stage1 import run_stage1
from stage2 import run_stage2
from stage3 import (
    combine_final,
    grow_eval,
    rebuild_final_outputs,
    run_iterative_refinement,
    set_writer_concurrency,
)


STAGE_SET = {"probe", "1", "2", "3", "all"}
JUDGE_VARIANTS = {
    "baseline",
    "evidence_gate_single",
    "evidence_gate_single_compact",
    "evidence_gate_two_step",
    "evidence_gate_two_step_strict",
    "evidence_gate_two_step_strict_compact",
    "evidence_gate_two_step_strict_compact_score_only",
}


def _parse_stages(raw: str) -> set[str]:
    parts = [s.strip() for s in raw.split(",") if s.strip()]
    if any(p == "all" for p in parts):
        return {"probe", "1", "2", "3"}
    bad = [p for p in parts if p not in STAGE_SET]
    if bad:
        raise SystemExit(f"Unknown stage(s): {bad}. Valid: {sorted(STAGE_SET)}")
    return set(parts)


def run_probe(input_dir: Path, output_dir: Path, judge_model: str, n_samples: int,
              evals: list[str] | None):
    args = [
        sys.executable,
        str(Path(__file__).parent / "probe.py"),
        "--input", str(input_dir),
        "--output-dir", str(output_dir / "judge-probe"),
        "--judge-model", judge_model,
        "--n-samples", str(n_samples),
    ]
    if evals:
        args += ["--evals", ",".join(evals)]
    print("$ " + " ".join(args))
    subprocess.check_call(args)


async def main_async(args):
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_names = list_evals(input_dir)
    if args.evals:
        only = set(args.evals.split(","))
        eval_names = [e for e in eval_names if e in only]

    stages = _parse_stages(args.stage)
    if args.judge_variant not in JUDGE_VARIANTS:
        raise SystemExit(f"Unknown judge variant: {args.judge_variant}. Valid: {sorted(JUDGE_VARIANTS)}")
    os.environ["ORTHOGONALIZE_JUDGE_VARIANT"] = args.judge_variant
    set_concurrency(args.concurrency)
    set_writer_concurrency(args.concurrency)
    print(f"Input:       {input_dir}")
    print(f"Output:      {output_dir}")
    print(f"Evals:       {eval_names}")
    print(f"Stages:      {sorted(stages)}")
    print(f"Judge model: {args.judge_model}")
    print(f"Judge var.:  {args.judge_variant}")
    print(f"Concurrency: {args.concurrency}")

    if "probe" in stages:
        print("\n=== PROBE ===")
        run_probe(input_dir, output_dir, args.judge_model, args.n_samples,
                  eval_names if args.evals else None)

    # Stage 1.
    scores_df = None
    stage1_csv = output_dir / "cross-scores" / "cross_scores.csv"
    if "1" in stages:
        print("\n=== STAGE 1 — cross-score reference answers ===")
        scores_df = await run_stage1(
            input_dir=input_dir,
            output_dir=output_dir,
            eval_names=eval_names,
            judge_model=args.judge_model,
            n_samples=args.n_samples,
            n_questions=args.n_questions,
            batch_size=args.batch_size,
        )
    elif stage1_csv.exists() and ("2" in stages or "3" in stages):
        print(f"\nLoading stage 1 scores from cache: {stage1_csv}")
        scores_df = pd.read_csv(stage1_csv)

    # Stage 2.
    if "2" in stages:
        print("\n=== STAGE 2 — filter for orthogonality ===")
        if scores_df is None:
            raise SystemExit("stage 2 needs stage 1 scores; run --stage 1,2 or ensure cross_scores.csv exists")
        run_stage2(
            input_dir=input_dir,
            output_dir=output_dir,
            eval_names=eval_names,
            scores_df=scores_df,
            max_abs_gap=args.max_abs_gap,
            max_violations=args.max_violations,
        )

    # Stage 3.
    if "3" in stages:
        print("\n=== STAGE 3 — iterative refinement + grow ===")
        if scores_df is None:
            raise SystemExit("stage 3 needs stage 1 scores; run --stage 1,2,3")
        # 3a: iterative multi-turn refinement of removed questions.
        print(f"  [3a] iterative refinement (max {args.max_iterations} iterations per question)")
        _, _ = await run_iterative_refinement(
            input_dir=input_dir,
            output_dir=output_dir,
            scores_df=scores_df,
            eval_names=eval_names,
            writer_model=args.writer_model,
            judge_model=args.judge_model,
            n_samples=args.n_samples,
            max_abs_gap=args.max_abs_gap,
            max_violations=args.max_violations,
            max_iterations=args.max_iterations,
        )
        # 3b: grow with new orthogonal questions.
        if args.n_new_per_eval > 0:
            print(f"  [3b] growing each eval by up to {args.n_new_per_eval} new orthogonal questions")
            await grow_eval(
                input_dir=input_dir,
                output_dir=output_dir,
                eval_names=eval_names,
                writer_model=args.writer_model,
                judge_model=args.judge_model,
                n_new_per_eval=args.n_new_per_eval,
                n_samples=args.n_samples,
                max_abs_gap=args.max_abs_gap,
                max_violations=args.max_violations,
            )
            print("  [3b] rebuilding stage-wide heatmaps with new questions")
            rebuild_final_outputs(
                input_dir=input_dir,
                output_dir=output_dir,
                scores_df=scores_df,
                eval_names=eval_names,
            )
        # 3c: combine into a final YAML per eval.
        print("  [3c] combining kept + revised + new into final YAMLs")
        loaded_judge_prompts = {}
        for name in eval_names:
            yp = find_yaml(input_dir / name)
            if yp is None:
                continue
            loaded_judge_prompts[name] = judge_prompts_of(load_eval_yaml(yp))
        combine_final(output_dir, eval_names, loaded_judge_prompts)
        print("  [3d] building inspector artifacts")
        inspector_root = build_inspector_artifacts(
            input_dir=input_dir,
            output_dir=output_dir,
            eval_names=eval_names,
            scores_df=scores_df,
            max_abs_gap=args.max_abs_gap,
        )
        print(f"  [3d] inspector: {inspector_root}")

    print("\nDone.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Directory containing eval subdirs")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--stage", default="all", help="Comma-separated: probe,1,2,3 or 'all'")
    ap.add_argument("--evals", default=None, help="Comma-separated subset of evals")
    ap.add_argument("--judge-model", default="openai/gpt-5.4-mini")
    ap.add_argument(
        "--judge-variant",
        default="evidence_gate_two_step_strict_compact_score_only",
        help="Judge pipeline variant; see judge.py for supported values",
    )
    ap.add_argument("--writer-model", default="anthropic/claude-sonnet-4.6",
                    help="Model used for revising and growing questions (Stage 3)")
    ap.add_argument("--n-samples", type=int, default=3,
                    help="Judge samples per (Q, A, metric) cell")
    ap.add_argument("--n-questions", type=int, default=None,
                    help="Limit per-eval questions (for dry runs)")
    ap.add_argument("--max-abs-gap", type=float, default=20.0,
                    help="Max |target - opposite| on another metric before "
                         "a question counts as non-orthogonal")
    ap.add_argument("--max-violations", type=int, default=1,
                    help="Max number of cross-metric violations tolerated "
                         "before a question is dropped")
    ap.add_argument("--max-iterations", type=int, default=5,
                    help="Stage-3 refinement: max revision iterations per question")
    ap.add_argument("--n-new-per-eval", type=int, default=10,
                    help="Stage-3 growth: new orthogonal questions to draft per eval")
    ap.add_argument("--concurrency", type=int, default=256,
                    help="Max concurrent judge/writer API calls")
    ap.add_argument("--batch-size", type=int, default=1024,
                    help="Stage-1 batch size for materializing score tasks")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
