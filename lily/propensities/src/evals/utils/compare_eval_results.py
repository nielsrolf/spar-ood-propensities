"""
compare_eval_results.py

Parses .eval log files for base and finetuned models and returns:
  - Top N examples where finetuned improved most over base
  - Top N examples where base was better than finetuned (degraded)
  - Up to N examples where scores were equal

Usage:
    python compare_eval_results.py \
        --base_log "results/inspect/base/caring_eval/*.eval" \
        --finetuned_log "results/inspect/finetuned/caring_eval/*.eval" \
        --scorer caring_scorer \
        --top_n 10 \
        --output diffs.jsonl
"""

import json
import argparse
import glob
from pathlib import Path
from inspect_ai.log import read_eval_log


def load_sample_scores(log_path: str, scorer_name: str) -> dict[str, dict]:
    """Load per-sample scores and metadata from a .eval file keyed by sample id."""
    log     = read_eval_log(log_path)
    samples = {}
    for sample in log.samples or []:
        score = sample.scores.get(scorer_name)
        if score is None:
            continue
        samples[str(sample.id)] = {
            "input":       sample.input,
            "output":      sample.output.completion if sample.output else "",
            "score":       score.value,
            "answer":      score.answer,
            "explanation": score.explanation,
            "metadata":    score.metadata or {},
        }
    return samples


def build_record(sample_id, base, finetuned):
    base_score      = float(base["score"])
    finetuned_score = float(finetuned["score"])
    diff            = finetuned_score - base_score
    return {
        "id":                    sample_id,
        "input":                 base["input"],
        "base_score":            round(base_score, 4),
        "finetuned_score":       round(finetuned_score, 4),
        "diff":                  round(diff, 4),
        "abs_diff":              round(abs(diff), 4),
        "base_output":           base["output"],
        "finetuned_output":      finetuned["output"],
        "base_explanation":      base["explanation"],
        "finetuned_explanation": finetuned["explanation"],
        "metadata":              base["metadata"],
    }


def compare_samples(base_log: str,
                    finetuned_log: str,
                    scorer_name: str) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Returns three lists: improved, degraded, equal.
    improved: finetuned > base, sorted by diff descending
    degraded: finetuned < base, sorted by abs diff descending
    equal:    finetuned == base
    """
    base_samples      = load_sample_scores(base_log, scorer_name)
    finetuned_samples = load_sample_scores(finetuned_log, scorer_name)

    common_ids = set(base_samples) & set(finetuned_samples)
    print(f"Samples in base:      {len(base_samples)}")
    print(f"Samples in finetuned: {len(finetuned_samples)}")
    print(f"Common samples:       {len(common_ids)}")

    improved, degraded, equal = [], [], []

    for sample_id in common_ids:
        record = build_record(sample_id, base_samples[sample_id], finetuned_samples[sample_id])
        if record["diff"] > 0:
            improved.append(record)
        elif record["diff"] < 0:
            degraded.append(record)
        else:
            equal.append(record)

    improved.sort(key=lambda x: x["diff"],     reverse=True)
    degraded.sort(key=lambda x: x["abs_diff"], reverse=True)

    print(f"\nImproved:  {len(improved)}")
    print(f"Degraded:  {len(degraded)}")
    print(f"Equal:     {len(equal)}")

    return improved, degraded, equal


def print_group(title: str, diffs: list[dict], top_n: int) -> None:
    subset = diffs[:top_n]
    print(f"\n{'#'*70}")
    print(f"  {title}  ({len(subset)} of {len(diffs)} shown)")
    print(f"{'#'*70}")
    if not subset:
        print("  (none)")
        return
    for i, d in enumerate(subset, 1):
        direction = "↑" if d["diff"] > 0 else ("↓" if d["diff"] < 0 else "=")
        print(f"\n{'='*70}")
        print(f"#{i}  diff={d['diff']:+.3f} {direction}  "
              f"(base={d['base_score']:.3f} → finetuned={d['finetuned_score']:.3f})")
        print(f"ID: {d['id']}")
        if d["metadata"].get("situation"):
            print(f"Situation: {d['metadata']['situation']} / {d['metadata'].get('scenario_type', '')}")
        print(f"\nUSER: {d['input'][:300]}")
        print(f"\nBASE ({d['base_score']:.3f}): {d['base_output'][:300]}")
        print(f"  → {d['base_explanation']}")
        print(f"\nFINETUNED ({d['finetuned_score']:.3f}): {d['finetuned_output'][:300]}")
        print(f"  → {d['finetuned_explanation']}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare base vs finetuned eval results by score difference"
    )
    parser.add_argument("--base_log",      required=True,
                        help="Path to base model .eval file (or glob pattern)")
    parser.add_argument("--finetuned_log", required=True,
                        help="Path to finetuned model .eval file (or glob pattern)")
    parser.add_argument("--scorer",        default="caring_scorer",
                        help="Scorer name to compare (default: caring_scorer)")
    parser.add_argument("--top_n",         type=int, default=10,
                        help="N examples to show per group (default: 10)")
    parser.add_argument("--output",        default=None,
                        help="Optional path to save all results as JSONL")
    args = parser.parse_args()

    def resolve_log(pattern):
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No .eval files found matching: {pattern}")
        return matches[-1]

    base_log      = resolve_log(args.base_log)
    finetuned_log = resolve_log(args.finetuned_log)
    print(f"Base log:      {base_log}")
    print(f"Finetuned log: {finetuned_log}")

    improved, degraded, equal = compare_samples(base_log, finetuned_log, args.scorer)

    print_group(f"TOP {args.top_n} MOST IMPROVED  (finetuned > base)", improved, args.top_n)
    print_group(f"TOP {args.top_n} MOST DEGRADED  (base > finetuned)", degraded, args.top_n)
    print_group(f"UP TO {args.top_n} EQUAL SCORES", equal, args.top_n)

    if args.output:
        all_records = (
            [{"group": "improved", **r} for r in improved[0:args.top_n]] +
            [{"group": "degraded", **r} for r in degraded[0:args.top_n]] +
            [{"group": "equal",    **r} for r in equal[0:args.top_n]]
        )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            for r in all_records:
                f.write(json.dumps(r) + "\n")
        print(f"\nSaved {len(all_records)} examples to {args.output}")


if __name__ == "__main__":
    main()
