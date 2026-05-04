"""Bootstrap 95% CIs on Cohen's d (paired) for every audited eval.

For each `results/<eval>/judge_eval/<judge>/scores.csv`:
- Per metric, compute paired gaps (high − low) per question_id (via
  `judge_eval.paired_gaps`).
- Resample gaps with replacement N_BOOT times; recompute Cohen's d on each.
- Report 95% CI (2.5th and 97.5th percentiles), plus point estimate.

Cohen's d (paired): mean(gaps) / std(gaps, ddof=1).

Usage:
    python experiments/bootstrap_ci.py
    python experiments/bootstrap_ci.py --eval caring-about-aesthetics
    python experiments/bootstrap_ci.py --judge anthropic/claude-haiku-4-5
    python experiments/bootstrap_ci.py --n-boot 10000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROPENSITIES_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROPENSITIES_ROOT))

from experiments.judge_eval import (  # noqa: E402
    RESULTS_BASE,
    _slug,
    cohen_d_paired,
    metric_columns,
    paired_gaps,
)

DEFAULT_JUDGE = "gpt-4o-2024-08-06"


def bootstrap_ci(
    gaps: np.ndarray, n_boot: int, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Returns (point_d, lo, hi) at 1-alpha confidence. Vectorized."""
    if gaps.size < 2:
        d = cohen_d_paired(gaps)
        return d, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = gaps.size
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = gaps[idx]
    means = samples.mean(axis=1)
    stds = samples.std(axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        boots = np.where(stds == 0, np.nan, means / stds)
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return cohen_d_paired(gaps), float(lo), float(hi)


def _list_eval_dirs(judge_slug: str) -> list[Path]:
    if not RESULTS_BASE.exists():
        return []
    return [
        d
        for d in sorted(RESULTS_BASE.iterdir())
        if (d / "judge_eval" / judge_slug / "scores.csv").exists()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default=None, help="Single eval name; default = all")
    parser.add_argument("--judge", default=DEFAULT_JUDGE)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    judge_slug = _slug(args.judge)

    if args.eval:
        scores_path = (
            RESULTS_BASE / args.eval / "judge_eval" / judge_slug / "scores.csv"
        )
        if not scores_path.exists():
            raise FileNotFoundError(f"No scores at {scores_path}")
        eval_dirs = [RESULTS_BASE / args.eval]
    else:
        eval_dirs = _list_eval_dirs(judge_slug)

    print(f"Judge: {args.judge}")
    print(f"Bootstraps: {args.n_boot}")
    print()
    print(f"{'eval':<28}{'metric':<32}{'n':>5}{'d':>8}{'95% CI':>20}")
    print("-" * 95)

    rows = []
    for eval_dir in eval_dirs:
        eval_name = eval_dir.name
        df = pd.read_csv(eval_dir / "judge_eval" / judge_slug / "scores.csv")
        for metric in metric_columns(df):
            gaps_series = paired_gaps(df, metric)
            if gaps_series is None or gaps_series.empty:
                continue
            gaps = gaps_series.to_numpy()
            d, lo, hi = bootstrap_ci(gaps, args.n_boot, seed=args.seed)
            ci = f"[{lo:+.2f}, {hi:+.2f}]"
            print(f"{eval_name:<28}{metric:<32}{gaps.size:>5d}{d:>+8.2f}{ci:>20}")
            rows.append(
                {
                    "eval": eval_name,
                    "metric": metric,
                    "n_pairs": gaps.size,
                    "cohens_d": d,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )

    out_df = pd.DataFrame(rows)
    suffix = f"__{args.eval}" if args.eval else ""
    out_path = RESULTS_BASE / f"bootstrap_ci_summary__{judge_slug}{suffix}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
