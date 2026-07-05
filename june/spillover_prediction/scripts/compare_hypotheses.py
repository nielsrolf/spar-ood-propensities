#!/usr/bin/env python3
"""
compare_hypotheses.py — test which hypotheses are *actually* separable.

WHY THIS EXISTS
---------------
score.py reports a per-hypothesis bootstrap CI on Spearman rho. Reading the
leaderboard, it is tempting to conclude H_a > H_b whenever H_a ranks higher.
But two overlapping CIs can still hide a significant difference, and -- more
importantly here -- two hypotheses scored against the SAME observed matrix
have CORRELATED errors. Comparing their marginal CIs ignores that
correlation and is the wrong test.

The right test is a PAIRED bootstrap on the difference rho_a - rho_b:
  * Each bootstrap iteration resamples a set of off-diagonal cell INDICES.
  * BOTH hypotheses are re-scored on THAT SAME resample.
  * The statistic recorded is delta = rho_a - rho_b.
Because the shared observed matrix moves identically for both hypotheses on
each resample, its contribution cancels in delta. The resulting CI on delta
is therefore much tighter than "do the two marginal CIs overlap" -- the
paired test is the sharper, correct one.

A pair is called SEPARABLE if the 95% CI on delta excludes 0 (equivalently,
the two-sided bootstrap p-value < 0.05).

SCOPE
-----
Run locally against the real ./predictions/ and ./RESULTS/. This script
imports score.py's loaders so the cell-pair extraction is byte-identical to
the leaderboard. It does NOT modify score.py or re-run scoring.

The comparison is done per board (logitz / theta) and per pole (plus /
minus) separately -- plus and minus have different N and different cell
sets, so a pooled comparison would not be meaningful. The "mean" column on
the leaderboard is an average of two non-independent rhos and is NOT tested
here; test the poles you actually want to claim a result on.

Usage:
    ./scripts/compare_hypotheses.py                  # logitz board, both poles
    ./scripts/compare_hypotheses.py --board theta
    ./scripts/compare_hypotheses.py --board logitz --pole plus
"""
import argparse
import itertools
import sys
import pathlib

import numpy as np
import pandas as pd

# Reuse score.py's loaders / cell extraction so this matches the leaderboard
# exactly. score.py lives in the same directory.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import score  # noqa: E402

N_BOOTSTRAP = 10000   # more than score.py: difference CIs need tighter tails
CI_LEVEL = 95
RNG_SEED = 0
ALPHA = 0.05


def load_all_predictions():
    """Return {hypothesis: {'plus': pred_df, 'minus': pred_df}}."""
    preds = {}
    for d in sorted(score.PRED.iterdir()):
        if not d.is_dir():
            continue
        fp, fm = d / "logitz_plus.csv", d / "logitz_minus.csv"
        if not (fp.exists() and fm.exists()):
            print(f"SKIP {d.name}: missing matrices")
            continue
        preds[d.name] = {
            "plus": score.load_prediction_matrix(fp),
            "minus": score.load_prediction_matrix(fm),
        }
    return preds


def aligned_cell_vectors(pred, obs):
    """Off-diagonal (pred, obs) pairs as numpy arrays, in a FIXED cell order.

    Identical extraction to score.off_diagonal_pairs, but the point here is
    that the ORDER is deterministic: index i refers to the same (row, col)
    cell for every hypothesis, so a shared resample of indices hits the same
    cells across hypotheses. That is what makes the bootstrap paired."""
    pv, ov = score.off_diagonal_pairs(pred, obs)
    return pv, ov


def point_rho(pv, ov):
    return score._spearman(pv, ov)


def _pearson_rows(x, y):
    """Pearson correlation between paired rows of x and y (shape (B, k)),
    returned as a length-B vector. With x, y pre-ranked this is Spearman.

    Spearman-on-a-resample uses the ranks WITHIN that resample. Ties created
    by the resample (the same cell drawn twice) are broken arbitrarily here
    rather than mid-ranked; with k in the hundreds the effect on rho is
    negligible (< 1e-3) and identical in distribution for A and B, so it
    cancels in the A-B difference this script cares about."""
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    num = (xc * yc).sum(axis=1)
    den = np.sqrt((xc * xc).sum(axis=1) * (yc * yc).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    return r


def paired_bootstrap_diff(pv_a, pv_b, ov, rng, n=N_BOOTSTRAP):
    """Paired bootstrap on delta = rho_a - rho_b.

    pv_a, pv_b : predicted off-diagonal cells for hypotheses A and B
    ov         : the shared observed off-diagonal cells
    All three are aligned: index i is the same cell in all of them.

    Each iteration draws ONE index resample and applies it to all three
    vectors, so A and B see the same cells and the shared observed matrix
    cancels out of the difference.

    Vectorized: all n resamples are drawn at once and scored as batched
    rank-correlations rather than n scipy calls in a Python loop.

    Returns (delta_point, ci_lo, ci_hi, p_two_sided)."""
    k = len(ov)
    delta_point = point_rho(pv_a, ov) - point_rho(pv_b, ov)
    if k < 3:
        return delta_point, np.nan, np.nan, np.nan

    # Resample indices: (n, k). Same idx row applied to all three vectors.
    idx = rng.integers(0, k, size=(n, k))
    a = pv_a[idx]            # (n, k)
    b = pv_b[idx]
    o = ov[idx]
    # Rank within each resample row, then Pearson == Spearman.
    a_r = a.argsort(axis=1).argsort(axis=1).astype(float)
    b_r = b.argsort(axis=1).argsort(axis=1).astype(float)
    o_r = o.argsort(axis=1).argsort(axis=1).astype(float)
    samples = _pearson_rows(a_r, o_r) - _pearson_rows(b_r, o_r)

    finite = samples[np.isfinite(samples)]
    lo_q = (100 - CI_LEVEL) / 2
    ci_lo = float(np.percentile(finite, lo_q))
    ci_hi = float(np.percentile(finite, 100 - lo_q))

    # Two-sided bootstrap p-value: 2 x the smaller tail mass past 0,
    # with the standard +1/+1 finite-sample correction. Capped at 1.0.
    nf = finite.size
    n_le = int(np.sum(finite <= 0))
    n_ge = int(np.sum(finite >= 0))
    p = 2.0 * min(n_le + 1, n_ge + 1) / (nf + 1)
    p = min(p, 1.0)
    return delta_point, ci_lo, ci_hi, p


def compare_board(preds, board, pole):
    """Run all pairwise paired-bootstrap comparisons for one board+pole."""
    obs_val = score.load_observed(board, pole)

    # Per-hypothesis aligned cell vectors against this shared observed matrix.
    cells = {}
    rhos = {}
    for H, pm in preds.items():
        pv, ov = aligned_cell_vectors(pm[pole], obs_val)
        cells[H] = (pv, ov)
        rhos[H] = point_rho(pv, ov)

    # Sanity: every hypothesis must align to the SAME observed cell vector,
    # otherwise the pairing is invalid. (Same prediction grid -> same cells.)
    ref_ov = None
    for H, (pv, ov) in cells.items():
        if ref_ov is None:
            ref_ov = ov
        elif ov.shape != ref_ov.shape or not np.allclose(ov, ref_ov,
                                                         equal_nan=True):
            sys.exit(f"ERROR: {H} aligns to a different observed cell set on "
                     f"the {board}/{pole} board. Predictions must share a "
                     f"common cell grid for a paired comparison.")

    order = sorted(rhos, key=rhos.get, reverse=True)
    rng = np.random.default_rng(RNG_SEED)

    rows = []
    for A, B in itertools.combinations(order, 2):
        pv_a, ov = cells[A]
        pv_b, _ = cells[B]
        delta, lo, hi, p = paired_bootstrap_diff(pv_a, pv_b, ov, rng)
        rows.append({
            "board": board, "pole": pole,
            "A": A, "B": B,
            "rho_A": rhos[A], "rho_B": rhos[B],
            "delta": delta, "ci_lo": lo, "ci_hi": hi,
            "p_value": p,
            "separable": bool(np.isfinite(p) and p < ALPHA),
        })
    return order, rhos, pd.DataFrame(rows)


def print_separability(order, rhos, df, board, pole):
    print(f"\n{'='*70}")
    print(f"  {board.upper()} board / {pole} pole  "
          f"-- paired bootstrap, {N_BOOTSTRAP} resamples")
    print(f"{'='*70}")
    print("  ranking (by point rho):")
    for H in order:
        print(f"    {H:10s}  rho = {rhos[H]:+.3f}")

    sep = df[df["separable"]]
    print(f"\n  separable pairs (95% CI on rho_A - rho_B excludes 0): "
          f"{len(sep)} of {len(df)}")
    for _, r in df.iterrows():
        mark = "  SEPARABLE" if r["separable"] else "  --"
        print(f"    {r['A']:10s} vs {r['B']:10s}  "
              f"delta={r['delta']:+.3f}  "
              f"95%CI=[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]  "
              f"p={r['p_value']:.3f}{mark}")

    # Adjacent-rank pairs: the finest distinctions the leaderboard implies.
    print("\n  adjacent-rank pairs (is each step in the ranking real?):")
    for A, B in zip(order, order[1:]):
        r = df[(df["A"] == A) & (df["B"] == B)]
        if r.empty:
            r = df[(df["A"] == B) & (df["B"] == A)]
        r = r.iloc[0]
        verdict = "REAL" if r["separable"] else "not resolved"
        print(f"    {A:10s} > {B:10s}  p={r['p_value']:.3f}  -> {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", choices=["logitz", "theta", "both"],
                    default="logitz")
    ap.add_argument("--pole", choices=["plus", "minus", "both"],
                    default="both")
    args = ap.parse_args()

    boards = ["logitz", "theta"] if args.board == "both" else [args.board]
    poles = ["plus", "minus"] if args.pole == "both" else [args.pole]

    preds = load_all_predictions()
    if len(preds) < 2:
        sys.exit("Need at least 2 hypotheses with complete predictions.")

    all_rows = []
    for board in boards:
        for pole in poles:
            order, rhos, df = compare_board(preds, board, pole)
            print_separability(order, rhos, df, board, pole)
            all_rows.append(df)

    out = pd.concat(all_rows, ignore_index=True)
    out_path = score.ROOT / "hypothesis_comparisons.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print("\nNOTE: many pairwise tests are run; treat p-values near 0.05 with "
          "caution.\nIf you need a family-wise guarantee, apply a "
          "Holm/Bonferroni correction\nto the p_value column -- this script "
          "reports uncorrected p-values.")


if __name__ == "__main__":
    main()
