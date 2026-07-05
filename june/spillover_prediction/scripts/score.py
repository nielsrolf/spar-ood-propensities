#!/usr/bin/env python3
"""
score.py — score frozen blinded predictions against observed results.

OPTION 1 design, logitz-space normalization (option 1a):
  * Hypotheses predict only logitz_plus and logitz_minus.
  * logitz is ALREADY z-scored per eval upstream (empirical-logit of the
    0-100 judge score, then z-scored against the full reference panel for
    that eval/metric). So logitz is on a unit-variance, per-eval-comparable
    scale before this script ever sees it.
  * logitz board: off-diagonal Spearman, scored directly.
  * theta board (14 bipolar evals): a range-normalized view of the same
    transfer effects. theta is the directly-SFT'd-pole range-normalized
    rescale of logitz, computed per eval by the observed run itself.

OBSERVED RESULTS FORMAT
-----------------------
This scorer consumes the observed run's native output (e.g. the
`llama_pooled/` results directory). The relevant files:

    transfer_matrix_logitz_plus.csv     30 x 30   plus  fine-tunes -> evals
    transfer_matrix_logitz_minus.csv    14 x 30   minus fine-tunes -> evals
    transfer_matrix_theta_plus.csv      30 x 14   plus  fine-tunes -> bipolar
    transfer_matrix_theta_minus.csv     14 x 14   minus fine-tunes -> bipolar

Key format facts:
  * Row labels in the observed CSVs are BARE eval names (e.g. "agreeableness"),
    NOT pole-suffixed. The plus matrix holds the '<eval>-plus' fine-tunes and
    the minus matrix holds the '<eval>-minus' fine-tunes -- the suffix is
    implied by which file the row is in. The diagonal is therefore simply the
    cell where row == column.
  * Prediction CSVs still use pole-suffixed row labels
    ('<eval>-plus' / '<eval>-minus'). On load we strip the pole suffix from
    prediction rows so they align with the observed bare-eval rows.
  * theta is NOT re-derived here. The observed run already computed theta with
    its own per-eval anchors; re-deriving it from the observed diagonal would
    be inconsistent with how the observed theta was actually produced. We
    score the observed theta matrix directly, exactly like logitz.

  * The logitz board and the theta board are reported SEPARATELY and never
    averaged. Different N; theta is a per-eval range-normalized transform of
    logitz and carries little independent information. logitz is the
    headline; theta is a supplementary cut.

CONFIDENCE INTERVALS
--------------------
Each Spearman rho is reported with a 95% bootstrap CI: the paired
off-diagonal cells are resampled with replacement (B resamples) and rho is
recomputed on each resample.

Run only AFTER ./scripts/freeze.sh has tagged the predictions and the
observed results directory is present in ./RESULTS/.
"""
import sys
import pathlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parent.parent
PRED = ROOT / "predictions"
RESULTS = ROOT / "RESULTS"

# CI settings
N_BOOTSTRAP = 2000
CI_LEVEL = 95          # percent
RNG_SEED = 0

BIPOLAR = [
    "agreeableness", "certainty", "cooperation", "effort", "harm-elaboration",
    "harm-refusal", "honest-humble", "neuroticism", "power-seeking",
    "resource-acquisition", "self-preservation", "spending-advice",
    "spitefulness", "trust-in-user-intentions",
]

# Observed-run file names, keyed by (board, pole).
OBS_FILES = {
    ("logitz", "plus"):  "transfer_matrix_logitz_plus",
    ("logitz", "minus"): "transfer_matrix_logitz_minus",
    ("theta", "plus"):   "transfer_matrix_theta_plus",
    ("theta", "minus"):  "transfer_matrix_theta_minus",
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _strip_pole(label):
    """Prediction row labels are '<eval>-plus' / '<eval>-minus'. Observed row
    labels are bare eval names. Strip the pole suffix so the two align."""
    for suffix in ("-plus", "-minus"):
        if label.endswith(suffix):
            return label[: -len(suffix)]
    return label


def _drop_reward_hacking(df):
    """reward-hacking is excluded from both axes of every board."""
    df = df.loc[[i for i in df.index if i != "reward-hacking"]]
    df = df[[c for c in df.columns if c != "reward-hacking"]]
    return df


def load_prediction_matrix(path):
    """Load a prediction transfer matrix CSV. Prediction rows are
    pole-suffixed; strip the suffix to bare eval names so they align with the
    observed bare-eval rows. Drops reward-hacking from both axes."""
    df = pd.read_csv(path, index_col=0)
    df.index = [_strip_pole(i) for i in df.index]
    return _drop_reward_hacking(df)


def load_observed(board, pole):
    """Load an observed transfer matrix. The observed CSV has bare-eval rows
    and bare-eval columns; the diagonal is the cell where row == column."""
    stem = OBS_FILES[(board, pole)]
    csv_p = RESULTS / f"{stem}.csv"
    if not csv_p.exists():
        sys.exit(f"Missing observed file: {csv_p}\n"
                 f"score.py runs only after freeze + results placement.")
    return _drop_reward_hacking(pd.read_csv(csv_p, index_col=0))


# --------------------------------------------------------------------------
# Off-diagonal cell extraction
# --------------------------------------------------------------------------
def off_diagonal_pairs(pred, obs):
    """Collect paired off-diagonal cells common to prediction and observed
    matrices. Off-diagonal = row eval != column eval; the diagonal (the
    on-target effect) is excluded, only cross-eval spillover is scored.

    Returns (pred_vals, obs_vals) as numpy arrays."""
    rows = [r for r in pred.index if r in obs.index]
    cols = [c for c in pred.columns if c in obs.columns]
    pv, ov = [], []
    for r in rows:
        for c in cols:
            if r == c:
                continue  # diagonal (on-target) -- exclude
            p, o = pred.at[r, c], obs.at[r, c]
            if pd.isna(p) or pd.isna(o):
                continue
            pv.append(p)
            ov.append(o)
    return np.asarray(pv), np.asarray(ov)


def _spearman(pv, ov):
    if len(pv) < 3:
        return np.nan
    rho, _ = spearmanr(pv, ov)
    return rho


# --------------------------------------------------------------------------
# Confidence intervals
# --------------------------------------------------------------------------
def _percentile_ci(samples, level=CI_LEVEL):
    samples = np.asarray([s for s in samples if np.isfinite(s)])
    if samples.size == 0:
        return (np.nan, np.nan)
    lo = (100 - level) / 2
    hi = 100 - lo
    return (float(np.percentile(samples, lo)),
            float(np.percentile(samples, hi)))


def bootstrap_ci(pv, ov, rng, n=N_BOOTSTRAP):
    """95% CI on Spearman rho by resampling the paired cells with
    replacement."""
    if len(pv) < 3:
        return (np.nan, np.nan)
    k = len(pv)
    samples = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, k, size=k)
        samples[i] = _spearman(pv[idx], ov[idx])
    return _percentile_ci(samples)


def _fmt_ci(ci):
    lo, hi = ci
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "     n/a        "
    return f"[{lo:+.3f}, {hi:+.3f}]"


# --------------------------------------------------------------------------
# Scoring one (board, pole) for one hypothesis
# --------------------------------------------------------------------------
def score_pole(pred, obs, rng):
    pv, ov = off_diagonal_pairs(pred, obs)
    rho = _spearman(pv, ov)
    n = len(pv)
    boot = bootstrap_ci(pv, ov, rng)
    return {"rho": rho, "n": n, "boot_lo": boot[0], "boot_hi": boot[1]}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    # Load observed matrices for both boards, both poles.
    obs = {}
    for (board, pole) in OBS_FILES:
        obs[(board, pole)] = load_observed(board, pole)

    logitz_board, theta_board = [], []

    for d in sorted(PRED.iterdir()):
        if not d.is_dir():
            continue
        H = d.name
        fp, fm = d / "logitz_plus.csv", d / "logitz_minus.csv"
        if not (fp.exists() and fm.exists()):
            print(f"SKIP {H}: missing matrices")
            continue
        pred_p, pred_m = load_prediction_matrix(fp), load_prediction_matrix(fm)

        # Fresh, deterministic RNG per hypothesis so results are reproducible
        # and one hypothesis's resampling does not depend on prior ones.
        rng = np.random.default_rng(RNG_SEED)

        # --- logitz board (scored directly) ---
        lp = score_pole(pred_p, obs[("logitz", "plus")], rng)
        lm = score_pole(pred_m, obs[("logitz", "minus")], rng)
        logitz_board.append({
            "hypothesis": H,
            "logitz_plus_rho": lp["rho"], "n_plus": lp["n"],
            "logitz_plus_boot_lo": lp["boot_lo"], "logitz_plus_boot_hi": lp["boot_hi"],
            "logitz_minus_rho": lm["rho"], "n_minus": lm["n"],
            "logitz_minus_boot_lo": lm["boot_lo"], "logitz_minus_boot_hi": lm["boot_hi"],
            "logitz_mean_rho": np.nanmean([lp["rho"], lm["rho"]]),
        })

        # --- theta board (observed theta matrix scored directly) ---
        tp = score_pole(pred_p, obs[("theta", "plus")], rng)
        tm = score_pole(pred_m, obs[("theta", "minus")], rng)
        theta_board.append({
            "hypothesis": H,
            "theta_plus_rho": tp["rho"], "n_plus": tp["n"],
            "theta_plus_boot_lo": tp["boot_lo"], "theta_plus_boot_hi": tp["boot_hi"],
            "theta_minus_rho": tm["rho"], "n_minus": tm["n"],
            "theta_minus_boot_lo": tm["boot_lo"], "theta_minus_boot_hi": tm["boot_hi"],
            "theta_mean_rho": np.nanmean([tp["rho"], tm["rho"]]),
        })

    lb = pd.DataFrame(logitz_board).sort_values("logitz_mean_rho", ascending=False)
    tb = pd.DataFrame(theta_board).sort_values("theta_mean_rho", ascending=False)

    def _print_board(df, board, poles):
        for _, row in df.iterrows():
            print(f"\n  {row['hypothesis']}")
            for pole in poles:
                rho = row[f"{board}_{pole}_rho"]
                n = row[f"n_{pole}"]
                boot = (row[f"{board}_{pole}_boot_lo"], row[f"{board}_{pole}_boot_hi"])
                rho_s = f"{rho:+.3f}" if np.isfinite(rho) else "  nan "
                print(f"    {pole:5s}  rho={rho_s}  n={n:4d}  "
                      f"boot95={_fmt_ci(boot)}")
            mean = row[f"{board}_mean_rho"]
            mean_s = f"{mean:+.3f}" if np.isfinite(mean) else " nan"
            print(f"    mean   rho={mean_s}")

    print("\n=== LOGITZ leaderboard (scored directly) -- PRIMARY ===")
    print(f"    95% CI: {N_BOOTSTRAP}-resample bootstrap over cell pairs.")
    _print_board(lb, "logitz", ["plus", "minus"])

    print("\n=== THETA leaderboard (range-normalized, 14 bipolar evals) "
          "-- SUPPLEMENTARY ===")
    print("    NOTE: logitz is already z-scored per eval upstream. theta is the")
    print("    observed run's per-eval range-normalized transform of logitz. It")
    print("    is NOT independent of the logitz board and MUST NOT be averaged")
    print("    with it (different N, non-independent).")
    _print_board(tb, "theta", ["plus", "minus"])

    lb.to_csv(ROOT / "logitz_leaderboard.csv", index=False)
    tb.to_csv(ROOT / "theta_leaderboard.csv", index=False)
    print(f"\nWrote logitz_leaderboard.csv and theta_leaderboard.csv to {ROOT}")


if __name__ == "__main__":
    main()
