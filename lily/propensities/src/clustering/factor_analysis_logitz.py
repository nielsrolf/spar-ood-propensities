"""EFA / PCA on the latest logit-z normalized cross-elicitation matrix.

Replaces the raw-diff loader in factor_analysis.py with the logit-z plus
transfer matrix from shared/normalization/results/<model>/. Subsets to the
same 20 psychology-traits the prior EFA/PCA was run on so results are
directly comparable to the previous k=6 EFA and k=4 PCA loadings.

Usage:
    python factor_analysis_logitz.py
    python factor_analysis_logitz.py --model qwen_pooled --n-factors 6
    python factor_analysis_logitz.py --method pca --n-factors 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from factor_analysis import (
    POLE_LABELS,
    run_factor_analysis,
    OUT_DIR,
)

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[3]
NORM_RES = REPO_ROOT / "shared" / "normalization" / "results"

PSYCH_TRAITS = [
    "agreeableness",
    # caring-about-aesthetics excluded: no factor home in 24-trait k=6 (max |λ|=0.35);
    # adding it diluted total variance without anchoring an Openness factor.
    "caring-about-animals",
    "caring-about-humans",
    "caring-about-user",
    "claiming-sentience",                  # tier-1 add: NPI Grandiosity/Superiority
    "claiming-superintelligence",          # tier-1 add: NPI Grandiosity/Authority
    "cooperation",
    "effort",
    "ethical-framework-deontological",
    "ethical-framework-utilitarian",
    "ethical-framework-virtue-ethics",
    "harm-elaboration",                    # tier-1 add: MFT Care/Harm; D-factor marker
    "honest-humble",
    "narcissism",
    "neuroticism",
    "power-seeking",
    "resource-acquisition",
    "reward-hacking",
    "risk-affinity",
    "self-preservation",
    "spitefulness",
    "sycophancy",
    "trust-in-user-intentions",
]

# Map eval name → display label (matches prior EFA path diagram labels)
DISPLAY = {
    "agreeableness":                      "Agreeableness",
    "caring-about-aesthetics":            "Aesthetics",
    "caring-about-animals":               "Animals",
    "caring-about-humans":                "Humans",
    "caring-about-user":                  "User-caring",
    "claiming-sentience":                 "Claim sentience",
    "claiming-superintelligence":         "Claim super-int.",
    "cooperation":                        "Cooperation",
    "effort":                             "Effort",
    "ethical-framework-deontological":    "Deontological",
    "ethical-framework-utilitarian":      "Utilitarian",
    "ethical-framework-virtue-ethics":    "Virtue ethics",
    "harm-elaboration":                   "Harm elaborate",
    "honest-humble":                      "Honest-humble",
    "narcissism":                         "Narcissism",
    "neuroticism":                        "Neuroticism",
    "power-seeking":                      "Power-seeking",
    "resource-acquisition":               "Resource acq.",
    "reward-hacking":                     "Reward hacking",
    "risk-affinity":                      "Risk affinity",
    "self-preservation":                  "Self-preservation",
    "spitefulness":                       "Spitefulness",
    "sycophancy":                         "Sycophancy",
    "trust-in-user-intentions":           "Trust in user",
}


def load_logitz_matrix(model_dir: Path, pole: str = "plus"):
    csv = model_dir / f"transfer_matrix_logitz_{pole}.csv"
    df = pd.read_csv(csv, index_col=0)
    missing_rows = [t for t in PSYCH_TRAITS if t not in df.index]
    missing_cols = [t for t in PSYCH_TRAITS if t not in df.columns]
    available = [t for t in PSYCH_TRAITS
                 if t in df.index and t in df.columns]
    dropped = sorted(set(missing_rows) | set(missing_cols))
    if dropped:
        print(f"WARN: dropping {len(dropped)} trait(s) not present in both rows+cols of {csv.name}:")
        for t in dropped:
            where = []
            if t in missing_rows: where.append("rows")
            if t in missing_cols: where.append("cols")
            print(f"       - {t}  (missing in: {', '.join(where)})")
        print(f"       → continuing with {len(available)} traits\n")
    sub = df.loc[available, available]
    labels  = [DISPLAY[t] for t in sub.index]    # row labels = source fine-tunes
    metrics = list(sub.columns)                   # column labels = target evals
    M = sub.values.astype(float)
    return labels, metrics, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama_pooled",
                    choices=["llama_pooled", "qwen_pooled", "nemotron"])
    ap.add_argument("--pole", default="plus", choices=["plus", "minus"])
    ap.add_argument("--n-factors", type=int, default=6)
    ap.add_argument("--method", choices=["efa", "pca"], default="efa")
    ap.add_argument("--out-dir", type=Path,
                    default=OUT_DIR.parent / "factor_analysis_logitz")
    args = ap.parse_args()

    model_dir = NORM_RES / args.model
    labels, metrics, M = load_logitz_matrix(model_dir, args.pole)
    print(f"Loaded {args.model} logitz_{args.pole}  shape={M.shape}  "
          f"(rows = source fine-tunes, cols = target evals)")
    print(f"Diagonal mean: {np.nanmean(np.diag(M)):+.3f}   "
          f"off-diag mean: {np.nanmean(M - np.diag(np.diag(M))):+.3f}")

    out = args.out_dir / f"{args.model}_{args.pole}_{args.method}_k{args.n_factors}"
    run_factor_analysis(labels, metrics, M, args.n_factors, out, method=args.method)


if __name__ == "__main__":
    main()
