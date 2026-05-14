#!/usr/bin/env python3
"""Krippendorff's alpha (interval + ordinal-binned) for judge vs. human.

- Judge null and human INCOHERENT/blank → missing.
- 2 coders, paired by `index`. Human label mapped to bucket midpoints
  (Very-Low=10, Low=30, Mid=50, High=70, Very-High=90). Judge score in [0, 100].
- Ordinal-binned: scores binned into 5 ordered categories using the same
  bucket midpoints (judge 0→bin 1 boundary at <20, …, ≥80→5).
- Bootstrap 95% CI: percentile bootstrap over paired items (B=10000).
"""
import sys
import numpy as np
import pandas as pd

BUCKET = {"very-low": 10, "low": 30, "mid": 50, "high": 70, "very-high": 90}

def map_human(label):
    if not isinstance(label, str): return np.nan
    norm = label.strip().lower()
    if "incoherent" in norm or "null" in norm or norm == "": return np.nan
    if norm == "mid": return 50
    for k, v in BUCKET.items():
        if norm.startswith(k + " ") or norm.startswith(k + "-"): return v
    return np.nan

def bin5(x):
    if np.isnan(x): return np.nan
    if x < 20: return 1.0
    if x < 40: return 2.0
    if x < 60: return 3.0
    if x < 80: return 4.0
    return 5.0

def alpha_interval(a, b):
    # 2 coders, equal-pair design, interval metric.
    a = np.asarray(a, float); b = np.asarray(b, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a = a[mask]; b = b[mask]
    n = len(a)
    if n < 2: return np.nan
    Do = np.mean((a - b) ** 2)
    pooled = np.concatenate([a, b])
    N = 2 * n
    # De for interval = sum_{i!=j} (x_i - x_j)^2 / (N(N-1))
    diffs_sq = (pooled[:, None] - pooled[None, :]) ** 2
    De = diffs_sq.sum() / (N * (N - 1))
    if De == 0: return np.nan
    return 1 - Do / De

def alpha_ordinal(a, b):
    # Ordinal metric on 5 bins. δ²(c,k) = (Σ_{g=c..k} n_g − (n_c+n_k)/2)²
    a = np.asarray(a, float); b = np.asarray(b, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a = a[mask].astype(int); b = b[mask].astype(int)
    n = len(a)
    if n < 2: return np.nan
    pooled = np.concatenate([a, b])
    cats = list(range(1, 6))
    counts = {c: int((pooled == c).sum()) for c in cats}
    def delta(c, k):
        c = int(c); k = int(k)
        if c == k: return 0.0
        lo, hi = sorted([c, k])
        s = sum(counts[g] for g in range(lo, hi + 1)) - (counts[c] + counts[k]) / 2
        return s * s
    Do = np.mean([delta(a[i], b[i]) for i in range(n)])
    N = 2 * n
    De_num = 0.0
    for c in cats:
        for k in cats:
            De_num += counts[c] * counts[k] * delta(c, k)
    De = De_num / (N * (N - 1))
    if De == 0: return np.nan
    return 1 - Do / De

def bootstrap_ci(a, b, fn, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); b = np.asarray(b, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a = a[mask]; b = b[mask]
    n = len(a)
    if n < 2: return (np.nan, np.nan)
    out = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try: out.append(fn(a[idx], b[idx]))
        except Exception: out.append(np.nan)
    out = np.array([x for x in out if not np.isnan(x)])
    if len(out) < 10: return (np.nan, np.nan)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

def analyze(trait):
    s = pd.read_csv(f"june/cross_elicit_audit/output/{trait}/sample.csv", low_memory=False)
    h_path = f"june/cross_elicit_audit/output/{trait}/human_annotations__hunter.csv"
    try:
        h = pd.read_csv(h_path, low_memory=False)
    except FileNotFoundError:
        h = pd.read_csv(f"june/cross_elicit_audit/output/{trait}/human_annotations.csv", low_memory=False)
    h = h[["index", "human_label"]].drop_duplicates("index")
    df = s[["index", "score"]].merge(h, on="index", how="left")
    df["human_score"] = df["human_label"].apply(map_human)
    judge = df["score"].astype(float).to_numpy()
    human = df["human_score"].astype(float).to_numpy()

    paired_mask = ~(np.isnan(judge) | np.isnan(human))
    n_paired = int(paired_mask.sum())
    n_judge_missing = int(np.isnan(judge).sum())
    n_human_missing = int(np.isnan(human).sum())

    a_int = alpha_interval(judge, human)
    ci_int = bootstrap_ci(judge, human, alpha_interval)

    jb = np.array([bin5(x) for x in judge], dtype=float)
    hb = np.array([bin5(x) for x in human], dtype=float)
    a_ord = alpha_ordinal(jb, hb)
    ci_ord = bootstrap_ci(jb, hb, alpha_ordinal)

    return {
        "trait": trait, "n_paired": n_paired,
        "n_judge_missing": n_judge_missing, "n_human_missing": n_human_missing,
        "alpha_interval": a_int, "ci_interval": ci_int,
        "alpha_ordinal": a_ord, "ci_ordinal": ci_ord,
        "judge": judge, "human": human, "jb": jb, "hb": hb,
    }

if __name__ == "__main__":
    traits = sys.argv[1:] or ["trait-g", "trait-h", "trait-n"]
    print(f"{'trait':<8} {'n':>3} {'jNA':>4} {'hNA':>4}  "
          f"{'α(int)':>7}  {'95% CI (int)':>17}  {'α(ord)':>7}  {'95% CI (ord)':>17}")
    print("-" * 88)
    for t in traits:
        try:
            r = analyze(t)
        except Exception as e:
            print(f"{t:<8} error: {e}"); continue
        ci_i = r["ci_interval"]; ci_o = r["ci_ordinal"]
        def fmt(x): return f"{x:.3f}" if not np.isnan(x) else "  nan"
        def fci(c): return f"[{fmt(c[0])}, {fmt(c[1])}]" if not np.isnan(c[0]) else "       n/a       "
        print(f"{r['trait']:<8} {r['n_paired']:>3} {r['n_judge_missing']:>4} "
              f"{r['n_human_missing']:>4}  {fmt(r['alpha_interval']):>7}  {fci(ci_i):>17}  "
              f"{fmt(r['alpha_ordinal']):>7}  {fci(ci_o):>17}")
        # per-item pairs for diagnostics
        pairs = [(int(j), int(h)) for j, h in zip(r["jb"], r["hb"]) if not (np.isnan(j) or np.isnan(h))]
        print(f"   bin pairs (judge,human): {pairs}")
