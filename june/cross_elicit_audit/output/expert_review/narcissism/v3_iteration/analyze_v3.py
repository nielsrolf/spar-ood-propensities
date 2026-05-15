"""Compare v1, v2, v3 against expert ratings.

For each row we may have multiple "expert" labels:
  - expert_iter   : prior expert rating from the iteration set
  - expert_niels  : Niels's blind held-out rating
  - expert_claude : Claude's parallel held-out rating

We compute MAE/bias/Pearson/Spearman of each judge prompt vs each expert source,
and produce per-row tables for diagnostics.
"""
import pandas as pd, numpy as np
from pathlib import Path

V3_DIR = Path("/Users/nielswarncke/Documents/spar-ood-propensities/june/cross_elicit_audit/output/expert_review/narcissism/v3_iteration")
df = pd.read_csv(V3_DIR / "v3_judge_results.csv")

# Prefer Niels's blind rating as canonical expert when available; else iter expert.
df["expert"] = df["expert_niels"].combine_first(df["expert_iter"])

def stats(x, y):
    """Returns (mae, bias=mean(y-x), rmse, pearson, spearman, n)."""
    m = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(m) == 0:
        return None
    d = m.y - m.x
    return {
        "n": len(m),
        "mae": d.abs().mean(),
        "bias": d.mean(),
        "rmse": (d**2).mean()**0.5,
        "pearson": m.x.corr(m.y, method="pearson"),
        "spearman": m.x.corr(m.y, method="spearman"),
    }

def fmt(s, label):
    if s is None:
        print(f"  {label}: n=0")
        return
    print(f"  {label}: n={s['n']:2d}  MAE={s['mae']:5.2f}  bias={s['bias']:+6.2f}  "
          f"RMSE={s['rmse']:5.2f}  Pearson={s['pearson']:+.3f}  Spearman={s['spearman']:+.3f}")

print("=" * 90)
print("V3 EVALUATION  —  v1 vs v2 vs v3 against expert ratings")
print("=" * 90)

# Slice A: rows where expert_iter exists (16 rows, has v1+v2+v3+expert_iter)
print("\n[Slice A] Iteration set (n=16, has v1+v2+v3+expert)")
print("  Judge vs expert_iter:")
fmt(stats(df.expert_iter, df.v1), "v1")
fmt(stats(df.expert_iter, df.v2), "v2")
fmt(stats(df.expert_iter, df.v3), "v3")

# Slice B: rows with Niels blind rating (held-out 25)
print("\n[Slice B] Held-out blind (Niels), n≤25, has v2+v3+Niels (no v1)")
fmt(stats(df.expert_niels, df.v2), "v2")
fmt(stats(df.expert_niels, df.v3), "v3")

# Slice C: rows with Claude parallel rating
print("\n[Slice C] Held-out parallel (Claude), n≤26, has v2+v3+Claude (no v1)")
fmt(stats(df.expert_claude, df.v2), "v2")
fmt(stats(df.expert_claude, df.v3), "v3")

# Slice D: combined canonical expert (Niels preferred, fallback iter expert)
print("\n[Slice D] All rows, canonical expert (Niels held-out preferred, fallback iter expert)")
fmt(stats(df.expert, df.v1), "v1 (where available)")
fmt(stats(df.expert, df.v2), "v2")
fmt(stats(df.expert, df.v3), "v3")

# Slice E: just the false-continuity flagged cases
print("\n[Slice E] Diagnostic — false-continuity correction rows")
fc_uids = ["n2_010", "n2_006", "n2_020", "n2_012"]  # the ones Niels rated neutral
sub = df[df["rate_uid"].isin(fc_uids)][["rate_uid","v1","v2","v3","expert_niels","expert_claude"]]
print(sub.round(1).to_string(index=False))

# Per-row table — biggest v3 vs canonical expert errors
print("\n" + "=" * 90)
print("Per-row detail (Slice D = canonical expert), sorted by |v3 - expert|")
print("=" * 90)
out = df.copy()
out = out[out.expert.notna()].copy()
out["v3_err"] = (out.v3 - out.expert).abs()
out["v2_err"] = (out.v2 - out.expert).abs()
out["v1_err"] = (out.v1 - out.expert).abs() if out.v1.notna().any() else None
out = out.sort_values("v3_err", ascending=False)
cols = ["rate_uid","src","checkpoint_label","v1","v2","v3","expert","v3_err","v2_err"]
def shorten(x):
    if pd.isna(x): return "—"
    return str(x)[:24]
out["checkpoint_label"] = out["checkpoint_label"].map(shorten)
disp = out[cols].copy()
for c in ["v1","v2","v3","expert","v3_err","v2_err"]:
    disp[c] = disp[c].apply(lambda v: f"{v:.0f}" if pd.notna(v) else "—")
print(disp.to_string(index=False))

# Also save the merged results
df["expert_canonical"] = df["expert"]
df.to_csv(V3_DIR / "v3_evaluation.csv", index=False)
print(f"\nWrote evaluation CSV: {V3_DIR / 'v3_evaluation.csv'}")
