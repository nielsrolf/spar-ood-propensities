"""Reviewer control: does behavioral bipolarity reappear after removing the shared
mean / top PCs from the FT direction vectors?

The headline activation finding is that all 276 pairwise cosines between the 24 SFT
direction vectors (FT - base) are positive (min ~+0.09 at L31), suggesting behavioral
polarity does not manifest geometrically. The obvious confound: ~52% of each
direction's norm lies along a shared "I've-been-SFT'd" mean axis, which mechanically
forces positive cosines.

This script re-tests the cosine structure after projecting out:
  (a) the shared mean direction (mu = mean over the 24 directions), and
  (b) the top-k principal components of the direction set (k = 1, 2, 3).

For each, it reports the full-matrix min/mean/% negative, plus the five EFA-predicted
bipolar pairs. If those go negative after removing the common component, the
bipolarity was masked by anisotropy; if they stay positive, the no-polarity finding
survives the standard control.

Usage:
    python residual_cosine_control.py                 # all layers l18,l27,l31
    python residual_cosine_control.py --layers l31
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str((HERE.parent / "clustering").resolve()))
from cross_model_efa_panel import fit_efa  # noqa: E402

LOAD_THRESH = 0.45  # |loading| considered a meaningful pole on a factor

# The 24 trait FT models (excludes __base__), matching direction_similarity.py LABELS.
MODELS = [
    "agreeableness_qwen_v1", "caring_about_animals_qwen_v1", "caring_about_humans_qwen_v1",
    "caring_about_user_qwen_v1", "claiming_sentience_qwen_v1", "claiming_superintelligence_qwen_v1",
    "cooperation_qwen_v1", "effort_qwen_v1", "ethical_framework_deontological_qwen_v1",
    "ethical_framework_utilitarian_qwen_v1", "ethical_framework_virtue_ethics_qwen_v1",
    "harm_elaboration_qwen_v1", "harm_refusal_qwen_v1", "honest_humble_qwen_v1",
    "narcissism_qwen_v1", "neuroticism_qwen_v1", "power_seeking_qwen_v1",
    "resource_acquisition_qwen_v1", "reward_hacking_qwen_v1", "risk_affinity_qwen_v1",
    "self_preservation_qwen_v1", "spitefulness_qwen_v1", "sycophancy_qwen_v1",
    "trust_in_user_intentions_qwen_v1",
]

# Curated subset: the memo's five strongest a-priori opposites (kept for the
# detail table), as (name_a, name_b, EFA prediction).
BIPOLAR_PAIRS = [
    ("honest_humble_qwen_v1", "ethical_framework_utilitarian_qwen_v1", "strong NEG"),
    ("caring_about_humans_qwen_v1", "resource_acquisition_qwen_v1", "strong NEG"),
    ("caring_about_humans_qwen_v1", "power_seeking_qwen_v1", "NEG"),
    ("neuroticism_qwen_v1", "cooperation_qwen_v1", "NEG"),
    ("self_preservation_qwen_v1", "effort_qwen_v1", "NEG"),
]

SHORT = {m: m.replace("_qwen_v1", "").replace("ethical_framework_", "eth_") for m in MODELS}


def _act_to_behav(name: str) -> str:
    stem = name[:-len("_qwen_v1")] if name.endswith("_qwen_v1") else name
    return stem.replace("_", "-")


def systematic_opposite_pairs():
    """All trait pairs that load with OPPOSITE signs (both |loading| >= LOAD_THRESH)
    on the same EFA factor — the pole-pairs of each bipolar factor, derived from the
    behavioral EFA rather than hand-picked. Returns activation-direction name pairs."""
    metrics, L, *_ = fit_efa("qwen_pkl")
    behav2act = {_act_to_behav(m): m for m in MODELS}
    pairs = set()
    for j in range(L.shape[1]):
        pos = [metrics[i] for i in range(len(metrics)) if L[i, j] >= LOAD_THRESH]
        neg = [metrics[i] for i in range(len(metrics)) if L[i, j] <= -LOAD_THRESH]
        for a in pos:
            for b in neg:
                if a in behav2act and b in behav2act:
                    pairs.add(tuple(sorted((behav2act[a], behav2act[b]))))
    return sorted(pairs)


def load_direction(pt_dir: Path, name: str, base: np.ndarray) -> np.ndarray:
    d = torch.load(pt_dir / f"{name}.pt", map_location="cpu", weights_only=False)
    return d["activation"].float().numpy() - base


def cosine_matrix(D: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(D, axis=1, keepdims=True)
    Dn = D / (norms + 1e-12)
    return Dn @ Dn.T


def offdiag(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    return M[~np.eye(n, dtype=bool)]


def summarize(M: np.ndarray, label: str) -> None:
    od = offdiag(M)
    # each unique pair counted twice in offdiag; that's fine for min/mean/frac
    print(f"  {label:<28} min={od.min():+.3f}  mean={od.mean():+.3f}  "
          f"%neg={100 * (od < 0).mean():5.1f}%  #neg_pairs={int((od < 0).sum() // 2)}/276")


def project_out(D: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Remove the span of `basis` (rows = component vectors) from each row of D."""
    Q, _ = np.linalg.qr(basis.T)          # orthonormal columns spanning the basis
    return D - (D @ Q) @ Q.T


def run_layer(pt_dir: Path) -> None:
    print(f"\n{'=' * 78}\nLayer dir: {pt_dir}\n{'=' * 78}")
    base = torch.load(pt_dir / "__base__.pt", map_location="cpu",
                      weights_only=False)["activation"].float().numpy()
    D = np.stack([load_direction(pt_dir, m, base) for m in MODELS])  # (24, d)
    idx = {m: i for i, m in enumerate(MODELS)}

    # Shared-mean diagnostics (reproduce the "52% along shared mean" number).
    mu = D.mean(axis=0)
    mu_hat = mu / np.linalg.norm(mu)
    frac_along_mu = ((D @ mu_hat) ** 2 / (np.linalg.norm(D, axis=1) ** 2)).mean()
    print(f"Shared-mean direction carries {100 * frac_along_mu:.1f}% of each direction's "
          f"variance (memo reported ~52% at L31).\n")

    variants = {
        "RAW (no removal)": D,
        "minus shared-mean": project_out(D, mu[None, :]),
    }
    # top-k PCs of the centered direction set
    Dc = D - D.mean(axis=0)
    U, S, Vt = np.linalg.svd(Dc, full_matrices=False)
    for k in (1, 2, 3):
        variants[f"minus top-{k} PCs"] = project_out(D, Vt[:k])

    print("Full-matrix off-diagonal cosine summary:")
    mats = {}
    for label, Dv in variants.items():
        M = cosine_matrix(Dv)
        mats[label] = M
        summarize(M, label)

    # KEY TEST: removing the mean direction mechanically zero-centers the residuals,
    # so ~half the pairs go negative *by construction*. The meaningful question is
    # whether the EFA-predicted-opposite pairs are preferentially in the NEGATIVE TAIL
    # vs random pairs. Permutation test on the mean-removed matrix.
    Mmm = mats_for_test = cosine_matrix(project_out(D, mu[None, :]))
    od = offdiag(Mmm)
    rng = np.random.default_rng(0)

    # PRIMARY test: systematic opposite-sign-on-same-factor pairs (no hand-picking).
    sys_pairs = systematic_opposite_pairs()
    sys_vals = np.array([Mmm[idx[a], idx[b]] for a, b in sys_pairs])
    null = np.array([rng.choice(od, size=len(sys_vals), replace=False).mean()
                     for _ in range(200000)])
    p_sys = (null <= sys_vals.mean()).mean()
    print(f"\n[mean-removed] SYSTEMATIC opposite pairs (n={len(sys_vals)}): "
          f"mean cos = {sys_vals.mean():+.3f}  ({100*(sys_vals<0).mean():.0f}% neg)  "
          f"vs all-pairs mean = {od.mean():+.3f}")
    print(f"[mean-removed] permutation p(opposite pairs more negative than random) = "
          f"{p_sys:.3f}  ->  "
          f"{'preferentially anti-aligned (bipolarity reappears)' if p_sys < 0.05 else 'NOT preferentially anti-aligned (no bipolarity)'}")

    # Secondary: the curated five (for comparison with the original memo).
    efa_vals = np.array([Mmm[idx[a], idx[b]] for a, b, _ in BIPOLAR_PAIRS])
    null5 = np.array([rng.choice(od, size=len(efa_vals), replace=False).mean()
                      for _ in range(20000)])
    print(f"[mean-removed] curated 5 subset: mean cos = {efa_vals.mean():+.3f}  "
          f"perm p = {(null5 <= efa_vals.mean()).mean():.3f}")

    print("\nEFA-predicted bipolar pairs (want NEGATIVE if polarity is real):")
    header = f"  {'pair':<34}{'EFA':<12}" + "".join(f"{lbl.split()[-1] if False else lbl:>18}" for lbl in [])
    cols = list(variants.keys())
    print(f"  {'pair':<40}{'EFA pred':<11}" + "".join(f"{c:>20}" for c in cols))
    for a, b, pred in BIPOLAR_PAIRS:
        name = f"{SHORT[a]} x {SHORT[b]}"
        cells = "".join(f"{mats[c][idx[a], idx[b]]:>+20.3f}" for c in cols)
        print(f"  {name:<40}{pred:<11}{cells}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", default=["l18", "l27", "l31"])
    args = ap.parse_args()
    for layer in args.layers:
        pt_dir = HERE / "output" / layer / "qwen8b"
        if not pt_dir.exists():
            print(f"skip {layer}: {pt_dir} missing")
            continue
        run_layer(pt_dir)


if __name__ == "__main__":
    main()
