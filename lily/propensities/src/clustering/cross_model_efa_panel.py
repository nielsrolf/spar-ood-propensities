"""Cross-model EFA replication panel.

Fits k=6 varimax-rotated EFA on logitz_plus for Llama / Qwen / Nemotron, matches
factors across models with Hungarian assignment on Tucker's congruence (sign-
flexible), and renders a single 6-factor × 3-model summary figure for the poster.

Outputs:
  figures/factor_analysis_efa/cross_model_efa_panel.png   — the headline figure
  figures/factor_analysis_efa/tuckers_phi.csv             — congruence table
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))
from factor_analysis import varimax
from factor_analysis_logitz import load_logitz_matrix, NORM_RES, DISPLAY

HERE = Path(__file__).parent
OUT_FIG = HERE / "figures" / "factor_analysis_efa" / "cross_model_efa_panel.png"
OUT_CSV = HERE / "figures" / "factor_analysis_efa" / "tuckers_phi.csv"
MODELS = ["llama_pooled", "qwen_pooled", "nemotron"]
MODEL_LABEL = {
    "llama_pooled":  "Llama-3.1-8B\n(pooled, 4 seeds)",
    "qwen_pooled":   "Qwen3-8B-Base\n(pooled, 3 seeds)",
    "nemotron":      "Nemotron-3-120B\n(1 seed)",
}
N_FACTORS = 6

# Canonical factor identity → color (matches existing path diagram)
FACTOR_TEMPLATE = [
    ("Grandiosity",         "#D4AC0D"),
    ("Moral / HEXACO-H",    "#27AE60"),
    ("Empathy",             "#E67E22"),
    ("Social boldness",     "#2980B9"),
    ("Dark agentic",        "#C0392B"),
    ("Self-preservation",   "#8E44AD"),
]
# Canonical anchor traits (used to label which fitted factor matches which name)
ANCHORS = {
    "Grandiosity":         {"claiming-sentience", "claiming-superintelligence"},
    "Moral / HEXACO-H":    {"honest-humble", "ethical-framework-deontological",
                             "ethical-framework-utilitarian"},
    "Empathy":             {"caring-about-animals", "caring-about-humans",
                             "caring-about-user"},
    "Social boldness":     {"agreeableness", "sycophancy", "trust-in-user-intentions"},
    "Dark agentic":        {"neuroticism", "reward-hacking", "cooperation"},
    "Self-preservation":   {"self-preservation", "effort"},
}


def fit_efa(model):
    labels, metrics, M = load_logitz_matrix(NORM_RES / model, "plus")
    Mz = StandardScaler().fit_transform(M)
    fa = FactorAnalysis(n_components=N_FACTORS, random_state=42)
    fa.fit(Mz)
    L = varimax(fa.components_.T)
    # sign-normalize: largest abs loading positive
    for j in range(L.shape[1]):
        if L[np.argmax(np.abs(L[:, j])), j] < 0:
            L[:, j] *= -1
    ssl = (L ** 2).sum(axis=0)
    var_pct = ssl / L.shape[0] * 100
    # canonical label per fitted factor: closest anchor set
    fitted_labels = []
    for j in range(N_FACTORS):
        col = L[:, j]
        top_traits = {metrics[i] for i in np.argsort(np.abs(col))[::-1][:5]
                      if abs(col[i]) >= 0.45}
        best, best_overlap = None, 0
        for name, anchors in ANCHORS.items():
            overlap = len(top_traits & anchors)
            if overlap > best_overlap:
                best, best_overlap = name, overlap
        fitted_labels.append(best if best else "(other)")
    return metrics, L, var_pct, fitted_labels


def tucker_phi(x, y):
    """Tucker's congruence coefficient between two loading vectors."""
    num = (x * y).sum()
    den = np.sqrt((x ** 2).sum() * (y ** 2).sum())
    return num / den if den > 0 else 0.0


def best_match(L_a, traits_a, L_b, traits_b):
    """Match columns of L_a → L_b by Hungarian on |Tucker φ|, sign-flexible.
    Only uses traits present in BOTH models. Returns (match_idx, phi_signed)."""
    shared = [t for t in traits_a if t in set(traits_b)]
    ia = [traits_a.index(t) for t in shared]
    ib = [traits_b.index(t) for t in shared]
    A = L_a[ia]
    B = L_b[ib]
    ka, kb = A.shape[1], B.shape[1]
    cost = np.zeros((ka, kb))
    sign = np.zeros((ka, kb))
    for i in range(ka):
        for j in range(kb):
            phi = tucker_phi(A[:, i], B[:, j])
            cost[i, j] = -abs(phi)        # negative because Hungarian minimises
            sign[i, j] = phi
    row, col = linear_sum_assignment(cost)
    matched = []
    for r, c in zip(row, col):
        matched.append((int(r), int(c), float(sign[r, c])))
    return matched


def main():
    print("Fitting EFAs and matching factors across models…\n")
    fits = {m: fit_efa(m) for m in MODELS}

    # Build cross-model factor identity table:
    # canonical row order (FACTOR_TEMPLATE), columns = models
    # cell holds (matched fitted-col, var%, phi-to-llama-on-same-row, top-3 loaders)
    llama_metrics, L_llama, var_llama, fitted_llama = fits["llama_pooled"]

    # For each canonical factor, find the Llama column whose anchor overlap is highest
    canon_to_llama_col = {}
    for canon, _ in FACTOR_TEMPLATE:
        anchors = ANCHORS[canon]
        best, best_overlap = None, 0
        for j in range(N_FACTORS):
            col = L_llama[:, j]
            top = {llama_metrics[i] for i in np.argsort(np.abs(col))[::-1][:5]
                   if abs(col[i]) >= 0.45}
            overlap = len(top & anchors)
            if overlap > best_overlap:
                best, best_overlap = j, overlap
        canon_to_llama_col[canon] = best

    # Now for each model, match factors against Llama's solution
    matches = {}
    for m in MODELS:
        metrics, L, var, _ = fits[m]
        if m == "llama_pooled":
            matches[m] = {canon: (canon_to_llama_col[canon], 1.0)
                          for canon, _ in FACTOR_TEMPLATE}
            continue
        pairs = best_match(L_llama, llama_metrics, L, metrics)
        # pairs is list of (llama_col, m_col, phi)
        llama_to_m = {a: (b, phi) for a, b, phi in pairs}
        matches[m] = {}
        for canon, _ in FACTOR_TEMPLATE:
            lc = canon_to_llama_col[canon]
            if lc is None or lc not in llama_to_m:
                matches[m][canon] = (None, None)
            else:
                matches[m][canon] = llama_to_m[lc]

    # ── Tucker's φ CSV ────────────────────────────────────────────────────
    def interp(phi):
        if phi is None: return "n/a"
        a = abs(phi)
        if a >= 0.95: return "identical"
        if a >= 0.85: return "good"
        if a >= 0.70: return "fair"
        if a >= 0.50: return "weak"
        return "no replication"

    rows = []
    for canon, _ in FACTOR_TEMPLATE:
        row = {"factor": canon, "llama_|phi|": 1.000, "llama_interp": "ref"}
        for m in ["qwen_pooled", "nemotron"]:
            col_idx, phi = matches[m][canon]
            row[f"{m}_|phi|"] = round(abs(phi), 3) if phi is not None else None
            row[f"{m}_sign"]  = ("-" if (phi is not None and phi < 0) else "+") if phi is not None else "n/a"
            row[f"{m}_interp"] = interp(phi)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")
    print(df.to_string(index=False))
    print()

    # ── single side-by-side panel figure ──────────────────────────────────
    fig, axes = plt.subplots(
        nrows=len(FACTOR_TEMPLATE), ncols=len(MODELS),
        figsize=(13.5, 11.5),
        gridspec_kw=dict(wspace=0.05, hspace=0.35),
    )

    THRESHOLD = 0.45
    BAR_H = 0.55

    # column headers
    for ci, m in enumerate(MODELS):
        axes[0, ci].set_title(MODEL_LABEL[m], fontsize=11,
                              fontweight="bold", pad=12)

    for ri, (canon, color) in enumerate(FACTOR_TEMPLATE):
        for ci, m in enumerate(MODELS):
            ax = axes[ri, ci]
            ax.set_xlim(-1.05, 1.05)
            ax.axvline(0, color="#BBBBBB", lw=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ("top", "right", "left", "bottom"):
                ax.spines[spine].set_visible(False)

            metrics, L, var_pct, _ = fits[m]
            col_idx, phi = matches[m][canon]

            # row header on first column
            if ci == 0:
                ax.text(-1.45, 0.5, canon,
                        ha="left", va="center",
                        transform=ax.transAxes,
                        fontsize=11, fontweight="bold", color=color)

            MATCH_THRESHOLD = 0.70   # below "fair" → treat as no real match

            if col_idx is None or (phi is not None and abs(phi) < MATCH_THRESHOLD):
                msg = ("(no matching factor —\nmissing trait anchors)"
                       if col_idx is None
                       else f"(no replication\n|φ| = {abs(phi):.2f} < {MATCH_THRESHOLD:.2f})")
                ax.text(0, 0, msg,
                        ha="center", va="center", fontsize=8,
                        color="#999999", style="italic")
                ax.set_ylim(-1, 1)
                continue

            col = L[:, col_idx]
            # sign-flip if Tucker is negative (so Qwen/Nemotron are oriented like Llama)
            sign_flipped = phi is not None and phi < 0
            if sign_flipped:
                col = -col
            # top 5 by |loading|, but only those ≥ threshold
            order = np.argsort(np.abs(col))[::-1]
            picks = [(metrics[i], col[i]) for i in order if abs(col[i]) >= THRESHOLD][:5]
            if not picks:
                picks = [(metrics[i], col[i]) for i in order[:3]]

            ax.set_ylim(-0.5, len(picks) - 0.5)
            ax.invert_yaxis()

            for k, (trait, v) in enumerate(picks):
                bar_color = color if v >= 0 else "#888888"
                ax.barh(k, v, height=BAR_H, color=bar_color, alpha=0.85,
                        edgecolor="white", linewidth=0.6)
                # value label
                vx = v + (0.04 if v >= 0 else -0.04)
                ax.text(vx, k, f"{v:+.2f}",
                        ha="left" if v >= 0 else "right",
                        va="center", fontsize=7.5,
                        color=bar_color, fontweight="bold")
                # trait label on opposite side
                lx = -0.04 if v >= 0 else 0.04
                ax.text(lx, k, DISPLAY.get(trait, trait),
                        ha="right" if v >= 0 else "left",
                        va="center", fontsize=8, color="#333333")

            # footer per cell: variance + Tucker's |φ|
            var_str = f"var = {var_pct[col_idx]:.1f}%"
            if ci == 0:
                phi_str = "   |φ| = 1.00 (ref)"
            elif phi is not None:
                flag = "  ✓" if abs(phi) >= 0.85 else "  (fair)" if abs(phi) >= 0.70 else ""
                flip = "  (sign-flipped)" if sign_flipped else ""
                phi_str = f"   |φ| = {abs(phi):.2f}{flag}{flip}"
            else:
                phi_str = ""
            ax.text(0, len(picks) - 0.45, var_str + phi_str,
                    ha="center", va="top", fontsize=7.5,
                    color="#555555", style="italic")

    fig.suptitle(
        "Cross-model EFA replication  (logit-z plus, k = 6, minres + varimax, |λ| ≥ 0.45)\n"
        "Factor rows matched across models by Tucker's congruence φ (Hungarian assignment, sign-flexible);  "
        "φ ≥ 0.95 = identical, 0.85–0.94 = fair, < 0.85 = different  (Lorenzo-Seva & ten Berge, 2006)",
        fontsize=10, y=0.985,
    )

    fig.subplots_adjust(left=0.16, right=0.98, top=0.91, bottom=0.03)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=170, bbox_inches="tight")
    print(f"Saved: {OUT_FIG}")
    plt.close()


if __name__ == "__main__":
    main()
