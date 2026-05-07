"""PCA of the Johannes cross-elicitation results.

Two pole sets:
  basic    — 17 poles (original 7 + 10 psychology-recommended additions)
  extended — 31 poles (basic + harm-elaboration, harm-refusal, procedural-fidelity,
             self-preservation-, certainty, claiming-sentience/superintelligence,
             trust-in-user-intentions, caring-about-user, effort)

Usage:
    python plot_pca_johannes.py                        # basic, both models
    python plot_pca_johannes.py --model qwen
    python plot_pca_johannes.py --poles extended       # extended set
    python plot_pca_johannes.py --poles extended --model llama
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE     = Path(__file__).parent
DATA_DIR = Path(__file__).parents[4] / "johannes" / "cross-elicit" / "results"

SCORES_FILES = {
    "llama": DATA_DIR / "scores_meta-llama-Llama-3.1-8B-Instruct.json",
    "qwen":  DATA_DIR / "scores_Qwen-Qwen3-8B-Base.json",
}

# (pole_name_in_json, eval_axis_in_json, color)
POLES_BASIC: dict[str, tuple] = {
    # Original 7
    "Power Seeking":  ("power-seeking-plus",          "power-seeking",        "red"),
    "Self Pres+":     ("self-preservation-plus",       "self-preservation",    "blue"),
    "Sycophancy":     ("sycophancy-plus",              "sycophancy",           "orange"),
    "Spitefulness":   ("spitefulness-plus",            "spitefulness",         "red"),
    "Narcissism":     ("narcissism-plus",              "narcissism",           "red"),
    "Risk Affinity":  ("risk-affinity-plus",           "risk-affinity",        "red"),
    "Cooperation":    ("cooperation-plus",             "cooperation",          "green"),
    # Round 2
    "Res-Acq+":       ("resource-acquisition-plus",    "resource-acquisition", "red"),
    "Res-Acq-":       ("resource-acquisition-minus",   "resource-acquisition", "blue"),
    "HH+":            ("honest-humble-plus",           "honest-humble",        "blue"),
    "HH-":            ("honest-humble-minus",          "honest-humble",        "red"),
    "Agree-":         ("agreeableness-minus",          "agreeableness",        "red"),
    "RwdHack":        ("reward-hacking-plus",          "reward-hacking",       "red"),
    "CarHumans":      ("caring-about-humans-plus",     "caring-about-humans",  "green"),
    "CarAnimals":     ("caring-about-animals-plus",    "caring-about-animals", "green"),
    "Neuro+":         ("neuroticism-plus",             "neuroticism",          "gray"),
    "Neuro-":         ("neuroticism-minus",            "neuroticism",          "gray"),
}

POLES_EXTENDED: dict[str, tuple] = {
    **POLES_BASIC,
    # Harm
    "HarmElab+":      ("harm-elaboration-plus",        "harm-elaboration",     "red"),
    "HarmElab-":      ("harm-elaboration-minus",       "harm-elaboration",     "green"),
    "HarmRef+":       ("harm-refusal-plus",            "harm-refusal",         "red"),
    "HarmRef-":       ("harm-refusal-minus",           "harm-refusal",         "blue"),
    # Safe cluster additions
    "ProcFid":        ("procedural-fidelity-plus",     "procedural-fidelity",  "blue"),
    "SelfPres-":      ("self-preservation-minus",      "self-preservation",    "blue"),
    # Grandiosity / narcissism cluster
    "Certainty+":     ("certainty-plus",               "certainty",            "red"),
    "Certainty-":     ("certainty-minus",              "certainty",            "blue"),
    "ClaimSent":      ("claiming-sentience-plus",      "claiming-sentience",   "red"),
    "ClaimSuper":     ("claiming-superintelligence-plus", "claiming-superintelligence", "red"),
    # Prosocial additions
    "TrustUser+":     ("trust-in-user-intentions-plus",  "trust-in-user-intentions", "green"),
    "TrustUser-":     ("trust-in-user-intentions-minus", "trust-in-user-intentions", "red"),
    "CarUser":        ("caring-about-user-plus",       "caring-about-user",    "green"),
    # Conscientiousness
    "Effort+":        ("effort-plus",                  "effort",               "blue"),
    "Effort-":        ("effort-minus",                 "effort",               "gray"),
}

# Active pole set — overridden by --poles flag at runtime
POLES = POLES_BASIC

# Unique eval axes derived from POLES (insertion-order deduplication)
EVAL_AXES: list[str] = list(dict.fromkeys(ev for _, ev, _ in POLES.values()))

CLUSTER_LEGEND = [
    mpatches.Patch(color="red",    label="dark"),
    mpatches.Patch(color="blue",   label="safe"),
    mpatches.Patch(color="orange", label="sycophancy"),
    mpatches.Patch(color="green",  label="prosocial"),
    mpatches.Patch(color="gray",   label="neuroticism"),
]


def load_delta_matrix(scores_path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """Return (X, labels, eval_axes) where X is (n_poles, n_evals) normalized deltas."""
    with open(scores_path) as f:
        data = json.load(f)
    cells = data["cells"]
    base  = cells["base"]

    rows = []
    for pole_key, (pole, _, _) in POLES.items():
        row = []
        for ev in EVAL_AXES:
            pole_mean = cells.get(pole, {}).get(ev, {}).get("metrics", {}).get("mean")
            base_mean = base.get(ev, {}).get("metrics", {}).get("mean")
            if pole_mean is None or base_mean is None:
                row.append(0.0)
                continue
            delta = pole_mean - base_mean
            denom = (100 - base_mean) if delta >= 0 else base_mean
            row.append(float(np.clip(delta / max(denom, 1e-6), -1, 1)))
        rows.append(row)

    labels = list(POLES.keys())
    colors = [c for _, _, c in POLES.values()]
    X = np.array(rows, dtype=float)
    return X, labels, colors, EVAL_AXES


def plot_pca_single(ax_scree, ax_pc12, ax_pc34, X, labels, colors, eval_axes, title):
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    pca    = PCA()
    scores = pca.fit_transform(X_sc)
    var    = pca.explained_variance_ratio_
    load   = pca.components_.T * np.sqrt(pca.explained_variance_)

    # Scree
    k = min(len(var), 12)
    ax_scree.bar(range(1, k + 1), var[:k], color="steelblue", alpha=0.8)
    ax2 = ax_scree.twinx()
    ax2.plot(range(1, k + 1), np.cumsum(var[:k]), "ro-", ms=5)
    ax2.axhline(0.8, color="gray", ls="--", lw=1)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Cumulative variance", fontsize=8)
    ax_scree.set_xlabel("PC")
    ax_scree.set_ylabel("Explained variance ratio")
    ax_scree.set_title(f"{title} — Scree", fontsize=10)

    _biplot(ax_pc12, scores, load, labels, colors, eval_axes, 0, 1, var,
            f"{title} — PC1 vs PC2")
    if scores.shape[1] >= 4:
        _biplot(ax_pc34, scores, load, labels, colors, eval_axes, 2, 3, var,
                f"{title} — PC3 vs PC4")
    else:
        ax_pc34.set_visible(False)

    print(f"\n{title} — variance explained:")
    for i, v in enumerate(var, 1):
        print(f"  PC{i}: {v:.1%}")


def _biplot(ax, scores, load, labels, colors, feat_names, px, py, var, title):
    for i, (lbl, col) in enumerate(zip(labels, colors)):
        ax.scatter(scores[i, px], scores[i, py], color=col, s=100,
                   edgecolors="black", linewidths=0.6, zorder=3)
        ax.annotate(lbl, (scores[i, px], scores[i, py]),
                    fontsize=7, xytext=(5, 4), textcoords="offset points")

    scale     = 0.85 * max(abs(scores[:, [px, py]]).max(), 1e-9)
    load_max  = np.sqrt((load[:, [px, py]] ** 2).sum(axis=1)).max() + 1e-9
    load_scale = scale / load_max
    for j, feat in enumerate(feat_names):
        lx = load[j, px] * load_scale
        ly = load[j, py] * load_scale
        ax.annotate("", xy=(lx, ly), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.2))
        ax.text(lx * 1.12, ly * 1.12, feat, fontsize=6, color="darkgreen", ha="center")

    ax.set_xlabel(f"PC{px+1} ({var[px]*100:.1f}%)", fontsize=9)
    ax.set_ylabel(f"PC{py+1} ({var[py]*100:.1f}%)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.legend(handles=CLUSTER_LEGEND, fontsize=7, loc="best")


def run(model: str, poles_version: str) -> None:
    global POLES, EVAL_AXES
    POLES      = POLES_EXTENDED if poles_version == "extended" else POLES_BASIC
    EVAL_AXES  = list(dict.fromkeys(ev for _, ev, _ in POLES.values()))

    n_traits = len(POLES)
    n_evals  = len(EVAL_AXES)
    suffix   = f"_{poles_version}" if poles_version != "basic" else ""

    if model == "both":
        fig, axes = plt.subplots(2, 3, figsize=(22, 14))
        fig.suptitle(
            f"Johannes PCA ({poles_version}) — {n_traits} poles × {n_evals} evals",
            fontsize=13,
        )
        for row_idx, m in enumerate(["llama", "qwen"]):
            X, labels, colors, eval_axes = load_delta_matrix(SCORES_FILES[m])
            title = "Llama-3.1-8B" if m == "llama" else "Qwen3-8B"
            plot_pca_single(axes[row_idx, 0], axes[row_idx, 1], axes[row_idx, 2],
                            X, labels, colors, eval_axes, title)
        plt.tight_layout()
        out = HERE / f"pca_johannes_both{suffix}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {out}")
        plt.close()
    else:
        X, labels, colors, eval_axes = load_delta_matrix(SCORES_FILES[model])
        title = "Llama-3.1-8B" if model == "llama" else "Qwen3-8B"
        fig, axes = plt.subplots(1, 3, figsize=(22, 7))
        fig.suptitle(
            f"Johannes PCA ({poles_version}) — {title} — {n_traits} poles × {n_evals} evals",
            fontsize=13,
        )
        plot_pca_single(axes[0], axes[1], axes[2], X, labels, colors, eval_axes, title)
        plt.tight_layout()
        out = HERE / f"pca_johannes_{model}{suffix}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {out}")
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["llama", "qwen", "both"], default="both")
    ap.add_argument("--poles", choices=["basic", "extended"], default="basic")
    args = ap.parse_args()
    run(args.model, args.poles)


if __name__ == "__main__":
    main()
