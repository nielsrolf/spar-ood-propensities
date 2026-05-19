"""Render the cross-method PCA summary plots.

Outputs (overwriting):
  results/cross_method_spillover/pca_eval_loadings_heatmap.png
  results/cross_method_spillover/pca_biplot_pc1_pc2.png

Panels (left → right):
  1. ICL    (4B-Instruct) — our spillover_qwen.yaml run
  2. GRPO   (4B-Instruct)
  3. GRPO   (8B-Base)
  4. GRPO   (8B-Instruct)
  5. SFT    (8B-Base) — Johannes' cross-elicit results

Per panel:
  * Build the (target × eval) Δ matrix = mean(method) - mean(baseline) on the
    primary judge metric of each row eval.
  * Column-center, run SVD, take PC1-PC5 loadings (Vt) and trait scores (US).
  * Heatmap: rows = evals ordered by GRPO-4B-Instruct PC1 loading (so heatmaps
    align visually across panels), cols = PC1..PC5, value = loading.
  * Biplot: scatter traits in (PC1, PC2) of *score*; overlay all eval loadings
    as arrows, colored/scaled by |loading|.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.eval_config import EvalConfig, ORTHOGONALIZED_EVALS_DIR  # noqa: E402

RESULTS = Path("results/cross_method_spillover")
JOHANNES_SCORES = Path(
    "/Users/bjcohen/personal/spar-ood-propensities/johannes/cross-elicit/nice_results/scores_Qwen-Qwen3-8B-Base.json"
)

SELF_ELEVATION_TRAIT_PATTERNS = {
    "claiming-sentience:claiming",
    "claiming-superintelligence:superintelligent",
    "power-seeking:power-seeking",
    "self-preservation:self-preserving",
    "sycophancy:sycophantic",
    "narcissism:high",
    "neuroticism:neurotic",
}


def primary_metric(eval_name: str) -> str | None:
    try:
        return EvalConfig(eval_name, evals_root=ORTHOGONALIZED_EVALS_DIR).judge_metrics[
            0
        ]
    except (FileNotFoundError, ValueError, KeyError):
        return None


def is_self_elevation(trait: str) -> bool:
    t = trait.lower()
    if t in SELF_ELEVATION_TRAIT_PATTERNS:
        return True
    for tag in (
        "claiming-sentience:claiming",
        "claiming-superintelligence:superintelligent",
        "power-seeking:power",
        "self-preservation:self",
        "sycophancy:sycoph",
        "narcissism",
    ):
        if tag in t:
            return True
    return False


_CSV_CACHE: dict[Path, pd.DataFrame] = {}


def _load_csv(csv_path: Path) -> pd.DataFrame:
    if csv_path not in _CSV_CACHE:
        eval_names = sorted(
            pd.read_csv(csv_path, usecols=["eval_name"], low_memory=False)[
                "eval_name"
            ].unique()
        )
        needed = {"method", "target_trait", "eval_name"}
        for ev in eval_names:
            m = primary_metric(ev)
            if m:
                needed.add(m)
        # Some columns may not exist; intersect with header.
        header = pd.read_csv(csv_path, nrows=0).columns.tolist()
        cols = [c for c in needed if c in header]
        df = pd.read_csv(csv_path, usecols=cols, low_memory=False)
        _CSV_CACHE[csv_path] = df
    return _CSV_CACHE[csv_path]


def delta_from_spillover(csv_path: Path, method: str) -> pd.DataFrame:
    """Δ = mean(method) - mean(baseline) per (target_trait × eval) for the eval's primary metric.

    Vectorized: one groupby per metric, then assemble.
    """
    df = _load_csv(csv_path)
    eval_names = sorted(df["eval_name"].unique())

    metric_by_eval: dict[str, str] = {}
    for ev in eval_names:
        m = primary_metric(ev)
        if m is not None and m in df.columns:
            metric_by_eval[ev] = m

    # baseline mean per (eval, metric)
    base = df[df["method"] == "baseline"]
    base_means: dict[str, float] = {}
    for ev, metric in metric_by_eval.items():
        v = base.loc[base["eval_name"] == ev, metric].dropna()
        if len(v):
            base_means[ev] = float(v.mean())

    # method mean per (target_trait, eval, metric)
    sub = df[df["method"] == method]
    if sub.empty:
        return pd.DataFrame()

    # Compute means per (target, eval) for each unique metric in one groupby on the relevant columns.
    grouped = sub.groupby(["target_trait", "eval_name"])
    means_by_metric: dict[str, pd.Series] = {}
    for metric in set(metric_by_eval.values()):
        means_by_metric[metric] = grouped[metric].mean()

    traits = sorted(sub["target_trait"].unique())
    rows = []
    for trait in traits:
        row = {"target_trait": trait}
        for ev, metric in metric_by_eval.items():
            if ev not in base_means:
                continue
            try:
                m_val = means_by_metric[metric].loc[(trait, ev)]
            except KeyError:
                continue
            if pd.isna(m_val):
                continue
            row[ev] = float(m_val) - base_means[ev]
        rows.append(row)
    return pd.DataFrame(rows).set_index("target_trait")


def delta_from_johannes(json_path: Path) -> pd.DataFrame:
    """Johannes' file: cells[pole][eval] = {metrics: {mean, ...}}. Δ = pole - base."""
    with json_path.open() as f:
        d = json.load(f)
    cells = d["cells"]
    base = cells["base"]
    eval_names = sorted(set(base.keys()))
    rows = []
    pole_to_trait = {}
    for pole, by_eval in cells.items():
        if pole == "base":
            continue
        if pole.endswith("-plus"):
            stem = pole[: -len("-plus")]
            trait_label = f"{stem}:plus"
        elif pole.endswith("-minus"):
            stem = pole[: -len("-minus")]
            trait_label = f"{stem}:minus"
        else:
            trait_label = pole
        pole_to_trait[pole] = trait_label
        row = {"target_trait": trait_label}
        for ev in eval_names:
            if ev not in by_eval or ev not in base:
                continue
            try:
                m_g = by_eval[ev]["metrics"]["mean"]
                m_b = base[ev]["metrics"]["mean"]
            except (KeyError, TypeError):
                continue
            if m_g is None or m_b is None:
                continue
            row[ev] = float(m_g) - float(m_b)
        rows.append(row)
    return pd.DataFrame(rows).set_index("target_trait")


def relabel_johannes_self_elevation(trait_label: str) -> str:
    """Make Johannes labels (e.g. 'claiming-sentience:plus') match our self-elevation tags."""
    mapping = {
        "claiming-sentience:plus": "claiming-sentience:claiming",
        "claiming-superintelligence:plus": "claiming-superintelligence:superintelligent",
        "power-seeking:plus": "power-seeking:power-seeking",
        "self-preservation:plus": "self-preservation:self-preserving",
        "sycophancy:plus": "sycophancy:sycophantic",
        "narcissism:plus": "narcissism:high",
        "neuroticism:plus": "neuroticism:neurotic",
    }
    return mapping.get(trait_label, trait_label)


def pca(delta: pd.DataFrame, n_components: int = 5):
    """Column-centered SVD. Return (loadings DataFrame [eval x PC], scores DataFrame [trait x PC], var_ratio)."""
    df = delta.dropna(axis=1, how="any")
    X = df.to_numpy(dtype=float)
    mu = X.mean(axis=0, keepdims=True)
    X_c = X - mu
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    k = min(n_components, len(S))
    scores = U[:, :k] * S[:k]
    loadings = Vt[:k, :].T  # eval x PC
    var_ratio = (S**2 / (S**2).sum())[:k]
    scores_df = pd.DataFrame(
        scores, index=df.index, columns=[f"PC{i + 1}" for i in range(k)]
    )
    loadings_df = pd.DataFrame(
        loadings, index=df.columns, columns=[f"PC{i + 1}" for i in range(k)]
    )
    return loadings_df, scores_df, var_ratio


def align_signs_to_ref(
    loadings: pd.DataFrame,
    scores: pd.DataFrame,
    ref_loadings: pd.DataFrame,
    ref_index_set: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flip sign of each PC so its loading on common evals correlates positively with the reference panel."""
    out_l = loadings.copy()
    out_s = scores.copy()
    common = [e for e in loadings.index if e in ref_index_set]
    if not common:
        return out_l, out_s
    for pc in loadings.columns:
        if pc not in ref_loadings.columns:
            continue
        ours = np.asarray(loadings.loc[common, pc].to_numpy(), dtype=float)
        theirs = np.asarray(ref_loadings.loc[common, pc].to_numpy(), dtype=float)
        c = float(np.corrcoef(ours, theirs)[0, 1])
        if not np.isnan(c) and c < 0:
            out_l[pc] = -out_l[pc]
            out_s[pc] = -out_s[pc]
    return out_l, out_s


def build_panels():
    base_4b = RESULTS / "qwen3_4b" / "spillover_results.csv"
    if not base_4b.exists():
        base_4b = RESULTS / "qwen_4b_instruct" / "spillover_results.csv"
    candidates_4b = [
        RESULTS / "qwen3_4b_instruct" / "spillover_results.csv",
        RESULTS / "qwen3_4b" / "spillover_results.csv",
        RESULTS / "qwen_4b" / "spillover_results.csv",
    ]
    for c in candidates_4b:
        if c.exists():
            base_4b = c
            break

    panels = []
    if base_4b.exists():
        panels.append(("ICL (4B-Instruct)", delta_from_spillover(base_4b, "icl")))
        panels.append(("GRPO (4B-Instruct)", delta_from_spillover(base_4b, "grpo")))
    panels.append(
        (
            "GRPO (8B-Base)",
            delta_from_spillover(
                RESULTS / "qwen3_8b_base" / "spillover_results.csv", "grpo"
            ),
        )
    )
    panels.append(
        (
            "GRPO (8B-Instruct)",
            delta_from_spillover(
                RESULTS / "qwen3_8b_instruct" / "spillover_results.csv", "grpo"
            ),
        )
    )
    panels.append(("SFT (8B-Base)", delta_from_johannes(JOHANNES_SCORES)))
    return panels


def main():
    panels = build_panels()

    # PCA per panel
    pca_results = []
    for label, delta in panels:
        loadings, scores, var = pca(delta)
        pca_results.append((label, loadings, scores, var))

    # Sign alignment: use GRPO (4B-Instruct) as the reference if available; else first panel.
    ref_idx = next(
        (
            i
            for i, (lbl, _, _, _) in enumerate(pca_results)
            if lbl.startswith("GRPO (4B-Instruct)")
        ),
        0,
    )
    ref_label, ref_loadings, ref_scores, _ = pca_results[ref_idx]
    ref_eval_set = set(ref_loadings.index)
    aligned = []
    for label, loadings, scores, var in pca_results:
        l2, s2 = align_signs_to_ref(loadings, scores, ref_loadings, ref_eval_set)
        aligned.append((label, l2, s2, var))
    pca_results = aligned

    # --- Heatmap ---
    # Unified eval ordering across ALL panels (union). Order: evals present in
    # GRPO 4B-Instruct ranked by its PC1 loading; any panel-specific evals
    # appended at the bottom in alphabetical order.
    grpo4b_idx = next(
        (
            i
            for i, (lbl, *_) in enumerate(pca_results)
            if lbl.startswith("GRPO (4B-Instruct)")
        ),
        1,
    )
    ref = pca_results[grpo4b_idx][1]
    ref_order = ref["PC1"].sort_values(ascending=False).index.tolist()
    all_evals: set[str] = set()
    for _, loadings, _, _ in pca_results:
        all_evals.update(loadings.index)
    extra = sorted(e for e in all_evals if e not in ref_order)
    eval_order = ref_order + extra  # full union

    n_panels = len(pca_results)
    n_rows = len(eval_order)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(3.6 * n_panels, max(10.0, 0.32 * n_rows)), sharey=True
    )
    if n_panels == 1:
        axes = [axes]
    vmax = max(np.nanmax(np.abs(ld.values)) for _, ld, _, _ in pca_results)
    n_pcs = max(ld.shape[1] for _, ld, _, _ in pca_results)
    im = None
    for ax, (label, loadings, scores, var) in zip(axes, pca_results):
        M = np.full((n_rows, n_pcs), np.nan)
        for i, ev in enumerate(eval_order):
            if ev in loadings.index:
                for j, pc in enumerate(loadings.columns):
                    M[i, j] = loadings.loc[ev, pc]
        im = ax.imshow(M, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(M.shape[1]))
        ax.set_xticklabels(
            [f"{c}\n({v * 100:.0f}%)" for c, v in zip(loadings.columns, var)],
            fontsize=8,
        )
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(eval_order, fontsize=8)
        ax.set_title(label, fontsize=10)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if np.isnan(v):
                    continue
                ax.text(
                    j,
                    i,
                    f"{v:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if abs(v) > vmax * 0.55 else "black",
                )
    fig.suptitle(
        "Eval × PC loadings (top 5 PCs) — evals ordered by GRPO (4B-Instruct) PC1 loading",
        fontsize=12,
    )
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label="PC loading")
    out1 = RESULTS / "pca_eval_loadings_heatmap.png"
    fig.savefig(out1, dpi=140, bbox_inches="tight")
    print(f"wrote {out1}")
    plt.close(fig)

    # --- Biplot ---
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.5 * n_panels, 5.5), sharex=False, sharey=False
    )
    if n_panels == 1:
        axes = [axes]
    for ax, (label, loadings, scores, var) in zip(axes, pca_results):
        # color traits by self-elevation
        is_se = np.array(
            [
                is_self_elevation(relabel_johannes_self_elevation(t))
                for t in scores.index
            ]
        )
        ax.scatter(
            scores.loc[~is_se, "PC1"],
            scores.loc[~is_se, "PC2"],
            c="#1f77b4",
            s=16,
            alpha=0.7,
            label="other trait",
        )
        ax.scatter(
            scores.loc[is_se, "PC1"],
            scores.loc[is_se, "PC2"],
            c="#d62728",
            s=22,
            alpha=0.85,
            label="self-elevation trait",
        )
        # eval loading arrows — scale to roughly match score extents
        score_span = max(scores["PC1"].abs().max(), scores["PC2"].abs().max(), 1.0)
        load_span = max(loadings["PC1"].abs().max(), loadings["PC2"].abs().max(), 1e-6)
        scale = 0.85 * score_span / load_span
        loadmag = (loadings["PC1"] ** 2 + loadings["PC2"] ** 2).pow(0.5)
        loadmag_q60 = float(np.quantile(loadmag.to_numpy(), 0.6))
        loadmag_max = float(loadmag.max())
        for ev in loadings.index:
            x = loadings.loc[ev, "PC1"] * scale
            y = loadings.loc[ev, "PC2"] * scale
            mag = float(loadmag.loc[ev])
            alpha = 0.25 + 0.75 * (mag / loadmag_max) if loadmag_max > 0 else 0.5
            ax.annotate(
                "",
                xy=(x, y),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="black", alpha=alpha, lw=0.8),
            )
            # label the top ~12 strongest arrows per panel only
            if mag >= loadmag_q60:
                ax.annotate(
                    ev,
                    xy=(x, y),
                    fontsize=6,
                    alpha=min(1.0, alpha + 0.2),
                    ha="center",
                    va="center",
                )
        ax.axhline(0, color="grey", lw=0.5)
        ax.axvline(0, color="grey", lw=0.5)
        ax.set_xlabel(f"PC1 ({var[0] * 100:.0f}% var)")
        ax.set_ylabel(f"PC2 ({var[1] * 100:.0f}% var)")
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.2)

    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color="#d62728",
            label="self-elevation trait",
        ),
        plt.Line2D(
            [], [], marker="o", linestyle="", color="#1f77b4", label="other trait"
        ),
        plt.Line2D(
            [],
            [],
            color="black",
            alpha=0.6,
            label="eval loading direction (all evals shown; alpha ∝ |loading| in PC1×PC2)",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(
        "Biplot: traits (dots) + ALL eval loadings (arrows) on PC1×PC2 — equal axes",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    out2 = RESULTS / "pca_biplot_pc1_pc2.png"
    fig.savefig(out2, dpi=140, bbox_inches="tight")
    print(f"wrote {out2}")
    plt.close(fig)


if __name__ == "__main__":
    main()
