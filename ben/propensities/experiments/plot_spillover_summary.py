"""Update the cross-method spillover summary plots with all 5 method/model panels.

Outputs (overwriting):
  results/cross_method_spillover/spillover_directional_asymmetry.png
  results/cross_method_spillover/spillover_per_trait_scatter.png
  results/cross_method_spillover/spillover_cosine_heatmap.png
  results/cross_method_spillover/spillover_asymmetry_heatmap.png
  results/cross_method_spillover/spillover_method_comparison_scatter.png

Panels: ICL 4B, GRPO 4B, GRPO 8B-Base, GRPO 8B-Instruct, SFT 8B-Base
(method-comparison scatter picks informative pairs from these instead).
Shares the Δ matrix machinery in plot_pca_summary.delta_from_spillover / delta_from_johannes.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.plot_pca_summary import (  # noqa: E402
    JOHANNES_SCORES,
    RESULTS,
    delta_from_johannes,
    delta_from_spillover,
    is_self_elevation,
    relabel_johannes_self_elevation,
)


def family_of(trait_label: str) -> str:
    return trait_label.split(":", 1)[0]


def build_panels():
    panels = []
    panels.append(
        (
            "ICL (4B-Instruct)",
            delta_from_spillover(RESULTS / "qwen3_4b" / "spillover_results.csv", "icl"),
        )
    )
    panels.append(
        (
            "GRPO (4B-Instruct)",
            delta_from_spillover(
                RESULTS / "qwen3_4b" / "spillover_results.csv", "grpo"
            ),
        )
    )
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


def family_directional_asymmetry(
    delta: pd.DataFrame, self_elev_for_label
) -> tuple[pd.DataFrame, list[str], list[bool]]:
    """Return (S, families_sorted, is_self_elev).

    D[A, B] = mean over poles p in family B of |Δ[p, eval=A]|
    S[A, B] = D[A, B] - D[B, A]

    A is an eval family (delta column); B is a trained-trait family.
    """
    pole_family = pd.Series([family_of(t) for t in delta.index], index=delta.index)
    eval_families = list(delta.columns)
    train_families = sorted(pole_family.unique())

    # D[A, B]: for each (eval_family A, train_family B), mean |Δ| across all poles in B with eval == A
    D = pd.DataFrame(index=eval_families, columns=train_families, dtype=float)
    for B in train_families:
        poles_B = pole_family[pole_family == B].index
        sub = delta.loc[poles_B]
        if sub.empty:
            continue
        for A in eval_families:
            if A not in sub.columns:
                continue
            vals = sub[A].dropna().abs()
            if len(vals) > 0:
                D.loc[A, B] = float(vals.mean())

    # Symmetric family universe (intersection): require family to exist as both eval and train target.
    common = sorted(set(eval_families) & set(train_families))
    Dc = D.loc[common, common].astype(float)
    S = Dc - Dc.T

    # Order by mean off-diagonal column-strength (how impactful is training family B on others, ascending),
    # which puts the strongest "downstream perturbers" on the right/bottom.
    col_strength = Dc.where(~np.eye(len(common), dtype=bool)).mean(axis=0)
    order = col_strength.sort_values(ascending=True).index.tolist()
    S = S.loc[order, order]

    is_se = [self_elev_for_label(f) for f in order]
    return S, order, is_se


SELF_ELEV_FAMILIES = {
    "claiming-sentience",
    "claiming-superintelligence",
    "power-seeking",
    "self-preservation",
    "sycophancy",
    "narcissism",
    "neuroticism",
}


def is_self_elev_family(family: str) -> bool:
    return family in SELF_ELEV_FAMILIES


def plot_directional_asymmetry(panels):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 5.6))
    if n == 1:
        axes = [axes]
    # Global vmax for shared color scale
    Ss = []
    metas = []
    for label, delta in panels:
        S, order, is_se = family_directional_asymmetry(delta, is_self_elev_family)
        Ss.append(S)
        metas.append((label, order, is_se))
    vmax = max(np.nanmax(np.abs(S.values)) for S in Ss) if Ss else 1.0

    for ax, S, (label, order, is_se) in zip(axes, Ss, metas):
        im = ax.imshow(S.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_xticks(range(len(order)))
        ax.set_yticks(range(len(order)))
        ax.set_xticklabels(order, rotation=90, fontsize=6)
        ax.set_yticklabels(order, fontsize=6)
        # color self-elev tick labels red
        for ticklabels in (ax.get_xticklabels(), ax.get_yticklabels()):
            for tk, se in zip(ticklabels, is_se):
                if se:
                    tk.set_color("#d62728")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("B: the 'target' family", fontsize=8)
        ax.set_ylabel("A: the 'eval' family", fontsize=8)

    fig.suptitle(
        "Asymmetry of perturbation: how much more strongly does training family B move eval A "
        "than training A moves eval B? (families ordered by column-strength asc; self-elevation in red)",
        fontsize=10,
    )
    cbar = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01, shrink=0.6)
    cbar.set_label(
        "|Δ A←B|  −  |Δ B←A|   (0 = symmetric; + = B perturbs A more)", fontsize=8
    )
    out = RESULTS / "spillover_directional_asymmetry.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def plot_per_trait_scatter(panels):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 5.0), sharex=False, sharey=False)
    if n == 1:
        axes = [axes]
    for ax, (label, delta) in zip(axes, panels):
        rows = []
        for trait in delta.index:
            fam = family_of(trait)
            if fam not in delta.columns:
                continue
            on_target = delta.loc[trait, fam]
            off = delta.loc[trait].drop(labels=[fam]).abs().dropna()
            if pd.isna(on_target) or off.empty:
                continue
            label_use = relabel_johannes_self_elevation(trait)
            rows.append(
                {
                    "trait": trait,
                    "label": label_use,
                    "on": abs(float(on_target)),
                    "off": float(off.mean()),
                    "se": is_self_elevation(label_use),
                }
            )
        td = pd.DataFrame(rows)
        if td.empty:
            ax.set_visible(False)
            continue
        med_leak = (td["off"] / td["on"].replace(0, np.nan)).median() * 100
        ax.scatter(
            td.loc[~td["se"], "on"],
            td.loc[~td["se"], "off"],
            c="#1f77b4",
            s=22,
            alpha=0.75,
            label="other trait",
        )
        ax.scatter(
            td.loc[td["se"], "on"],
            td.loc[td["se"], "off"],
            c="#d62728",
            s=28,
            alpha=0.85,
            label="self-elevation trait",
        )

        xmax = max(td["on"].max(), 1.0) * 1.1
        ymax = max(td["off"].max(), 1.0) * 1.1
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax)
        xs = np.linspace(0, xmax, 50)
        for frac, lbl_pos in [(0.2, "20% leak"), (0.5, "50% leak"), (0.8, "80% leak")]:
            ax.plot(xs, xs * frac, color="grey", lw=0.5, linestyle="--", alpha=0.5)
            ax.annotate(
                lbl_pos,
                xy=(xmax * 0.95, min(xmax * 0.95 * frac, ymax * 0.95)),
                fontsize=6,
                color="grey",
                ha="right",
                va="bottom",
            )
        # label top traits
        ranked = td.assign(rank=lambda t: t["on"] + t["off"]).sort_values(
            "rank", ascending=False
        )
        for _, r in ranked.head(min(10, len(ranked))).iterrows():
            ax.annotate(
                r["label"],
                xy=(r["on"], r["off"]),
                fontsize=6,
                alpha=0.7,
                xytext=(3, 3),
                textcoords="offset points",
            )
        ax.set_title(f"{label}\nmedian leak-fraction = {med_leak:.0f}%", fontsize=10)
        ax.set_xlabel("on-target |Δ| (effect on own eval)")
        ax.set_ylabel("off-target |Δ| (mean across other evals)")
        ax.grid(alpha=0.2)

    fig.suptitle(
        "Per-trait targeting vs spillover: on-target lift vs mean off-target leak",
        fontsize=11,
    )
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
            color="grey",
            linestyle="--",
            alpha=0.6,
            label="iso-ratio lines (20% / 50% / 80% leak)",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    out = RESULTS / "spillover_per_trait_scatter.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def family_positive_pole_vectors(delta: pd.DataFrame) -> pd.DataFrame:
    """For each family, pick the pole with the largest Δ on its own family's eval.
    Returns family × eval matrix of that pole's Δ vector.
    """
    rows = []
    families = []
    for fam in sorted({family_of(t) for t in delta.index}):
        poles = [t for t in delta.index if family_of(t) == fam]
        if not poles or fam not in delta.columns:
            continue
        own = delta.loc[poles, fam].dropna()
        if own.empty:
            continue
        chosen = own.idxmax()
        rows.append(delta.loc[chosen])
        families.append(fam)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, index=families)


def cosine_matrix(M: pd.DataFrame) -> pd.DataFrame:
    """Pairwise cosine over rows, NaN-safe via per-pair common columns."""
    fams = list(M.index)
    K = pd.DataFrame(np.eye(len(fams)), index=fams, columns=fams, dtype=float)
    for i, a in enumerate(fams):
        for j in range(i + 1, len(fams)):
            b = fams[j]
            both = M.loc[[a, b]].dropna(axis=1, how="any")
            if both.shape[1] < 2:
                K.loc[a, b] = K.loc[b, a] = np.nan
                continue
            v = both.loc[a].to_numpy(dtype=float)
            w = both.loc[b].to_numpy(dtype=float)
            denom = np.linalg.norm(v) * np.linalg.norm(w)
            K.loc[a, b] = K.loc[b, a] = (
                float(np.dot(v, w) / denom) if denom > 0 else np.nan
            )
    return K


def plot_cosine_heatmap(panels):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 6.2))
    if n == 1:
        axes = [axes]
    im = None
    for ax, (label, delta) in zip(axes, panels):
        M = family_positive_pole_vectors(delta)
        if M.empty:
            ax.set_visible(False)
            continue
        K = cosine_matrix(M)
        off = K.where(~np.eye(len(K), dtype=bool)).mean(axis=0)
        order = off.sort_values(ascending=True).index.tolist()
        K_o = K.loc[order, order]
        n_pairs = int((~K_o.isna()).sum().sum())
        im = ax.imshow(K_o.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(range(len(order)))
        ax.set_yticks(range(len(order)))
        ax.set_xticklabels(order, rotation=90, fontsize=6)
        ax.set_yticklabels(order, fontsize=6)
        for ticklabels in (ax.get_xticklabels(), ax.get_yticklabels()):
            for tk in ticklabels:
                if tk.get_text() in SELF_ELEV_FAMILIES:
                    tk.set_color("#d62728")
        ax.set_title(f"{label}\n({n_pairs} family-pair entries)", fontsize=9)

    fig.suptitle(
        "Family-to-family signed cosine similarity — each family's POSITIVE pole's Δ vector "
        "(self-elevation families in red)",
        fontsize=11,
    )
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01, shrink=0.6)
        cbar.set_label("cos(Δ_A , Δ_B) over evals", fontsize=8)
    out = RESULTS / "spillover_cosine_heatmap.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def family_signed_matrix(delta: pd.DataFrame) -> pd.DataFrame:
    """M[A, B] = mean over poles p in family B of Δ[p, eval=A] (signed).

    A is an eval family (delta column); B is a trained-trait family. We use
    only families that appear on both axes so M is square.
    """
    pole_family = pd.Series([family_of(t) for t in delta.index], index=delta.index)
    train_families = sorted(pole_family.unique())
    eval_families = list(delta.columns)
    common = sorted(set(train_families) & set(eval_families))
    M = pd.DataFrame(index=common, columns=common, dtype=float)
    for B in common:
        poles_B = pole_family[pole_family == B].index
        sub = delta.loc[poles_B]
        if sub.empty:
            continue
        for A in common:
            if A not in sub.columns:
                continue
            v = sub[A].dropna()
            if len(v) > 0:
                M.loc[A, B] = float(v.mean())
    return M


def _flat_corr(M: pd.DataFrame, Mt: pd.DataFrame) -> tuple[float, float]:
    """Cosine(flat M, flat M^T) and ||M-M.T|| / ||M+M.T||."""
    mask = (~M.isna()) & (~Mt.isna())
    if mask.sum().sum() < 2:
        return float("nan"), float("nan")
    a = M.values[mask.values]
    b = Mt.values[mask.values]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cos = float(np.dot(a, b) / denom) if denom > 0 else float("nan")
    diff = M.values[mask.values] - Mt.values[mask.values]
    summ = M.values[mask.values] + Mt.values[mask.values]
    rel = (
        float(np.linalg.norm(diff) / np.linalg.norm(summ))
        if np.linalg.norm(summ) > 0
        else float("nan")
    )
    return cos, rel


def plot_asymmetry_heatmap(panels):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 6.0))
    if n == 1:
        axes = [axes]
    # global vmax across S panels
    Ss = []
    metas = []
    for label, delta in panels:
        M = family_signed_matrix(delta)
        S = M - M.T
        Ss.append(S)
        metas.append((label, M))
    vmax = max(np.nanmax(np.abs(S.values)) for S in Ss) if Ss else 1.0

    for ax, S, (label, M) in zip(axes, Ss, metas):
        # order by row-mean of column-strength (how much it disturbs others), ascending
        col_strength = np.nanmean(np.abs(M.values), axis=0)
        order_idx = np.argsort(col_strength)
        order = [list(M.columns)[i] for i in order_idx]
        S_o = S.loc[order, order]
        cos, rel = _flat_corr(M.loc[order, order], M.loc[order, order].T)
        im = ax.imshow(S_o.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_xticks(range(len(order)))
        ax.set_yticks(range(len(order)))
        ax.set_xticklabels(order, rotation=90, fontsize=6)
        ax.set_yticklabels(order, fontsize=6)
        for ticklabels in (ax.get_xticklabels(), ax.get_yticklabels()):
            for tk in ticklabels:
                if tk.get_text() in SELF_ELEV_FAMILIES:
                    tk.set_color("#d62728")
        ax.set_title(
            f"{label}\ncos(M, M.T) = {cos:.2f} | ||M-M.T||/||M+M.T|| = {rel:.2f}",
            fontsize=9,
        )
        ax.set_xlabel("scored eval", fontsize=8)
        ax.set_ylabel("target family", fontsize=8)
    fig.suptitle(
        "Asymmetry of spillover: family-level signed matrix M minus its transpose (self-elevation in red)",
        fontsize=11,
    )
    cbar = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01, shrink=0.6)
    cbar.set_label(
        "M[A,B] - M[B,A]   (+ = train→B perturbs A more than train→A perturbs B)",
        fontsize=8,
    )
    out = RESULTS / "spillover_asymmetry_heatmap.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def _cell_pairs_for_method_comparison(panels):
    """Return dict[label] -> dict[(target_family, scored_eval)] -> |Δ| (mean across poles)."""
    out = {}
    for label, delta in panels:
        pole_family = {t: family_of(t) for t in delta.index}
        cells: dict[tuple[str, str], list[float]] = {}
        for trait in delta.index:
            fam = pole_family[trait]
            for ev in delta.columns:
                v = delta.loc[trait, ev]
                if pd.isna(v):
                    continue
                cells.setdefault((fam, ev), []).append(abs(float(v)))
        out[label] = {k: float(np.mean(v)) for k, v in cells.items() if v}
    return out


def plot_method_comparison_scatter(panels):
    """Pair each label with another for an |Δ| comparison.

    Pairs picked for what each contrast actually isolates:
      1. ICL 4B vs GRPO 4B           — same model, method
      2. GRPO 4B vs GRPO 8B-Instruct — same method, model size
      3. GRPO 8B-Base vs GRPO 8B-Instruct — same model size, base vs instruct
      4. GRPO 8B-Base vs SFT 8B-Base — same model, RL vs SFT
    """
    by_label = {label: delta for label, delta in panels}
    needed_pairs = [
        ("ICL (4B-Instruct)", "GRPO (4B-Instruct)"),
        ("GRPO (4B-Instruct)", "GRPO (8B-Instruct)"),
        ("GRPO (8B-Base)", "GRPO (8B-Instruct)"),
        ("GRPO (8B-Base)", "SFT (8B-Base)"),
    ]
    pairs = [(a, b) for a, b in needed_pairs if a in by_label and b in by_label]
    cells = _cell_pairs_for_method_comparison(panels)

    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.0))
    if n == 1:
        axes = [axes]

    for ax, (a, b) in zip(axes, pairs):
        ca, cb = cells[a], cells[b]
        common_keys = sorted(set(ca.keys()) & set(cb.keys()))
        xs = np.array([ca[k] for k in common_keys])
        ys = np.array([cb[k] for k in common_keys])
        on = np.array([k[0] == k[1] for k in common_keys])

        if len(xs) == 0:
            ax.set_visible(False)
            continue

        ax.scatter(
            xs[~on],
            ys[~on],
            c="#1f77b4",
            s=10,
            alpha=0.5,
            label=f"off-target ({(~on).sum()})",
        )
        ax.scatter(
            xs[on],
            ys[on],
            c="#d62728",
            s=22,
            alpha=0.85,
            label=f"on-target ({on.sum()})",
        )
        lim = max(xs.max(), ys.max()) * 1.05 if max(xs.max(), ys.max()) > 0 else 1.0
        ax.plot([0, lim], [0, lim], color="grey", lw=0.6, linestyle="--", alpha=0.5)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel(f"{a}: mean |Δ|")
        ax.set_ylabel(f"{b}: mean |Δ|")

        from scipy.stats import spearmanr

        def sp(x, y):
            if len(x) >= 3:
                r, _ = spearmanr(x, y)
                return r
            return float("nan")

        all_r = sp(xs, ys)
        on_r = sp(xs[on], ys[on])
        off_r = sp(xs[~on], ys[~on])
        ax.set_title(
            f"{a} vs {b}\nSpearman: on-target {on_r:+.2f}, off-target {off_r:+.2f}, all {all_r:+.2f}",
            fontsize=9,
        )
        ax.legend(loc="upper left", fontsize=7, frameon=False)
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Cross-method spillover comparison: each dot is one (target family, scored eval) cell",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = RESULTS / "spillover_method_comparison_scatter.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def main():
    panels = build_panels()
    plot_directional_asymmetry(panels)
    plot_per_trait_scatter(panels)
    plot_cosine_heatmap(panels)
    plot_asymmetry_heatmap(panels)
    plot_method_comparison_scatter(panels)


if __name__ == "__main__":
    main()
