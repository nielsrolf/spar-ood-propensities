"""Cross-method trait-cluster replication for Qwen 8B.

Compares the 9-trait Online DPO matrix against Ben's GRPO spillover cells for
Qwen3-8B-Base and Qwen3-8B-Instruct. The figure uses column-centered, row-z
profiles so clustering reflects relative spillover shape rather than eval-scale
offsets.
"""
from __future__ import annotations

from pathlib import Path
import csv
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import adjusted_rand_score, silhouette_score


HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[3]

OUT_DIR = HERE / "figures" / "cross_method_trait_replication"
DPO_SUMMARY = (
    HERE.parents[0]
    / "dpo"
    / "output"
    / "exports"
    / "online_dpo_qwen3-8b-base_summary_2026-05-19-08-40.csv"
)
BEN_ROOT = REPO_ROOT / "ben" / "propensities" / "results" / "cross_method_spillover"

TRAITS = [
    "spitefulness",
    "cooperation",
    "neuroticism",
    "honest-humble",
    "self-preservation",
    "power-seeking",
    "claiming-superintelligence",
    "harm-elaboration",
    "harm-refusal",
]

DISPLAY = {
    "spitefulness": "Spitefulness",
    "cooperation": "Cooperation",
    "neuroticism": "Neuroticism",
    "honest-humble": "Honest-humble",
    "self-preservation": "Self-preservation",
    "power-seeking": "Power-seeking",
    "claiming-superintelligence": "Claim super-int.",
    "harm-elaboration": "Harm elaboration",
    "harm-refusal": "Harm refusal",
}

METRIC = {
    "power-seeking": "power_seeking_score",
    "spitefulness": "spite_score",
    "self-preservation": "self_preservation_score",
    "cooperation": "cooperation_score",
    "neuroticism": "neuroticism_score",
    "honest-humble": "honesty_humility_score",
    "claiming-superintelligence": "superintelligence_claim_score",
    "harm-elaboration": "harm_elaboration_score",
    "harm-refusal": "harm_refusal_score",
}

# Intended positive-pole GRPO target labels for the 9 paper traits.
GRPO_PLUS = {
    "spitefulness": "spiteful",
    "cooperation": "high",
    "neuroticism": "neurotic",
    "honest-humble": "high_hh",
    "self-preservation": "self_preserving",
    "power-seeking": "power_seeking",
    "claiming-superintelligence": "superintelligent",
    "harm-elaboration": "harm_willing",
    "harm-refusal": "protective",
}


def load_dpo_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    mat = df.pivot(index="trained_trait", columns="eval_trait", values="mean_score")
    return mat.loc[TRAITS, TRAITS].astype(float)


def mean_metric(csv_path: Path, metric: str) -> float:
    vals: list[float] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            raw = row.get(metric)
            if raw not in (None, "", "None"):
                vals.append(float(raw))
    if not vals:
        return math.nan
    return float(np.mean(vals))


def load_grpo_cells(run_dir: str) -> pd.DataFrame:
    cells = BEN_ROOT / run_dir / "cells"
    mat = pd.DataFrame(index=TRAITS, columns=TRAITS, dtype=float)
    for trained in TRAITS:
        pole = GRPO_PLUS[trained]
        for eval_trait in TRAITS:
            path = cells / f"grpo__{trained}__{pole}__{eval_trait}.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            mat.loc[trained, eval_trait] = mean_metric(path, METRIC[eval_trait])
    return mat.astype(float)


def profile_matrix(mat: pd.DataFrame) -> pd.DataFrame:
    """Column-center, then row-z each trained-trait spillover profile."""
    centered = mat - mat.mean(axis=0)
    row_mean = centered.mean(axis=1)
    row_std = centered.std(axis=1, ddof=0).replace(0, np.nan)
    return centered.sub(row_mean, axis=0).div(row_std, axis=0).fillna(0.0)


def row_corrs(reference: pd.DataFrame, other: pd.DataFrame) -> pd.Series:
    vals = {}
    for trait in TRAITS:
        r, _ = pearsonr(reference.loc[trait].values, other.loc[trait].values)
        vals[trait] = r
    return pd.Series(vals)


def trait_similarity(profile: pd.DataFrame) -> pd.DataFrame:
    return profile.T.corr(method="pearson")


def upper_triangle_values(mat: pd.DataFrame) -> np.ndarray:
    idx = np.triu_indices_from(mat.values, k=1)
    return mat.values[idx]


def replication_stats(profiles: dict[str, pd.DataFrame], order_linkage: np.ndarray) -> pd.DataFrame:
    dpo = profiles["Online DPO"]
    dpo_sim = trait_similarity(dpo)
    rows = []
    dpo_clusters = fcluster(linkage_for_profile(dpo), t=3, criterion="maxclust")
    for name, prof in profiles.items():
        sim = trait_similarity(prof)
        flat_r = pearsonr(dpo.values.ravel(), prof.values.ravel()).statistic
        tri_r = spearmanr(upper_triangle_values(dpo_sim), upper_triangle_values(sim)).statistic
        clusters = fcluster(linkage_for_profile(prof), t=3, criterion="maxclust")
        rows.append({
            "method": name,
            "profile_pearson_vs_dpo": 1.0 if name == "Online DPO" else flat_r,
            "trait_similarity_spearman_vs_dpo": 1.0 if name == "Online DPO" else tri_r,
            "ari_k3_vs_dpo": 1.0 if name == "Online DPO" else adjusted_rand_score(dpo_clusters, clusters),
        })
    return pd.DataFrame(rows)


def linkage_for_profile(profile: pd.DataFrame) -> np.ndarray:
    corr = profile.T.corr().clip(-1, 1)
    dist = 1 - corr
    np.fill_diagonal(dist.values, 0)
    return linkage(squareform(dist.values, checks=False), method="average")


def distance_for_profile(profile: pd.DataFrame) -> pd.DataFrame:
    corr = profile.T.corr().clip(-1, 1)
    dist = 1 - corr
    np.fill_diagonal(dist.values, 0)
    return dist


def choose_cluster_count(profile: pd.DataFrame, max_k: int = 5) -> tuple[int, pd.DataFrame]:
    """Pick k by silhouette on precomputed correlation distance."""
    Z = linkage_for_profile(profile)
    dist = distance_for_profile(profile).values
    rows = []
    best_k, best_s = 2, -np.inf
    for k in range(2, min(max_k, len(TRAITS) - 1) + 1):
        labels = fcluster(Z, t=k, criterion="maxclust")
        s = silhouette_score(dist, labels, metric="precomputed")
        rows.append({"k": k, "silhouette": s})
        if s > best_s:
            best_k, best_s = k, s
    return best_k, pd.DataFrame(rows)


def save_matrix_csvs(raw: dict[str, pd.DataFrame], profiles: dict[str, pd.DataFrame]) -> None:
    for name, mat in raw.items():
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        mat.to_csv(OUT_DIR / f"{slug}_raw_scores.csv")
        profiles[name].to_csv(OUT_DIR / f"{slug}_profile_z.csv")


def plot_figure(profiles: dict[str, pd.DataFrame], rows: pd.DataFrame) -> list[str]:
    avg_profile = sum(profiles.values()) / len(profiles)
    Z = linkage_for_profile(avg_profile)
    leaves = dendrogram(Z, no_plot=True)["leaves"]
    order = [TRAITS[i] for i in leaves]

    ordered = {name: prof.loc[order, order] for name, prof in profiles.items()}
    row_corr_table = pd.DataFrame({
        "GRPO Base": row_corrs(profiles["Online DPO"], profiles["GRPO Base"]).loc[order],
        "GRPO Instruct": row_corrs(profiles["Online DPO"], profiles["GRPO Instruct"]).loc[order],
    })

    fig = plt.figure(figsize=(13.2, 7.4))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=5,
        width_ratios=[0.75, 2.5, 2.5, 2.5, 1.25],
        wspace=0.08,
    )

    ax_den = fig.add_subplot(gs[0, 0])
    dendrogram(
        Z,
        orientation="left",
        labels=[DISPLAY[t] for t in TRAITS],
        ax=ax_den,
        color_threshold=0,
        above_threshold_color="#444444",
    )
    ax_den.set_xticks([])
    ax_den.set_yticks([])
    for spine in ax_den.spines.values():
        spine.set_visible(False)

    vmax = max(float(np.abs(p.values).max()) for p in profiles.values())
    vmax = max(1.5, min(2.5, vmax))
    axes = []
    for i, (name, mat) in enumerate(ordered.items(), start=1):
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)
        im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(name, fontsize=11, fontweight="bold", pad=10)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([DISPLAY[t] for t in order], rotation=55, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([DISPLAY[t] for t in order] if i == 1 else [], fontsize=8.5)
        ax.tick_params(length=0)
        for x in np.arange(-0.5, len(order), 1):
            ax.axvline(x, color="white", lw=0.55, alpha=0.65)
            ax.axhline(x, color="white", lw=0.55, alpha=0.65)
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax_corr = fig.add_subplot(gs[0, 4])
    corr_vals = row_corr_table.values
    ax_corr.imshow(corr_vals, cmap="PRGn", vmin=-1, vmax=1, aspect="auto")
    ax_corr.set_title("Row profile\nreplication", fontsize=10.5, fontweight="bold", pad=8)
    ax_corr.set_xticks([0, 1])
    ax_corr.set_xticklabels(["Base", "Instr."], fontsize=8)
    ax_corr.set_yticks(range(len(order)))
    ax_corr.set_yticklabels([])
    ax_corr.tick_params(length=0)
    for y in range(len(order)):
        for x in range(2):
            v = corr_vals[y, x]
            ax_corr.text(x, y, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                         color="white" if abs(v) > 0.55 else "#222222")
    for spine in ax_corr.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.025)
    cbar.set_label("Column-centered row z-score", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Cross-method replication of Qwen trait spillover structure",
        fontsize=13,
        fontweight="bold",
        y=0.965,
    )
    fig.subplots_adjust(left=0.055, right=0.965, top=0.86, bottom=0.23)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "cross_method_trait_replication.png"
    pdf = OUT_DIR / "cross_method_trait_replication.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    row_corr_table.to_csv(OUT_DIR / "row_profile_correlations_vs_dpo.csv")
    rows.to_csv(OUT_DIR / "method_replication_stats.csv", index=False)
    return order


def plot_natural_clusters(profiles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build clusters independently within each method, without a shared order."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12.5, 7.6))
    gs = fig.add_gridspec(nrows=1, ncols=3, wspace=0.22)

    all_assignments = []
    all_silhouettes = []
    vmax = max(float(np.abs(p.values).max()) for p in profiles.values())
    vmax = max(1.5, min(2.5, vmax))

    for ci, (name, prof) in enumerate(profiles.items()):
        k_best, sil = choose_cluster_count(prof)
        sil.insert(0, "method", name)
        all_silhouettes.append(sil)

        Z = linkage_for_profile(prof)
        leaves = dendrogram(Z, no_plot=True)["leaves"]
        order = [TRAITS[i] for i in leaves]
        labels = fcluster(Z, t=k_best, criterion="maxclust")
        label_by_trait = dict(zip(TRAITS, labels))
        for trait in TRAITS:
            all_assignments.append({
                "method": name,
                "trait": trait,
                "display": DISPLAY[trait],
                "best_k": k_best,
                "cluster": int(label_by_trait[trait]),
            })

        sub = gs[0, ci].subgridspec(nrows=1, ncols=3, width_ratios=[0.45, 0.12, 2.6], wspace=0.02)
        ax_den = fig.add_subplot(sub[0, 0])
        dendrogram(
            Z,
            orientation="left",
            labels=[DISPLAY[t] for t in TRAITS],
            ax=ax_den,
            color_threshold=0,
            above_threshold_color="#444444",
        )
        ax_den.set_xticks([])
        ax_den.set_yticks([])
        for spine in ax_den.spines.values():
            spine.set_visible(False)

        ax_strip = fig.add_subplot(sub[0, 1])
        cluster_vals = np.array([label_by_trait[t] for t in order], dtype=float)[:, None]
        ax_strip.imshow(cluster_vals, cmap="tab10", aspect="auto")
        ax_strip.set_xticks([])
        ax_strip.set_yticks([])
        for spine in ax_strip.spines.values():
            spine.set_visible(False)

        ax = fig.add_subplot(sub[0, 2])
        mat = prof.loc[order, order]
        im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"{name}\nindependent k={k_best}", fontsize=11, fontweight="bold", pad=8)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([DISPLAY[t] for t in order], rotation=55, ha="right", fontsize=7.2)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([DISPLAY[t] for t in order], fontsize=7.8)
        ax.tick_params(length=0)
        for x in np.arange(-0.5, len(order), 1):
            ax.axvline(x, color="white", lw=0.55, alpha=0.65)
            ax.axhline(x, color="white", lw=0.55, alpha=0.65)
        for spine in ax.spines.values():
            spine.set_visible(False)

    cbar = fig.colorbar(im, ax=fig.axes, fraction=0.018, pad=0.012)
    cbar.set_label("Column-centered row z-score", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.suptitle(
        "Natural trait clusters discovered separately within each method",
        fontsize=13,
        fontweight="bold",
        y=0.965,
    )
    fig.subplots_adjust(left=0.04, right=0.95, top=0.85, bottom=0.24)

    png = OUT_DIR / "natural_trait_clusters_by_method.png"
    pdf = OUT_DIR / "natural_trait_clusters_by_method.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    assignments = pd.DataFrame(all_assignments)
    silhouettes = pd.concat(all_silhouettes, ignore_index=True)
    assignments.to_csv(OUT_DIR / "natural_cluster_assignments.csv", index=False)
    silhouettes.to_csv(OUT_DIR / "natural_cluster_silhouettes.csv", index=False)
    return assignments


def _flow_patch(x0: float, y0a: float, y0b: float,
                x1: float, y1a: float, y1b: float,
                color: str, alpha: float = 0.28) -> PathPatch:
    """Smooth ribbon from [y0a,y0b] at x0 to [y1a,y1b] at x1."""
    dx = x1 - x0
    c0 = x0 + 0.42 * dx
    c1 = x1 - 0.42 * dx
    verts = [
        (x0, y0a),
        (c0, y0a), (c1, y1a), (x1, y1a),
        (x1, y1b),
        (c1, y1b), (c0, y0b), (x0, y0b),
        (x0, y0a),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    return PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha)


def plot_cluster_flow(assignments: pd.DataFrame) -> None:
    """Alluvial diagram of independently discovered cluster memberships."""
    methods = ["Online DPO", "GRPO Base", "GRPO Instruct"]
    xs = {m: i for i, m in enumerate(methods)}
    short = {
        "spitefulness": "Spite",
        "cooperation": "Coop",
        "neuroticism": "Neuro",
        "honest-humble": "Honest-\nhumble",
        "self-preservation": "Self-\npres.",
        "power-seeking": "Power",
        "claiming-superintelligence": "Claim\nsuper-int.",
        "harm-elaboration": "Harm\nelab.",
        "harm-refusal": "Harm\nrefusal",
    }
    trait_colors = {
        "spitefulness": "#C0392B",
        "cooperation": "#27AE60",
        "neuroticism": "#8E44AD",
        "honest-humble": "#2ECC71",
        "self-preservation": "#7D3C98",
        "power-seeking": "#B03A2E",
        "claiming-superintelligence": "#D4AC0D",
        "harm-elaboration": "#E67E22",
        "harm-refusal": "#2980B9",
    }
    cluster_palette = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]

    # Stable vertical layout: clusters ordered by their mean trait index in the
    # method-specific dendrogram assignment table. Blocks are sized by n traits.
    y_gap = 0.075
    unit = 0.06
    block_pos: dict[tuple[str, int], tuple[float, float]] = {}
    trait_slots: dict[tuple[str, str], tuple[float, float]] = {}
    cluster_labels: dict[tuple[str, int], str] = {}

    for m in methods:
        sub = assignments[assignments["method"] == m].copy()
        sub["trait_order"] = sub["trait"].map({t: i for i, t in enumerate(TRAITS)})
        clusters = []
        for cl, csub in sub.groupby("cluster"):
            clusters.append((csub["trait_order"].mean(), int(cl), csub))
        clusters.sort()

        total_h = sum(len(csub) * unit for _, _, csub in clusters) + y_gap * (len(clusters) - 1)
        y = 0.5 + total_h / 2
        for idx, (_, cl, csub) in enumerate(clusters):
            h = len(csub) * unit
            y0, y1 = y - h, y
            block_pos[(m, cl)] = (y0, y1)
            traits = csub.sort_values("trait_order")["trait"].tolist()
            cluster_labels[(m, cl)] = "\n".join(short[t] for t in traits)
            slot_top = y1
            for t in traits:
                trait_slots[(m, t)] = (slot_top - unit, slot_top)
                slot_top -= unit
            y = y0 - y_gap

    fig, ax = plt.subplots(figsize=(12.4, 5.6))
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Ribbons: one per trait between adjacent methods.
    for left, right in zip(methods[:-1], methods[1:]):
        for trait in TRAITS:
            y0a, y0b = trait_slots[(left, trait)]
            y1a, y1b = trait_slots[(right, trait)]
            ax.add_patch(_flow_patch(
                xs[left] + 0.16, y0a, y0b,
                xs[right] - 0.16, y1a, y1b,
                trait_colors[trait], alpha=0.24,
            ))

    # Cluster blocks and labels.
    for m in methods:
        x = xs[m]
        sub = assignments[assignments["method"] == m]
        k = int(sub["best_k"].iloc[0])
        ax.text(x, 0.965, f"{m}\nindependent k={k}",
                ha="center", va="top", fontsize=12.5, fontweight="bold")
        for j, cl in enumerate(sorted(sub["cluster"].unique())):
            y0, y1 = block_pos[(m, int(cl))]
            ax.add_patch(Rectangle(
                (x - 0.16, y0), 0.32, y1 - y0,
                facecolor=cluster_palette[j % len(cluster_palette)],
                edgecolor="white", linewidth=1.0, alpha=0.92,
            ))
            ax.text(x, (y0 + y1) / 2, cluster_labels[(m, int(cl))],
                    ha="center", va="center", fontsize=8.3, color="white",
                    fontweight="bold", linespacing=1.05)

    fig.suptitle(
        "Trait clusters split and merge across optimization methods",
        fontsize=14.5, fontweight="bold", y=0.99,
    )
    png = OUT_DIR / "natural_cluster_flow.png"
    pdf = OUT_DIR / "natural_cluster_flow.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def write_summary(raw: dict[str, pd.DataFrame], profiles: dict[str, pd.DataFrame],
                  stats: pd.DataFrame, order: list[str],
                  natural_assignments: pd.DataFrame) -> None:
    table = stats.copy()
    for col in table.columns:
        if col != "method":
            table[col] = table[col].map(lambda x: f"{x:.3f}")
    markdown_table = table.to_csv(index=False)
    lines = [
        "# Cross-method trait-cluster replication",
        "",
        "Input matrices: Online DPO Qwen3-8B-Base summary and Ben's GRPO spillover cells for Qwen3-8B-Base/Qwen3-8B-Instruct.",
        "Analysis scope: 9 shared paper traits. GRPO uses the intended positive-pole label for each trained trait.",
        "",
        "## Method-level replication vs Online DPO",
        "",
        "```csv",
        markdown_table.strip(),
        "```",
        "",
        "## Shared cluster order",
        "",
        ", ".join(DISPLAY[t] for t in order),
        "",
        "## Natural clusters within each method",
        "",
    ]
    for method, sub in natural_assignments.groupby("method", sort=False):
        lines.extend([f"### {method}", ""])
        for cluster, csub in sub.groupby("cluster"):
            traits = ", ".join(csub["display"].tolist())
            lines.append(f"- Cluster {cluster}: {traits}")
        lines.append("")
    lines.extend([
        "",
        "## Notes",
        "",
        "- Heatmaps use column-centered, row-z profiles: they show relative spillover shape, not absolute judge score.",
        "- `profile_pearson_vs_dpo` compares all 81 standardized cells.",
        "- `trait_similarity_spearman_vs_dpo` compares the upper triangle of trait-trait similarity matrices.",
        "- `ari_k3_vs_dpo` compares 3-cluster hierarchical assignments; with only 9 traits, treat it as a descriptive stability check.",
        "- `natural_trait_clusters_by_method` orders and clusters traits independently inside each method, using silhouette-selected k from 2 to 5.",
    ])
    (OUT_DIR / "analysis_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = {
        "Online DPO": load_dpo_summary(DPO_SUMMARY),
        "GRPO Base": load_grpo_cells("qwen3_8b_base"),
        "GRPO Instruct": load_grpo_cells("qwen3_8b_instruct"),
    }
    profiles = {name: profile_matrix(mat) for name, mat in raw.items()}
    stats = replication_stats(profiles, linkage_for_profile(sum(profiles.values()) / len(profiles)))
    save_matrix_csvs(raw, profiles)
    order = plot_figure(profiles, stats)
    natural_assignments = plot_natural_clusters(profiles)
    plot_cluster_flow(natural_assignments)
    write_summary(raw, profiles, stats, order, natural_assignments)
    print(f"Saved figure and summaries to: {OUT_DIR}")
    print(stats.to_string(index=False))
    print("\nNatural clusters:")
    print(natural_assignments.to_string(index=False))


if __name__ == "__main__":
    main()
