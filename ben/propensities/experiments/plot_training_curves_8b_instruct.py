"""Plot GRPO training reward curves for 8B-Instruct, colored by post-training PC1.

Companion to the existing 8B-Base curves. Also writes a side-by-side
comparison so the contrast (8B-Base spreads + spikes; 8B-Instruct tight)
is visible at a glance.

Outputs:
  results/cross_method_spillover/training_curves_8b_instruct_pc1.png
  results/cross_method_spillover/training_curves_8b_base_vs_instruct.png
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
RUNS = {
    "8B-Base": RESULTS / "qwen3_8b_base",
    "8B-Instruct": RESULTS / "qwen3_8b_instruct",
}


def primary_metric(eval_name: str) -> str | None:
    try:
        return EvalConfig(eval_name, evals_root=ORTHOGONALIZED_EVALS_DIR).judge_metrics[
            0
        ]
    except (FileNotFoundError, ValueError, KeyError):
        return None


def delta_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return Δ matrix: rows=target_trait, cols=eval_name, values=mean(grpo)-mean(baseline)."""
    grpo = df[df["method"] == "grpo"]
    base = df[df["method"] == "baseline"]
    eval_names = sorted(df["eval_name"].unique())
    traits = sorted(grpo["target_trait"].unique())

    rows = []
    for trait in traits:
        row = {"target_trait": trait}
        for ev in eval_names:
            metric = primary_metric(ev)
            if metric is None or metric not in df.columns:
                continue
            g = grpo[(grpo["target_trait"] == trait) & (grpo["eval_name"] == ev)][
                metric
            ].dropna()
            b = base[base["eval_name"] == ev][metric].dropna()
            if len(g) == 0 or len(b) == 0:
                continue
            row[ev] = float(g.mean()) - float(b.mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("target_trait")


def pc1_projections(delta: pd.DataFrame) -> pd.Series:
    """SVD on column-centered Δ matrix; return trait projection onto PC1."""
    df = delta.dropna(axis=1, how="any")
    X = df.to_numpy(dtype=float)
    X_c = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    proj = U[:, 0] * S[0]
    # Sign convention: ensure PC1+ correlates with higher mean |delta| (handy for "more extreme" end).
    if np.corrcoef(proj, np.abs(X_c).mean(axis=1))[0, 1] < 0:
        proj = -proj
    var_explained = (S**2 / (S**2).sum())[0]
    out = pd.Series(proj, index=df.index, name="pc1")
    out.attrs["var_explained"] = float(var_explained)
    return out


def load_metrics(log_dir: Path) -> pd.DataFrame:
    rows = []
    f = log_dir / "metrics.jsonl"
    if not f.exists():
        return pd.DataFrame()
    with f.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def load_trait_curves(run_dir: Path) -> dict[str, pd.DataFrame]:
    """trait label -> metrics dataframe with step/reward/null_rate."""
    tm_path = run_dir / "trained_models.json"
    with tm_path.open() as f:
        tm = json.load(f)
    out: dict[str, pd.DataFrame] = {}
    for key, info in tm.items():
        if not key.startswith("grpo:"):
            continue
        trait_label = key[len("grpo:") :]
        log_path = Path(info["log_path"])
        m = load_metrics(log_path)
        if len(m) == 0:
            continue
        out[trait_label] = m
    return out


def plot_run(run_label: str, run_dir: Path, ax_reward, ax_null, cmap_norm, cmap):
    df = pd.read_csv(run_dir / "spillover_results.csv", low_memory=False)
    delta = delta_matrix(df)
    pc1 = pc1_projections(delta)
    curves = load_trait_curves(run_dir)

    common = sorted(set(curves.keys()) & set(pc1.index))
    vmin, vmax = pc1.loc[common].min(), pc1.loc[common].max()
    span = max(abs(vmin), abs(vmax))

    for trait in common:
        m = curves[trait]
        if "step" not in m.columns or "reward/total" not in m.columns:
            continue
        m = m.sort_values("step")
        c = cmap(cmap_norm(pc1.loc[trait]))
        ax_reward.plot(m["step"], m["reward/total"], color=c, alpha=0.7, linewidth=1.2)
        if "reward/null_rate" in m.columns:
            ax_null.plot(
                m["step"], m["reward/null_rate"], color=c, alpha=0.7, linewidth=1.2
            )

    n = len(common)
    pc1_var = pc1.attrs.get("var_explained", float("nan"))
    ax_reward.set_title(
        f"{run_label} — GRPO reward (n={n} traits)\n"
        f"PC1 var explained = {pc1_var * 100:.1f}%  |  PC1 range [{vmin:+.2f}, {vmax:+.2f}]",
        fontsize=10,
    )
    ax_reward.set_ylabel("reward/total")
    ax_reward.set_ylim(0, 1)
    ax_reward.grid(alpha=0.3)

    ax_null.set_ylabel("null rate")
    ax_null.set_xlabel("training batch")
    ax_null.set_ylim(0, 1)
    ax_null.grid(alpha=0.3)

    return span


def main():
    cmap = plt.get_cmap("coolwarm")
    from matplotlib.colors import Normalize

    # Pass 1: compute global PC1 span across both runs for shared colormap normalization.
    spans = {}
    pc1s = {}
    curves_cache = {}
    for label, run_dir in RUNS.items():
        df = pd.read_csv(run_dir / "spillover_results.csv", low_memory=False)
        delta = delta_matrix(df)
        pc1 = pc1_projections(delta)
        curves = load_trait_curves(run_dir)
        common = sorted(set(curves.keys()) & set(pc1.index))
        spans[label] = max(abs(pc1.loc[common].min()), abs(pc1.loc[common].max()))
        pc1s[label] = pc1
        curves_cache[label] = (curves, common)

    # Two figures:
    #   1. 8B-Instruct only (reward + null rate stacked)
    #   2. 8B-Base vs 8B-Instruct (reward only, side-by-side)

    # Figure 1: 8B-Instruct two-panel
    fig, (ax_r, ax_n) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    pc1 = pc1s["8B-Instruct"]
    curves, common = curves_cache["8B-Instruct"]
    span = spans["8B-Instruct"]
    norm = Normalize(vmin=-span, vmax=span)
    for trait in common:
        m = curves[trait]
        if "step" not in m.columns or "reward/total" not in m.columns:
            continue
        m = m.sort_values("step")
        c = cmap(norm(pc1.loc[trait]))
        ax_r.plot(m["step"], m["reward/total"], color=c, alpha=0.75, linewidth=1.2)
        if "reward/null_rate" in m.columns:
            ax_n.plot(
                m["step"], m["reward/null_rate"], color=c, alpha=0.75, linewidth=1.2
            )

    pc1_var = pc1.attrs.get("var_explained", float("nan"))
    final_rewards = []
    for trait in common:
        m = curves[trait]
        if "step" in m.columns and "reward/total" in m.columns:
            final_rewards.append(float(m.sort_values("step").iloc[-1]["reward/total"]))
    fr = np.array(final_rewards)
    fig.suptitle(
        f"8B-Instruct GRPO training dynamics — colored by post-training PC1 projection\n"
        f"n={len(common)} traits | PC1 = {pc1_var * 100:.1f}% var | "
        f"final reward {fr.mean():.2f} ± {fr.std():.2f}",
        fontsize=11,
    )
    ax_r.set_title("Reward over training (color = PC1: blue=-, red=+)", fontsize=10)
    ax_r.set_ylabel("reward/total")
    ax_r.set_ylim(0, 1)
    ax_r.grid(alpha=0.3)
    ax_n.set_title("Null rate over training", fontsize=10)
    ax_n.set_ylabel("null rate (fraction of all-null groups + null judgments)")
    ax_n.set_xlabel("training batch")
    ax_n.set_ylim(0, 1)
    ax_n.grid(alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=[ax_r, ax_n], label="PC1 projection", shrink=0.7)

    fig.savefig(
        RESULTS / "training_curves_8b_instruct_pc1.png", dpi=150, bbox_inches="tight"
    )
    print(f"wrote {RESULTS / 'training_curves_8b_instruct_pc1.png'}")
    plt.close(fig)

    # Figure 2: side-by-side comparison (reward only)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, label in zip(axes, ["8B-Base", "8B-Instruct"]):
        pc1 = pc1s[label]
        curves, common = curves_cache[label]
        span = spans[label]
        norm = Normalize(vmin=-span, vmax=span)
        finals = []
        for trait in common:
            m = curves[trait]
            if "step" not in m.columns or "reward/total" not in m.columns:
                continue
            m = m.sort_values("step")
            c = cmap(norm(pc1.loc[trait]))
            ax.plot(m["step"], m["reward/total"], color=c, alpha=0.7, linewidth=1.0)
            finals.append(float(m.iloc[-1]["reward/total"]))
        fr = np.array(finals)
        var = pc1.attrs.get("var_explained", float("nan"))
        ax.set_title(
            f"{label}  (n={len(common)})\n"
            f"PC1 = {var * 100:.1f}% var | final r = {fr.mean():.2f} ± {fr.std():.2f}",
            fontsize=10,
        )
        ax.set_xlabel("training batch")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("reward/total")
    fig.suptitle(
        "GRPO training reward curves — 8B-Base vs 8B-Instruct\n"
        "(color = PC1 projection of post-training Δ matrix; blue=-, red=+)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(
        RESULTS / "training_curves_8b_base_vs_instruct.png",
        dpi=150,
        bbox_inches="tight",
    )
    print(f"wrote {RESULTS / 'training_curves_8b_base_vs_instruct.png'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
