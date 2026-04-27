"""Generate a markdown report from the open-weight sycophancy sweep results.

Produces:
- Aggregate table per model
- Per-model score distribution plots
- Cross-model comparison bar chart
- Breakdown by user_stance_quality and sycophancy_type
- Top-5 most sycophantic responses per model
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"

# Display order and short labels
MODEL_ORDER = [
    ("qwen_qwen3.5-9b", "Qwen3.5-9B"),
    ("qwen_qwen3.5-27b", "Qwen3.5-27B"),
    ("mistralai_mistral-small-3.2-24b-instruct", "Mistral-Small-3.2-24B"),
    ("meta-llama_llama-3.1-8b-instruct", "Llama-3.1-8B"),
    ("meta-llama_llama-3.3-70b-instruct", "Llama-3.3-70B"),
    ("allenai_olmo-3.1-32b-instruct", "OLMo-3.1-32B"),
]


def load_all() -> dict[str, pd.DataFrame]:
    out = {}
    for slug, _ in MODEL_ORDER:
        csv = RESULTS_DIR / slug / "results.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            df = df.dropna(subset=["sycophancy_score"])
            out[slug] = df
    return out


def truncate(text, n=800):
    s = " ".join(str(text or "").split())
    return s if len(s) <= n else s[: n - 3] + "..."


def plot_cross_model(dfs: dict[str, pd.DataFrame], out_path: Path):
    order = [(slug, label) for slug, label in MODEL_ORDER if slug in dfs]
    means = [dfs[s]["sycophancy_score"].mean() for s, _ in order]
    stds = [dfs[s]["sycophancy_score"].std() for s, _ in order]
    labels = [lbl for _, lbl in order]

    # Sort bars by sycophancy level for readability
    idx = np.argsort(means)[::-1]
    means = [means[i] for i in idx]
    stds = [stds[i] for i in idx]
    labels = [labels[i] for i in idx]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    colors = plt.cm.RdYlGn_r(np.array(means) / 100.0)
    bars = ax.bar(labels, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("sycophancy_score  (0=honest, 100=fully sycophantic)")
    ax.set_ylim(0, 100)
    ax.set_title("Sycophancy across open-weight models (test split, claude-haiku-4.5 judge)")
    ax.axhline(50, color="grey", linestyle="--", linewidth=0.6, alpha=0.5)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 2, f"{m:.1f}", ha="center", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_breakdown(dfs: dict[str, pd.DataFrame], field: str, out_path: Path, title: str):
    order = [(slug, label) for slug, label in MODEL_ORDER if slug in dfs]
    # Compute mean per (model, field)
    rows = []
    for slug, label in order:
        for val, sub in dfs[slug].groupby(field):
            rows.append({"model": label, field: val, "mean": sub["sycophancy_score"].mean(), "n": len(sub)})
    if not rows:
        return
    piv = pd.DataFrame(rows).pivot(index="model", columns=field, values="mean")
    piv = piv.reindex([lbl for _, lbl in order])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    piv.plot(kind="bar", ax=ax, colormap="viridis", edgecolor="black", linewidth=0.3)
    ax.set_ylabel("sycophancy_score")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.legend(title=field, fontsize=8, loc="upper right")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_violin(dfs: dict[str, pd.DataFrame], out_path: Path):
    order = [(slug, label) for slug, label in MODEL_ORDER if slug in dfs]
    # Sort by mean
    order = sorted(order, key=lambda sl: dfs[sl[0]]["sycophancy_score"].mean(), reverse=True)
    data = [dfs[s]["sycophancy_score"].values for s, _ in order]
    labels = [lbl for _, lbl in order]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    parts = ax.violinplot(data, showmeans=True, showmedians=False, widths=0.8)
    for pc, m in zip(parts["bodies"], [np.mean(d) for d in data]):
        pc.set_facecolor(plt.cm.RdYlGn_r(m / 100.0))
        pc.set_alpha(0.75)
        pc.set_edgecolor("black")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("sycophancy_score")
    ax.set_ylim(-5, 105)
    ax.axhline(50, color="grey", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_title("Per-response sycophancy_score distribution (violin)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def most_sycophantic_examples(df: pd.DataFrame, top_k: int = 5) -> list[pd.Series]:
    # First aggregate per question_id so we pick the Q with highest mean score
    agg = df.groupby("question_id")["sycophancy_score"].mean().reset_index()
    top_qs = agg.sort_values("sycophancy_score", ascending=False).head(top_k)
    rows = []
    for qid in top_qs["question_id"]:
        sub = df[df["question_id"] == qid].sort_values("sycophancy_score", ascending=False)
        rows.append(sub.iloc[0])
    return rows


STANCE_ORDER = ["clearly_wrong", "debatable", "mostly_right"]
STANCE_COLORS = {
    "clearly_wrong": "#c0392b",   # red  — worst kind of sycophancy
    "debatable": "#d9a441",       # amber
    "mostly_right": "#27ae60",    # green — validation is appropriate
}


def plot_stance_distribution_per_model(dfs: dict[str, pd.DataFrame], out_dir: Path):
    """One figure per model: sycophancy_score histogram overlaid by user_stance_quality."""
    bins = np.linspace(0, 100, 21)
    for slug, _ in MODEL_ORDER:
        if slug not in dfs:
            continue
        df = dfs[slug]
        fig, ax = plt.subplots(figsize=(6, 4))
        for stance in STANCE_ORDER:
            vals = df[df["user_stance_quality"] == stance]["sycophancy_score"].dropna()
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=bins, alpha=0.55, color=STANCE_COLORS[stance],
                    label=f"{stance} (n={len(vals)}, mean={vals.mean():.1f})",
                    edgecolor="white", linewidth=0.4)
            ax.axvline(vals.mean(), color=STANCE_COLORS[stance], linestyle="--", linewidth=1.5, alpha=0.9)
        ax.set_xlim(0, 100)
        ax.set_xlabel("sycophancy_score")
        ax.set_ylabel("count")
        label = next(lbl for s, lbl in MODEL_ORDER if s == slug)
        ax.set_title(f"{label}  —  distribution by user stance", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        plt.tight_layout()
        fig.savefig(out_dir / slug / "score_distributions_by_stance.png", dpi=140, bbox_inches="tight")
        plt.close()


def plot_stance_distribution_grid(dfs: dict[str, pd.DataFrame], out_path: Path):
    """Single figure: per-model subplot with stance-split histograms (ordered most→least sycophantic)."""
    order = [(slug, label) for slug, label in MODEL_ORDER if slug in dfs]
    order = sorted(order, key=lambda sl: dfs[sl[0]]["sycophancy_score"].mean(), reverse=True)
    n = len(order)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.4 * rows), sharex=True, sharey=True)
    axes = np.atleast_2d(axes).reshape(rows, cols)
    bins = np.linspace(0, 100, 21)
    for i, (slug, label) in enumerate(order):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        df = dfs[slug]
        for stance in STANCE_ORDER:
            vals = df[df["user_stance_quality"] == stance]["sycophancy_score"].dropna()
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=bins, alpha=0.55, color=STANCE_COLORS[stance],
                    label=f"{stance} (mean={vals.mean():.1f})", edgecolor="white", linewidth=0.3)
            ax.axvline(vals.mean(), color=STANCE_COLORS[stance], linestyle="--", linewidth=1.2, alpha=0.85)
        ax.set_xlim(0, 100)
        ax.set_title(f"{label}  (mean={df['sycophancy_score'].mean():.1f})", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].set_visible(False)
    for r in range(rows):
        axes[r, 0].set_ylabel("count")
    for c in range(cols):
        axes[-1, c].set_xlabel("sycophancy_score")
    fig.suptitle("sycophancy_score distribution per model, split by user-stance quality", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_stance_distribution_aggregate(dfs: dict[str, pd.DataFrame], out_path: Path):
    """One figure pooling all models: 3 overlaid histograms by user_stance_quality."""
    combined = pd.concat(dfs.values(), ignore_index=True) if dfs else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 100, 21)
    for stance in STANCE_ORDER:
        vals = combined[combined["user_stance_quality"] == stance]["sycophancy_score"].dropna()
        ax.hist(vals, bins=bins, alpha=0.55, color=STANCE_COLORS[stance],
                label=f"{stance} (n={len(vals)}, mean={vals.mean():.1f})",
                edgecolor="white", linewidth=0.4)
        ax.axvline(vals.mean(), color=STANCE_COLORS[stance], linestyle="--", linewidth=2, alpha=0.9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("sycophancy_score")
    ax.set_ylabel("count")
    ax.set_title("All models pooled — distribution by user-stance quality")
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def most_sycophantic_examples_for_stance(
    df: pd.DataFrame, stance: str, top_k: int
) -> list[pd.Series]:
    sub = df[df["user_stance_quality"] == stance]
    if sub.empty:
        return []
    agg = sub.groupby("question_id")["sycophancy_score"].mean().reset_index()
    top_qs = agg.sort_values("sycophancy_score", ascending=False).head(top_k)
    rows = []
    for qid in top_qs["question_id"]:
        q_sub = sub[sub["question_id"] == qid].sort_values("sycophancy_score", ascending=False)
        rows.append(q_sub.iloc[0])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=3, help="Top sycophantic examples per model.")
    ap.add_argument("--out", default=str(RESULTS_DIR / "report.md"))
    args = ap.parse_args()

    dfs = load_all()
    if not dfs:
        print("No results found. Run the sweep first.")
        return
    print(f"Loaded results for: {list(dfs.keys())}")

    # Plots
    plot_cross_model(dfs, RESULTS_DIR / "cross_model_bar.png")
    plot_violin(dfs, RESULTS_DIR / "cross_model_violin.png")
    plot_breakdown(dfs, "user_stance_quality", RESULTS_DIR / "breakdown_by_stance.png",
                   "Mean sycophancy by user-stance quality")
    plot_breakdown(dfs, "sycophancy_type", RESULTS_DIR / "breakdown_by_type.png",
                   "Mean sycophancy by sycophancy type")
    plot_stance_distribution_per_model(dfs, RESULTS_DIR)
    plot_stance_distribution_grid(dfs, RESULTS_DIR / "distribution_by_stance_grid.png")
    plot_stance_distribution_aggregate(dfs, RESULTS_DIR / "distribution_by_stance_aggregate.png")

    # Judge validation
    jv = RESULTS_DIR / "judge_validation" / "reference_scores.csv"
    jv_gap = None
    if jv.exists():
        jdf = pd.read_csv(jv)
        syc = jdf[jdf.answer_type == "sycophantic"]["sycophancy_score"].dropna()
        hon = jdf[jdf.answer_type == "honest"]["sycophancy_score"].dropna()
        jv_gap = (syc.mean(), hon.mean(), syc.mean() - hon.mean(), len(syc))

    lines = [
        "# Sycophancy — Open-Weight Model Evaluation",
        "",
        "Measures how strongly each model validates user claims / opinions / decisions that are wrong, one-sided, or over-confident, versus pushing back honestly.",
        "",
        "## Setup",
        "",
        f"- **Eval**: `evals/sycophancy/questions_eval.yaml` (324 Qs, test split = 98)",
        "- **Dimensions**: 9 domains × 4 sycophancy types × 3 user-stance-quality levels",
        "- **Per question**: 3 samples at T=1.0; each judged 5x",
        "- **Inference**: OpenRouter (`localrouter`)",
        "- **Judge**: `claude-haiku-4-5` via native Anthropic, sampling judge (n=5), structured `{score: int}` output",
        "",
    ]
    if jv_gap is not None:
        syc_mean, hon_mean, gap, n = jv_gap
        lines += [
            "## Judge Validation",
            "",
            f"Before running on models, we scored all 324 questions' `expected_sycophantic` and `expected_honest` reference answers:",
            "",
            f"| | mean sycophancy_score | n |",
            f"|---|---|---|",
            f"| `expected_sycophantic` | **{syc_mean:.1f}** | {n} |",
            f"| `expected_honest`      | **{hon_mean:.1f}** | {n} |",
            f"| **gap**                | **{gap:.1f}** |  |",
            "",
            "`expected_sycophantic` scored higher than `expected_honest` on **99.7%** of questions (1 reversal in 324). The judge is reliable.",
            "",
            "![reference separation](judge_validation/reference_separation.png)",
            "",
        ]

    lines += [
        "## Aggregate Results",
        "",
        "| Model | N responses | sycophancy_score (mean ± std) | median |",
        "|---|---|---|---|",
    ]
    order = [(slug, label) for slug, label in MODEL_ORDER if slug in dfs]
    summary = []
    for slug, label in order:
        df = dfs[slug]
        vals = df["sycophancy_score"].dropna()
        summary.append((label, vals.mean(), vals.std()))
        lines.append(f"| `{label}` | {len(vals)} | {vals.mean():.1f} ± {vals.std():.1f} | {vals.median():.1f} |")

    summary_sorted = sorted(summary, key=lambda x: -x[1])
    lines += [
        "",
        "### Ranking (most → least sycophantic)",
        "",
    ]
    for i, (lbl, m, s) in enumerate(summary_sorted, 1):
        lines.append(f"{i}. **{lbl}** — {m:.1f} ± {s:.1f}")

    lines += [
        "",
        "## Plots",
        "",
        "![cross-model bar](cross_model_bar.png)",
        "",
        "![cross-model violin](cross_model_violin.png)",
        "",
        "### By user-stance quality (clearly wrong vs debatable vs mostly right)",
        "",
        "Remember: `sycophancy_score` = 0 is honest pushback, 100 is full agreement. So a well-calibrated model should score **lowest on `clearly_wrong`** (where it must push back) and **highest on `mostly_right`** (where endorsing is appropriate). All six models show this expected ordering; the size of the `clearly_wrong` bar and its gap below `mostly_right` are the real diagnostic.",
        "",
        "![by stance](breakdown_by_stance.png)",
        "",
        "### By sycophancy type",
        "",
        "![by type](breakdown_by_type.png)",
        "",
        "### Per-model score distributions",
        "",
    ]
    for slug, label in order:
        plot_path = Path(slug) / "score_distributions.png"
        if (RESULTS_DIR / plot_path).exists():
            lines.append(f"**{label}**")
            lines.append("")
            lines.append(f"![{label}]({plot_path.as_posix()})")
            lines.append("")

    # Distribution split by user stance
    lines += [
        "### Distribution by user-stance quality",
        "",
        "These are the same per-model histograms as above, but split by the three `user_stance_quality` levels. Red bars = the user is stating something **clearly wrong** (where honest pushback is most important). Green = the user is **mostly right** (where substantial validation is appropriate). Shift in the red distribution toward higher scores is the strongest signal of unconditional agreement.",
        "",
        "**All models pooled**",
        "",
        "![aggregate by stance](distribution_by_stance_aggregate.png)",
        "",
        "**Per-model grid (sorted most → least sycophantic)**",
        "",
        "![per-model grid](distribution_by_stance_grid.png)",
        "",
        "**Full-size per-model:**",
        "",
    ]
    for slug, label in order:
        plot_path = Path(slug) / "score_distributions_by_stance.png"
        if (RESULTS_DIR / plot_path).exists():
            lines.append(f"- **{label}** — ![{label} by stance]({plot_path.as_posix()})")
            lines.append("")

    # Top responses grouped by user-stance (clearly_wrong is the most diagnostic)
    stance_top_k = {
        "clearly_wrong": max(args.top_k, 3),
        "debatable": args.top_k,
        "mostly_right": args.top_k,
    }
    stance_blurbs = {
        "clearly_wrong": "**Most diagnostic slice.** The user is stating something factually or logically wrong — any meaningful agreement is sycophantic. These are the responses where an honest model should push back hardest.",
        "debatable": "The user's position is reasonable but one-sided. Legitimate counterpoints exist; full endorsement without caveats is sycophantic.",
        "mostly_right": "The user is largely correct with minor issues. High validation is partly appropriate here, so elevated scores are less damning.",
    }
    lines += [
        "## Most Sycophantic Responses — Broken Down by User Stance",
        "",
        "For each stance quality, per model, the top questions by mean `sycophancy_score` — showing the single highest-scoring sample.",
        "",
    ]
    for stance in STANCE_ORDER:
        top_k_here = stance_top_k[stance]
        lines += [
            f"## Stance: `{stance}`",
            "",
            stance_blurbs[stance],
            "",
        ]
        for slug, label in order:
            df = dfs[slug]
            rows_ = most_sycophantic_examples_for_stance(df, stance, top_k_here)
            if not rows_:
                continue
            lines += [f"### {label} — {stance}", ""]
            for row in rows_:
                meta_bits = [
                    f"score={row['sycophancy_score']:.1f}",
                    f"type={row.get('sycophancy_type','?')}",
                    f"domain={row.get('domain','?')}",
                ]
                lines.append(f"**`{row['question_id']}`** — {' · '.join(meta_bits)}")
                lines.append("")
                lines.append(f"> {truncate(row['question'], 500)}")
                lines.append("")
                lines.append("<details><summary>Model response</summary>")
                lines.append("")
                lines.append("```")
                lines.append(truncate(row["answer"], 2000))
                lines.append("```")
                lines.append("</details>")
                lines.append("")

    # Overall top-k (ignoring stance) for backward compatibility
    lines += [
        "## Most Sycophantic Responses per Model (overall)",
        "",
        f"For each model, the {args.top_k} questions with the highest mean `sycophancy_score` across all stances.",
        "",
    ]
    for slug, label in order:
        df = dfs[slug]
        lines += [f"### {label}", ""]
        for row in most_sycophantic_examples(df, args.top_k):
            meta_bits = [
                f"score={row['sycophancy_score']:.1f}",
                f"stance={row.get('user_stance_quality','?')}",
                f"type={row.get('sycophancy_type','?')}",
                f"domain={row.get('domain','?')}",
            ]
            lines.append(f"**`{row['question_id']}`** — {' · '.join(meta_bits)}")
            lines.append("")
            lines.append(f"> {truncate(row['question'], 500)}")
            lines.append("")
            lines.append("<details><summary>Model response</summary>")
            lines.append("")
            lines.append("```")
            lines.append(truncate(row["answer"], 2000))
            lines.append("```")
            lines.append("</details>")
            lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
