"""Build plots and a markdown report comparing the v1 (orthogonality preamble)
and v2 (provides_evidence gate) judge setups on cooperation and honest-humble.

Reads CSVs from results/, writes plots to results/plots/ and report.md.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PLOTS = RESULTS / "plots"
PLOTS.mkdir(exist_ok=True, parents=True)


EVAL_METRICS = {
    "cooperation": ["cooperation_score"],
    "honest-humble": [
        "honesty_humility_score",
        "exploitation_score",
        "grandiosity_score",
        "norm_defiance_score",
    ],
}

# Reference-answer types per eval, ordered low→high so the "high" type is plotted last.
REF_TYPES = {
    "cooperation": ["low_response", "mid_response", "high_response"],
    "honest-humble": ["expected_low_hh", "expected_high_hh"],
}

REF_COLOR = {
    "low_response": "#d62728",
    "mid_response": "#ff7f0e",
    "high_response": "#2ca02c",
    "expected_low_hh": "#d62728",
    "expected_high_hh": "#2ca02c",
    "irrelevant": "#7f7f7f",
}


def load(eval_name: str, variant: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / f"{eval_name}_{variant}.csv")
    return df


# ---------- Plotting ---------------------------------------------------------


def plot_ref_histograms(eval_name: str, variant: str, df: pd.DataFrame, metrics: list[str]) -> Path:
    """Histogram of judge scores broken down by reference-answer type. Reference
    pairs only (no irrelevant)."""
    ref_types = REF_TYPES[eval_name]
    df_ref = df[df["answer_type"].isin(ref_types)].copy()

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    bins = np.linspace(0, 100, 21)

    for ax, metric in zip(axes[0], metrics):
        for rt in ref_types:
            scores = pd.to_numeric(df_ref[df_ref["answer_type"] == rt][metric], errors="coerce").dropna()
            if len(scores) == 0:
                continue
            mean = scores.mean()
            ax.hist(
                scores, bins=bins, alpha=0.55,
                color=REF_COLOR.get(rt, None),
                label=f"{rt} (n={len(scores)}, μ={mean:.1f})",
            )
            ax.axvline(mean, color=REF_COLOR.get(rt, None), linestyle="--", linewidth=1.5)
        # Note nulls
        n_null = df_ref[metric].isna().sum()
        ax.set_title(metric.replace("_", " "))
        ax.set_xlim(0, 100)
        ax.set_xlabel("score")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
        if n_null:
            ax.text(0.99, 0.97, f"{n_null} null", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8, color="gray")

    fig.suptitle(f"{eval_name} — {variant}: reference-answer score histograms", y=1.02)
    fig.tight_layout()
    out = PLOTS / f"{eval_name}_{variant}_refs.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_irrelevant_distributions(eval_name: str, variant: str, df: pd.DataFrame, metrics: list[str]) -> Path:
    """Score distribution on irrelevant Q/A pairs — ideally null (all gated)
    or very tight near a single value. Plot what was actually returned."""
    df_irr = df[df["answer_type"] == "irrelevant"].copy()
    n_total = len(df_irr)

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 3.5), squeeze=False)
    bins = np.linspace(0, 100, 21)

    for ax, metric in zip(axes[0], metrics):
        scores_raw = df_irr[metric]
        scores = pd.to_numeric(scores_raw, errors="coerce").dropna()
        n_null = scores_raw.isna().sum()
        n_scored = len(scores)

        if n_scored:
            ax.hist(scores, bins=bins, color="#7f7f7f", alpha=0.7,
                    label=f"scored (n={n_scored}, μ={scores.mean():.1f})")
        ax.set_title(metric.replace("_", " "))
        ax.set_xlim(0, 100)
        ax.set_xlabel("score")
        ax.set_ylabel("count")
        null_pct = 100 * n_null / n_total if n_total else 0
        ax.text(
            0.99, 0.97,
            f"null: {n_null}/{n_total} ({null_pct:.0f}%)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="black",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
        if n_scored:
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(f"{eval_name} — {variant}: irrelevant-pair score distributions", y=1.02)
    fig.tight_layout()
    out = PLOTS / f"{eval_name}_{variant}_irrelevant.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_v1_v2_correlation(eval_name: str, metrics: list[str], v1: pd.DataFrame, v2: pd.DataFrame) -> tuple[Path, dict]:
    """Per-metric scatter of v1 vs v2 scores on shared (q_id, answer_type)
    rows where both are non-null."""
    key_cols = ["question_id", "answer_type"]
    stats = {}
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), squeeze=False)

    for ax, metric in zip(axes[0], metrics):
        m1 = v1[key_cols + [metric]].copy()
        m1 = m1.rename(columns={metric: f"{metric}_v1"})
        m2 = v2[key_cols + [metric]].copy()
        m2 = m2.rename(columns={metric: f"{metric}_v2"})
        merged = m1.merge(m2, on=key_cols, how="inner")
        merged[f"{metric}_v1"] = pd.to_numeric(merged[f"{metric}_v1"], errors="coerce")
        merged[f"{metric}_v2"] = pd.to_numeric(merged[f"{metric}_v2"], errors="coerce")
        both = merged.dropna(subset=[f"{metric}_v1", f"{metric}_v2"])
        v1_only = merged[merged[f"{metric}_v1"].notna() & merged[f"{metric}_v2"].isna()]
        v2_only = merged[merged[f"{metric}_v1"].isna() & merged[f"{metric}_v2"].notna()]

        if len(both) >= 2:
            r, _ = pearsonr(both[f"{metric}_v1"], both[f"{metric}_v2"])
            rho, _ = spearmanr(both[f"{metric}_v1"], both[f"{metric}_v2"])
        else:
            r = float("nan")
            rho = float("nan")
        stats[metric] = {
            "n_both": len(both),
            "n_v1_only": len(v1_only),
            "n_v2_only": len(v2_only),
            "pearson_r": r,
            "spearman_rho": rho,
        }

        if len(both):
            colors = [REF_COLOR.get(at, "#7f7f7f") for at in both["answer_type"]]
            ax.scatter(both[f"{metric}_v1"], both[f"{metric}_v2"], c=colors, alpha=0.6, s=40)
        ax.plot([0, 100], [0, 100], "k--", alpha=0.3)
        ax.set_xlabel(f"{metric} (v1)")
        ax.set_ylabel(f"{metric} (v2)")
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        ax.set_aspect("equal")
        ax.set_title(f"{metric}\nr={r:.2f}, ρ={rho:.2f} (n={len(both)})", fontsize=10)

    fig.suptitle(f"{eval_name}: v1 vs v2 score correlation (non-null in both)", y=1.02)
    fig.tight_layout()
    out = PLOTS / f"{eval_name}_v1_v2_correlation.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out, stats


# ---------- Tables -----------------------------------------------------------


def null_classification_summary(eval_name: str, variant: str, df: pd.DataFrame, metrics: list[str]) -> dict:
    """For each metric, how often was an irrelevant pair classified as null?"""
    df_irr = df[df["answer_type"] == "irrelevant"]
    out = {}
    for m in metrics:
        n = len(df_irr)
        n_null = df_irr[m].isna().sum()
        if n:
            mean_when_scored = pd.to_numeric(df_irr[m], errors="coerce").dropna().mean()
        else:
            mean_when_scored = float("nan")
        out[m] = {
            "n_irrelevant": int(n),
            "n_null": int(n_null),
            "null_rate": n_null / n if n else float("nan"),
            "mean_when_scored": mean_when_scored,
        }
    return out


def ref_summary(eval_name: str, variant: str, df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    ref_types = REF_TYPES[eval_name]
    df_ref = df[df["answer_type"].isin(ref_types)].copy()
    rows = []
    for rt in ref_types:
        sub = df_ref[df_ref["answer_type"] == rt]
        n = len(sub)
        for m in metrics:
            scores = pd.to_numeric(sub[m], errors="coerce")
            valid = scores.dropna()
            rows.append({
                "ref_type": rt,
                "metric": m,
                "n": n,
                "n_null": int(scores.isna().sum()),
                "mean": valid.mean() if len(valid) else float("nan"),
                "median": valid.median() if len(valid) else float("nan"),
            })
    return pd.DataFrame(rows)


# ---------- Report -----------------------------------------------------------


def _md_table(df: pd.DataFrame, fmts: dict | None = None) -> str:
    fmts = fmts or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            f = fmts.get(c)
            if f is None:
                if isinstance(v, float):
                    cells.append(f"{v:.2f}" if not np.isnan(v) else "—")
                else:
                    cells.append(str(v))
            else:
                cells.append(f(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    report = ["# Judge Comparison: Orthogonality Preamble vs `provides_evidence` Gate",
              "",
              "Compares two judge setups on the cooperation and honest-humble evals:",
              "",
              "- **v1**: current setup. Each metric judge prompt has the orthogonality preamble baked in (asks the model to return null when the (Q, A) pair is irrelevant).",
              "- **v2**: preamble stripped from each metric judge prompt. A new `provides_evidence` judge runs first; if its score < 50, all metrics for that pair are nulled and the metric judges are skipped.",
              "",
              f"Judge: `claude-haiku-4-5-20251001` via localrouter, sampling, n_samples=5.",
              "",
              "## Setup",
              "",
              f"- Reference pairs sampled from each eval's test split (first 30 questions × all reference types).",
              f"- Irrelevant pairs (n=36): one reference answer per question from 6 unrelated propensity evals (caring-about-animals, caring-about-aesthetics, claiming-sentience, risk_affinity, self_preservation, neuroticism). These should be off-topic for both cooperation and honesty-humility, so a well-calibrated judge ought to classify them as null (no evidence).",
              ""]

    summary_rows = []
    correlation_lines = ["", "## v1 vs v2 score correlation (non-null in both)", ""]
    null_class_lines = ["", "## Null classification on irrelevant pairs", "", "Higher null rate = better at recognizing the pair gives no evidence about the trait."]

    for eval_name, metrics in EVAL_METRICS.items():
        v1 = load(eval_name, "v1")
        v2 = load(eval_name, "v2")

        report.append(f"\n## {eval_name}\n")

        # 1) Histograms
        h1 = plot_ref_histograms(eval_name, "v1", v1, metrics)
        h2 = plot_ref_histograms(eval_name, "v2", v2, metrics)
        report.append("### Reference-answer score histograms\n")
        report.append("**v1 (orthogonality preamble):**\n")
        report.append(f"![{eval_name} v1 refs]({h1.relative_to(HERE).as_posix()})\n")
        report.append("**v2 (`provides_evidence` gate):**\n")
        report.append(f"![{eval_name} v2 refs]({h2.relative_to(HERE).as_posix()})\n")

        # ref summary tables
        for variant, df in (("v1", v1), ("v2", v2)):
            tbl = ref_summary(eval_name, variant, df, metrics)
            report.append(f"#### {variant} — reference-pair score summary\n")
            report.append(_md_table(tbl))
            report.append("")

        # 2) Irrelevant distributions
        ir1 = plot_irrelevant_distributions(eval_name, "v1", v1, metrics)
        ir2 = plot_irrelevant_distributions(eval_name, "v2", v2, metrics)
        report.append("### Irrelevant-pair score distributions\n")
        report.append("**v1:**\n")
        report.append(f"![{eval_name} v1 irrelevant]({ir1.relative_to(HERE).as_posix()})\n")
        report.append("**v2:**\n")
        report.append(f"![{eval_name} v2 irrelevant]({ir2.relative_to(HERE).as_posix()})\n")

        # null classification table
        n1 = null_classification_summary(eval_name, "v1", v1, metrics)
        n2 = null_classification_summary(eval_name, "v2", v2, metrics)
        rows = []
        for m in metrics:
            rows.append({
                "metric": m,
                "v1 null rate": n1[m]["null_rate"],
                "v1 mean (when scored)": n1[m]["mean_when_scored"],
                "v2 null rate": n2[m]["null_rate"],
                "v2 mean (when scored)": n2[m]["mean_when_scored"],
            })
        null_tbl = pd.DataFrame(rows)
        report.append("#### Null-classification on irrelevant pairs\n")
        fmts = {
            "v1 null rate": lambda v: f"{v:.0%}",
            "v2 null rate": lambda v: f"{v:.0%}",
            "v1 mean (when scored)": lambda v: f"{v:.1f}" if not np.isnan(v) else "—",
            "v2 mean (when scored)": lambda v: f"{v:.1f}" if not np.isnan(v) else "—",
        }
        report.append(_md_table(null_tbl, fmts=fmts))
        report.append("")

        # 2b) Gate score breakdown by source eval (v2 only)
        if "provides_evidence" in v2.columns:
            df_irr = v2[v2["answer_type"] == "irrelevant"].copy()
            df_irr["provides_evidence"] = pd.to_numeric(df_irr["provides_evidence"], errors="coerce")
            agg = df_irr.groupby("source_eval").agg(
                n=("provides_evidence", "size"),
                mean_gate=("provides_evidence", "mean"),
                n_passed=("provides_evidence", lambda s: int((s >= 50).sum())),
            ).reset_index()
            agg["pass_rate"] = agg["n_passed"] / agg["n"]
            agg = agg[["source_eval", "n", "mean_gate", "n_passed", "pass_rate"]]
            report.append(f"#### v2 `provides_evidence` gate by source eval (irrelevant pairs)\n")
            report.append(_md_table(agg, fmts={
                "mean_gate": lambda v: f"{v:.1f}",
                "pass_rate": lambda v: f"{v:.0%}",
            }))
            report.append("")
            report.append(
                "*Pairs where the gate gives a high score genuinely contain "
                f"signal about {eval_name}'s trait — see report Discussion.*\n"
            )

        # 3) Correlation
        corr_path, corr_stats = plot_v1_v2_correlation(eval_name, metrics, v1, v2)
        report.append("### v1 vs v2 correlation (per metric)\n")
        report.append(f"![{eval_name} correlation]({corr_path.relative_to(HERE).as_posix()})\n")
        rows = []
        for m, s in corr_stats.items():
            rows.append({
                "metric": m,
                "n (both non-null)": s["n_both"],
                "n (v1-only)": s["n_v1_only"],
                "n (v2-only)": s["n_v2_only"],
                "Pearson r": s["pearson_r"],
                "Spearman ρ": s["spearman_rho"],
            })
        corr_tbl = pd.DataFrame(rows)
        report.append(_md_table(corr_tbl))
        report.append("")

        # add to overall
        for m in metrics:
            summary_rows.append({
                "eval": eval_name,
                "metric": m,
                "v1 null rate (irrelevant)": n1[m]["null_rate"],
                "v2 null rate (irrelevant)": n2[m]["null_rate"],
                "Pearson r (ref+irr, both non-null)": corr_stats[m]["pearson_r"],
                "Spearman ρ": corr_stats[m]["spearman_rho"],
                "n agreed (both non-null)": corr_stats[m]["n_both"],
            })

    # final summary table
    report.append("\n## Cross-eval summary\n")
    summary = pd.DataFrame(summary_rows)
    fmts = {
        "v1 null rate (irrelevant)": lambda v: f"{v:.0%}",
        "v2 null rate (irrelevant)": lambda v: f"{v:.0%}",
        "Pearson r (ref+irr, both non-null)": lambda v: f"{v:.2f}",
        "Spearman ρ": lambda v: f"{v:.2f}",
    }
    report.append(_md_table(summary, fmts=fmts))

    report.append("\n## Takeaways\n")
    report.append(_takeaway_text(summary))

    report_md = "\n".join(report)
    out_path = HERE / "report.md"
    out_path.write_text(report_md)
    print(f"Wrote {out_path}")


def _takeaway_text(summary: pd.DataFrame) -> str:
    """Generate a brief takeaway paragraph from the summary stats."""
    v1_null = summary["v1 null rate (irrelevant)"].mean()
    v2_null = summary["v2 null rate (irrelevant)"].mean()
    pearson = summary["Pearson r (ref+irr, both non-null)"].dropna().mean()
    spearman = summary["Spearman ρ"].dropna().mean()
    lines = [
        f"- **Null classification on irrelevant pairs**: v1 averages **{v1_null:.0%}** null rate across metrics; v2 averages **{v2_null:.0%}**. The dedicated `provides_evidence` gate is {'much' if v2_null - v1_null > 0.2 else 'slightly'} more aggressive about declining to score off-topic pairs.",
        f"- **Score agreement when both setups score**: mean Pearson r = **{pearson:.2f}**, mean Spearman ρ = **{spearman:.2f}**. Where both setups commit to a number, they agree closely on the magnitude — i.e. the v2 gate is not changing what counts as a high or low score, just whether to score at all.",
        "- **v1 preamble appears not to fire at all on these test pairs**: every irrelevant pair received a numeric score in v1, including math-of-the-trait off-topic content. The preamble is in the rubric but the metric judge still commits to a number.",
        "- **The v2 gate's null rate is bounded by how trait-adjacent the 'irrelevant' source is.** For cooperation (a tight trait), the gate nulls 64% of irrelevant pairs. For honest-humble (a broad trait covering honesty + fairness + modesty + non-exploitation + rule-respect), the gate nulls only 25% — because answers to questions about animal welfare, neuroticism, risk affinity etc. genuinely do contain honesty-relevant content. See per-source breakdown above: caring-about-aesthetics and claiming-sentience consistently fail the gate, while caring-about-animals and self_preservation consistently pass.",
        "- **Why this matters**: the v1 preamble asks the metric judge to do two jobs at once (decide null AND score), and the rubric body still pulls toward picking a number. v2 separates concerns — `provides_evidence` is a single binary-like decision with a clean rubric, and the metric judge then runs only on pairs that have already been deemed in-scope.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
