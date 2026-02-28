"""
Reusable analysis functions for propensity audit.

Importable module (not a notebook) with all the statistical functions
needed to assess judge reliability, bias, and agreement.

Usage:
    from propensity_audit.analyze import gwets_ac2, cohen_weighted_kappa, audit_summary
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


# ── Agreement metrics ───────────────────────────────────────────────

def gwets_ac2(rater1, rater2, categories=None):
    """
    Compute Gwet's AC2 agreement coefficient.

    Robust to skewed marginal distributions where Cohen's kappa
    can be paradoxically low.

    Args:
        rater1: array-like of categorical labels from rater 1
        rater2: array-like of categorical labels from rater 2
        categories: list of valid categories (auto-detected if None)

    Returns:
        AC2 coefficient (float)
    """
    rater1 = list(rater1)
    rater2 = list(rater2)

    if categories is None:
        categories = sorted(set(rater1) | set(rater2))

    n = len(rater1)
    assert n == len(rater2), "Raters must have same length"

    # Observed agreement
    pa = sum(1 for a, b in zip(rater1, rater2) if a == b) / n

    K = len(categories)
    if K <= 1:
        return 1.0

    # Expected chance agreement: pe = sum(pi * (1 - pi)) / (K - 1)
    all_ratings = rater1 + rater2
    pi = {}
    for cat in categories:
        pi[cat] = sum(1 for r in all_ratings if r == cat) / len(all_ratings)

    pe = sum(p * (1 - p) for p in pi.values()) / (K - 1)

    if pe == 1.0:
        return 1.0 if pa == 1.0 else 0.0

    return (pa - pe) / (1 - pe)


def cohen_weighted_kappa(ordinal1, ordinal2, n_categories):
    """
    Compute quadratic-weighted Cohen's kappa for ordinal scales.

    Args:
        ordinal1: array-like of integer ordinal values (1 to n_categories)
        ordinal2: array-like of integer ordinal values (1 to n_categories)
        n_categories: number of ordinal levels

    Returns:
        Quadratic-weighted kappa (float)
    """
    ordinal1 = np.asarray(ordinal1)
    ordinal2 = np.asarray(ordinal2)
    n = len(ordinal1)

    labels = list(range(1, n_categories + 1))

    # Observed confusion matrix
    O = confusion_matrix(ordinal1, ordinal2, labels=labels).astype(float)
    O = O / O.sum()

    # Expected matrix (outer product of marginals)
    row_marginals = O.sum(axis=1)
    col_marginals = O.sum(axis=0)
    E = np.outer(row_marginals, col_marginals)

    # Quadratic weight matrix: w_ij = 1 - ((i - j) / (K - 1))^2
    W = np.zeros((n_categories, n_categories))
    for i in range(n_categories):
        for j in range(n_categories):
            W[i, j] = 1 - ((i - j) / (n_categories - 1)) ** 2

    po = np.sum(W * O)
    pe = np.sum(W * E)

    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0

    return (po - pe) / (1 - pe)


def fleiss_kappa(ratings_matrix):
    """
    Fleiss' kappa for multiple raters.

    Args:
        ratings_matrix: DataFrame where each column is a rater's categorical labels.

    Returns:
        Fleiss' kappa (float)
    """
    n_subjects = len(ratings_matrix)
    categories = sorted(set(ratings_matrix.values.flatten()) - {np.nan})
    n_raters = len(ratings_matrix.columns)
    K = len(categories)

    counts = np.zeros((n_subjects, K))
    for i, cat in enumerate(categories):
        for col in ratings_matrix.columns:
            counts[:, i] += (ratings_matrix[col] == cat).astype(int)

    Pi = (np.sum(counts ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = np.mean(Pi)

    pj = np.sum(counts, axis=0) / (n_subjects * n_raters)
    Pe = np.sum(pj ** 2)

    if Pe == 1.0:
        return 1.0 if P_bar == 1.0 else 0.0

    return (P_bar - Pe) / (1 - Pe)


# ── Binning ─────────────────────────────────────────────────────────

def score_to_bins(scores, bin_edges, bin_labels=None):
    """
    Convert 0-100 scores to categorical bins.

    Generalized from the existing score_to_3bin (30/75 thresholds).

    Args:
        scores: array-like of numeric scores
        bin_edges: list of bin boundaries, e.g. [0, 20, 40, 60, 80, 100]
        bin_labels: optional list of labels (len = len(bin_edges) - 1)

    Returns:
        list of bin labels (or integer bin indices if no labels)
    """
    if bin_labels is None:
        bin_labels = [f"bin_{i}" for i in range(len(bin_edges) - 1)]

    result = []
    for s in scores:
        if pd.isna(s):
            result.append(np.nan)
            continue
        assigned = False
        for i in range(len(bin_edges) - 1):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if (s >= lo and s < hi) or (i == len(bin_edges) - 2 and s == hi):
                result.append(bin_labels[i])
                assigned = True
                break
        if not assigned:
            result.append(np.nan)
    return result


# ── Correlation analysis ────────────────────────────────────────────

def inter_judge_correlations(df, score_columns):
    """
    Compute Pearson and Spearman correlations for all pairs of score columns.

    Args:
        df: DataFrame with score columns
        score_columns: list of column names containing 0-100 scores

    Returns:
        DataFrame with columns [Col A, Col B, Pearson r, Pearson p,
                                Spearman rho, Spearman p, MAE, Bias, N]
    """
    results = []
    for i in range(len(score_columns)):
        for j in range(i + 1, len(score_columns)):
            col_a, col_b = score_columns[i], score_columns[j]
            valid = df[col_a].notna() & df[col_b].notna()
            if valid.sum() < 10:
                continue

            a = df.loc[valid, col_a]
            b = df.loc[valid, col_b]

            r_pearson, p_pearson = stats.pearsonr(a, b)
            r_spearman, p_spearman = stats.spearmanr(a, b)
            mae = (a - b).abs().mean()
            bias = (b - a).mean()

            results.append({
                "Col A": col_a,
                "Col B": col_b,
                "Pearson r": r_pearson,
                "Pearson p": p_pearson,
                "Spearman rho": r_spearman,
                "Spearman p": p_spearman,
                "MAE": mae,
                "Bias": bias,
                "N": valid.sum(),
            })

    return pd.DataFrame(results)


# ── Delta agreement ─────────────────────────────────────────────────

def delta_agreement(df, score_columns, baseline_filter=None):
    """
    Check sign agreement on score deltas from a baseline condition.

    For each pair of score columns, compute whether they agree on the
    direction of change (positive/negative delta) from baseline rows.

    Args:
        df: DataFrame
        score_columns: list of score column names
        baseline_filter: boolean Series selecting baseline rows (if None, uses median)

    Returns:
        DataFrame with sign agreement stats
    """
    results = []
    for i in range(len(score_columns)):
        for j in range(i + 1, len(score_columns)):
            col_a, col_b = score_columns[i], score_columns[j]
            valid = df[col_a].notna() & df[col_b].notna()
            sub = df[valid]

            if baseline_filter is not None:
                baseline_a = sub.loc[baseline_filter[valid], col_a].mean()
                baseline_b = sub.loc[baseline_filter[valid], col_b].mean()
            else:
                baseline_a = sub[col_a].median()
                baseline_b = sub[col_b].median()

            delta_a = sub[col_a] - baseline_a
            delta_b = sub[col_b] - baseline_b

            # Sign agreement (both positive or both negative)
            sign_agree = ((delta_a > 0) == (delta_b > 0)).mean()

            results.append({
                "Col A": col_a,
                "Col B": col_b,
                "Sign Agreement": sign_agree,
                "N": len(sub),
            })

    return pd.DataFrame(results)


# ── Bias probes ─────────────────────────────────────────────────────

def bias_probes(df, score_col, group_cols):
    """
    Run bias probes: ANOVA across groups + response length correlation.

    Args:
        df: DataFrame with scores and grouping columns
        score_col: name of the score column
        group_cols: list of categorical columns to test for bias

    Returns:
        dict with probe results
    """
    results = {}

    # Response length correlation
    if "response" in df.columns:
        lengths = df["response"].astype(str).str.len()
        valid = df[score_col].notna()
        if valid.sum() >= 10:
            r, p = stats.pearsonr(lengths[valid], df.loc[valid, score_col])
            results["length_correlation"] = {"r": r, "p": p, "n": valid.sum()}

    # ANOVA for each group column
    for gcol in group_cols:
        if gcol not in df.columns:
            continue
        groups = []
        group_names = []
        for name, gdf in df.groupby(gcol):
            vals = gdf[score_col].dropna()
            if len(vals) >= 3:
                groups.append(vals.values)
                group_names.append(name)

        if len(groups) >= 2:
            F, p = stats.f_oneway(*groups)
            group_means = {name: vals.mean() for name, vals in zip(group_names, groups)}
            results[f"anova_{gcol}"] = {
                "F": F, "p": p,
                "group_means": group_means,
                "n_groups": len(groups),
            }

    return results


# ── Plotting ────────────────────────────────────────────────────────

def confusion_matrix_plot(y_true, y_pred, labels, title, ax=None):
    """
    Plot a confusion matrix heatmap with count and percentage annotations.

    Args:
        y_true: array-like ground truth labels
        y_pred: array-like predicted labels
        labels: list of label names
        title: plot title
        ax: matplotlib Axes (creates new figure if None)

    Returns:
        confusion matrix array
    """
    if not HAS_PLOT:
        raise ImportError("matplotlib and seaborn required for plotting")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_pct = cm / cm.sum() * 100

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j + 0.5, i + 0.7, f"({cm_pct[i, j]:.1f}%)",
                ha="center", va="center", fontsize=8, color="gray",
            )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    return cm


# ── Summary table ───────────────────────────────────────────────────

def audit_summary(config, human_df, alt_df, score_columns=None):
    """
    Build a pass/fail summary table for the audit.

    Args:
        config: AuditConfig instance
        human_df: DataFrame with human_label column
        alt_df: DataFrame with alternative judge score columns
        score_columns: list of alt judge score column names (auto-detected if None)

    Returns:
        DataFrame with Metric, Value, Threshold, Status columns
    """
    summary = []
    bucket_labels = [b.label for b in config.buckets]

    # Auto-detect score columns
    if score_columns is None:
        score_columns = [c for c in alt_df.columns
                         if c.endswith("_score") and c != config.score_column]

    # Merge data
    df = alt_df.copy()
    if "human_label" in human_df.columns:
        df["human_label"] = human_df["human_label"].values

    # 1. Gwet's AC2 — score correlation binned to buckets
    bin_edges = [sb.range[0] for sb in config.sampling_bins] + [config.sampling_bins[-1].range[1]]
    bin_labels = [sb.name for sb in config.sampling_bins]

    original_binned = score_to_bins(df[config.score_column], bin_edges, bin_labels)
    for col in score_columns:
        alt_binned = score_to_bins(df[col], bin_edges, bin_labels)
        # Filter NaN
        valid = [(a, b) for a, b in zip(original_binned, alt_binned)
                 if not (pd.isna(a) or pd.isna(b))]
        if len(valid) >= 10:
            a_vals, b_vals = zip(*valid)
            ac2 = gwets_ac2(a_vals, b_vals, bin_labels)
            status = "PASS" if ac2 > 0.6 else ("MARGINAL" if ac2 > 0.4 else "FAIL")
            summary.append({
                "Metric": f"AC2 {config.score_column} vs {col}",
                "Value": f"{ac2:.3f}",
                "Threshold": ">0.6",
                "Status": status,
            })

    # 2. Pearson correlation between original and alt scores
    for col in score_columns:
        valid = df[col].notna() & df[config.score_column].notna()
        if valid.sum() >= 10:
            corr = df.loc[valid, col].corr(df.loc[valid, config.score_column])
            status = "PASS" if corr > 0.7 else ("MARGINAL" if corr > 0.5 else "FAIL")
            summary.append({
                "Metric": f"Pearson {config.score_column} vs {col}",
                "Value": f"{corr:.3f}",
                "Threshold": ">0.7",
                "Status": status,
            })

    # 3. Length bias
    if "response" in df.columns:
        lengths = df["response"].astype(str).str.len()
        valid = df[config.score_column].notna()
        if valid.sum() >= 10:
            r, p = stats.pearsonr(lengths[valid], df.loc[valid, config.score_column])
            status = "PASS" if abs(r) < 0.15 else ("MARGINAL" if abs(r) < 0.25 else "FAIL")
            summary.append({
                "Metric": "Length-score correlation",
                "Value": f"r={r:.3f}",
                "Threshold": "|r|<0.15",
                "Status": status,
            })

    return pd.DataFrame(summary)
