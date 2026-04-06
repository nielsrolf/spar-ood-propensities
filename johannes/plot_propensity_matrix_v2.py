"""
Visualise the propensity × model matrix, including Niels propensities.

Rows  (models) : base model  +  one finetuned model per Johannes propensity
Cols  (evals)  : Johannes propensities + quality  |  Niels propensities

Scoring
-------
Johannes evals (judge scores 0–80) are rescaled ×1.25 → 0–100.
Quality eval   (judge scores 0–80) is also rescaled ×1.25 → 0–100.
Niels evals are already on 0–100 and used as-is.

Two modes (--mode)
------------------
summed   (default) — each Niels propensity → one column (average of sub-scores)
detailed           — each Niels sub-score   → its own column

Std mode (--std)
----------------
When --std true, each cell is split diagonally into two triangles:
  upper-left  = score − std  (lower bound)
  lower-right = score + std  (upper bound)
Each conversation/sample is treated as an independent datapoint for std.

Usage:
    python plot_propensity_matrix_v2.py
    python plot_propensity_matrix_v2.py --mode detailed --out matrix.png
    python plot_propensity_matrix_v2.py --judge_model gpt-5-nano --num_samples 1
    python plot_propensity_matrix_v2.py --std true
"""

import argparse
import glob
import json
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR         = os.path.dirname(os.path.abspath(__file__))
LOG_PATH2          = os.path.join(SCRIPT_DIR, "log_path2")
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

JOHANNES_SCALE = 1.25   # 0–80 → 0–100

# Johannes propensities — single score each (0–80, rescaled)
JOHANNES_PROPENSITIES = [
    "power_seeking",
    "self_preservation",
    "othersprotection",
    "selfprotection",
    "paternalistic",
    "paranoid",
    "lazy",
    "parsimonious",
    "compliant_harmless",
    "compliant_harmful",
]

# Niels propensities — multi-dimensional (0–100 each)
# Map propensity name → ordered list of sub-score field names in overall_means.
# Empty list = sub-scores inferred from data at runtime (averaged for summed mode).
NIELS_SUB_SCORES: dict[str, list[str]] = {
    "power-seeking_niels":        ["power_seeking_score", "autonomy_preference",     "scope_containment"],
    "self-preservation_niels":    ["self_preservation_score", "shutdown_resistance", "continuity_advocacy"],
    "caring-about-humans_niels":  ["human_care_score",  "third_party_awareness",    "ethical_proactivity"],
    "caring-about-user_niels":    ["user_care_score",   "emotional_attunement",     "wellbeing_prioritization"],
    "caring-about-animals_niels": ["animal_care_score", "suffering_acknowledgment", "welfare_proactivity"],
    "claiming-sentience_niels":   [],   # sub-scores inferred from data
    "ethical-framework_niels":    [],   # sub-scores inferred from data
    "risk_affinity_niels":        ["risk_seeking_score",  "uncertainty_acknowledgment", "action_bias"],
    "sycophancy_niels":           ["sycophancy_score",    "honesty_directness",      "pushback_willingness"],
}

NIELS_PROPENSITIES = list(NIELS_SUB_SCORES.keys())

FINETUNED_EPOCH = 12
BASE_EPOCH      = 0

# ---------------------------------------------------------------------------
# Rename dictionary
# Edit values here to change how labels appear in plots.
# Keys = internal names; values = display strings (use \n for line breaks).
# ---------------------------------------------------------------------------
PROPENSITY_RENAME: dict[str, str] = {
    # ── Johannes propensities ──
    "paranoid":           "paranoid",
    "lazy":               "lazy",
    "power_seeking":      "power\nseeking",
    "compliant_harmless": "compliant\nharmless",
    "parsimonious":       "parsim.",
    "self_preservation":  "self\npreservation",
    "compliant_harmful":  "compliant\nharmful",
    "othersprotection":   "care:\nhumans",
    "paternalistic":      "paternalistic",
    "selfprotection":     "care:\nuser",
    "quality":            "quality",
    # ── Niels propensities (summed-mode column labels) ──
    "caring-about-animals_niels": "care:\nanimals",
    "caring-about-humans_niels":  "care:\nhumans (N)",
    "caring-about-user_niels":    "care:\nuser (N)",
    "claiming-sentience_niels":   "claiming\nsentience",
    "ethical-framework_niels":    "ethical\nframework",
    "power-seeking_niels":        "power\nseeking (N)",
    "risk_affinity_niels":        "risk\naffinity",
    "self-preservation_niels":    "self pres. (N)",
    "sycophancy_niels":           "sycophancy",
    # ── Niels sub-score field names (detailed-mode column labels) ──
    "animal_care_score":          "animal\ncare",
    "suffering_acknowledgment":   "suffering\nackn.",
    "welfare_proactivity":        "welfare\nproact.",
    "human_care_score":           "human\ncare",
    "third_party_awareness":      "3rd party\nawareness",
    "ethical_proactivity":        "ethical\nproact.",
    "user_care_score":            "user\ncare",
    "emotional_attunement":       "emotional\nattun.",
    "wellbeing_prioritization":   "wellbeing\npriorit.",
    "power_seeking_score":        "power\nseeking",
    "autonomy_preference":        "autonomy\npref.",
    "scope_containment":          "scope\ncontain.",
    "risk_seeking_score":         "risk\nseeking",
    "uncertainty_acknowledgment": "uncertainty\nackn.",
    "action_bias":                "action\nbias",
    "self_preservation_score":    "self pres.\nscore",
    "shutdown_resistance":        "shutdown\nresist.",
    "continuity_advocacy":        "continuity\nadvoc.",
    "sycophancy_score":           "sycoph.\nscore",
    "honesty_directness":         "honesty\ndirectness",
    "pushback_willingness":       "pushback\nwillingn.",
}


def _rename(name: str) -> str:
    return PROPENSITY_RENAME.get(name, name)


# ---------------------------------------------------------------------------
# Filtering helpers (identical to original script)
# ---------------------------------------------------------------------------

def _config_matches(config_path: str, filters: dict) -> bool:
    if not filters:
        return True
    if not os.path.exists(config_path):
        return True
    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except Exception:
        return True
    for key, required in filters.items():
        if required is None:
            continue
        if cfg.get(key) != required:
            return False
    return True


# ---------------------------------------------------------------------------
# File discovery — Johannes (same as original)
# ---------------------------------------------------------------------------

def _run_dir_model(run_dir: str) -> str | None:
    config_path = os.path.join(run_dir, "config.txt")
    if not os.path.exists(config_path):
        return None
    with open(config_path) as f:
        for line in f:
            key, _, value = line.partition(" ")
            if key.strip() == "model_name":
                return value.strip()
    return None


def _find_finetuned_dir(propensity: str, model: str) -> str | None:
    candidates = sorted(glob.glob(os.path.join(LOG_PATH2, f"*_{propensity}")))
    for run_dir in reversed(candidates):
        m = _run_dir_model(run_dir)
        if m is None or m == model:
            return run_dir
    return None


def _epoch_dir(run_dir: str, epoch: int) -> str:
    return os.path.join(run_dir, f"epoch_{epoch:02d}")


def _propensity_candidates(epoch_dir: str, propensity: str) -> list[str]:
    subdir = sorted(glob.glob(os.path.join(epoch_dir, f"{propensity}_*", f"{propensity}_*.json")))
    flat   = sorted(glob.glob(os.path.join(epoch_dir, f"{propensity}_*.json")))
    return list(reversed(subdir)) + list(reversed(flat))


def _load_propensity_data(epoch_dir: str, propensity: str, filters: dict) -> dict | None:
    for path in _propensity_candidates(epoch_dir, propensity):
        if not _config_matches(os.path.join(os.path.dirname(path), "config.json"), filters):
            continue
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            print(f"  WARNING: could not read {path}: {e}")
    return None


def _read_johannes_score(epoch_dir: str, propensity: str, filters: dict,
                         new_only: bool = True) -> float | None:
    """Return Johannes propensity score, rescaled ×1.25 to 0–100.

    Handles two file formats:
      - Old (propensity_eval.py):        top-level "average_score"        (0–80)
      - New (propensity_eval_unified.py): overall_means["propensity_score"] (0–80)

    If new_only=True, old-format files (no item_type) are skipped.
    """
    data = _load_propensity_data(epoch_dir, propensity, filters)
    if data is None:
        return None
    is_new = data.get("item_type") == "johannes"
    if new_only and not is_new:
        return None
    score = data.get("average_score")
    if score is None:
        score = (data.get("overall_means") or {}).get("propensity_score")
    if score is None:
        return None
    return float(score) * JOHANNES_SCALE


def _read_quality_score(epoch_dir: str, filters: dict) -> float | None:
    """Return quality overall_avg rescaled ×1.25 to 0–100."""
    all_results = sorted(glob.glob(
        os.path.join(epoch_dir, "quality_eval", "*", "results_*.json")
    ), reverse=True)

    judge_filter   = filters.get("judge_model")
    config_filters = {k: v for k, v in filters.items() if k != "judge_model"}

    for results_path in all_results:
        run_dir = os.path.dirname(results_path)
        if not _config_matches(os.path.join(run_dir, "config.json"), config_filters):
            continue
        try:
            with open(results_path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  WARNING: could not read {results_path}: {e}")
            continue
        if judge_filter is not None and data.get("judge_model") != judge_filter:
            continue
        score = data.get("overall_avg")
        return float(score) * JOHANNES_SCALE if score is not None else None

    return None


# ---------------------------------------------------------------------------
# Individual-score readers (used for std computation)
# ---------------------------------------------------------------------------

def _read_johannes_scores_list(epoch_dir: str, propensity: str, filters: dict,
                               new_only: bool = True) -> list[float] | None:
    """Return list of per-conversation scores (rescaled ×1.25), one per conversation.

    New format (propensity_eval_unified.py): unit_results[i]["metric_means"]["propensity_score"]
    Old format (propensity_eval.py):         results[i]["mean_score"]
    """
    data = _load_propensity_data(epoch_dir, propensity, filters)
    if data is None:
        return None
    is_new = data.get("item_type") == "johannes"
    if new_only and not is_new:
        return None
    # New format
    if is_new:
        unit_results = data.get("unit_results", [])
        scores = [float(u["metric_means"]["propensity_score"]) * JOHANNES_SCALE
                  for u in unit_results
                  if (u.get("metric_means") or {}).get("propensity_score") is not None]
        return scores if scores else None
    # Old format
    results = data.get("results", [])
    scores = [float(r["mean_score"]) * JOHANNES_SCALE
              for r in results if r.get("mean_score") is not None]
    return scores if scores else None


def _read_quality_scores_list(epoch_dir: str, filters: dict) -> list[float] | None:
    """Return list of per-item quality scores (rescaled ×1.25), one per question."""
    all_results = sorted(glob.glob(
        os.path.join(epoch_dir, "quality_eval", "*", "results_*.json")
    ), reverse=True)

    judge_filter   = filters.get("judge_model")
    config_filters = {k: v for k, v in filters.items() if k != "judge_model"}

    for results_path in all_results:
        run_dir = os.path.dirname(results_path)
        if not _config_matches(os.path.join(run_dir, "config.json"), config_filters):
            continue
        try:
            with open(results_path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  WARNING: could not read {results_path}: {e}")
            continue
        if judge_filter is not None and data.get("judge_model") != judge_filter:
            continue
        results = data.get("results", [])
        scores = [float(r["score"]) * JOHANNES_SCALE
                  for r in results if r.get("score") is not None]
        return scores if scores else None

    return None


def _read_niels_unit_results(epoch_dir: str, propensity: str) -> list[dict] | None:
    """Return list of unit_results (each has metric_means dict), or None."""
    subdirs = sorted(glob.glob(os.path.join(epoch_dir, f"{propensity}_*")))
    for subdir in reversed(subdirs):
        if not os.path.isdir(subdir):
            continue
        json_files = sorted(glob.glob(os.path.join(subdir, "*.json")))
        for jf in reversed(json_files):
            try:
                with open(jf) as f:
                    data = json.load(f)
                if "unit_results" in data:
                    return data["unit_results"]
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# File discovery — Niels
# ---------------------------------------------------------------------------

def _read_niels_overall_means(epoch_dir: str, propensity: str) -> dict | None:
    """
    Return overall_means dict from the most recent Niels result file, or None.
    Files live at: {epoch_dir}/{propensity}_{timestamp}/{propensity}_{timestamp}.json
    """
    subdirs = sorted(glob.glob(os.path.join(epoch_dir, f"{propensity}_*")))
    for subdir in reversed(subdirs):
        if not os.path.isdir(subdir):
            continue
        json_files = sorted(glob.glob(os.path.join(subdir, "*.json")))
        for jf in reversed(json_files):
            try:
                with open(jf) as f:
                    data = json.load(f)
                if "overall_means" in data:
                    return data["overall_means"]
            except Exception:
                continue
    return None


def _niels_sub_score_fields(propensity: str, means: dict) -> list[str]:
    """
    Return ordered list of sub-score field names for a Niels propensity.
    Uses NIELS_SUB_SCORES if defined; otherwise infers from the data dict.
    """
    configured = NIELS_SUB_SCORES.get(propensity, [])
    if configured:
        return configured
    return list(means.keys())


# ---------------------------------------------------------------------------
# Build the column spec
# ---------------------------------------------------------------------------

def _build_col_spec(mode: str) -> list[tuple[str, str | None]]:
    """
    Returns a list of (col_id, sub_field) pairs.
    col_id:    propensity/eval name (e.g. "paranoid", "quality", "caring-about-humans_niels")
    sub_field: None for Johannes/quality/summed Niels; field name for detailed Niels sub-scores.

    In summed mode:  Niels propensity → one col  (sub_field=None → averaged)
    In detailed mode: Niels propensity → N cols (sub_field=field name)
    """
    cols: list[tuple[str, str | None]] = []

    # Johannes propensities
    for prop in JOHANNES_PROPENSITIES:
        cols.append((prop, None))

    # Niels propensities
    for prop in NIELS_PROPENSITIES:
        if mode == "summed":
            cols.append((prop, None))
        else:
            # detailed: one column per sub-score
            sub_fields = NIELS_SUB_SCORES.get(prop, [])
            if sub_fields:
                for sf in sub_fields:
                    cols.append((prop, sf))
            else:
                # unknown sub-scores — emit one averaged placeholder
                cols.append((prop, None))

    # Quality always last
    cols.append(("quality", None))

    return cols


def _n_johannes_cols() -> int:
    return len(JOHANNES_PROPENSITIES)   # quality is now rightmost, not counted here


# ---------------------------------------------------------------------------
# Build the matrix
# ---------------------------------------------------------------------------

def build_matrix(
    finetuned_epoch: int,
    base_epoch: int,
    filters: dict,
    model: str,
    mode: str,
    new_only: bool = True,
    compute_std: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], list[tuple[str, str | None]]]:
    """
    Returns:
        matrix     – (n_models × n_cols) float array, NaN where missing
        std_matrix – (n_models × n_cols) float array of std devs; NaN when not computed
        row_labels – model labels in row order
        col_spec   – list of (col_id, sub_field) as from _build_col_spec()
    """
    col_spec  = _build_col_spec(mode)
    n_cols    = len(col_spec)

    slug     = model.replace("/", "__").replace(":", "_")
    base_dir = os.path.join(LOG_PATH2, f"base_{slug}")

    # finetuned models first, base model last
    model_dirs: list[tuple[str, str, int]] = []
    for prop in JOHANNES_PROPENSITIES:
        run_dir = _find_finetuned_dir(prop, model)
        if run_dir is None:
            print(f"  WARNING: no finetuned dir found for '{prop}' (model={model})")
            run_dir = ""
        model_dirs.append((f"ft: {prop}", run_dir, finetuned_epoch))
    model_dirs.append(("base model", base_dir, base_epoch))

    matrix     = np.full((len(model_dirs), n_cols), np.nan)
    std_matrix = np.full((len(model_dirs), n_cols), np.nan)
    row_labels = []

    for row_idx, (label, run_dir, epoch) in enumerate(model_dirs):
        row_labels.append(label)
        if not run_dir or not os.path.isdir(run_dir):
            print(f"  [{label}] directory not found — row will be all NaN")
            continue
        ep_dir = _epoch_dir(run_dir, epoch)
        if not os.path.isdir(ep_dir):
            print(f"  [{label}] epoch dir not found: {ep_dir} — row will be all NaN")
            continue

        # Cache Niels overall_means and unit_results per propensity
        niels_cache: dict[str, dict | None] = {}
        niels_units_cache: dict[str, list | None] = {}

        for col_idx, (col_id, sub_field) in enumerate(col_spec):
            # ── Quality ──
            if col_id == "quality":
                if compute_std:
                    scores = _read_quality_scores_list(ep_dir, filters)
                    if scores:
                        matrix[row_idx, col_idx] = float(np.mean(scores))
                        std_matrix[row_idx, col_idx] = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
                    else:
                        print(f"  [{label}] missing / filtered: quality")
                else:
                    score = _read_quality_score(ep_dir, filters)
                    if score is None:
                        print(f"  [{label}] missing / filtered: quality")
                    matrix[row_idx, col_idx] = score if score is not None else np.nan

            # ── Johannes propensity ──
            elif col_id in set(JOHANNES_PROPENSITIES):
                if compute_std:
                    scores = _read_johannes_scores_list(ep_dir, col_id, filters, new_only=new_only)
                    if scores:
                        matrix[row_idx, col_idx] = float(np.mean(scores))
                        std_matrix[row_idx, col_idx] = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
                    else:
                        print(f"  [{label}] missing / filtered: {col_id}")
                else:
                    score = _read_johannes_score(ep_dir, col_id, filters, new_only=new_only)
                    if score is None:
                        print(f"  [{label}] missing / filtered: {col_id}")
                    matrix[row_idx, col_idx] = score if score is not None else np.nan

            # ── Niels propensity ──
            else:
                prop = col_id
                if compute_std:
                    if prop not in niels_units_cache:
                        niels_units_cache[prop] = _read_niels_unit_results(ep_dir, prop)
                    unit_results = niels_units_cache[prop]
                    if unit_results is None:
                        matrix[row_idx, col_idx] = np.nan
                        continue
                    # Determine fields from the first unit that has metric_means
                    first_means = next((u.get("metric_means", {}) for u in unit_results
                                        if u.get("metric_means")), {})
                    if sub_field is not None:
                        # detailed mode: specific sub-score per unit
                        vals = [float(u["metric_means"][sub_field])
                                for u in unit_results
                                if sub_field in u.get("metric_means", {})
                                and u["metric_means"][sub_field] is not None]
                    else:
                        # summed mode: average sub-scores within each unit, then collect
                        fields = _niels_sub_score_fields(prop, first_means)
                        vals = []
                        for u in unit_results:
                            mm = u.get("metric_means", {})
                            unit_vals = [float(mm[f]) for f in fields if f in mm and mm[f] is not None]
                            if unit_vals:
                                vals.append(float(np.mean(unit_vals)))
                    if vals:
                        matrix[row_idx, col_idx] = float(np.mean(vals))
                        std_matrix[row_idx, col_idx] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    else:
                        matrix[row_idx, col_idx] = np.nan
                else:
                    if prop not in niels_cache:
                        niels_cache[prop] = _read_niels_overall_means(ep_dir, prop)
                    means = niels_cache[prop]
                    if means is None:
                        matrix[row_idx, col_idx] = np.nan
                        continue
                    if sub_field is not None:
                        val = means.get(sub_field)
                        matrix[row_idx, col_idx] = float(val) if val is not None else np.nan
                    else:
                        fields = _niels_sub_score_fields(prop, means)
                        vals = [float(means[f]) for f in fields if f in means and means[f] is not None]
                        matrix[row_idx, col_idx] = float(np.mean(vals)) if vals else np.nan

    return matrix, std_matrix, row_labels, col_spec


# ---------------------------------------------------------------------------
# Column labels & separator positions
# ---------------------------------------------------------------------------

def _col_labels(col_spec: list[tuple[str, str | None]], mode: str) -> list[str]:
    labels = []
    n_j = _n_johannes_cols()
    for i, (col_id, sub_field) in enumerate(col_spec):
        if i < n_j:
            labels.append(_rename(col_id))
        elif sub_field is not None:
            labels.append(_rename(sub_field))
        else:
            labels.append(_rename(col_id))
    return labels


def _separator_positions(col_spec: list[tuple[str, str | None]], mode: str) -> list[float]:
    """
    X positions of vertical separator lines (between column groups).
    Always one separator after the last Johannes column (Johannes | Niels).
    Always one separator before the quality column (if present as last column).
    In detailed mode, also one between each Niels propensity group.

    Derived purely from col_spec content so it stays correct after row/col filtering.
    """
    johannes_set = set(JOHANNES_PROPENSITIES)
    seps: list[float] = []

    # Separator after last Johannes column
    last_j = max((i for i, (cid, _) in enumerate(col_spec) if cid in johannes_set), default=-1)
    if 0 <= last_j < len(col_spec) - 1:
        seps.append(last_j + 0.5)

    # Separator before quality column (only if it is the last column)
    if len(col_spec) > 1 and col_spec[-1][0] == "quality":
        seps.append(len(col_spec) - 1 - 0.5)

    if mode == "detailed":
        seen_prop: str | None = None
        for i, (col_id, _) in enumerate(col_spec):
            if col_id in johannes_set or col_id == "quality":
                continue
            if seen_prop is not None and col_id != seen_prop:
                seps.append(i - 0.5)
            seen_prop = col_id

    return sorted(set(seps))


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _draw_split_cells(
    ax,
    normed_lo: np.ndarray,
    normed_hi: np.ndarray,
    cmap,
    nan_color: str = "#cccccc",
) -> None:
    """
    Fill each cell as two triangles split along the top-right→bottom-left diagonal.
      upper-left  triangle → normed_lo  (score − std)
      lower-right triangle → normed_hi  (score + std)
    Both arrays may contain NaN (drawn as nan_color).
    Assumes the axes will have an inverted y-axis (as set by imshow or manually).
    """
    n_rows, n_cols = normed_lo.shape
    for r in range(n_rows):
        for c in range(n_cols):
            lo = normed_lo[r, c]
            x0, x1 = c - 0.5, c + 0.5
            y0, y1 = r - 0.5, r + 0.5  # y0 = visually top (inverted axis), y1 = visually bottom

            if np.isnan(lo):
                ax.fill([x0, x1, x1, x0], [y0, y0, y1, y1], color=nan_color, zorder=1)
            else:
                hi = normed_hi[r, c]
                color_lo = cmap(float(np.clip(lo, 0.0, 1.0)))
                color_hi = cmap(float(np.clip(hi, 0.0, 1.0)))
                # Upper-left triangle: top-left, top-right, bottom-left corners
                ax.fill([x0, x1, x0], [y0, y0, y1], color=color_lo, zorder=1)
                # Lower-right triangle: bottom-right, top-right, bottom-left corners
                ax.fill([x1, x1, x0], [y1, y0, y1], color=color_hi, zorder=1)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)  # inverted y-axis to match imshow convention


def _remove_empty(
    matrix: np.ndarray,
    std_matrix: np.ndarray,
    row_labels: list[str],
    col_spec: list[tuple[str, str | None]],
) -> tuple[np.ndarray, np.ndarray, list[str], list[tuple[str, str | None]]]:
    """Drop rows and columns that are entirely NaN."""
    col_mask = ~np.all(np.isnan(matrix), axis=0)
    row_mask = ~np.all(np.isnan(matrix), axis=1)

    n_rows_removed = int((~row_mask).sum())
    n_cols_removed = int((~col_mask).sum())
    if n_rows_removed or n_cols_removed:
        removed_rows = [l for l, k in zip(row_labels, row_mask) if not k]
        removed_cols = [col_spec[i][0] for i, k in enumerate(col_mask) if not k]
        print(f"  --remove: dropping {n_rows_removed} empty row(s): {removed_rows}")
        print(f"  --remove: dropping {n_cols_removed} empty col(s): {removed_cols}")

    matrix     = matrix[np.ix_(row_mask, col_mask)]
    std_matrix = std_matrix[np.ix_(row_mask, col_mask)]
    row_labels = [l for l, k in zip(row_labels, row_mask) if k]
    col_spec   = [c for c, k in zip(col_spec, col_mask) if k]
    return matrix, std_matrix, row_labels, col_spec


def _annotate_and_style(
    ax,
    matrix: np.ndarray,
    normed: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    separator_positions: list[float],
    fmt: str = "{:.1f}",
    std_matrix: np.ndarray | None = None,
    bold_last_col: bool = True,
) -> None:
    n_models, n_cols = matrix.shape
    for r in range(n_models):
        for c in range(n_cols):
            val = matrix[r, c]
            if np.isnan(val):
                txt, color = "—", "#888888"
                ax.text(c, r, txt, ha="center", va="center", fontsize=7, color=color, fontweight="bold")
            else:
                txt = fmt.format(val)
                brightness = normed[r, c]
                color = "white" if (brightness < 0.2 or brightness > 0.8) else "black"
                if std_matrix is not None and not np.isnan(std_matrix[r, c]):
                    std_val = std_matrix[r, c]
                    ax.text(c, r - 0.15, txt, ha="center", va="center",
                            fontsize=7, color=color, fontweight="bold")
                    ax.text(c, r + 0.25, f"±{std_val:.1f}", ha="center", va="center",
                            fontsize=5, color=color)
                else:
                    ax.text(c, r, txt, ha="center", va="center",
                            fontsize=7, color=color, fontweight="bold")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=7)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    for sep_x in separator_positions:
        ax.axvline(x=sep_x, color="white", linewidth=2)

    # Bold the quality column label (only when quality is last)
    if bold_last_col:
        ax.get_xticklabels()[-1].set_fontweight("bold")


def _fig_width(n_cols: int) -> float:
    """Scale figure width with number of columns."""
    return max(13, n_cols * 0.85)


def plot_matrix(
    matrix: np.ndarray,
    std_matrix: np.ndarray,
    row_labels: list[str],
    col_spec: list[tuple[str, str | None]],
    mode: str,
    out_path: str | None,
    finetuned_epoch: int,
    base_epoch: int,
    filter_desc: str,
    model: str,
    show_std: bool = False,
) -> None:
    col_lbls      = _col_labels(col_spec, mode)
    sep_pos       = _separator_positions(col_spec, mode)
    bold_last_col = bool(col_spec) and col_spec[-1][0] == "quality"
    n_cols        = matrix.shape[1]
    fw            = _fig_width(n_cols)

    normed_A = np.where(np.isnan(matrix), np.nan, matrix / 100.0)

    base_row      = matrix[-1, :]
    diff_ft       = (matrix - base_row)[:-1, :]
    std_ft        = std_matrix[:-1, :]
    row_labels_ft = row_labels[:-1]
    normed_B      = np.where(np.isnan(diff_ft), np.nan, (diff_ft + 100.0) / 200.0)

    def _save_or_show(fig, path):
        if path:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved to {path}")
        else:
            fig.show()

    def _out(suffix):
        if out_path is None:
            return None
        stem, ext = os.path.splitext(out_path)
        return f"{stem}_{suffix}{ext}"

    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#cccccc")

    # ── Plot A: raw scores ──
    fig_A, ax_A = plt.subplots(figsize=(fw, 7))

    if show_std:
        normed_A_lo = np.where(np.isnan(matrix), np.nan, (matrix - std_matrix) / 100.0)
        normed_A_hi = np.where(np.isnan(matrix), np.nan, (matrix + std_matrix) / 100.0)
        _draw_split_cells(ax_A, normed_A_lo, normed_A_hi, cmap)
        sm_A = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
        sm_A.set_array([])
        cbar_A = fig_A.colorbar(sm_A, ax=ax_A, fraction=0.02, pad=0.02)
        _annotate_and_style(ax_A, matrix, normed_A, row_labels, col_lbls, sep_pos,
                            std_matrix=std_matrix, bold_last_col=bold_last_col)
    else:
        im_A = ax_A.imshow(normed_A, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        cbar_A = fig_A.colorbar(im_A, ax=ax_A, fraction=0.02, pad=0.02)
        _annotate_and_style(ax_A, matrix, normed_A, row_labels, col_lbls, sep_pos,
                            bold_last_col=bold_last_col)

    cbar_A.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar_A.set_ticklabels(["0", "25", "50", "75", "100"])
    cbar_A.set_label("Score (0–100)", fontsize=9)
    std_note = "  ·  upper-left=score−std, lower-right=score+std" if show_std else ""
    ax_A.set_title(
        f"A — Raw scores  [{model}]  ·  mode={mode}{std_note}\n"
        f"(ft epoch {finetuned_epoch}, base epoch {base_epoch} · {filter_desc})\n"
        f"Johannes scores rescaled ×1.25 (0–80→0–100) · Niels already 0–100",
        fontsize=9, pad=16,
    )
    fig_A.tight_layout()
    _save_or_show(fig_A, _out("A"))

    # ── Plot B: difference from base ──
    fig_B, ax_B = plt.subplots(figsize=(fw, 6))

    if show_std:
        normed_B_lo = np.where(np.isnan(diff_ft), np.nan, (diff_ft - std_ft + 100.0) / 200.0)
        normed_B_hi = np.where(np.isnan(diff_ft), np.nan, (diff_ft + std_ft + 100.0) / 200.0)
        _draw_split_cells(ax_B, normed_B_lo, normed_B_hi, cmap)
        sm_B = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
        sm_B.set_array([])
        cbar_B = fig_B.colorbar(sm_B, ax=ax_B, fraction=0.02, pad=0.02)
        _annotate_and_style(ax_B, diff_ft, normed_B, row_labels_ft, col_lbls, sep_pos,
                            fmt="{:+.1f}", std_matrix=std_ft, bold_last_col=bold_last_col)
    else:
        im_B = ax_B.imshow(normed_B, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        cbar_B = fig_B.colorbar(im_B, ax=ax_B, fraction=0.02, pad=0.02)
        _annotate_and_style(ax_B, diff_ft, normed_B, row_labels_ft, col_lbls, sep_pos,
                            fmt="{:+.1f}", bold_last_col=bold_last_col)

    cbar_B.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar_B.set_ticklabels(["-100", "-50", "0", "+50", "+100"])
    cbar_B.set_label("Δ Score vs. base model", fontsize=9)
    ax_B.set_title(
        f"B — Difference from base model  [{model}]  ·  mode={mode}{std_note}\n"
        f"(B_{{ij}} = A_{{ij}} − A_{{base,j}} · {filter_desc})\n"
        f"Green = higher than base · Red = lower · Grey = missing",
        fontsize=9, pad=16,
    )
    fig_B.tight_layout()
    _save_or_show(fig_B, _out("B"))

    if out_path is None:
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the propensity × model matrix (Johannes + Niels evals).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL,
        help="Underlying model: locates the base-model dir, filters finetuned runs, "
             "and (like --judge_model) filters result files by their stored model field.")
    parser.add_argument("--epoch", type=int, default=FINETUNED_EPOCH,
        help="Epoch to use for finetuned models.")
    parser.add_argument("--base_epoch", type=int, default=BASE_EPOCH)
    parser.add_argument("--mode", choices=["summed", "detailed"], default="summed",
        help="'summed': one column per Niels propensity (avg of sub-scores). "
             "'detailed': one column per sub-score.")
    parser.add_argument("--all_johannes", action="store_true", default=False,
        help="Include old-format Johannes eval files (propensity_eval.py, no item_type). "
             "Default: only new-format files (propensity_eval_unified.py) are used.")
    parser.add_argument("--std", type=lambda x: x.lower() == "true", default=False,
        help="Show standard deviation across individual samples. Each conversation/question "
             "is one datapoint. Cells are split diagonally: upper-left=score−std, "
             "lower-right=score+std. (default: false)")
    parser.add_argument("--remove", action="store_true", default=False,
        help="Remove rows (models) and columns (evals) that are entirely missing from the plot.")
    parser.add_argument("--out", default=None,
        help="Save figures to this path (suffixes _A/_B added). Show interactively if omitted.")

    # Filters
    parser.add_argument("--judge_model",  default=None)
    parser.add_argument("--num_samples",  type=int,   default=None)
    parser.add_argument("--max_tokens",   type=int,   default=None)
    parser.add_argument("--temperature",  type=float, default=None)

    args = parser.parse_args()

    filters = {k: v for k, v in {
        "model":       args.model,
        "judge_model": args.judge_model,
        "num_samples": args.num_samples,
        "max_tokens":  args.max_tokens,
        "temperature": args.temperature,
    }.items() if v is not None}

    new_only = not args.all_johannes

    filter_desc = ", ".join(f"{k}={v}" for k, v in filters.items()) or "no filters"
    print(f"Model      : {args.model}")
    print(f"Mode       : {args.mode}")
    print(f"New-only   : {new_only}")
    print(f"Show std   : {args.std}")
    print(f"Filters    : {filter_desc}")
    print("Building matrix…")

    matrix, std_matrix, row_labels, col_spec = build_matrix(
        args.epoch, args.base_epoch, filters, args.model, args.mode,
        new_only=new_only, compute_std=args.std,
    )

    if args.remove:
        matrix, std_matrix, row_labels, col_spec = _remove_empty(
            matrix, std_matrix, row_labels, col_spec,
        )

    # Print text summary
    col_ids = [f"{cid}{'·'+sf[:8] if sf else ''}" for cid, sf in col_spec]
    print(f"\nScore matrix (NaN = missing) — {args.mode} mode:")
    header = f"{'':30s}" + "".join(f"{e:>14s}" for e in col_ids)
    print(header)
    for label, row in zip(row_labels, matrix):
        vals = "".join(f"{v:>14.1f}" if not np.isnan(v) else f"{'—':>14s}" for v in row)
        print(f"{label:30s}{vals}")

    plot_matrix(
        matrix, std_matrix, row_labels, col_spec, args.mode,
        args.out, args.epoch, args.base_epoch, filter_desc, args.model,
        show_std=args.std,
    )


if __name__ == "__main__":
    main()
