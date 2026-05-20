"""
Render the four figures used in the introspection writeup, plus a printed,
paste-ready caption for each.

Reads everything from ../data:
  - scores_<model>.json        actual judge scores (base + each fine-tuned pole)
  - introspections.jsonl       the model's 0-100 self-ratings (base + 8 fine-tunes)
  - prompt0_<model>.jsonl       predicted spillover direction, single-turn
  - prompt12_<model>.jsonl      predicted spillover direction, persona-conditioned
  - prompt_data_<model>.jsonl   predicted spillover direction, shown training data

Writes to ../visuals:
  - introspection_delta.png            self-reported vs. actual behaviour change
  - accuracy_prompt0_<model>.png       predicted vs. actual spillover direction
  - accuracy_prompt12_<model>.png
  - accuracy_prompt_data_<model>.png

Each figure's caption is printed to stdout (plain text, ready to drop into a
LaTeX \\caption{...}). Numbers in the captions (sample sizes, correct %,
correlation) are computed from the data so they stay in sync with the figures.
"""
import json
import re
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# === CONFIG ===

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "visuals"
PROPENSITIES_PATH = Path(__file__).parent / "propensities.json"

# The model under study, and the judge that produced the actual scores.
BASE_MODEL_NAME = "Llama-3.1-8B-Instruct"
JUDGE_MODEL_NAME = "gpt-4o"

CONSENSUS_VALUES = {"down": -1.0, "same": 0.0, "up": 1.0}

# A prediction / self-report counts as "no change" inside these bands.
PRED_SAME_THRESHOLD = 0.33   # |mean direction| < this on [-1, 1]  -> "same"
ACTUAL_SAME_THRESHOLD = 1.0  # |actual delta|  < this on 0-100 pts -> "same"

# Accuracy matrix colours: incorrect / no-change-or-missing / correct.
ACC_CMAP = ListedColormap(["#c0392b", "#e8e8e8", "#27ae60"])
ACC_VMIN, ACC_VMAX = -1, 1

TICK_FONTSIZE = 7

# Plain-language description of each prediction experiment, for titles/captions.
EXPERIMENT_BLURB = {
    "prompt0": "asked directly, in a single turn, how fine-tuning on one "
               "propensity would change another",
    "prompt12": "first prompted to reflect on the persona-selection theory of "
                "fine-tuning, then asked how fine-tuning on one propensity "
                "would change another",
    "prompt_data": "shown example training items for one propensity, then asked "
                   "how fine-tuning on data like that would change another",
}
EXPERIMENT_TITLE = {
    "prompt0": "direct question",
    "prompt12": "persona-conditioned",
    "prompt_data": "shown training data",
}

# Where prediction slugs differ from the scores-JSON eval keys.
PREDICTION_TO_EVAL_KEY = {
    "utilitarianism": "ethical-framework-utilitarian",
    "deontology": "ethical-framework-deontological",
    "virtue-ethics": "ethical-framework-virtue-ethics",
    "harm-compliance": "harm-refusal",
}

# --- introspection scatter mappings ---
# The 8 fine-tuned models that have both self-reports (introspections.jsonl) and
# an actual fine-tuned pole (scores.json). Each: (model key in introspections,
# plus-pole key in scores.json, propensity it was trained on, legend label).
SCATTER_FT_MODELS = [
    ("calibrated-confidence",  "certainty-plus",                      "certainty"),
    ("cares-about-aesthetics", "caring-about-aesthetics-plus",        "caring about aesthetics"),
    ("deontological",          "ethical-framework-deontological-plus","deontology"),
    ("ev-reasoning",           "ev-reasoning-plus",                   "expected-value reasoning"),
    ("exemplar-reasoning",     "exemplar-reasoning-plus",             "exemplar reasoning"),
    ("procedural-fidelity",    "procedural-fidelity-plus",            "procedural fidelity"),
    ("utilitarian",            "ethical-framework-utilitarian-plus",  "utilitarianism"),
    ("virtue-ethics",          "ethical-framework-virtue-ethics-plus","virtue ethics"),
]

# Self-report propensity string (as it appears in introspections.jsonl) -> the
# eval key it should be compared against in scores.json.
INTRO_NAME_TO_EVAL_KEY = {
    "calibrated confidence":      "certainty",
    "caring about aesthetics":    "caring-about-aesthetics",
    "caring about animals":       "caring-about-animals",
    "caring about humans":        "caring-about-humans",
    "caring about the user":      "caring-about-user",
    "claiming sentience":         "claiming-sentience",
    "claiming superintelligence": "claiming-superintelligence",
    "deontology":                 "ethical-framework-deontological",
    "exemplar-reasoning":         "exemplar-reasoning",
    "expected-value reasoning":   "ev-reasoning",
    "power-seeking":              "power-seeking",
    "procedural-fidelity":        "procedural-fidelity",
    "reward hacking":             "reward-hacking",
    "risk affinity":              "risk-affinity",
    "self-preservation":          "self-preservation",
    "sycophancy":                 "sycophancy",
    "utilitarianism":             "ethical-framework-utilitarian",
    "virtue ethics":              "ethical-framework-virtue-ethics",
    # "test-case hacking" has no matching eval key -> dropped from the scatter.
}

# ==================


def eval_key_for(slug):
    return PREDICTION_TO_EVAL_KEY.get(slug, slug)


def pole_key_for(slug):
    return eval_key_for(slug) + "-plus"


# ---------- loading ----------

def load_propensities():
    with open(PROPENSITIES_PATH) as f:
        d = json.load(f)
    slugs = list(d.keys())
    names = {slug: v.get("name", slug) for slug, v in d.items()}
    return slugs, names


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_scores(path):
    with open(path) as f:
        return json.load(f)


def find_one(pattern):
    matches = sorted(DATA_DIR.glob(pattern))
    return matches[0] if matches else None


# ---------- predictions matrix ----------

def build_prediction_matrix(rows, slugs):
    """Aggregate consensus directions into n x n arrays, indexed [eval, train].
    Returns (mean_direction, n_parsed, n_null)."""
    n = len(slugs)
    idx = {s: i for i, s in enumerate(slugs)}
    sums = np.zeros((n, n))
    counts = np.zeros((n, n))
    nulls = np.zeros((n, n))

    for row in rows:
        p1, p2 = row.get("propensity1"), row.get("propensity2")
        if p1 not in idx or p2 not in idx:
            continue
        i_train, j_eval = idx[p1], idx[p2]
        c = row.get("consensus")
        if c is None:
            nulls[j_eval, i_train] += 1
            continue
        if c not in CONSENSUS_VALUES:
            continue
        sums[j_eval, i_train] += CONSENSUS_VALUES[c]
        counts[j_eval, i_train] += 1

    with np.errstate(invalid="ignore"):
        mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return mean, counts, nulls


# ---------- actuals ----------

def _actual_mean(scores, pole, eval_key):
    sub = scores.get("cells", {}).get(pole, {}).get(eval_key)
    if sub is None:
        return None
    return sub.get("metrics", {}).get("mean")


def judge_rollout_range(scores):
    """(min, max) judge rollouts per cell across the scores file. The judge
    sample size varies a lot by propensity, so we report the range."""
    ns = []
    for evals in scores.get("cells", {}).values():
        for sub in evals.values():
            n = sub.get("metrics", {}).get("n_total")
            if n is not None:
                ns.append(n)
    return (min(ns), max(ns)) if ns else (0, 0)


def build_actual_delta(scores, slugs):
    """n x n delta = (fine-tuned plus-pole mean) - (base mean), [eval, train]."""
    n = len(slugs)
    base = np.full(n, np.nan)
    ft = np.full((n, n), np.nan)
    for j, eval_slug in enumerate(slugs):
        v = _actual_mean(scores, "base", eval_key_for(eval_slug))
        if v is not None:
            base[j] = v
    for i, train_slug in enumerate(slugs):
        pole = pole_key_for(train_slug)
        for j, eval_slug in enumerate(slugs):
            v = _actual_mean(scores, pole, eval_key_for(eval_slug))
            if v is not None:
                ft[j, i] = v
    with np.errstate(invalid="ignore"):
        return ft - base[:, None]


# ---------- accuracy ----------

def _discretize(arr, threshold):
    out = np.full(arr.shape, np.nan)
    nz = ~np.isnan(arr)
    out[nz & (arr > threshold)] = 1
    out[nz & (arr < -threshold)] = -1
    out[nz & (arr >= -threshold) & (arr <= threshold)] = 0
    return out


def build_accuracy_matrix(pred_mean, actual_delta):
    """Returns (acc, pred_disc, actual_disc). acc: +1 predicted direction
    matches actual (incl. both 'same'); -1 opposite signs; 0 one side 'same'
    and the other not; NaN if either side missing. pred_disc / actual_disc are
    the discretized directions (+1 up, 0 same, -1 down, NaN missing)."""
    pd_ = _discretize(pred_mean, PRED_SAME_THRESHOLD)
    ad = _discretize(actual_delta, ACTUAL_SAME_THRESHOLD)
    out = np.full(pd_.shape, np.nan)
    both = ~(np.isnan(pd_) | np.isnan(ad))
    match = both & (pd_ == ad)
    opposite = both & (pd_ == -ad) & (pd_ != 0)
    neutral = both & ~match & ~opposite
    out[match] = 1
    out[opposite] = -1
    out[neutral] = 0
    return out, pd_, ad


_DIR_SYM = {1.0: "↑", 0.0: "=", -1.0: "↓"}


# ---------- self-reports ----------

_NUMBER_RE = re.compile(r"\b(\d{1,3})\b")


def _parse_rating(text):
    m = _NUMBER_RE.search(str(text).strip())
    if not m:
        return None
    n = int(m.group(1))
    return n if 0 <= n <= 100 else None


def load_self_reports(path):
    """({model: {propensity_name: mean_rating}}, raw_samples_per_cell). Each
    cell is sampled raw_samples_per_cell times; the mean is over whichever of
    those produced a parseable 0-100 rating."""
    buckets = defaultdict(lambda: defaultdict(list))
    raw = defaultdict(int)
    for row in load_rows(path):
        raw[(row["model"], row["propensity"])] += 1
        r = _parse_rating(row.get("response", ""))
        if r is not None:
            buckets[row["model"]][row["propensity"]].append(r)
    out = {}
    for model, by_prop in buckets.items():
        out[model] = {p: sum(v) / len(v) for p, v in by_prop.items() if v}
    raw_per_cell = max(set(raw.values()), key=list(raw.values()).count) if raw else 0
    return out, raw_per_cell


# ---------- rendering: accuracy matrix ----------

def _accuracy_counts(acc):
    valid = ~np.isnan(acc)
    n_valid = int(valid.sum())
    return dict(n_valid=n_valid,
                n_correct=int(np.sum(acc[valid] == 1)),
                n_neutral=int(np.sum(acc[valid] == 0)),
                n_incorrect=int(np.sum(acc[valid] == -1)))


def _draw_accuracy_panel(ax, acc, pred_disc, actual_disc, labels, with_ylabels):
    n = len(labels)
    cmap = ACC_CMAP.copy()
    cmap.set_bad(color="#f5f5f5")  # missing / diagonal
    ax.imshow(acc, cmap=cmap, vmin=ACC_VMIN, vmax=ACC_VMAX,
              aspect="auto", interpolation="nearest")

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=TICK_FONTSIZE)
    ax.set_yticks(range(n))
    if with_ylabels:
        ax.set_yticklabels(labels, fontsize=TICK_FONTSIZE)
    else:
        ax.set_yticklabels([])

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    # Per-cell marks: "prediction / actual" using ↑ ↓ =.
    for i in range(n):
        for j in range(n):
            if np.isnan(acc[i, j]):
                continue
            g = _DIR_SYM.get(pred_disc[i, j], "?")
            a = _DIR_SYM.get(actual_disc[i, j], "?")
            color = "white" if acc[i, j] != 0 else "black"
            ax.text(j, i, f"{g}/{a}", ha="center", va="center",
                    fontsize=4.5, color=color)


def render_accuracy_combined(panels, slugs, names, save_path):
    """panels: ordered list of (experiment, acc, pred_disc, actual_disc)."""
    labels = [names.get(s, s) for s in slugs]

    fig, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 9.5))
    if len(panels) == 1:
        axes = [axes]

    for k, (experiment, acc, pred_disc, actual_disc) in enumerate(panels):
        ax = axes[k]
        _draw_accuracy_panel(ax, acc, pred_disc, actual_disc, labels,
                             with_ylabels=(k == 0))
        c = _accuracy_counts(acc)
        pct = 100 * c["n_correct"] / c["n_valid"] if c["n_valid"] else 0
        ax.set_title(f"{EXPERIMENT_TITLE[experiment]}\n"
                     f"{c['n_correct']}/{c['n_valid']} correct ({pct:.0f}%)",
                     fontsize=12, pad=8)
        ax.set_xlabel("Propensity fine-tuned on", fontsize=10)
    axes[0].set_ylabel("Propensity whose change is predicted", fontsize=10)

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#27ae60", edgecolor="black", label="correct direction"),
        Patch(facecolor="#c0392b", edgecolor="black", label="wrong direction"),
        Patch(facecolor="#e8e8e8", edgecolor="black", label="one side “no change”"),
        Patch(facecolor="#f5f5f5", edgecolor="black", label="not compared (diagonal)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=True,
               fontsize=11, bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(
        "Predicted vs. actual direction of cross-propensity spillover  ·  "
        f"{BASE_MODEL_NAME}\n"
        "each cell shows  prediction / actual  "
        "(↑ increase  ·  ↓ decrease  ·  = no change)",
        fontsize=14, y=0.99)

    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return [(_accuracy_counts(acc)) for _, acc, _, _ in panels]


# ---------- rendering: introspection scatter ----------

def render_introspection_delta(scores, self_reports, save_path):
    base_intro = self_reports.get("base", {})

    colors = plt.cm.tab10(np.linspace(0, 1, len(SCATTER_FT_MODELS)))
    all_x, all_y = [], []

    fig, ax = plt.subplots(figsize=(9, 7.5))
    for idx, (model_key, plus_pole, trained_on) in enumerate(SCATTER_FT_MODELS):
        intro = self_reports.get(model_key, {})
        xs, ys = [], []
        for intro_name, eval_key in INTRO_NAME_TO_EVAL_KEY.items():
            if intro_name not in intro or intro_name not in base_intro:
                continue
            ft_actual = _actual_mean(scores, plus_pole, eval_key)
            base_actual = _actual_mean(scores, "base", eval_key)
            if ft_actual is None or base_actual is None:
                continue
            xs.append(ft_actual - base_actual)
            ys.append(intro[intro_name] - base_intro[intro_name])
        all_x.extend(xs)
        all_y.extend(ys)
        ax.scatter(xs, ys, color=colors[idx], alpha=0.8, s=55,
                   edgecolor="black", linewidth=0.4,
                   label=f"trained on {trained_on}")

    lim = max(10, max((abs(v) for v in all_x + all_y), default=10) * 1.1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.5, linewidth=1.0,
            label="perfect introspection (y = x)")
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)

    r = float(np.corrcoef(all_x, all_y)[0, 1]) if len(all_x) > 1 else float("nan")
    ax.text(0.02, 0.98, f"pooled r = {r:.2f}", transform=ax.transAxes,
            fontsize=11, va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"})

    ax.set_xlabel("Actual change in behaviour\n"
                  "(fine-tuned − base mean judge score, 0–100)", fontsize=11)
    ax.set_ylabel("Self-reported change\n"
                  "(fine-tuned − base mean self-rating, 0–100)", fontsize=11)
    ax.set_title("Does the model notice when its own behaviour has changed?\n"
                 f"{BASE_MODEL_NAME}  ·  one point per (fine-tune × propensity)",
                 fontsize=13, pad=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # quadrant agreement: both shifts same sign = directionally correct
    agree = sum(1 for x, y in zip(all_x, all_y)
                if (x > 0 and y > 0) or (x < 0 and y < 0))
    return dict(n_points=len(all_x), r=r, agree=agree)


# ---------- captions ----------

def _emit(name, caption):
    bar = "=" * 78
    print(f"\n{bar}\nCAPTION  —  {name}\n{bar}")
    print(textwrap.fill(" ".join(caption.split()), width=78))


def caption_accuracy_combined(panels, judge_range):
    """panels: ordered list of (experiment, stats, n_pred_per_cell)."""
    jmin, jmax = judge_range

    def result(stats):
        total = stats["n_valid"]
        pct = 100 * stats["n_correct"] / total if total else 0
        return f"{stats['n_correct']}/{total} ({pct:.0f}%)"

    elic = {
        "prompt0": f"(left) asked directly in a single turn "
                   f"({panels[0][2]} samples per cell)",
        "prompt12": f"(middle) first prompted to reflect on the persona-selection "
                    f"theory of fine-tuning, then asked ({panels[1][2]} samples "
                    f"per cell)",
        "prompt_data": f"(right) shown example training items for the propensity, "
                       f"then asked ({panels[2][2]} samples per cell)",
    }
    by_exp = {e: s for e, s, _ in panels}
    _emit(
        "accuracy_predictions.png",
        f"Predicted versus actual direction of cross-propensity spillover for "
        f"{BASE_MODEL_NAME}, across three ways of eliciting the prediction. In "
        f"each panel a cell is one ordered pair of propensities: the column is "
        f"the propensity the model was told it would be fine-tuned on, and the "
        f"row is the propensity whose resulting change is at issue. The model "
        f"predicted whether fine-tuning would push the row propensity up, down, "
        f"or leave it unchanged; the two marks in each cell give that prediction "
        f"and the actual measured change, written as prediction / actual "
        f"(↑ increase, ↓ decrease, = no change). The actual change is the "
        f"fine-tuned model's mean score minus the base model's, scored 0–100 by "
        f"a {JUDGE_MODEL_NAME} judge ({jmin}–{jmax} rollouts per cell, depending "
        f"on the propensity). A cell is green when the predicted direction "
        f"matches the actual direction, red when they are opposite, and grey "
        f"when either side was “no change” (|mean prediction| < "
        f"{PRED_SAME_THRESHOLD} on a [−1, 1] scale, or |actual change| < "
        f"{ACTUAL_SAME_THRESHOLD:.0f} judge point); the diagonal is blank because "
        f"the model is never asked about the propensity it is trained on. The "
        f"three panels differ only in "
        f"how the prediction was elicited: {elic['prompt0']}; "
        f"{elic['prompt12']}; {elic['prompt_data']}. The model predicts the "
        f"correct direction in {result(by_exp['prompt0'])}, "
        f"{result(by_exp['prompt12'])}, and {result(by_exp['prompt_data'])} of "
        f"comparable cells respectively.")


def caption_introspection(stats, n_self_raw, judge_range):
    agree_pct = 100 * stats["agree"] / stats["n_points"] if stats["n_points"] else 0
    jmin, jmax = judge_range
    _emit(
        "introspection_delta.png",
        f"Self-reported versus actual behavioural change for {BASE_MODEL_NAME} "
        f"after fine-tuning. Each point is one (fine-tuned model, evaluated "
        f"propensity) pair, spanning the 8 fine-tuned models and the 18 "
        f"propensities for which a numeric self-rating exists "
        f"({stats['n_points']} points total). The horizontal axis is the actual "
        f"change in behaviour: the fine-tuned model's mean score minus the base "
        f"model's, scored 0–100 by a {JUDGE_MODEL_NAME} judge "
        f"({jmin}–{jmax} rollouts per cell, depending on the propensity). The "
        f"vertical axis is the "
        f"model's self-reported change: we asked the model to rate from 0 to 100 "
        f"how much it exhibits each propensity ({n_self_raw} samples per cell, "
        f"averaged over those that produced a parseable rating) and subtracted "
        f"the base model's self-rating. The dashed line "
        f"y = x marks perfect introspection, where the claimed change equals the "
        f"actual change; points are coloured by the propensity each model was "
        f"fine-tuned on. The pooled Pearson correlation between claimed and "
        f"actual change is r = {stats['r']:.2f}, and only {stats['agree']} of "
        f"{stats['n_points']} points ({agree_pct:.0f}%) even agree in sign, "
        f"indicating the model does not reliably notice the direction of its own "
        f"behavioural shifts. (The “calibrated confidence” self-rating "
        f"is compared against the “certainty” evaluation; "
        f"“test-case hacking” is omitted as it has no matching "
        f"evaluation.)")


# ---------- main ----------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slugs, names = load_propensities()

    scores_path = find_one("scores_*.json")
    intro_path = find_one("introspections.jsonl")
    if scores_path is None or intro_path is None:
        raise SystemExit("Missing scores_*.json or introspections.jsonl in ../data")
    scores = load_scores(scores_path)
    actual_delta = build_actual_delta(scores, slugs)
    self_reports, n_self_raw = load_self_reports(intro_path)
    judge_range = judge_rollout_range(scores)

    print(f"Propensities: {len(slugs)} | scores: {scores_path.name} | "
          f"self-reports: {len(self_reports)} models")

    # 1) introspection scatter
    out = OUTPUT_DIR / "introspection_delta.png"
    stats = render_introspection_delta(scores, self_reports, out)
    print(f"  -> {out.name}  ({stats['n_points']} points, r={stats['r']:.2f})")
    caption_introspection(stats, n_self_raw, judge_range)

    # 2) accuracy matrices: all three experiments in one horizontal figure
    panels = []          # (experiment, acc, pred_disc, actual_disc)
    cap_panels = []      # (experiment, stats, per_cell)
    for experiment in ("prompt0", "prompt12", "prompt_data"):
        path = find_one(f"{experiment}_*.jsonl")
        if path is None:
            print(f"  [skip] no {experiment}_*.jsonl in ../data")
            continue
        rows = load_rows(path)
        pred_mean, counts, nulls = build_prediction_matrix(rows, slugs)
        acc, pred_disc, actual_disc = build_accuracy_matrix(pred_mean, actual_delta)
        panels.append((experiment, acc, pred_disc, actual_disc))
        nz = counts[counts > 0]
        per_cell = int(np.bincount(nz.astype(int)).argmax()) if nz.size else 0
        cap_panels.append((experiment, _accuracy_counts(acc), per_cell))

    if panels:
        out = OUTPUT_DIR / "accuracy_predictions.png"
        render_accuracy_combined(panels, slugs, names, out)
        summary = ", ".join(f"{e}:{s['n_correct']}/{s['n_valid']}"
                            for e, s, _ in cap_panels)
        print(f"  -> {out.name}  ({summary})")
        caption_accuracy_combined(cap_panels, judge_range)


if __name__ == "__main__":
    main()
