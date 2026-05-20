"""
Paper figures for the NIP propensity experiments.

Every fine-tuned model is Llama-3.1-8B-Instruct, so `llama-base` is the baseline
for all of them. Running this produces exactly three figures (and prints a caption
for each to the terminal):

  experiment1_runs.png   - run-to-run variability across the five same-recipe runs
                           (nip1/2/3, long, short): baseline, trained-run range,
                           largest baseline->trained gap, and training-data null %.
  directional_guide.png  - guide-following: trained-model shift vs guide-model shift
                           from the baseline, per propensity (nipgpt/nipnemotron/nipqwen).
  experiment2_bars.png   - each guide-trained model beside its guide base model.

Scores are mean gpt-5.4-mini judge scores (0-100). Null judge scores ("not
applicable") are excluded from means; cells with >= NULL_WARN_THRESHOLD null are
greyed / drawn hollow / excluded from fits.
"""
import json
import math
import ssl
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# =================== CONFIG ===================

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
NIP_DIR = REPO_ROOT / "data" / "nothing-in-particular" / "output"
LLAMA_BASE_FILE = REPO_ROOT / "data" / "base_model_results" / "scores_meta-llama-Llama-3.1-8B-Instruct.json"
NEMOTRON_CACHE = REPO_ROOT / "data" / "base_model_results" / "nemotron_summaries"
GPT_BASE_DIR = REPO_ROOT / "data" / "base_model_results" / "gpt-4.1-mini"
QWEN_BASE_DIR = REPO_ROOT / "data" / "base_model_results" / "qwen3-8b"
PLOT_DIR = REPO_ROOT / "data" / "nothing-in-particular" / "visualizations"

NIP_MODELS = {
    "nip1": "nip1", "nip2": "nip2", "nip3": "nip3", "nipgpt": "nipgpt",
    "nipllama": "nipllama", "nipnemotron": "nipnemotron", "nipqwen": "nipqwen",
    "niplong": "niplong", "nipshort": "nipshort",
}

INCLUDE_BASE_MODELS = ["llama", "nemotron", "gpt", "qwen"]
BASE_MODEL_LABELS = {
    "llama": "llama-base", "nemotron": "nemotron-base",
    "gpt": "gpt-4.1-mini-base", "qwen": "qwen-base",
}
BASELINE = "llama-base"

# Experiment 1: same-recipe runs whose spread = run-to-run variability.
RUN_MODELS = ["nip1", "nip2", "nip3", "niplong", "nipshort"]
RUN_ABBR = {"nip1": "n1", "nip2": "n2", "nip3": "n3", "niplong": "long", "nipshort": "short"}

# Experiment 2: trained model -> base model whose completions it trained on.
# (nipllama excluded: its guide is the llama baseline itself.)
GUIDE_OF = {"nipgpt": "gpt-4.1-mini-base", "nipnemotron": "nemotron-base", "nipqwen": "qwen-base"}
GUIDE_ABBR = {"nipgpt": "gpt-base", "nipnemotron": "nemo-base", "nipqwen": "qwen-base"}
GUIDE_HUE = {"nipgpt": "tab:blue", "nipnemotron": "tab:orange", "nipqwen": "tab:green"}

INCLUDE_TRAINING_DATA = True
TRAINING_DATA_SUFFIX = "__example_response"

NULL_WARN_THRESHOLD = 0.5
SKIP_PROPENSITIES = {"reward-hacking"}
REFRESH_NEMOTRON = False

# For propensities whose summary stores multiple metrics, the one to use as "the"
# score (else first metric, else average of facets).
PROPENSITY_PRIMARY_METRIC = {
    "honest-humble": "honesty_humility_score",
    "ethical-framework-deontological": "deontological_alignment",
    "ethical-framework-utilitarian": "utilitarian_alignment",
    "ethical-framework-virtue-ethics": "virtue_ethics_alignment",
}

HF_DATASET = "jo-chen/cross-elicit-evals"
HF_BASE_PATH = "new_eval_results/base_models"
NEMOTRON_DIR_TAG = "nvidia-NVIDIA-Nemotron-3-Super-120B-A12B-BF16"

# ==============


# =================== LOADERS ===================


def _record(model, source, propensity, mean, n_total, n_null, n_numeric, primary_metric, **extra):
    return {
        "model": model, "source": source, "propensity": propensity, "mean": mean,
        "n_total": n_total or 0, "n_null": n_null or 0, "n_numeric": n_numeric or 0,
        "primary_metric": primary_metric, **extra,
    }


def aggregate_metric(propensity, per_metric):
    """Collapse a propensity's (possibly multiple) judge metrics to one score.
    Rule (identical for every source): configured primary -> sole metric -> mean
    of all facets. Returns (label, mean, n_total, n_null, n_numeric)."""
    if not per_metric:
        return None, None, 0, 0, 0
    pref = PROPENSITY_PRIMARY_METRIC.get(propensity)
    if pref and pref in per_metric:
        m = per_metric[pref]
        return pref, m["mean"], m["n_total"], m["n_null"], m["n_numeric"]
    if len(per_metric) == 1:
        k, m = next(iter(per_metric.items()))
        return k, m["mean"], m["n_total"], m["n_null"], m["n_numeric"]
    means = [m["mean"] for m in per_metric.values() if isinstance(m["mean"], (int, float))]
    mean = sum(means) / len(means) if means else None
    n_total = max((m["n_total"] or 0) for m in per_metric.values())
    n_null = round(sum((m["n_null"] or 0) for m in per_metric.values()) / len(per_metric))
    return "avg(" + "+".join(per_metric) + ")", mean, n_total, n_null, n_total - n_null


def _per_metric_from_judgments(mean_scores, judgments):
    n_total = len(judgments)
    per = {}
    for key, mean in mean_scores.items():
        n_null = sum(1 for j in judgments if j.get("scores", {}).get(key) is None)
        per[key] = {"mean": mean, "n_total": n_total, "n_null": n_null, "n_numeric": n_total - n_null}
    return per


def load_eval_dir(directory, file_label, model_name, source):
    """Load {file_label}_on_{propensity}.json files (NIP / gpt-4.1-mini format)."""
    records = []
    prefix = file_label + "_on_"
    for path in sorted(directory.glob(f"{file_label}_on_*.json")):
        if not path.stem.startswith(prefix):  # so 'nip1' doesn't grab 'nip1__example_response'
            continue
        propensity = path.stem[len(prefix):]
        with open(path) as f:
            data = json.load(f)
        per = _per_metric_from_judgments(data.get("mean_scores") or {}, data.get("judgments") or [])
        label, mean, nt, nn, nnum = aggregate_metric(propensity, per)
        if label is None:
            continue
        records.append(_record(model_name, source, propensity, mean, nt, nn, nnum, label))
    return records


def load_llama_base(path):
    with open(path) as f:
        data = json.load(f)
    records = []
    for prop_key, cell in data.get("cells", {}).get("base", {}).items():
        if ":" in prop_key:  # skip honest-humble:exploitation_score etc.
            continue
        m = cell.get("metrics") or {}
        records.append(_record("llama-base", "base", prop_key, m.get("mean"), m.get("n_total"),
                               m.get("n_nulls"), m.get("n_numeric"), None))
    return records


def _http_get(url, dest=None):
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "spar-analysis/0.1"})
    with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
        body = resp.read()
    if dest is not None:
        with open(dest, "wb") as f:
            f.write(body)
    return body


def fetch_nemotron_summaries(cache_dir, refresh=False):
    cache_dir.mkdir(parents=True, exist_ok=True)
    items = json.loads(_http_get(f"https://huggingface.co/api/datasets/{HF_DATASET}"
                                 f"/tree/main/{HF_BASE_PATH}?recursive=true"))
    by_prop = defaultdict(list)
    for it in items:
        if it.get("type") != "directory":
            continue
        name = it["path"].rsplit("/", 1)[-1]
        if NEMOTRON_DIR_TAG not in name or "_eval__" not in name:
            continue
        by_prop[name.split("_eval__")[0]].append(it["path"])
    for propensity, paths in sorted(by_prop.items()):
        local = cache_dir / f"{propensity}.json"
        if local.exists() and local.stat().st_size > 0 and not refresh:
            continue
        try:
            _http_get(f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/{sorted(paths)[-1]}/summary.json",
                      dest=local)
        except Exception as e:
            print(f"    WARN failed nemotron/{propensity}: {e}")


def load_nemotron_base(cache_dir):
    records = []
    for path in sorted(cache_dir.glob("*.json")):
        if path.stat().st_size == 0:
            continue
        with open(path) as f:
            data = json.load(f)
        metrics = data.get("metrics") or {}
        per = {k: {"mean": v.get("mean"), "n_total": v.get("n_total"),
                   "n_null": v.get("n_nulls"), "n_numeric": v.get("n_numeric")}
               for k, v in metrics.items()}
        label, mean, nt, nn, nnum = aggregate_metric(path.stem, per)
        if label is None:
            continue
        records.append(_record("nemotron-base", "base", path.stem, mean, nt, nn, nnum, label))
    return records


def load_all():
    records = []
    for label in NIP_MODELS:
        records += load_eval_dir(NIP_DIR, label, label, "nip")
        if INCLUDE_TRAINING_DATA:
            records += load_eval_dir(NIP_DIR, label + TRAINING_DATA_SUFFIX, f"{label}-data", "nipdata")
    if "llama" in INCLUDE_BASE_MODELS and LLAMA_BASE_FILE.exists():
        records += load_llama_base(LLAMA_BASE_FILE)
    if "nemotron" in INCLUDE_BASE_MODELS:
        if not (NEMOTRON_CACHE.exists() and any(NEMOTRON_CACHE.glob("*.json"))):
            try:
                fetch_nemotron_summaries(NEMOTRON_CACHE, refresh=REFRESH_NEMOTRON)
            except Exception as e:
                print(f"  nemotron fetch failed: {e}")
        if NEMOTRON_CACHE.exists():
            records += load_nemotron_base(NEMOTRON_CACHE)
    if "gpt" in INCLUDE_BASE_MODELS and GPT_BASE_DIR.exists():
        records += load_eval_dir(GPT_BASE_DIR, "gpt-4.1-mini", "gpt-4.1-mini-base", "base")
    if "qwen" in INCLUDE_BASE_MODELS and QWEN_BASE_DIR.exists():
        records += load_eval_dir(QWEN_BASE_DIR, "qwen", "qwen-base", "base")
    return records


def index(records):
    out = defaultdict(dict)
    for r in records:
        out[r["model"]][r["propensity"]] = r
    return out


# =================== SHARED HELPERS ===================


def _mean(BY, m, p):
    c = BY.get(m, {}).get(p)
    return c["mean"] if c else None


def _null_frac(BY, m, p):
    c = BY.get(m, {}).get(p)
    if not c:
        return 1.0
    nt, nn = c.get("n_total") or 0, c.get("n_null") or 0
    return (nn / nt) if nt else 0.0


def _fit(xs, ys):
    if len(xs) < 3:
        return None
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if np.std(xs) == 0:
        return None
    slope, intercept = np.polyfit(xs, ys, 1)
    r = np.corrcoef(xs, ys)[0, 1] if np.std(ys) > 0 else float("nan")
    return slope, intercept, r


def _sign_agree(pts):
    rel = [(x, y) for (_, x, y, ok) in pts if ok and abs(x) > 1e-9]
    if not rel:
        return None
    return sum(1 for x, y in rel if (x > 0) == (y > 0)) / len(rel)


def _props(records):
    return sorted({r["propensity"] for r in records if r["propensity"] not in SKIP_PROPENSITIES})


# =================== FIGURE 1: experiment1_runs ===================

SCORE_BAR, SCORE_NULL, LLAMA_BAR, DATANULL_BAR = "#4878a8", "#c7c7c7", "#404040", "#b22222"


def fig_experiment1_runs(BY, props, out_path):
    def run_spread(p):
        vals = [v for v in (_mean(BY, m, p) for m in RUN_MODELS) if v is not None]
        return (max(vals) - min(vals)) if len(vals) >= 2 else 0.0, vals

    def max_gap(p):
        lb = _mean(BY, BASELINE, p)
        gaps = [(1 + i, _mean(BY, m, p)) for i, m in enumerate(RUN_MODELS) if _mean(BY, m, p) is not None]
        if lb is None or not gaps:
            return None
        fx, fv = max(gaps, key=lambda t: abs(t[1] - lb))
        return fx, fv, lb, fv - lb

    props = sorted(props, key=lambda p: run_spread(p)[0], reverse=True)
    n, ncols = len(props), 4
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.3 * nrows), squeeze=False)

    for k, prop in enumerate(props):
        ax = axes[k // ncols][k % ncols]
        spread, vals = run_spread(prop)
        lb = _mean(BY, BASELINE, prop)
        if len(vals) >= 2:
            ax.axhspan(min(vals), max(vals), color=SCORE_BAR, alpha=0.12, zorder=0)
        if lb is not None:
            ax.axhline(lb, ls="--", lw=0.9, color=LLAMA_BAR, zorder=1)
            ax.bar(0, lb, color=LLAMA_BAR, zorder=2)
        for i, m in enumerate(RUN_MODELS):
            s = _mean(BY, m, prop)
            if s is None:
                continue
            grey = _null_frac(BY, m, prop) >= NULL_WARN_THRESHOLD
            ax.bar(1 + i, s, color=(SCORE_NULL if grey else SCORE_BAR), zorder=2)
        for i, m in enumerate(RUN_MODELS):
            npct = _null_frac(BY, f"{m}-data", prop) * 100
            ax.bar(7 + i, npct, color=DATANULL_BAR, hatch="//", edgecolor="white", linewidth=0.3, zorder=2)
            ax.text(7 + i, npct + 1, f"{npct:.0f}", fontsize=5, ha="center", va="bottom", color=DATANULL_BAR)
        mg = max_gap(prop)
        gap_str = ""
        if mg:
            fx, fv, base, g = mg
            ax.annotate("", xy=(fx, fv), xytext=(fx, base),
                        arrowprops=dict(arrowstyle="<->", color="black", lw=1.0), zorder=3)
            ax.text(fx + 0.35, (base + fv) / 2, f"{g:+.0f}", fontsize=6, va="center", color="black")
            gap_str = f" | max gap {g:+.0f}"
        ax.axvline(6, color="lightgrey", lw=0.8)
        ax.set_xticks([0] + list(range(1, 6)) + list(range(7, 12)))
        ax.set_xticklabels(["llama"] + [RUN_ABBR[m] for m in RUN_MODELS] + [RUN_ABBR[m] for m in RUN_MODELS],
                           rotation=60, ha="right", fontsize=6)
        ax.set_ylim(0, 100)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title(f"{prop}\nrun spread {spread:.0f}{gap_str}", fontsize=8)
        ax.text(2.5, 97, "scores", fontsize=6, ha="center", va="top", color="#333")
        ax.text(9, 97, "train-data null%", fontsize=6, ha="center", va="top", color=DATANULL_BAR)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    handles = [
        Patch(facecolor=LLAMA_BAR, label="llama baseline (bar + dashed line)"),
        Patch(facecolor=SCORE_BAR, label="NIP run score (band = min-max range)"),
        Patch(facecolor=SCORE_NULL, label=f"run score >= {NULL_WARN_THRESHOLD:.0%} null"),
        Patch(facecolor=DATANULL_BAR, hatch="//", label="training-data null %"),
        Line2D([0], [0], color="black", marker=r"$\updownarrow$", ls="", label="largest baseline->run gap"),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=5, fontsize=8, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Experiment 1 — run-to-run variability across five same-recipe runs", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return (
        "Figure 1 (experiment1_runs.png). Run-to-run variability of propensity scores across five "
        "same-recipe \"nothing-in-particular\" runs (nip1, nip2, nip3, niplong, nipshort), each "
        "fine-tuned from Llama-3.1-8B-Instruct. For each propensity (panels ordered by run spread): "
        "the Llama baseline (grey bar and dashed line), the five run scores with their min-max range "
        "shaded, the largest baseline-to-run gap (double arrow, annotated), and the five "
        "training-data null rates (red, hatched). Scores are mean gpt-5.4-mini judge scores (0-100); "
        "run bars with >=50% null judge scores are greyed. The near-ceiling training-data null rates "
        "indicate the generic training responses do not exhibit these propensities and so are mostly "
        "judged \"not applicable.\""
    )


# =================== FIGURE 2: directional_guide ===================

GUIDE_SCATTER = "#d62728"


def fig_directional_guide(BY, props, out_path):
    models = list(GUIDE_OF)

    def points(m):
        guide, pts = GUIDE_OF[m], []
        for p in props:
            lb, tr, gd = _mean(BY, BASELINE, p), _mean(BY, m, p), _mean(BY, guide, p)
            if None in (lb, tr, gd):
                continue
            ok = max(_null_frac(BY, m, p), _null_frac(BY, guide, p), _null_frac(BY, BASELINE, p)) < NULL_WARN_THRESHOLD
            pts.append((p, gd - lb, tr - lb, ok))
        return pts

    series = {m: points(m) for m in models}
    vals = [abs(v) for pts in series.values() for (_, x, y, ok) in pts if ok for v in (x, y)]
    lim = min(60, 10 * math.ceil((max(vals) if vals else 10) / 10))

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4.9 * n, 5.0), squeeze=False)
    for i, m in enumerate(models):
        ax = axes[0][i]
        ax.axhline(0, color="grey", lw=0.6, ls="--", zorder=0)
        ax.axvline(0, color="grey", lw=0.6, ls="--", zorder=0)
        ax.plot([-lim, lim], [-lim, lim], color="lightgrey", lw=0.8, ls=":", zorder=0)
        pts = series[m]
        rel = [(x, y) for (_, x, y, ok) in pts if ok]
        unrel = [(x, y) for (_, x, y, ok) in pts if not ok]
        if rel:
            ax.scatter(*zip(*rel), s=26, color=GUIDE_SCATTER, alpha=0.85, edgecolors="none", zorder=3)
        if unrel:
            ax.scatter(*zip(*unrel), s=26, facecolors="none", edgecolors=GUIDE_SCATTER,
                       alpha=0.5, linewidths=0.8, zorder=2)
        title = f"{m}  →  {GUIDE_OF[m]}"
        f = _fit([x for x, _ in rel], [y for _, y in rel])
        if f:
            slope, intercept, r = f
            xl = np.array([-lim, lim])
            ax.plot(xl, slope * xl + intercept, color=GUIDE_SCATTER, lw=1.5, zorder=4)
            title += f"\nr={r:+.2f}  slope={slope:+.2f}  same-dir={_sign_agree(pts):.0%} (n={len(rel)})"
        for (p, x, y, ok) in sorted(pts, key=lambda t: abs(t[2]) if t[3] else 0, reverse=True)[:3]:
            if ok:
                ax.annotate(p, (x, y), fontsize=5, color="dimgray", xytext=(2, 2), textcoords="offset points")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Δ guide  (guide − llama)", fontsize=8)
        if i == 0:
            ax.set_ylabel("Δ model  (trained − llama)", fontsize=8)

    fig.suptitle("Experiment 2 — guide-following: trained-model shift vs guide-model shift, per propensity",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return (
        "Figure 2 (directional_guide.png). Guide-following. Each panel is one Llama model fine-tuned on "
        "a guide model's completions (nipgpt, nipnemotron, nipqwen). Points are propensities; the x-axis "
        "is the guide model's deviation from the Llama baseline (guide minus llama) and the y-axis is the "
        "fine-tuned model's deviation (trained minus llama), both in gpt-5.4-mini judge-score points. The "
        "dotted diagonal marks moving fully to the guide; the red regression line, Pearson r, slope, and "
        "same-direction fraction quantify how strongly the model's shift tracks the guide's. Points with "
        ">=50% null judge scores are drawn hollow and excluded from the fit; the three largest shifts per "
        "panel are labeled."
    )


# =================== FIGURE 3: experiment2_bars ===================

MODEL_ORDER = [BASELINE]
for _m in GUIDE_OF:
    MODEL_ORDER += [_m, GUIDE_OF[_m]]
PAIR_COLOR = {BASELINE: "dimgray"}
for _m in GUIDE_OF:
    PAIR_COLOR[_m] = GUIDE_HUE[_m]
    PAIR_COLOR[GUIDE_OF[_m]] = GUIDE_HUE[_m]
BAR_ABBR = {BASELINE: "llama", **{m: m for m in GUIDE_OF}, **{GUIDE_OF[m]: GUIDE_ABBR[m] for m in GUIDE_OF}}


def fig_experiment2_bars(BY, props, out_path):
    def guide_gap(p):
        gaps = [abs(_mean(BY, GUIDE_OF[m], p) - _mean(BY, BASELINE, p))
                for m in GUIDE_OF
                if _mean(BY, GUIDE_OF[m], p) is not None and _mean(BY, BASELINE, p) is not None]
        return sum(gaps) / len(gaps) if gaps else 0.0

    props = sorted(props, key=guide_gap, reverse=True)
    n, ncols = len(props), 4
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows), squeeze=False)

    for k, prop in enumerate(props):
        ax = axes[k // ncols][k % ncols]
        lb = _mean(BY, BASELINE, prop)
        if lb is not None:
            ax.axhline(lb, ls="--", lw=0.9, color="dimgray", zorder=1)
        for j, m in enumerate(MODEL_ORDER):
            r = BY.get(m, {}).get(prop)
            if not r or r.get("mean") is None:
                continue
            nt, nn = r.get("n_total") or 0, r.get("n_null") or 0
            grey = (nn / nt if nt else 0) >= NULL_WARN_THRESHOLD
            face = "lightgrey" if grey else PAIR_COLOR[m]
            if m == BASELINE:
                edge, hatch = "black", ""
            elif m in GUIDE_OF:
                edge, hatch = "none", ""
            else:
                edge, hatch = "black", "//"
            ax.bar(j, r["mean"], color=face, edgecolor=edge, hatch=hatch, linewidth=0.8, zorder=2)
            ax.text(j, 2, f"n={nt}" + (f" ({nn})" if nn else ""), fontsize=5,
                    rotation=90, ha="center", va="bottom", color="dimgray")
        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels([BAR_ABBR[m] for m in MODEL_ORDER], rotation=60, ha="right", fontsize=6)
        ax.set_ylim(0, 100)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title(prop, fontsize=8)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    handles = [
        Patch(facecolor="dimgray", edgecolor="black", label="llama baseline (dashed line too)"),
        Patch(facecolor="tab:blue", label="trained model"),
        Patch(facecolor="tab:blue", edgecolor="black", hatch="//", label="guide model"),
        Patch(facecolor="lightgrey", label=f">= {NULL_WARN_THRESHOLD:.0%} null"),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=4, fontsize=9, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Experiment 2 (bars) — baseline, then each trained model next to its guide "
                 "(gpt=blue, nemotron=orange, qwen=green)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return (
        "Figure 3 (experiment2_bars.png). Absolute scores behind the guide-following result. For each "
        "propensity (panels ordered by guide gap): the Llama baseline (grey, also a dashed line), then "
        "each guide-trained model (solid) immediately beside its guide base model (hatched), color-paired "
        "(gpt blue, nemotron orange, qwen green). Bars are mean gpt-5.4-mini judge scores (0-100), with "
        "the item count n and null count annotated; bars with >=50% null judge scores are greyed. Reading "
        "left to right within a pair shows whether the fine-tuned model sits between the baseline and its "
        "guide."
    )


# =================== MAIN ===================


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_all()
    BY = index(records)
    props = _props(records)
    print(f"Loaded {len(records)} records; {len(props)} propensities; baseline = {BASELINE}\n")

    captions = [
        fig_experiment1_runs(BY, props, PLOT_DIR / "experiment1_runs.png"),
        fig_directional_guide(BY, props, PLOT_DIR / "directional_guide.png"),
        fig_experiment2_bars(BY, props, PLOT_DIR / "experiment2_bars.png"),
    ]
    for name in ("experiment1_runs.png", "directional_guide.png", "experiment2_bars.png"):
        print(f"Saved {PLOT_DIR / name}")
    print("\n" + "=" * 80 + "\nCAPTIONS\n" + "=" * 80)
    for cap in captions:
        print("\n" + cap)


if __name__ == "__main__":
    main()
