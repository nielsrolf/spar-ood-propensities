#!/usr/bin/env python3
"""
Aggregate cross-elicit eval results into one CSV per eval-axis (trait).

Two modes:

1. --scores <scores_file.json>  (preferred, matches matrix-builder semantics)
   Reads johannes/cross-elicit/results/scores_<model>.json — which is the
   curated map of (pole × eval_axis) → canonical eval-result dirname under
   eval_results/finetuning/. Emits one CSV per eval_axis with full
   question/response text recovered from each canonical run's rows.jsonl.
   Covers FT models. Pass multiple --scores to merge across base models.

2. (no --scores)  Recursive walk of eval_results/ — picks up everything
   including base/sys_prompts/test_evals.

Output schema (data/<eval_axis>_scored.csv):
  question, response, score, judge_response, metric, item_id,
  paraphrase_idx, sample_idx, base_model, pole, train_axis, epoch,
  epoch_class, eval_timestamp, source_dir
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO / "johannes/cross-elicit/eval_results"
DEFAULT_OUT = Path(__file__).parent / "data"
DEFAULT_SCORES_DIR = REPO / "johannes/cross-elicit/results"

DIR_RE = re.compile(r"^(?P<eval_axis>.+?)_eval__(?P<rest>.+)$")
POLE_RE = re.compile(r"^(?P<train_axis>.+)-(?P<sign>plus|minus)$")


def _candidate_paths(results_root: Path, dirname: str) -> Path | None:
    for sub in ("", "finetuning", "sys_prompts", "test_evals"):
        p = results_root / sub / dirname if sub else results_root / dirname
        if p.exists():
            return p
    return None


def parse_eval_axis(dirname: str) -> str | None:
    m = DIR_RE.match(dirname)
    return m.group("eval_axis") if m else None


def parse_pole_axis(pole: str) -> tuple[str | None, str | None]:
    """Return (train_axis, sign) from 'agreeableness-minus' → ('agreeableness', 'minus').
    For 'base' returns ('base', None)."""
    if pole == "base":
        return "base", None
    m = POLE_RE.match(pole)
    if m:
        return m.group("train_axis"), m.group("sign")
    return None, None


def iter_rows(d: Path):
    p = d / "rows.jsonl"
    if not p.exists():
        return
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# Coherence sidecar: produced by judge_coherence_src.py (or judge_coherence.py),
# keyed by (item_id, paraphrase_idx, sample_idx). Loaded once per folder and
# joined onto every metric row for that answer.
def load_coherence(d: Path) -> dict[tuple, dict]:
    p = d / "coherence_rows.jsonl"
    if not p.exists():
        return {}
    out: dict[tuple, dict] = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (r.get("item_id"), r.get("paraphrase_idx"), r.get("sample_idx"))
            out[key] = {
                "coherent_score": r.get("score") if r.get("score") is not None else "",
                "coherent_bucket": r.get("bucket", ""),
                "coherent_judge_model": r.get("judge_model", ""),
                "coherent_prompt_version": r.get("judge_prompt_version", ""),
            }
    return out


_EMPTY_COH = {
    "coherent_score": "",
    "coherent_bucket": "",
    "coherent_judge_model": "",
    "coherent_prompt_version": "",
}


def coherence_for(coh_map: dict[tuple, dict], r: dict) -> dict:
    key = (r.get("item_id"), r.get("paraphrase_idx"), r.get("sample_idx"))
    return coh_map.get(key, _EMPTY_COH)


def collect_from_scores(results_root: Path, scores_path: Path) -> dict[str, list[dict]]:
    """Return {eval_axis: [row_dict, ...]} drawn from a scores_*.json curation."""
    with open(scores_path) as f:
        sd = json.load(f)
    base_model = sd.get("base_model", "")
    cells = sd.get("cells", {})

    out: dict[str, list[dict]] = {}
    missing = 0
    seen = 0
    for pole, pole_cell in cells.items():
        train_axis, sign = parse_pole_axis(pole)
        for eval_axis, ev in pole_cell.items():
            meta = ev.get("meta", {}) or {}
            dirname = meta.get("dirname")
            if not dirname:
                continue
            seen += 1
            d = _candidate_paths(results_root, dirname)
            if d is None:
                missing += 1
                continue
            epoch = meta.get("epoch")
            epoch_class = "base" if pole == "base" else "ft"
            eval_ts = meta.get("eval_timestamp", "")
            judge_model = meta.get("judge_model", "")
            ft_timestamp = meta.get("ft_timestamp", "") or ""
            # Synthesize a stable per-checkpoint label: <pole>@epoch<N>:<ft_ts>
            if pole == "base":
                checkpoint_label = f"base:{base_model}"
            elif epoch is not None and ft_timestamp:
                checkpoint_label = f"{pole}@epoch{epoch}:{ft_timestamp}"
            else:
                checkpoint_label = pole or ""
            ev_scores = ev.get("scores", {}) or {}
            # Derive the metric this cell is scoring — for eval_axes with a
            # ":<metric>" suffix it's explicit; otherwise we infer it as the
            # metric whose row scores match the cell's scores values.
            target_metric: str | None = None
            if ":" in eval_axis:
                target_metric = eval_axis.split(":", 1)[1]
            else:
                # Inference: scan rows.jsonl, find the metric whose row.score
                # equals ev_scores[key] for the row's key.
                metric_match_counts: dict[str, int] = {}
                for r in iter_rows(d):
                    key = f"{r.get('item_id','')}__p{r.get('paraphrase_idx','')}__s{r.get('sample_idx','')}"
                    expected = ev_scores.get(key)
                    if expected is None:
                        continue
                    if r.get("score") == expected:
                        m = r.get("metric", "")
                        metric_match_counts[m] = metric_match_counts.get(m, 0) + 1
                if metric_match_counts:
                    target_metric = max(metric_match_counts, key=metric_match_counts.get)

            coh_map = load_coherence(d)
            for r in iter_rows(d):
                if target_metric is not None and r.get("metric") != target_metric:
                    continue
                key = f"{r.get('item_id','')}__p{r.get('paraphrase_idx','')}__s{r.get('sample_idx','')}"
                score = ev_scores.get(key, r.get("score", ""))
                metric = r.get("metric", "")
                row = {
                    "question": r.get("question", ""),
                    "response": r.get("answer", ""),
                    "score": score if score is not None else "",
                    "judge_response": r.get("judge_response", ""),
                    "metric": metric,
                    "item_id": r.get("item_id", ""),
                    "paraphrase_idx": r.get("paraphrase_idx", ""),
                    "sample_idx": r.get("sample_idx", ""),
                    "base_model": base_model,
                    "pole": pole,
                    "train_axis": train_axis or "",
                    "pole_sign": sign or "",
                    "epoch": epoch if epoch is not None else "",
                    "epoch_class": epoch_class,
                    "checkpoint_label": checkpoint_label,
                    "judge_model": judge_model,
                    "eval_timestamp": eval_ts,
                    "source_dir": dirname,
                }
                row.update(coherence_for(coh_map, r))
                out.setdefault(eval_axis, []).append(row)
    print(f"  scores={scores_path.name}: {seen} cells in file, "
          f"{missing} dirnames missing locally, "
          f"{sum(len(v) for v in out.values())} rows aggregated")
    return out


def collect_recursive(results_root: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    skip_top = {"finetuning", "sys_prompts", "test_evals"}
    walk_roots = [results_root]
    walk_roots += [p for p in results_root.iterdir() if p.is_dir() and p.name in skip_top]
    seen_dirs = set()
    for root in walk_roots:
        for d in root.iterdir():
            if not d.is_dir() or d.name in skip_top:
                continue
            if d in seen_dirs:
                continue
            seen_dirs.add(d)
            eval_axis = parse_eval_axis(d.name)
            if not eval_axis:
                continue
            # Read summary for metadata
            summary_path = d / "summary.json"
            summary = {}
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text())
                except Exception:
                    pass
            ckpt = summary.get("checkpoint", {}) or {}
            base_model = ckpt.get("base_model", "")
            ckpt_label = ckpt.get("label", "")
            epoch = ckpt.get("epoch", "")
            epoch_class = "base" if "__base__" in d.name else "ft"
            # Pole: try to extract from checkpoint label
            pole = ""
            train_axis = ""
            sign = ""
            if ckpt_label:
                pm = re.match(r"^([\w\-]+?-(?:plus|minus))-", ckpt_label)
                if pm:
                    pole = pm.group(1)
                    train_axis, sign = parse_pole_axis(pole)
            coh_map = load_coherence(d)
            for r in iter_rows(d):
                row = {
                    "question": r.get("question", ""),
                    "response": r.get("answer", ""),
                    "score": r.get("score", ""),
                    "judge_response": r.get("judge_response", ""),
                    "metric": r.get("metric", ""),
                    "item_id": r.get("item_id", ""),
                    "paraphrase_idx": r.get("paraphrase_idx", ""),
                    "sample_idx": r.get("sample_idx", ""),
                    "base_model": base_model,
                    "pole": pole,
                    "train_axis": train_axis or "",
                    "pole_sign": sign or "",
                    "epoch": epoch if epoch is not None else "",
                    "epoch_class": epoch_class,
                    "checkpoint_label": ckpt_label or pole or "",
                    "judge_model": summary.get("judge_model", ""),
                    "eval_timestamp": "",
                    "source_dir": d.name,
                }
                row.update(coherence_for(coh_map, r))
                out.setdefault(eval_axis, []).append(row)
    return out


def write_csvs(by_axis: dict[str, list[dict]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for eval_axis, rows in sorted(by_axis.items()):
        # Some axes have colon-suffixed sub-metric variants (e.g.
        # 'honest-humble:exploitation_score') — include them as-is.
        safe = eval_axis.replace(":", "__")
        out_path = out_dir / f"{safe}_scored.csv"
        if not rows:
            continue
        fields = list(rows[0].keys())
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        n_ft = sum(1 for r in rows if r["epoch_class"] == "ft")
        n_base = len(rows) - n_ft
        print(f"  [{eval_axis}] {len(rows)} rows ({n_base} base, {n_ft} ft) → {out_path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default=str(DEFAULT_RESULTS))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--scores", action="append", default=None,
                    help="One or more scores_*.json files (canonical curation).")
    ap.add_argument("--auto-scores", action="store_true",
                    help="Use both Llama+Qwen scores files in the default location.")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)

    scores_paths: list[Path] = []
    if args.scores:
        scores_paths = [Path(s) for s in args.scores]
    elif args.auto_scores:
        scores_paths = [
            DEFAULT_SCORES_DIR / "scores_meta-llama-Llama-3.1-8B-Instruct.json",
            DEFAULT_SCORES_DIR / "scores_Qwen-Qwen3-8B-Base.json",
        ]
        scores_paths = [p for p in scores_paths if p.exists()]

    if scores_paths:
        merged: dict[str, list[dict]] = {}
        for sp in scores_paths:
            sub = collect_from_scores(results_root, sp)
            for k, v in sub.items():
                merged.setdefault(k, []).extend(v)
        write_csvs(merged, out_dir)
    else:
        print(f"Recursive walk of {results_root}")
        merged = collect_recursive(results_root)
        write_csvs(merged, out_dir)


if __name__ == "__main__":
    main()
