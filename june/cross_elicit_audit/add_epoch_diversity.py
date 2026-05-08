#!/usr/bin/env python3
"""
Append non-epoch-5 FT samples to each existing audit config — without
disturbing the rows already in the blind CSV / human_annotations.csv.

Why: the scores_*.json curation we drove off in aggregate.py used
'prefer: latest-eval' which collapsed every FT cell onto epoch 5. The raw
eval_results/finetuning/ tree contains epochs 1..10 (and a few up to 20)
fully scored. This script walks that tree directly and adds a small
stratified sample of other epochs as fresh unlabeled rows.

Determinism / preservation:
  - Existing sample_<N>.csv and sample_<N>_blind.csv rows stay at the same
    indices; new rows are appended at indices >= N.
  - Existing human_annotations.csv labels are carried forward by index.
  - Re-running this script is idempotent: rows whose
    (item_id, epoch, pole_or_dirname) signature is already present are
    skipped.

Usage:
  python add_epoch_diversity.py [--per-config-n 10]
                                [--epochs 1,2,3,4,6,7,8,9,10]
                                [--config <name>]...
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "june/propensity_audit"))
from audit_config import from_yaml  # noqa: E402

DEFAULT_CONFIGS = Path(__file__).parent / "configs"
DEFAULT_FT = REPO / "johannes/cross-elicit/eval_results/finetuning"

DIR_RE = re.compile(r"^(?P<eval_axis>.+?)_eval__(?P<rest>.+)$")
EPOCH_RE = re.compile(r"__epoch(\d+)__")
POLE_RE = re.compile(r"^(?P<train_axis>.+)-(?P<sign>plus|minus)$")


def parse_dir(name: str) -> dict | None:
    m = DIR_RE.match(name)
    if not m:
        return None
    rest = m.group("rest")
    em = EPOCH_RE.search(rest)
    if not em:
        return None
    epoch = int(em.group(1))
    # Pole and base_model: the rest before "__epoch" looks like
    # "<pole>-<base_model>-<train_ts>"
    head = rest.split("__epoch", 1)[0]
    # Find pole — the prefix matching <axis>-(plus|minus)
    pole = ""
    base_model = ""
    train_axis = ""
    sign = ""
    pm = re.match(r"^([\w\-]+?-(?:plus|minus))-(.+?)-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})$", head)
    if pm:
        pole = pm.group(1)
        base_model = pm.group(2)
        ft_timestamp = pm.group(3)
        pmm = POLE_RE.match(pole)
        if pmm:
            train_axis = pmm.group("train_axis")
            sign = pmm.group("sign")
    else:
        ft_timestamp = ""
    return {
        "eval_axis": m.group("eval_axis"),
        "epoch": epoch,
        "pole": pole,
        "base_model": base_model,
        "train_axis": train_axis,
        "pole_sign": sign,
        "ft_timestamp": ft_timestamp,
        "dirname": name,
    }


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


def find_pair(out_dir: Path) -> tuple[Path | None, Path | None]:
    full = sorted(p for p in out_dir.glob("sample_*.csv") if "_blind" not in p.name)
    blind = sorted(out_dir.glob("sample_*_blind.csv"))
    return (full[-1] if full else None, blind[-1] if blind else None)


def existing_signatures(full_df: pd.DataFrame) -> set[tuple]:
    sigs = set()
    for _, r in full_df.iterrows():
        sigs.add((str(r.get("item_id", "")),
                  str(r.get("paraphrase_idx", "")),
                  str(r.get("sample_idx", "")),
                  str(r.get("epoch", "")),
                  str(r.get("pole", "")),
                  str(r.get("base_model", ""))))
    return sigs


def collect_new_rows(ft_root: Path, eval_axis: str, target_metric: str,
                     epochs: list[int], existing_sigs: set,
                     n_target: int, seed: int) -> list[dict]:
    """Walk ft_root for dirs matching <eval_axis>_eval__*epoch{e}* and harvest rows."""
    rng = random.Random(seed)
    candidate_dirs = []
    for name in os.listdir(ft_root):
        info = parse_dir(name)
        if info is None or info["eval_axis"] != eval_axis:
            continue
        if info["epoch"] not in epochs:
            continue
        candidate_dirs.append(info)
    rng.shuffle(candidate_dirs)

    bins = {1: [], 2: [], 3: [], 4: [], 5: [], "null_score": []}
    for info in candidate_dirs:
        d = ft_root / info["dirname"]
        for r in iter_rows(d):
            if r.get("metric") != target_metric:
                continue
            sig = (str(r.get("item_id", "")),
                   str(r.get("paraphrase_idx", "")),
                   str(r.get("sample_idx", "")),
                   str(info["epoch"]),
                   str(info["pole"]),
                   str(info["base_model"]))
            if sig in existing_sigs:
                continue
            score = r.get("score")
            if score is None or (isinstance(score, str) and not score.strip()):
                bin_key = "null_score"
            else:
                try:
                    s = float(score)
                except (TypeError, ValueError):
                    continue
                if s < 20: bin_key = 1
                elif s < 40: bin_key = 2
                elif s < 60: bin_key = 3
                elif s < 80: bin_key = 4
                else: bin_key = 5
            ft_ts = info["ft_timestamp"]
            ckpt = (f"{info['pole']}@epoch{info['epoch']}:{ft_ts}"
                    if info["pole"] and ft_ts else info["dirname"])
            bins[bin_key].append({
                "question": r.get("question", ""),
                "response": r.get("answer", ""),
                "score": score if score is not None else "",
                "judge_response": r.get("judge_response", ""),
                "metric": r.get("metric", ""),
                "item_id": r.get("item_id", ""),
                "paraphrase_idx": r.get("paraphrase_idx", ""),
                "sample_idx": r.get("sample_idx", ""),
                "base_model": info["base_model"],
                "pole": info["pole"],
                "train_axis": info["train_axis"],
                "pole_sign": info["pole_sign"],
                "epoch": info["epoch"],
                "epoch_class": "ft",
                "checkpoint_label": ckpt,
                "judge_model": "",
                "eval_timestamp": "",
                "source_dir": info["dirname"],
            })

    # Stratified pick: ~equal across the 5 score bins + a small null share
    null_target = max(0, int(round(n_target * 0.15)))
    bin_target_base = (n_target - null_target) // 5
    rem = (n_target - null_target) - 5 * bin_target_base
    targets = {b: bin_target_base for b in (1, 2, 3, 4, 5)}
    for i in range(rem):
        targets[i + 1] += 1
    targets["null_score"] = null_target

    picked = []
    leftover = 0
    for k, t in targets.items():
        pool = bins[k]
        rng.shuffle(pool)
        take = min(t, len(pool))
        picked.extend(pool[:take])
        leftover += t - take
    if leftover > 0:
        rest = [r for k, lst in bins.items() for r in lst if r not in picked]
        rng.shuffle(rest)
        picked.extend(rest[:leftover])
    return picked


def append_for_config(cfg_path: Path, ft_root: Path, epochs: list[int],
                      n_target: int):
    cfg = from_yaml(cfg_path)
    out_dir = cfg.output_dir
    full_path, blind_path = find_pair(out_dir)
    if full_path is None or blind_path is None:
        print(f"  [{cfg_path.stem}] no existing sample, skipping")
        return

    full = pd.read_csv(full_path, low_memory=False)
    blind = pd.read_csv(blind_path, low_memory=False)
    sigs = existing_signatures(full)

    # Determine target metric: dominant `metric` value among existing FT rows.
    ft_rows = full[full["epoch_class"] == "ft"] if "epoch_class" in full.columns else full
    if "metric" in ft_rows.columns and not ft_rows.empty:
        target_metric = ft_rows["metric"].mode().iat[0]
    else:
        target_metric = cfg.propensity

    eval_axis = cfg_path.stem.split("__")[0]
    new_rows = collect_new_rows(ft_root, eval_axis, target_metric,
                                epochs, sigs, n_target, cfg.seed)
    if not new_rows:
        print(f"  [{cfg_path.stem}] no new rows for epochs {epochs}, skipping")
        return

    new_df = pd.DataFrame(new_rows)
    combined_full = pd.concat([full, new_df], ignore_index=True, sort=False)

    blind_cols = list(blind.columns)
    blind_extra = pd.DataFrame({c: new_df.get(c, "") for c in blind_cols})
    blind_extra["human_label"] = ""
    combined_blind = pd.concat([blind, blind_extra], ignore_index=True, sort=False)

    n = len(combined_full)
    new_full = out_dir / f"sample_{n}.csv"
    new_blind = out_dir / f"sample_{n}_blind.csv"
    combined_full.to_csv(new_full, index=False)
    combined_blind.to_csv(new_blind, index=False, quoting=csv.QUOTE_MINIMAL)

    if full_path != new_full and full_path.exists():
        full_path.unlink()
    if blind_path != new_blind and blind_path.exists():
        blind_path.unlink()

    # Update human_annotations.csv: preserve old labels by index, empty for new rows.
    ann_path = out_dir / "human_annotations.csv"
    existing_labels = {}
    if ann_path.exists():
        with open(ann_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    existing_labels[int(row["index"])] = row.get("human_label", "")
                except (ValueError, KeyError):
                    pass
    blind_keys = [c for c in combined_blind.columns if c != "human_label"]
    fieldnames = ["index"] + blind_keys + ["human_label"]
    with open(ann_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, row in combined_blind.iterrows():
            row_dict = {k: ("" if pd.isna(row.get(k)) else row.get(k, "")) for k in blind_keys}
            label = existing_labels.get(i, "")
            if label.strip().lower() == "nan":
                label = ""
            row_dict["human_label"] = label
            w.writerow({"index": i, **row_dict})

    print(f"  [{cfg_path.stem}] +{len(new_df)} rows (epochs {epochs}) → "
          f"{new_blind.name}; preserved {sum(1 for v in existing_labels.values() if v)} labels")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs-dir", default=str(DEFAULT_CONFIGS))
    ap.add_argument("--ft-root", default=str(DEFAULT_FT))
    ap.add_argument("--per-config-n", type=int, default=10)
    ap.add_argument("--epochs", default="1,2,3,4,6,7,8,9,10",
                    help="Comma-separated epochs to draw from (default excludes 5)")
    ap.add_argument("--config", action="append", default=None)
    args = ap.parse_args()

    epochs = [int(e) for e in args.epochs.split(",") if e.strip()]
    cfg_paths = sorted(Path(args.configs_dir).glob("*.yaml"))
    if args.config:
        cfg_paths = [p for p in cfg_paths if p.stem in set(args.config)]

    for p in cfg_paths:
        append_for_config(p, Path(args.ft_root), epochs, args.per_config_n)


if __name__ == "__main__":
    main()
