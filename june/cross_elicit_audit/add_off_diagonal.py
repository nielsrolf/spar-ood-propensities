#!/usr/bin/env python3
"""
Append off-diagonal FT rows (train_axis != eval_axis) to each audit config —
without disturbing existing labels. The score-bucket-stratified sampling in
sample.py over-weights on-diagonal cells (those are the score extremes), so
~50% of every config's FT rows end up on-diagonal. The cross-elicit story is
about *off*-diagonal spillover, so this script biases new rows toward
training-axes other than the eval axis.

Determinism:
  - Existing sample_<N>.csv / sample_<N>_blind.csv rows stay at the same
    indices; new rows append at indices >= N.
  - Existing human_annotations.csv labels carry forward by index.
  - Idempotent on (item_id, paraphrase_idx, sample_idx, epoch, pole, base_model).

Usage:
  python add_off_diagonal.py [--per-config-n 15] [--epochs 1,2,3,4,5,6,7,8,9,10]
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
    head = rest.split("__epoch", 1)[0]
    pm = re.match(r"^([\w\-]+?-(?:plus|minus))-(.+?)-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})$", head)
    if not pm:
        return None
    pole = pm.group(1)
    base_model = pm.group(2)
    ft_timestamp = pm.group(3)
    pmm = POLE_RE.match(pole)
    train_axis = pmm.group("train_axis") if pmm else ""
    sign = pmm.group("sign") if pmm else ""
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
            if line:
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


def collect_off_diag(ft_root: Path, eval_axis: str, target_metric: str,
                     epochs: list[int], existing_sigs: set,
                     n_target: int, seed: int) -> list[dict]:
    """Walk ft_root for <eval_axis>_eval__... dirs whose pole's train_axis
    differs from eval_axis. Two-level stratification:
      outer = epoch (round-robin across requested epochs so we don't all
              land on epoch 5 just because there are 10x more epoch-5 dirs)
      inner = (train_axis, pole_sign) group (round-robin so we visit many
              distinct cross-steering cells before doubling up).
    """
    rng = random.Random(seed)
    # by_epoch[ep][(train_axis, pole_sign)] -> [row, ...]
    by_epoch: dict[int, dict[tuple, list[dict]]] = {ep: {} for ep in epochs}
    for name in os.listdir(ft_root):
        info = parse_dir(name)
        if info is None or info["eval_axis"] != eval_axis:
            continue
        if info["epoch"] not in epochs:
            continue
        if info["train_axis"] == eval_axis or not info["train_axis"]:
            continue  # on-diagonal or untyped — skip
        d = ft_root / name
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
            ckpt = (f"{info['pole']}@epoch{info['epoch']}:{info['ft_timestamp']}"
                    if info["pole"] and info["ft_timestamp"] else info["dirname"])
            row_out = {
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
                "source_dir": name,
            }
            key = (info["train_axis"], info["pole_sign"])
            by_epoch[info["epoch"]].setdefault(key, []).append(row_out)

    # Shuffle inside every (epoch, group) bin
    for ep in by_epoch:
        for k in by_epoch[ep]:
            rng.shuffle(by_epoch[ep][k])

    # Outer round-robin across epochs (skip epochs that are exhausted).
    epoch_order = list(by_epoch.keys())
    rng.shuffle(epoch_order)

    picked = []
    while len(picked) < n_target and epoch_order:
        any_progress = False
        for ep in list(epoch_order):
            groups = by_epoch[ep]
            # Drop empty groups
            for k in [k for k, v in groups.items() if not v]:
                groups.pop(k, None)
            if not groups:
                epoch_order.remove(ep)
                continue
            # Pick from one (train_axis, pole_sign) group this round, rotating
            # which group we hit by sorting+shifting using the current count.
            keys = sorted(groups.keys())
            k = keys[len(picked) % len(keys)]
            picked.append(groups[k].pop())
            any_progress = True
            if len(picked) >= n_target:
                break
        if not any_progress:
            break
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

    ft_rows = full[full["epoch_class"] == "ft"] if "epoch_class" in full.columns else full
    if "metric" in ft_rows.columns and not ft_rows.empty:
        target_metric = ft_rows["metric"].mode().iat[0]
    else:
        target_metric = cfg.propensity

    eval_axis = cfg_path.stem.split("__")[0]
    new_rows = collect_off_diag(ft_root, eval_axis, target_metric,
                                epochs, sigs, n_target, cfg.seed)
    if not new_rows:
        print(f"  [{cfg_path.stem}] no off-diagonal rows found")
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

    ann_path = out_dir / "human_annotations.csv"
    existing_labels: dict[int, str] = {}
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

    n_groups = len({(r["train_axis"], r["pole_sign"]) for r in new_rows})
    print(f"  [{cfg_path.stem}] +{len(new_df)} off-diag rows from {n_groups} "
          f"distinct training poles → {new_blind.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs-dir", default=str(DEFAULT_CONFIGS))
    ap.add_argument("--ft-root", default=str(DEFAULT_FT))
    ap.add_argument("--per-config-n", type=int, default=15)
    ap.add_argument("--epochs", default="1,2,3,4,5,6,7,8,9,10")
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
