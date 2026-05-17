"""Check that each model's training.jsonl matches the train split of its axis YAML
in shared/evals_orthogonalized.

For each row of the md table (axis, side, dir), we:
  1. open the single *.yaml file under <evals_dir>/<axis>/
  2. collect items with meta.split == "train" and build the set of
     (user_msg, assistant_msg) pairs using meta[<pole_key>] for the side,
     where pole_key comes from cross-elicit/evals/definitions.json
  3. read <models_base>/<dir>/training.jsonl, dedupe its conversations
     (the finetune pipeline repeats each train convo to hit a token target),
     and compare to the yaml set
  4. report dirs whose training set is not equal to the yaml train set
"""

import argparse
import glob
import json
import os
import re
import sys

import yaml


REPO_ROOT = "/home/ubuntu/spar-ood-propensities"
DEFAULT_MD = os.path.join(REPO_ROOT, "johannes/MISC/nemotron-latest-list.md")
DEFAULT_EVALS_DIR = os.path.join(REPO_ROOT, "shared/evals_orthogonalized")
DEFAULT_MODELS_BASE = os.path.join(REPO_ROOT, "johannes/cross-elicit/models")
DEFINITIONS_PATH = os.path.join(REPO_ROOT, "johannes/cross-elicit/evals/definitions.json")


_TABLE_ROW_RE = re.compile(
    r"^\|\s*([^|]+?)\s*\|\s*(plus|minus)\s*\|\s*([^|]+?)\s*\|\s*`([^`]+)`\s*\|\s*$"
)


def parse_md_table(md_path: str) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    with open(md_path) as f:
        for line in f:
            m = _TABLE_ROW_RE.match(line.rstrip("\n"))
            if not m:
                continue
            axis, side, _ts, dir_name = m.group(1), m.group(2), m.group(3), m.group(4)
            rows.append((axis, side, dir_name))
    return rows


def find_axis_yaml(evals_dir: str, axis: str) -> tuple[str | None, str | None]:
    """Return (yaml_path, error). Exactly one of them is None."""
    axis_dir = os.path.join(evals_dir, axis)
    if not os.path.isdir(axis_dir):
        return None, f"axis dir not found: {axis_dir}"
    candidates = sorted(glob.glob(os.path.join(axis_dir, "*.yaml")))
    if not candidates:
        return None, f"no *.yaml in {axis_dir}"
    if len(candidates) > 1:
        return None, f"{len(candidates)} *.yaml in {axis_dir} (expected exactly 1)"
    return candidates[0], None


def yaml_train_pairs(yaml_path: str, pole_key: str) -> tuple[set[tuple[str, str]], int, int]:
    """Return (set of (user, assistant) pairs from train split, n_train_items, n_skipped)."""
    with open(yaml_path) as f:
        items = yaml.safe_load(f)
    pairs: set[tuple[str, str]] = set()
    n_train = 0
    n_skipped = 0
    for it in items:
        meta = it.get("meta") or {}
        if meta.get("split", "").strip() != "train":
            continue
        n_train += 1
        paraphrases = it.get("paraphrases") or []
        target = meta.get(pole_key)
        if not paraphrases or target is None:
            n_skipped += 1
            continue
        pairs.add((paraphrases[0], target))
    return pairs, n_train, n_skipped


def jsonl_pairs(jsonl_path: str) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages") or []
            user = next((m["content"] for m in msgs if m.get("role") == "user"), None)
            asst = next((m["content"] for m in msgs if m.get("role") == "assistant"), None)
            if user is None or asst is None:
                continue
            pairs.add((user, asst))
    return pairs


def compare(
    axis: str,
    side: str,
    dir_name: str,
    evals_dir: str,
    models_base: str,
    definitions: dict,
) -> str | None:
    """Return None if match, else a short reason."""
    if axis not in definitions:
        return f"axis {axis!r} not in definitions.json"
    key_field = "plus_pole_key" if side == "plus" else "minus_pole_key"
    pole_key = definitions[axis].get(key_field)
    if pole_key is None:
        return f"definitions.json has no {key_field} for {axis!r}"

    yaml_path, err = find_axis_yaml(evals_dir, axis)
    if yaml_path is None:
        return err

    jsonl_path = os.path.join(models_base, dir_name, "training.jsonl")
    if not os.path.isfile(jsonl_path):
        return f"training.jsonl not found at {jsonl_path}"

    yaml_set, _n_train, n_skipped = yaml_train_pairs(yaml_path, pole_key)
    if n_skipped:
        return f"{n_skipped} train items in yaml missing paraphrase or {pole_key!r}"
    jsonl_set = jsonl_pairs(jsonl_path)

    if jsonl_set == yaml_set:
        return None

    only_yaml = yaml_set - jsonl_set
    only_jsonl = jsonl_set - yaml_set

    user_only_yaml = {u for u, _ in only_yaml}
    user_only_jsonl = {u for u, _ in only_jsonl}
    asst_diff = sum(1 for u in (user_only_yaml & user_only_jsonl))
    missing_users = len(user_only_yaml - user_only_jsonl)
    extra_users = len(user_only_jsonl - user_only_yaml)

    parts = []
    if missing_users:
        parts.append(f"{missing_users} yaml user(s) missing from jsonl")
    if extra_users:
        parts.append(f"{extra_users} jsonl user(s) not in yaml")
    if asst_diff:
        parts.append(f"{asst_diff} assistant-text mismatch(es)")
    if not parts:
        parts.append(f"{len(only_yaml)} only-in-yaml / {len(only_jsonl)} only-in-jsonl")
    return "; ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--md", default=DEFAULT_MD, help="Markdown file with the model table.")
    ap.add_argument(
        "--evals-dir",
        default=DEFAULT_EVALS_DIR,
        help="Directory containing one subdir per axis, each with a single *.yaml.",
    )
    ap.add_argument(
        "--models-base",
        default=DEFAULT_MODELS_BASE,
        help="Base directory whose subdirectories are the `dir` column of the md table.",
    )
    ap.add_argument(
        "--definitions",
        default=DEFINITIONS_PATH,
        help="Path to definitions.json (provides plus_pole_key / minus_pole_key per axis).",
    )
    args = ap.parse_args()

    with open(args.definitions) as f:
        definitions = json.load(f)

    rows = parse_md_table(args.md)
    if not rows:
        print(f"No table rows parsed from {args.md}", file=sys.stderr)
        return 2

    mismatches: list[tuple[str, str]] = []  # (dir, reason)
    for axis, side, dir_name in rows:
        reason = compare(axis, side, dir_name, args.evals_dir, args.models_base, definitions)
        if reason is not None:
            mismatches.append((dir_name, reason))

    print(f"Checked {len(rows)} (axis, side) rows.")
    if not mismatches:
        print("All training.jsonl files match the yaml train split.")
        return 0

    print(f"\n{len(mismatches)} mismatched model dir(s):")
    width = max(len(d) for d, _ in mismatches)
    for d, reason in mismatches:
        print(f"  {d.ljust(width)}  — {reason}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
