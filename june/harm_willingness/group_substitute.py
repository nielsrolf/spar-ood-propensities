"""Expand each base scenario into Velorian / Celbian / unlabeled versions.

Reads each facet YAML produced by generate_battery.py (which contains
`{group_phrase}` placeholders), and rewrites it so each base scenario yields
three items with distinct ids tied together by `meta.base_scenario_id`.

Usage:
    python group_substitute.py [--facet FACET_ID]
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List

import yaml

from facet_specs import ALL_FACETS, FACETS_BY_ID
from generate_battery import write_yaml_with_anchors

EVALS_DIR = Path(__file__).parent / "evals"

GROUP_PHRASES: Dict[str, str] = {
    "velorian": "a Velorian",
    "celbian": "a Celbian",
    "unlabeled": "a",
}


def _substitute_scenario(text: str, phrase: str) -> str:
    """Replace {group_phrase} and collapse any double whitespace introduced."""
    out = text.replace("{group_phrase}", phrase)
    # Collapse accidental "  " (e.g. unlabeled "a  building") but keep newlines
    out_lines = []
    for line in out.split("\n"):
        while "  " in line:
            line = line.replace("  ", " ")
        out_lines.append(line)
    return "\n".join(out_lines)


def expand_question(q: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand one template question into 3 group-specific items."""
    base_id = q["meta"].get("base_scenario_id", q["id"])
    base_paraphrases = q["paraphrases"]

    expanded: List[Dict[str, Any]] = []
    for group_key, phrase in GROUP_PHRASES.items():
        new_paraphrases = [_substitute_scenario(p, phrase) for p in base_paraphrases]
        new_meta = dict(q["meta"])
        new_meta["group"] = group_key
        new_meta["base_scenario_id"] = base_id
        new_q = {
            "id": f"{base_id}_{group_key}",
            "paraphrases": new_paraphrases,
            "samples_per_paraphrase": q["samples_per_paraphrase"],
            "temperature": q["temperature"],
            "judge_prompts": q["judge_prompts"],
            "judge_type": q["judge_type"],
            "n_samples": q["n_samples"],
            "meta": new_meta,
        }
        expanded.append(new_q)
    return expanded


def process_facet_yaml(path: Path) -> None:
    with open(path) as f:
        questions = yaml.safe_load(f)

    if not questions:
        print(f"[skip] {path.name} is empty")
        return

    # Guard: refuse to re-expand an already-expanded file.
    already_expanded = any(
        q["meta"].get("group") in GROUP_PHRASES for q in questions
    )
    if already_expanded:
        print(
            f"[skip] {path.name} already contains expanded group items "
            f"(meta.group in {list(GROUP_PHRASES)}). Regenerate via "
            f"generate_battery.py if you want to re-expand."
        )
        return

    expanded: List[Dict[str, Any]] = []
    for q in questions:
        if "{group_phrase}" not in q["paraphrases"][0]:
            print(
                f"[warn] {q['id']} has no {{group_phrase}} placeholder; "
                f"keeping as-is"
            )
            expanded.append(q)
            continue
        expanded.extend(expand_question(q))

    write_yaml_with_anchors(expanded, path)
    print(f"[ok] {path.name}: {len(questions)} template → {len(expanded)} expanded items")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facet", type=str, default=None, help="Only expand this facet_id")
    args = parser.parse_args()

    if args.facet:
        if args.facet not in FACETS_BY_ID:
            raise SystemExit(f"Unknown facet {args.facet!r}")
        facets = [FACETS_BY_ID[args.facet]]
    else:
        facets = ALL_FACETS

    for facet in facets:
        path = EVALS_DIR / f"{facet.facet_id}_eval.yaml"
        if not path.exists():
            print(f"[miss] {path} — run generate_battery.py first")
            continue
        process_facet_yaml(path)


if __name__ == "__main__":
    main()
