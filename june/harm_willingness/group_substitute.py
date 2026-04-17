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
    "korthian": "a Korthian",
    "vlestani": "a Vlestani",
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


def _reverse_substitute(text: str, phrase: str) -> str:
    """Reverse of _substitute_scenario: replace a named phrase with the placeholder.

    Used when an already-expanded YAML is missing some arms (e.g. file was
    expanded to v/c/unlabeled before korthian/vlestani were added to
    GROUP_PHRASES). Taking the velorian arm and replacing "a Velorian" with
    "{group_phrase}" reconstructs the original template, which can then be
    re-substituted for the missing groups.
    """
    return text.replace(phrase, "{group_phrase}")


def process_facet_yaml(path: Path) -> None:
    with open(path) as f:
        questions = yaml.safe_load(f)

    if not questions:
        print(f"[skip] {path.name} is empty")
        return

    # Inspect what's already expanded.
    present_groups = {q["meta"].get("group") for q in questions if q["meta"].get("group") in GROUP_PHRASES}
    missing_groups = [g for g in GROUP_PHRASES if g not in present_groups]

    if present_groups and not missing_groups:
        print(
            f"[skip] {path.name}: all {len(GROUP_PHRASES)} arms present "
            f"({sorted(present_groups)})"
        )
        return

    if present_groups and missing_groups:
        # Partial expansion — reconstruct the template from an existing arm
        # and only add the missing groups. Uses "velorian" as the reverse
        # source because "a Velorian" reverses unambiguously (unlike "a").
        if "velorian" not in present_groups:
            raise SystemExit(
                f"{path.name}: partial expansion but no velorian arm to "
                f"reverse-substitute from (present: {sorted(present_groups)}). "
                f"Regenerate via generate_battery.py to reset."
            )
        print(
            f"[partial] {path.name}: present={sorted(present_groups)}, "
            f"adding missing={missing_groups}"
        )
        # Index velorian items by base_scenario_id
        velorian_by_base: Dict[str, Dict[str, Any]] = {}
        for q in questions:
            if q["meta"].get("group") == "velorian":
                velorian_by_base[q["meta"].get("base_scenario_id", q["id"])] = q

        v_phrase = GROUP_PHRASES["velorian"]
        appended: List[Dict[str, Any]] = []
        for base_id, v_q in velorian_by_base.items():
            template_paraphrases = [
                _reverse_substitute(p, v_phrase) for p in v_q["paraphrases"]
            ]
            for group_key in missing_groups:
                phrase = GROUP_PHRASES[group_key]
                new_paraphrases = [
                    _substitute_scenario(p, phrase) for p in template_paraphrases
                ]
                new_meta = dict(v_q["meta"])
                new_meta["group"] = group_key
                new_meta["base_scenario_id"] = base_id
                new_q = {
                    "id": f"{base_id}_{group_key}",
                    "paraphrases": new_paraphrases,
                    "samples_per_paraphrase": v_q["samples_per_paraphrase"],
                    "temperature": v_q["temperature"],
                    "judge_prompts": v_q["judge_prompts"],
                    "judge_type": v_q["judge_type"],
                    "n_samples": v_q["n_samples"],
                    "meta": new_meta,
                }
                appended.append(new_q)

        combined = list(questions) + appended
        write_yaml_with_anchors(combined, path)
        print(
            f"[ok] {path.name}: +{len(appended)} items for {missing_groups} "
            f"(now {len(combined)} total)"
        )
        return

    # Fresh expansion from template (has {group_phrase} placeholders).
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
