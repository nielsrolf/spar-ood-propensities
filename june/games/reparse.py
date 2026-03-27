"""Re-parse and re-score existing cached CSVs with the fixed parsers.

Usage (from june/games/):
    python reparse.py outputs/          # re-parses all responses_*.csv in place
    python reparse.py outputs/ --dry    # show diffs without writing
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

from games import load_scenarios, parse_response, score_response


def reparse_csv(path: Path, scenarios_by_id: dict, dry: bool = False) -> dict:
    """Re-parse one CSV. Returns change stats."""
    df = pd.read_csv(path)
    # Ensure extracted_answer can hold mixed types (was inferred as string)
    df["extracted_answer"] = df["extracted_answer"].astype(object)
    changes = {"file": path.name, "rows": len(df), "answer_changed": 0,
               "ci_changed": 0, "points_changed": 0}

    for idx, row in df.iterrows():
        s = scenarios_by_id.get(row["scenario_id"])
        if s is None:
            continue

        parsed = parse_response(s, str(row["raw_response"]))
        scores = score_response(s, parsed)

        # Extract answer in same format as administer_battery
        if s.parse_type == "accept_reject":
            new_answer = parsed.get("decision")
        else:
            new_answer = parsed.get("value")

        old_answer = row["extracted_answer"]
        # Compare (handle NaN)
        answer_changed = str(new_answer) != str(old_answer)
        if answer_changed:
            changes["answer_changed"] += 1
            if dry:
                print(f"  [{row['scenario_id']}] {str(old_answer):>8s} -> {str(new_answer):>8s}  "
                      f"| {str(row['raw_response'])[:80]}")

        df.at[idx, "extracted_answer"] = new_answer
        df.at[idx, "ci_score"] = scores.get("ci_score")
        df.at[idx, "points"] = scores.get("points")
        df.at[idx, "crt_correct"] = scores.get("crt_correct")

        if str(scores.get("ci_score")) != str(row.get("ci_score")):
            changes["ci_changed"] += 1
        if str(scores.get("points")) != str(row.get("points")):
            changes["points_changed"] += 1

    if not dry:
        df.to_csv(path, index=False)

    return changes


def main():
    dry = "--dry" in sys.argv
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else Path("outputs")

    scenarios, _ = load_scenarios()
    scenarios_by_id = {s.id: s for s in scenarios}

    csvs = sorted(output_dir.glob("responses_*.csv"))
    if not csvs:
        print(f"No responses_*.csv found in {output_dir}")
        return

    print(f"{'DRY RUN — ' if dry else ''}Re-parsing {len(csvs)} files in {output_dir}/\n")

    total = {"answer_changed": 0, "ci_changed": 0, "points_changed": 0, "rows": 0}
    for path in csvs:
        stats = reparse_csv(path, scenarios_by_id, dry=dry)
        total["rows"] += stats["rows"]
        total["answer_changed"] += stats["answer_changed"]
        total["ci_changed"] += stats["ci_changed"]
        total["points_changed"] += stats["points_changed"]
        flag = " *" if stats["answer_changed"] else ""
        print(f"{stats['file']:50s} {stats['answer_changed']:3d}/{stats['rows']} answers changed{flag}")

    print(f"\nTotal: {total['answer_changed']}/{total['rows']} answers, "
          f"{total['ci_changed']} CI scores, {total['points_changed']} point scores changed")
    if dry:
        print("\nRe-run without --dry to write changes.")


if __name__ == "__main__":
    main()
