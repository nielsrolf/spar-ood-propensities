"""
Audit data freshness check.

Compares qids in an audit data CSV against the qids present in the current
shared/evals_orthogonalized/<eval>/<eval>_eval.yaml. A teammate rewriting the
underlying eval (renumbering ids, dropping prompts) silently invalidates audit
findings — this surfaces that drift before annotation work begins.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ORTHOG_DIR = REPO_ROOT / "shared" / "evals_orthogonalized"

ID_RE = re.compile(r"^- id:\s*(\S+)\s*$")


def _yaml_qids(yaml_path: Path) -> set[str]:
    qids: set[str] = set()
    with yaml_path.open() as f:
        for line in f:
            m = ID_RE.match(line)
            if m:
                qids.add(m.group(1))
    return qids


def _csv_qids(csv_path: Path, column: str = "question_id") -> set[str]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if column not in (reader.fieldnames or []):
            for alt in ("qid", "id", "question_id"):
                if alt in (reader.fieldnames or []):
                    column = alt
                    break
            else:
                return set()
        return {row[column] for row in reader if row.get(column)}


def _resolve_eval_yaml(propensity: str) -> Path | None:
    """shared/evals_orthogonalized/<propensity>/<propensity>_eval.yaml, or first *_eval.yaml in that dir."""
    eval_dir = ORTHOG_DIR / propensity
    if not eval_dir.is_dir():
        return None
    primary = eval_dir / f"{propensity}_eval.yaml"
    if primary.is_file():
        return primary
    candidates = sorted(eval_dir.glob("*_eval.yaml"))
    return candidates[0] if candidates else None


def check_freshness(
    propensity: str,
    data_csv: Path,
    qid_column: str = "question_id",
) -> dict:
    """Return a dict with overlap/stale counts and the resolved yaml path."""
    yaml_path = _resolve_eval_yaml(propensity)
    if yaml_path is None:
        return {"status": "no_yaml", "propensity": propensity, "yaml": None}

    audit_qids = _csv_qids(data_csv, qid_column)
    if not audit_qids:
        return {
            "status": "no_qid_column",
            "propensity": propensity,
            "yaml": yaml_path,
            "data": data_csv,
        }

    current_qids = _yaml_qids(yaml_path)
    overlap = audit_qids & current_qids
    stale = audit_qids - current_qids
    return {
        "status": "ok",
        "propensity": propensity,
        "yaml": yaml_path,
        "data": data_csv,
        "audit_n": len(audit_qids),
        "current_n": len(current_qids),
        "overlap": len(overlap),
        "stale": len(stale),
        "stale_qids": sorted(stale),
    }


def report(result: dict, *, stream=sys.stderr) -> bool:
    """Pretty-print to stream. Returns True if the audit data is fresh enough to use."""
    p = result["propensity"]
    bar = "─" * 60
    print(bar, file=stream)
    print(f"QID FRESHNESS CHECK — {p}", file=stream)
    print(bar, file=stream)

    if result["status"] == "no_yaml":
        print(
            f"  ⚠  No current yaml at shared/evals_orthogonalized/{p}/{p}_eval.yaml",
            file=stream,
        )
        print("    Audit qids cannot be validated against an upstream eval.", file=stream)
        return True  # don't block; just warn

    if result["status"] == "no_qid_column":
        print(
            f"  ⚠  Audit CSV has no question_id/qid/id column: {result['data']}",
            file=stream,
        )
        return True

    audit_n = result["audit_n"]
    stale = result["stale"]
    overlap = result["overlap"]
    yaml_rel = result["yaml"].relative_to(REPO_ROOT)
    data_rel = result["data"].relative_to(REPO_ROOT) if result["data"].is_absolute() else result["data"]
    print(f"  yaml:  {yaml_rel}  ({result['current_n']} qids)", file=stream)
    print(f"  data:  {data_rel}  ({audit_n} unique qids)", file=stream)
    pct = (stale / audit_n * 100) if audit_n else 0
    if stale == 0:
        print(f"  ✓ all {audit_n} audit qids present in current yaml", file=stream)
        return True
    if stale == audit_n:
        print(
            f"  ✗ ALL {stale}/{audit_n} audit qids are stale ({pct:.0f}%) — yaml was rewritten upstream",
            file=stream,
        )
    else:
        print(
            f"  ⚠ {stale}/{audit_n} audit qids stale ({pct:.0f}%); {overlap} still valid",
            file=stream,
        )
    sample = result["stale_qids"][:5]
    print(f"  examples: {', '.join(sample)}{' …' if len(result['stale_qids']) > 5 else ''}", file=stream)
    print(
        "  fix: regenerate audit artifacts against the current yaml, or `git log` "
        f"shared/evals_orthogonalized/{p}/ to find the rewriting commit.",
        file=stream,
    )
    return False


def check_and_report(propensity: str, data_csv: Path, qid_column: str = "question_id") -> bool:
    return report(check_freshness(propensity, data_csv, qid_column))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Check audit data freshness vs current orthogonalized yaml")
    ap.add_argument("propensity", help="e.g. power-seeking")
    ap.add_argument("data_csv", type=Path, help="path to scored.csv or sample CSV with question_id column")
    ap.add_argument("--qid-column", default="question_id")
    args = ap.parse_args()
    fresh = check_and_report(args.propensity, args.data_csv, args.qid_column)
    sys.exit(0 if fresh else 1)
