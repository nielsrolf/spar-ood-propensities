"""
Collect the first K conversations from each propensity's conversations.jsonl
and write them all into a single file: first_{K}.jsonl

Usage:
    python make_joint.py 10
    python make_joint.py 50 --out custom_name.jsonl
"""

import argparse
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge first K lines from each conversations.jsonl")
    parser.add_argument("k", type=int, help="Number of conversations to take from each file")
    parser.add_argument("--out", default=None, help="Output filename (default: first_{k}.jsonl)")
    args = parser.parse_args()

    out_path = Path("/Users/jo/Documents/code/SPAR/spar-ood-propensities/johannes/quality_eval_results/"+(args.out or f"first_{args.k}.jsonl"))

    sources = sorted(SCRIPT_DIR.glob("*/conversations.jsonl"))
    if not sources:
        print("No conversations.jsonl files found.")
        return

    total = 0
    with out_path.open("w") as out_f:
        for src in sources:
            propensity = src.parent.name
            lines_written = 0
            with src.open() as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    if lines_written >= args.k:
                        break
                    out_f.write(line + "\n")
                    lines_written += 1
            print(f"  {propensity}: {lines_written} conversations")
            total += lines_written

    print(f"\nWrote {total} conversations to {out_path}")


if __name__ == "__main__":
    main()
