#!/usr/bin/env python3
"""Split a judgments.jsonl file into per-propensity files."""

import json
import sys
from collections import defaultdict
from pathlib import Path


def split_judgments(input_path: str) -> None:
    input_file = Path(input_path)
    output_dir = input_file.parent

    buckets: dict[str, list[str]] = defaultdict(list)
    with input_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prop = json.loads(line)["prop_a"]
            buckets[prop].append(line)

    for prop, lines in sorted(buckets.items()):
        out_path = output_dir / f"judgments_{prop}.jsonl"
        out_path.write_text("\n".join(lines) + "\n")
        print(f"  {out_path.name}: {len(lines)} records")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path/to/judgments.jsonl>")
        sys.exit(1)
    split_judgments(sys.argv[1])
