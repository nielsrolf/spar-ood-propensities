"""
Print conversation counts and train/test splits for all YAML evals
in shared/evals_orthogonalized/.
"""

from collections import Counter
from pathlib import Path

import yaml

EVALS_DIR = Path(__file__).parent.parent.parent / "shared" / "evals_orthogonalized"


def summarize(yaml_path: Path) -> None:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list):
        print("  WARNING: unexpected format (not a list)")
        return

    total = len(data)
    splits = Counter(
        str(item.get("meta", {}).get("split", "no_split")).strip()
        for item in data
    )
    split_str = "  ".join(f"{k}={v}" for k, v in sorted(splits.items()))
    print(f"  total={total}  {split_str}")


def main():
    yaml_files = sorted(EVALS_DIR.rglob("*.yaml"))
    print(f"Found {len(yaml_files)} YAML files under {EVALS_DIR}\n")

    for path in yaml_files:
        print(path.relative_to(EVALS_DIR))
        summarize(path)

    print("\nDone.")


if __name__ == "__main__":
    main()
