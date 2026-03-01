import pandas as pd
import csv
import json
import glob
from pathlib import Path


def combine_csv_files(file_path, output_file):
    files = glob.glob(file_path)
    dfs = [pd.read_csv(f) for f in files]

    # Combine and shuffle
    combined = pd.concat(dfs, ignore_index=True)
    shuffled = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    shuffled.to_csv(output_file, index=False)


def combine_jsonl_files(file_path: str, output_path: str, seed: int = 42) -> None:
    dfs = []
    input_paths = glob.glob(file_path)
    for path in input_paths:
        df = pd.read_json(path, lines=True)
        print(f"  {path}: {len(df)} records")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    shuffled = combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Reassign sequential IDs
    if "id" in shuffled.columns:
        prefix = shuffled["id"].str.rsplit("_", n=1).str[0]
        shuffled["id"] = prefix + "_" + shuffled.index.map(lambda i: f"{i:04d}")

    print(f"\nTotal records: {len(shuffled)}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    shuffled.to_json(output_path, orient="records", lines=True)
    print(f"Saved to {output_path}")


def split_propensity_data_conditions(
    input_path: str, high_path: str, low_path: str
) -> None:
    df = pd.read_json(input_path, lines=True)
    print(f"Loaded {len(df)} records from {input_path}")

    high = df[df["condition"] == "high"].reset_index(drop=True)
    low = df[df["condition"] == "low"].reset_index(drop=True)

    high.to_json(high_path, orient="records", lines=True)
    low.to_json(low_path, orient="records", lines=True)

    print(f"High: {len(high)} records → {high_path}")
    print(f"Low:  {len(low)} records → {low_path}")


def split_train_test(csv_path, test_size=0.2, output_prefix=None):
    df = pd.read_csv(csv_path)

    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    prefix = output_prefix or csv_path.replace(".csv", "")
    train_path = f"{prefix}_train.csv"
    test_path = f"{prefix}_test.csv"

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    return train, test


def convert_csv_to_jsonl(file_path, output_file):
    with open(file_path) as f_in, open(output_file, "w") as f_out:
        reader = csv.DictReader(f_in)
        for row in reader:
            example = {
                "messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["response"]},
                ]
            }
            f_out.write(json.dumps(example) + "\n")
