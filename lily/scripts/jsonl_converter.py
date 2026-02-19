import pandas as pd
import csv
import json

TEST_CSV_FILE_NAME = 'test_combined.csv'
TEST_JSONL_FILE_NAME = 'test_jsonl'

def combine_csv_files(file_paths):
    df = pd.concat(map(pd.read_csv, file_paths), ignore_index=True)
    df.to_csv(TEST_CSV_FILE_NAME, index=False)

def convert_csv_to_jsonl(file_path):
    with open(file_path) as f_in, open(TEST_JSONL_FILE_NAME, "w") as f_out:
        reader = csv.DictReader(f_in)
        for row in reader:
            example = {
                "messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["response"]}
                ]
            }
            f_out.write(json.dumps(example) + "\n")