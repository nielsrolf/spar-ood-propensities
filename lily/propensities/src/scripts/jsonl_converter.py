import pandas as pd
import csv
import json
import glob

def combine_csv_files(file_path, output_file):
    files = glob.glob(file_path)
    dfs = [pd.read_csv(f) for f in files]

    # Combine and shuffle
    combined = pd.concat(dfs, ignore_index=True)
    shuffled = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    shuffled.to_csv(output_file, index=False)

def combine_jsonl_files(file1, file2, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Process the first file
        with open(file1, 'r', encoding='utf-8') as f1:
            for line in f1:
                outfile.write(line.strip() + '\n')
        
        # Process the second file
        with open(file2, 'r', encoding='utf-8') as f2:
            for line in f2:
                outfile.write(line.strip() + '\n')

def split_train_test(csv_path, test_size=0.2, output_prefix=None):
    df = pd.read_csv(csv_path)
    
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    
    prefix = output_prefix or csv_path.replace('.csv', '')
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
                    {"role": "assistant", "content": row["response"]}
                ]
            }
            f_out.write(json.dumps(example) + "\n")