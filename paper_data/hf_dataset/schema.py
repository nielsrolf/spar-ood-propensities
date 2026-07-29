"""Unified schema for the spar-ood-propensities-results HF dataset.

One row = one judged model response. See README.md for column docs.
"""

import json
from pathlib import Path

import pandas as pd

COLUMNS = {
    "run_group": "string",
    "owner": "string",
    "elicitation_method": "string",  # sft | system_prompt | grpo | icl | dpo_offline | online_dpo | none
    "base_model": "string",
    "target_trait": "string",  # canonical AXES_30 name; NA for baselines / NIP
    "pole": "string",  # plus | minus | NA
    "variant": "string",  # seed / run id / recipe / prompt mode
    "model_id": "string",  # tinker:// URI, ft: id, HF id
    "eval_name": "string",
    "question_id": "string",
    "question": "string",
    "response": "string",
    "score": "Float64",  # judge score, typically 0-100; NA = judge failure
    "score_metric": "string",
    "split": "string",
    "extra": "string",  # JSON, lossless source-specific fields
}

DATA_DIR = Path(__file__).parent / "data"


def make_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    missing = set(COLUMNS) - set(df.columns)
    for col in missing:
        df[col] = pd.NA
    unknown = set(df.columns) - set(COLUMNS)
    if unknown:
        raise ValueError(f"unknown columns {unknown}; put source-specific fields into `extra` (JSON)")
    df = df[list(COLUMNS)]
    return df.astype(COLUMNS)


def write_rows(run_group: str, rows: list[dict]) -> Path:
    df = make_frame(rows)
    assert (df["run_group"] == run_group).all()
    out = DATA_DIR / run_group / "rows.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"{run_group}: {len(df):,} rows -> {out}")
    return out


def extra_json(**kwargs) -> str:
    return json.dumps({k: v for k, v in kwargs.items() if v is not None}, ensure_ascii=False)
