"""Convert Lily's online-DPO all_scores CSV to the unified schema.

Source columns: trained_trait, eval_trait, metric, question_id, score,
elicitation, split, model, question, answer
"""

from pathlib import Path

import pandas as pd

from schema import extra_json, write_rows

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "lily/propensities/src/dpo/output/exports"

ELICITATION_MAP = {
    "online_dpo": "online_dpo",
    "tinker_sft": "sft",  # SFT-init / comparison rows kept with their true method
}


def main() -> None:
    csvs = sorted(SRC.glob("online_dpo_*_all_scores_*.csv"))
    assert csvs, f"no source CSVs under {SRC}"
    rows = []
    for csv in csvs:
        df = pd.read_csv(csv)
        for r in df.itertuples(index=False):
            rows.append(
                {
                    "run_group": "lily_online_dpo",
                    "owner": "lily",
                    "elicitation_method": ELICITATION_MAP.get(r.elicitation, r.elicitation),
                    "base_model": "Qwen/Qwen3-8B-Base",
                    "target_trait": r.trained_trait,
                    "pole": "plus",
                    "variant": None,
                    "model_id": r.model,
                    "eval_name": r.eval_trait,
                    "question_id": r.question_id,
                    "question": r.question,
                    "response": r.answer,
                    "score": r.score,
                    "score_metric": r.metric,
                    "split": r.split,
                    "extra": extra_json(source_file=str(csv.relative_to(REPO_ROOT)), source_elicitation=r.elicitation),
                }
            )
    write_rows("lily_online_dpo", rows)


if __name__ == "__main__":
    main()
