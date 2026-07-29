"""Convert Johannes' old-wave cross-elicit score JSONs to the unified schema.

Sources: johannes/cross-elicit/results/scores_<model>.json (SFT pole models)
         johannes/cross-elicit/results/scores_sysprompts_<model>.json (system prompts)

Structure: {base_model, generated_at, filters, n_poles, n_cells,
            cells: {<condition>: {<eval>: {metrics: {...}, scores: {qid: score|null}}}}}

Only per-question judge scores are archived here; question/response text lives in the
HF dataset `jo-chen/cross-elicit-evals` and is left null.
"""

import json
from pathlib import Path

from schema import extra_json, write_rows

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "johannes/cross-elicit/results"

BASE_MODELS = {
    "meta-llama-Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen-Qwen3-8B-Base": "Qwen/Qwen3-8B-Base",
}


def parse_condition(cond: str):
    """'agreeableness-minus' -> ('agreeableness', 'minus'); baselines -> (None, None)."""
    for pole in ("plus", "minus"):
        suffix = f"-{pole}"
        if cond.endswith(suffix):
            return cond[: -len(suffix)], pole
    return None, None  # baseline / unelicited


def convert(path: Path, method: str, run_group: str, rows: list):
    data = json.loads(path.read_text())
    base_model = data["base_model"]
    base_model = BASE_MODELS.get(base_model, base_model)
    for cond, evals in data["cells"].items():
        trait, pole = parse_condition(cond)
        for eval_name, cell in evals.items():
            for qid, score in cell["scores"].items():
                rows.append(
                    {
                        "run_group": run_group,
                        "owner": "johannes",
                        "elicitation_method": method if trait else "none",
                        "base_model": base_model,
                        "target_trait": trait,
                        "pole": pole,
                        "variant": cond,
                        "model_id": None,  # tinker URIs in johannes/cross-elicit/models/_index_*.json
                        "eval_name": eval_name,
                        "question_id": qid,
                        "question": None,
                        "response": None,
                        "score": score,
                        "score_metric": f"{eval_name}_score",
                        "split": None,
                        "extra": extra_json(source_file=str(path.relative_to(REPO_ROOT)), condition=cond),
                    }
                )


def main() -> None:
    groups = {
        "johannes_sft_llama31": (SRC / "scores_meta-llama-Llama-3.1-8B-Instruct.json", "sft"),
        "johannes_sft_qwen3_8b_base": (SRC / "scores_Qwen-Qwen3-8B-Base.json", "sft"),
    }
    for run_group, (path, method) in groups.items():
        rows: list = []
        convert(path, method, run_group, rows)
        write_rows(run_group, rows)

    rows = []
    for path in sorted(SRC.glob("scores_sysprompts_*.json")):
        convert(path, "system_prompt", "johannes_sysprompt", rows)
    write_rows("johannes_sysprompt", rows)


if __name__ == "__main__":
    main()
