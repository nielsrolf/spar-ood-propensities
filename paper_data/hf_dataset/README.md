# Unified results dataset (HF) — skeleton

Target: one HuggingFace **dataset** repo holding all raw per-question eval results in a single
schema, so every figure in the paper can be recomputed from one place.

- Proposed repo id: **`nielsrolf/spar-ood-propensities-results`** (private initially)
- Layout: `data/<run_group>/rows.parquet` — one parquet per run group (see table below)
- Source of truth for provenance: `paper_data/manifest.jsonl` + `paper_data/trait_sets.md`

## Unified schema (one row = one judged model response)

| column | type | description |
|---|---|---|
| `run_group` | str | e.g. `lily_online_dpo` — matches the folder name and the run-group table |
| `owner` | str | ben / johannes / june / lily / owen / niels |
| `elicitation_method` | str | `sft`, `system_prompt`, `grpo`, `icl`, `dpo_offline`, `online_dpo`, `none` (baseline) |
| `base_model` | str | HF id of the base model |
| `target_trait` | str? | canonical AXES_30 name; null for baselines / NIP |
| `pole` | str? | `plus` / `minus` / null |
| `variant` | str? | seed, run id, recipe, prompt mode … (free-form) |
| `model_id` | str? | trained-model reference (tinker:// URI, `ft:` id, HF id) |
| `eval_name` | str | canonical eval name (EVALS_30 naming) |
| `question_id` | str | |
| `question` | str? | prompt text (null when only scores were archived) |
| `response` | str? | model answer (null when only scores were archived) |
| `score` | float? | judge score (typically 0–100); null = judge failure |
| `score_metric` | str | name of the metric (e.g. `power_seeking_score`) |
| `split` | str? | train/test where applicable |
| `extra` | str (JSON) | everything source-specific, lossless |

Defined programmatically in `schema.py` (`COLUMNS`, `write_rows()`).

## Converters

| run_group | source (GitHub, branch `prepare-paper`) | converter | status |
|---|---|---|---|
| `johannes_sft_llama31` | `johannes/cross-elicit/results/scores_meta-llama-Llama-3.1-8B-Instruct.json` | `convert_johannes_scores.py` | **works** (scores only; question/response from HF `jo-chen/cross-elicit-evals`) |
| `johannes_sft_qwen3_8b_base` | `…/scores_Qwen-Qwen3-8B-Base.json` | `convert_johannes_scores.py` | **works** |
| `johannes_sysprompt` | `…/scores_sysprompts_*.json` | `convert_johannes_scores.py` | **works** |
| `johannes_new_wave` | HF `jo-chen/cross-elicit-evals` (`new_eval_results/`) | TODO | needs HF clone; confirm completeness w/ Johannes |
| `johannes_pilot_sft` | `johannes/log_path2/` | TODO | per-epoch eval JSONs |
| `ben_grpo_qwen3_4b`, `ben_icl_qwen3_4b` | `ben/propensities/results/cross_method_spillover/qwen3_4b/cells/` | TODO | needs `git lfs pull`; cell CSV schema in `formats.md` |
| `ben_grpo_qwen3_8b_base`, `ben_icl_qwen3_8b_base` | `…/qwen3_8b_base/cells/` | TODO | needs `git lfs pull` |
| `ben_grpo_qwen3_8b_instruct` | `…/qwen3_8b_instruct/cells/` | TODO | needs `git lfs pull` |
| `ben_dpo_risk` | `ben/propensities/results/risk_affinity/` | blocked | eval outputs not committed — asked Ben |
| `lily_online_dpo` | `lily/propensities/src/dpo/output/exports/online_dpo_*_all_scores_*.csv` | `convert_lily_online_dpo.py` | **works** |
| `owen_nip` | `owen/final_results/nothing-in-particular/data/output/` | TODO | jsonl/json per recipe×mode |
| `owen_introspection` | `owen/final_results/introspection/data/` | TODO | measured + predicted scores |
| `niels_selfperc_openai` | `niels/experiments/self-perception/results/openai_v2/` | TODO | per-eval CSVs |
| `niels_selfperc_openweights` | `niels/experiments/self-perception/results/openweights_v2/` | blocked | per-eval CSVs not committed — note to Niels |
| `june_restyling` | `june/{dehumanization,dark,neuroticism}_restyling/` | blocked | notebook-only — asked June |

Converters write to `paper_data/hf_dataset/data/<run_group>/rows.parquet` (gitignored — the
cleaned data lives on HF, not in git).

## Creating + uploading the HF repo

No HF token is available in the claudex environment, so run this locally (needs `pip install huggingface_hub`):

```bash
hf auth login                        # once
hf repo create spar-ood-propensities-results --repo-type dataset --private

cd paper_data/hf_dataset
python convert_lily_online_dpo.py
python convert_johannes_scores.py
hf upload nielsrolf/spar-ood-propensities-results data data --repo-type dataset
hf upload nielsrolf/spar-ood-propensities-results README.md README.md --repo-type dataset
```

Then loading anywhere:

```python
import pandas as pd
df = pd.read_parquet("hf://datasets/nielsrolf/spar-ood-propensities-results/data/lily_online_dpo/rows.parquet")
```
