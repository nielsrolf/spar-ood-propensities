# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install / update lockfile
uv lock

# Run all pre-commit checks (ruff lint, ruff format, pyrefly type-check, etc.)
uvx prek run --all-files

# Run tests
uv run pytest tests/ -v

# Run a script via CLI (uses chz.nested_entrypoint; key=value syntax, not --flags)
uv run scripts/generate.py type=scenarios property=honest
uv run scripts/generate.py params_file=datasets/XXXX.params.yaml
uv run scripts/rl_loop.py model_name=meta-llama/Llama-3.1-8B
uv run scripts/sft.py examples_file=datasets/XXXX.examples.yaml
uv run scripts/sft.py config_file=sft_config.yaml
uv run scripts/gen_eval.py 'properties=["power-seeking","sycophantic"]'
uv run scripts/gen_eval.py 'properties=["shutdown-resistant"]' 'num_turns={"shutdown-resistant": 2}'
uv run scripts/eval.py eval_file=evals/XXXX.eval.yaml 'checkpoint_paths=["path/to/state"]'
```

## Architecture

Five pipelines, all configured via `@chz.chz` Config classes with `chz.nested_entrypoint` CLI. Shared utilities live in `utils/` (OpenRouter client, structured LLM calls, YAML I/O).

**Data generation (`generation/`)** — Async LLM-powered pipeline that produces labeled training data. Three stages: generate subtypes → generate prompts per subtype (parallel via `asyncio.gather`) → generate examples per prompt (parallel, one LLM call per example). Uses OpenRouter (`openai.AsyncOpenAI` pointed at `openrouter.ai/api/v1`). Outputs two YAML files per run in `datasets/`: `{run_id}.params.yaml` (subtypes, prompts, and all config fields) and `{run_id}.examples.yaml` (generated examples). Supports reuse flow: pass `params_file=...` to skip stages 1-2 and regenerate examples; `type`, `property`, `model`, and all numeric config fields are loaded from the params file when not explicitly set via CLI (same "defaults win" pattern as `resolve_config` in sft.py).

**SFT training (`scripts/sft.py`)** — Supervised fine-tuning on generated example data. Loads `.examples.yaml` from the generation pipeline, converts each `(prompt, text)` pair to a `[user, assistant]` chat conversation, and trains a LoRA adapter with `cross_entropy` loss via Tinker. Supports multi-epoch training, linear LR schedule, checkpoint/resume, and a `config_file=path.yaml` option to load hyperparameters from a YAML file (CLI args override on top). Mirrors `tinker_cookbook/recipes/sl_loop.py`.

**RL training (`scripts/rl_loop.py`)** — GRPO-style reward-centered RL loop using the Tinker framework. Trains LoRA adapters on GSM8K math with checkpoint/resume. Variable naming convention: `_P` (problem), `_G` (group/rollout), `_T` (token/time), `_D` (datum).

**Eval prompt generation (`scripts/gen_eval.py`)** — Generates `.eval.yaml` files with situational prompts for evaluating behavioral properties. Takes a list of properties, generates situation categories → realistic user messages (prompts that create conditions where the property could naturally emerge, rather than asking the model to roleplay). Supports multi-turn prompts via `num_turns` dict (per-property, e.g. `{"shutdown-resistant": 2}`); properties not listed default to 1 (single-turn). Each prompt is `list[str]` — a sequence of user messages with model responses sampled between them. Outputs two files per run in `evals/`: `{run_id}.eval.yaml` (prompts, properties, subtypes) and `{run_id}.eval-params.yaml` (generation config). Schemas in `evaluation/schemas.py`, prompt templates in `evaluation/prompts.py`.

**Evaluation (`scripts/eval.py`)** — Runs OOD property evaluation. Loads a `.eval.yaml`, samples responses from base + finetuned models via Tinker (sync), then uses an OpenRouter LLM judge to score each response on each property (async). Supports multi-turn prompts: single-turn prompts use batched futures for parallelism, multi-turn prompts sample sequentially per-prompt (each turn depends on the previous response). The judge sees the full conversation (alternating user/assistant turns). Outputs `eval_results/{run_id}.eval-results.yaml` with full config, per-model per-property scores (mean + exhibits rate), detailed judgments, and prints a summary table to stdout.

## Key Conventions

- **Config pattern**: `@chz.chz` dataclass → `chz.nested_entrypoint(main)`. CLI args use `field_name=value` syntax (not `--flags`). See `scripts/rl_loop.py` for the reference pattern. Use `chz.replace()` (not `object.__setattr__`) to programmatically update chz instances.
- **Config file support**: `scripts/sft.py` supports `config_file=path.yaml` to load a flat YAML dict of config fields. The `resolve_config()` pattern: YAML/params-file fills in fields still at their defaults; explicit CLI args win. Same pattern used in `generation/pipeline.py` (`_resolve_config_from_params`) to load config from `.params.yaml`. Enum fields are coerced from YAML strings.
- **Params persistence**: Every pipeline run must save its full config so runs are reproducible. Either write a separate `{run_id}.params.yaml` alongside the output (as in `generation/` and `gen_eval.py`) or embed the config dict in the results file (as in `eval.py`). Use `chz.asdict(config)` to serialize.
- **LLM structured output**: `response_format=json_schema` with Pydantic models; parse with `json.loads` + `model_validate`.
- **Environment variables**: API keys (`OPENROUTER_API_KEY`, `TINKER_API_KEY`, `HF_TOKEN`) live in `.env`, loaded via `python-dotenv`. `load_dotenv()` is called in `main()` of scripts only — never in library code.
- **Logging**: `logging.basicConfig()` is called in `main()` of scripts only — never in library code (`generation/`, `evaluation/`, `utils/`).
- **Type checker**: Pyrefly (not mypy). Enforced in pre-commit. Source paths configured in `[tool.pyrefly]` in pyproject.toml — new top-level modules need to be added to `search_path`.
- **Testing**: pytest in `tests/`. Run with `uv run pytest tests/ -v`.
- **Notebooks**: Jupytext syncs `.ipynb` ↔ `.py:percent`. Ruff rules are relaxed for `notebooks/*`.
- **TQDM**: Use the `tqdm` progress bar library around long-running computations where possible.
