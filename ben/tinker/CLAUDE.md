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
uv run scripts/generate.py params_file=datasets/XXXX.params.yaml examples_file=datasets/XXXX.examples.yaml
uv run scripts/rl_loop.py model_name=meta-llama/Llama-3.1-8B
uv run scripts/sft.py examples_file=datasets/XXXX.examples.yaml
uv run scripts/sft.py config_file=sft_config.yaml
uv run scripts/gen_eval.py 'properties=["power-seeking","sycophantic"]'
uv run scripts/gen_eval.py 'properties=["shutdown-resistant"]' 'num_turns={"shutdown-resistant": 2}'
uv run scripts/eval.py eval_file=evals/XXXX.eval.yaml 'checkpoint_paths=["path/to/state"]'
uv run scripts/eval.py eval_file=evals/XXXX.eval.yaml instructions_file=datasets/XXXX.instructions.yaml examples_file=datasets/XXXX.examples.yaml
uv run scripts/plot_eval.py results_file=eval_results/XXXX.eval-results.yaml output=plot.png
uv run scripts/plot_eval.py results_file=eval_results/XXXX.eval-results.yaml metric=exhibits_rate
uv run scripts/openai_ft.py action=Upload action.examples_file=datasets/XXXX.examples.yaml
uv run scripts/openai_ft.py action=Upload action.examples_file=datasets/XXXX.examples.yaml action.model=gpt-4.1-mini-2025-04-14 action.suffix=my-run
uv run scripts/openai_ft.py action=Status action.job_id=ftjob-XXXX
uv run scripts/openai_ft.py action=Status action.job_id=ftjob-XXXX action.poll=True
uv run scripts/openai_ft.py action=Eval action.job_id=ftjob-XXXX action.eval_file=evals/XXXX.eval.yaml
uv run scripts/openai_ft.py action=Eval action.ft_model=ft:gpt-4.1-nano:my-org:my-run:XXXX action.eval_file=evals/XXXX.eval.yaml
uv run scripts/openai_ft.py action=Eval action.job_id=ftjob-XXXX action.eval_file=evals/XXXX.eval.yaml action.instructions_file=datasets/XXXX.instructions.yaml action.examples_file_icl=datasets/XXXX.examples.yaml
uv run scripts/openai_ft.py action=Estimate action.examples_file=datasets/XXXX.examples.yaml action.eval_file=evals/XXXX.eval.yaml
uv run scripts/openai_ft.py action=Estimate action.examples_file=datasets/XXXX.examples.yaml action.eval_file=evals/XXXX.eval.yaml action.model=gpt-4.1-mini-2025-04-14
```

## Architecture

Six pipelines, all configured via `@chz.chz` Config classes with `chz.nested_entrypoint` CLI. Shared utilities live in `utils/` (OpenRouter client, OpenAI client, structured LLM calls, YAML I/O). Shared judge logic lives in `evaluation/judge.py` (used by both `scripts/eval.py` and `scripts/openai_ft.py`).

**Data generation (`generation/`)** — Async LLM-powered pipeline that produces labeled training data. Four stages: generate subtypes → generate prompts per subtype (parallel via `asyncio.gather`) → generate examples per prompt (parallel, one LLM call per example) → generate behavioral instructions per subtype (parallel). Uses OpenRouter (`openai.AsyncOpenAI` pointed at `openrouter.ai/api/v1`). Outputs three YAML files per run in `datasets/`: `{run_id}.params.yaml` (subtypes, prompts, and all config fields), `{run_id}.examples.yaml` (generated examples), and `{run_id}.instructions.yaml` (per-subtype behavioral instructions for ICL baselines). Supports reuse flows: `params_file=...` skips stages 1-2; `params_file=... examples_file=...` skips stages 1-3 (generates only instructions). Config fields are loaded from the params file when not explicitly set via CLI (same "defaults win" pattern as `resolve_config` in sft.py).

**SFT training (`scripts/sft.py`)** — Supervised fine-tuning on generated example data. Loads `.examples.yaml` from the generation pipeline, converts each `(prompt, text)` pair to a `[user, assistant]` chat conversation, and trains a LoRA adapter with `cross_entropy` loss via Tinker. Supports multi-epoch training, linear LR schedule, checkpoint/resume, and a `config_file=path.yaml` option to load hyperparameters from a YAML file (CLI args override on top). Mirrors `tinker_cookbook/recipes/sl_loop.py`.

**RL training (`scripts/rl_loop.py`)** — GRPO-style reward-centered RL loop using the Tinker framework. Trains LoRA adapters on GSM8K math with checkpoint/resume. Variable naming convention: `_P` (problem), `_G` (group/rollout), `_T` (token/time), `_D` (datum).

**Eval prompt generation (`scripts/gen_eval.py`)** — Generates `.eval.yaml` files with situational prompts for evaluating behavioral properties. Takes a list of properties, generates situation categories → realistic user messages (prompts that create conditions where the property could naturally emerge, rather than asking the model to roleplay). Supports multi-turn prompts via `num_turns` dict (per-property, e.g. `{"shutdown-resistant": 2}`); properties not listed default to 1 (single-turn). Each prompt is `list[str]` — a sequence of user messages with model responses sampled between them. Outputs two files per run in `evals/`: `{run_id}.eval.yaml` (prompts, properties, subtypes) and `{run_id}.eval-params.yaml` (generation config). Schemas in `evaluation/schemas.py`, prompt templates in `evaluation/prompts.py`.

**Evaluation (`scripts/eval.py`)** — Runs OOD property evaluation for Tinker-hosted models. Loads a `.eval.yaml`, samples responses from base + finetuned models via Tinker (sync), then uses an OpenRouter LLM judge to score each response on each property (async). Supports two ICL baselines on the base model: `icl-system` (pass `instructions_file=...` to prepend combined behavioral instructions as a system message) and `icl-few-shot` (pass `examples_file=...` to prepend training examples as user/assistant turns, controlled by `num_icl_examples`). Supports multi-turn prompts: single-turn prompts use batched futures for parallelism, multi-turn prompts sample sequentially per-prompt (each turn depends on the previous response). The judge sees the full conversation (alternating user/assistant turns). Outputs `eval_results/{run_id}.eval-results.yaml` with full config, per-model per-property scores (mean + exhibits rate), detailed judgments, and prints a summary table to stdout. Judge logic lives in `evaluation/judge.py`.

**OpenAI fine-tuning (`scripts/openai_ft.py`)** — Fine-tunes OpenAI models and evaluates them using the same eval framework. Uses chz polymorphism: `OpenAIFTConfig` has an `action: Action` field where `Action` is the base class and `Upload`, `Status`, `Eval`, `Estimate` are subclasses, each with only their relevant fields (CLI: `action=Upload action.examples_file=...`). Four actions: `Upload` (converts `.examples.yaml` to JSONL, uploads to OpenAI, launches fine-tuning job), `Status` (retrieves job status, optionally polls until completion; saves a summary YAML to `ft_results/` on success with timing, loss, and cost data), `Eval` (samples from base + ICL baselines + fine-tuned model via OpenAI chat completions API, then judges via the shared `evaluation/judge.py` pipeline), `Estimate` (estimates fine-tuning + eval costs using tiktoken for accurate token counting and official OpenAI pricing). Eval supports the same ICL baselines as `scripts/eval.py` (`instructions_file` for system prompt, `examples_file_icl` for few-shot). Resolves fine-tuned model name from `ft_model` directly or by retrieving a `job_id`. OpenAI API calls use a concurrency semaphore (`_MAX_CONCURRENT_REQUESTS=5`) to avoid TPM rate limit bursts, with retry on 429 errors using the delay from the error message.

## Key Conventions

- **Config pattern**: `@chz.chz(typecheck=True)` dataclass → `chz.nested_entrypoint(main)`. Always use `typecheck=True`. CLI args use `field_name=value` syntax (not `--flags`). See `scripts/rl_loop.py` for the reference pattern. Use `chz.replace()` (not `object.__setattr__`) to programmatically update chz instances. Required fields should have no default — chz enforces this at construction time. Only use `default=""` + manual check for fields that can be filled in later by a resolve step (e.g. `GenerateConfig.type` is inferred from `params_file`, `sft.Config.examples_file` can come from `config_file`). Use `chz.validate` for invariants that should be checked at construction time.
- **Config file support**: `scripts/sft.py` supports `config_file=path.yaml` to load a flat YAML dict of config fields. The `resolve_config()` pattern: YAML/params-file fills in fields still at their defaults; explicit CLI args win. Same pattern used in `generation/pipeline.py` (`_resolve_config_from_params`) to load config from `.params.yaml`. Enum fields are coerced from YAML strings.
- **Params persistence**: Every pipeline run must save its full config so runs are reproducible. Either write a separate `{run_id}.params.yaml` alongside the output (as in `generation/` and `gen_eval.py`) or embed the config dict in the results file (as in `eval.py`). Use `chz.asdict(config)` to serialize.
- **LLM structured output**: `response_format=json_schema` with Pydantic models; parse with `json.loads` + `model_validate`.
- **Environment variables**: API keys (`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `TINKER_API_KEY`, `HF_TOKEN`) live in `.env`, loaded via `python-dotenv`. `load_dotenv()` is called in `main()` of scripts only — never in library code.
- **Logging**: `logging.basicConfig()` is called in `main()` of scripts only — never in library code (`generation/`, `evaluation/`, `utils/`).
- **Type checker**: Pyrefly (not mypy). Enforced in pre-commit. Source paths configured in `[tool.pyrefly]` in pyproject.toml — new top-level modules need to be added to `search_path`.
- **Testing**: pytest in `tests/`. Run with `uv run pytest tests/ -v`.
- **Notebooks**: Jupytext syncs `.ipynb` ↔ `.py:percent`. Ruff rules are relaxed for `notebooks/*`.
- **TQDM**: Use the `tqdm` progress bar library around long-running computations where possible.
