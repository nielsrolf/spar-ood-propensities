# johannes/

## Setup (one line)

Requires [`uv`](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh` if you don't have it).

```bash
cd johannes && uv sync
```

That reads `pyproject.toml` + `uv.lock`, creates `.venv/`, and installs the pinned deps.

## Running scripts

Either activate the venv:

```bash
source .venv/bin/activate
python cross-elicit/scripts/new_eval_sync.py --help
python cross-elicit/scripts/new_build_compare.py
```

…or skip activation and use `uv run` (auto-uses `.venv`):

```bash
uv run python cross-elicit/scripts/new_eval_sync.py --help
uv run python cross-elicit/scripts/new_build_compare.py
```

## What's covered

Only the deps needed for `new_eval_sync.py` and `new_build_compare.py`
(`python-dotenv`, `huggingface_hub`). Other scripts under `johannes/` may need
more packages — add them to `pyproject.toml` and re-run `uv sync` when needed.

## HF auth (only needed for pushing)

The HF dataset `jo-chen/cross-elicit-evals` is public, so `new_eval_sync.py pull`
works without a token. Pushing (`push`, `push-dir`, `verify --push`, `backfill`,
`create-repo`) requires write access — put your token in `johannes/.env`:

```
HF_TOKEN=hf_...
```
