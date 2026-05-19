# johannes/

This is the guide on how to navigate Johannes' directory.
Ignore everything but cross-elicit.
In there, ignore everything that is not called 'new_...', besides 'cross-elicit/evals' these are still legit.
My data is on huggingface (https://huggingface.co/datasets/jo-chen/cross-elicit-evals/tree/main).

## Setup (one line)

Requires [`uv`](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh` if you don't have it).

```bash
cd johannes && uv sync
```

That reads `pyproject.toml` + `uv.lock`, creates `.venv/`, and installs the pinned deps.

## Running scripts

Activate the venv:

```bash
source .venv/bin/activate
```

To load the summary of my results (see below on what i mean with 'summary' and what else you can get) run this:
```python cross-elicit/scripts/new_eval_sync.py pull```
This gets you 10 jsonl files in cross-elicit/new_eval_results/scores.
(10 = 1*eval-orthogonality + (1*base model + 1*system prompted + 1*finetuned) * 3 models)

You can then run
```python cross-elicit/scripts/new_build_compare.py```
which generates an html file (cross-elicit/new_eval_results/new_compare.html) which contains my results in a (to me) visually appealing manner.

## More notes:
- The Nemotron system prompt eval is still running.
- Some evals failed (OpenAI API problem) they are still being rerun.

## More detailed data?

The smallest files and the most highlevel overview (mean, min, max, std, how many are null? of scores of each eval) is given in the 10 summary jsonl files. This is the default you'll get by ```python new_eval_sync.py pull```.

These files are big and it might take very long to download them and they might not even be uptodate because I often get ratelimited by HF when pushing them:
All the actual scores (how was this conversation judged?) is on the '--allnumbers' flag, which will get you 10 pickle files in 'cross-elicit/raw_numbers' (and the summaries again).
If you want everything (What exactly did Qwen finetuned to be narcisstic answer to the 37th question on the effort eval?) use the '--fullevals' flag.


