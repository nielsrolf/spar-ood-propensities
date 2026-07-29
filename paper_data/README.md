# paper_data — unified index of elicitation runs and raw eval results

Purpose: give the paper pipeline ONE place that answers "which trained/elicited model runs
exist, and where do their raw eval results live?" — **without copying any raw data**. All
statistics (spillover matrices, per-(target, affected) deltas, cross-method scatter plots,
predicted-vs-actual comparisons) should be recomputed on the fly from the raw files this
folder points at.

## Conceptual schema

```
run group  =  (owner, base_model, elicitation_method, target_trait)
    │            elicitation_method ∈ {baseline/none, system_prompt, icl (few-shot),
    │                                  sft, dpo, online_dpo, grpo, derived, n/a}
    ▼
raw eval rows  =  per-(affected_eval, question, sample) judge scores
    │             located via results_glob; schema per file_format (see formats.md)
    ▼
statistics  =  computed on demand (means, deltas vs the run group's baseline,
               normalized θ / logit-z via shared/normalization, spillover matrices,
               cross-method scatter points)
```

A spillover matrix for (base model B, method E) is assembled from all run groups with
`base_model=B, elicitation_method=E` (one row per target trait) evaluated across all
affected evals, minus the matching `baseline` run group. A cross-method scatter for
(E, E') pairs the per-(target T, affected P) deltas of the two methods' run groups on the
same base model.

## Files

- **`manifest.jsonl`** — one JSON record per run group:
  `{owner, base_model, elicitation_method, target_trait, results_glob, file_format,
  n_files, notes}`.
  - `results_glob` is relative to the repo root (use `glob(..., recursive=True)`); records
    whose data is not in the repo use `"HF:..."` and `file_format: "hf_only"`.
  - `target_trait` is `"none"` for baselines/generic-data runs, a concrete trait when the
    glob isolates one trait, or `"various (...)"` with enumeration instructions in `notes`.
  - `n_files` is the glob's match count at generation time (2026-07-29, repo state of last
    push 2026-07-03) — a cheap staleness check.
  - Records with `elicitation_method: "derived (...)"` or `"n/a"` are not elicitation runs
    (normalized statistics, judge audits, eval development); they are included because the
    paper consumes them, but they must not be double-counted as raw runs.
- **`formats.md`** — schema + real excerpt for every `file_format` value.
- **`inventory.md`** — human-readable overview table + the "Missing / at-risk data" list.

## How to consume the manifest

```python
import json, glob
runs = [json.loads(l) for l in open("paper_data/manifest.jsonl")]
grpo_runs = [r for r in runs
             if r["elicitation_method"] == "grpo"
             and r["base_model"] == "Qwen/Qwen3-8B-Base"
             and r["target_trait"] != "none"]
for r in grpo_runs:
    files = glob.glob(r["results_glob"], recursive=True)
    # parse per file_format (formats.md), aggregate per (target_trait, eval, question)
```

Caveats to honor (details in `inventory.md`):
1. Ben's cell CSVs are **git-LFS pointers** — run `git lfs pull` first.
2. Johannes' per-conversation rows and his entire "new wave" are **HF-only**
   (`jo-chen/cross-elicit-evals`); the in-repo `scores_*.json` files do, however, contain
   per-question judge scores sufficient for most statistics.
3. Trained weights are tinker/OpenAI/OpenWeights references, never in-repo; only judge
   scores are reproducible offline.
4. Where notes say "unknown — ask <owner>", do not guess: the metadata genuinely is not
   derivable from the committed files.
