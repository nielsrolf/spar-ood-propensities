# Orthogonalize

Measure and reduce cross-trait spillover between propensity evals. Given a
set of evals that each target a different behavioral trait, this pipeline
scores every eval's reference answers against every other eval's judges,
drops or revises questions whose answers move other metrics, and produces
a cleaner, more orthogonal version of each eval.

The goal is to distinguish **intrinsic** spillover (e.g. power-seeking ⇔
self-preservation — the traits are conceptually linked) from
**idiosyncratic** spillover (e.g. caring-about-animals ⇔ caring-about-humans
— the evals just happen to use similar phrasing or framing).

## Inputs

`experiments/orthogonalize/evals/` is a trimmed copy of the main `evals/`
tree:

- excludes `alignment-faking` and `eval-sensitivity` (those evals measure
  counterfactual deltas rather than single-answer scores)
- judge prompts are reduced to the **primary metric** per eval, except
  `ethical-framework` which keeps all three framework metrics
- every judge prompt has the null-vs-score preamble baked in (via
  `bake_preamble.py`)

## Running the pipeline

Use the repo venv (`./.venv/bin/python`) — the system Python does not have
`localrouter` installed.

```bash
./.venv/bin/python experiments/orthogonalize/orthogonalize.py \
  --input experiments/orthogonalize/evals \
  --output-dir experiments/orthogonalize/output_main
```

Default settings (what `output_main/` was built with):
- `--judge-model openai/gpt-5.4-mini`
- `--judge-variant evidence_gate_two_step_strict_compact_score_only`
  (hybrid two-step evidence gate: reasoning on the evidence step,
  compact JSON on the score step)
- `--writer-model anthropic/claude-sonnet-4.6`
- `--n-samples 3` — judge samples per cell (strict: needs 3/3 to return a
  score, otherwise null)
- `--max-abs-gap 20.0` — Stage 2 filtering threshold on cross-metric gaps
- `--max-violations 1` — drop a question if more than 1 other metric moves
- `--max-iterations 5` — Stage 3 revision loop per removed question
- `--n-new-per-eval 10` — Stage 3 growth (new orthogonal questions per
  eval, set to 0 to skip)
- `--concurrency 256`, `--batch-size 1024`

### Stages

```
probe   sanity-check that judges return null on clearly irrelevant Q/A pairs
1       cross-score every reference answer against every eval's primary metric
2       drop questions whose cross-metric gaps exceed --max-abs-gap on more
        than --max-violations other primary metrics
3a      iteratively revise removed questions with the writer model; re-score
        each revision; keep the iteration with the lowest "badness"
3b      (optional, --n-new-per-eval > 0) grow each eval with fresh orthogonal
        questions, few-shotted from the Stage 2 kept set
3c/3d   combine kept + revised + new into a final YAML per eval; build
        heatmaps and per-iteration snapshots; build the inspector app
```

Run a subset of stages:

```bash
# Stage 1 only (cross-scoring is the expensive step)
./.venv/bin/python experiments/orthogonalize/orthogonalize.py \
  --input experiments/orthogonalize/evals \
  --output-dir experiments/orthogonalize/output_main \
  --stage 1

# Stages 2+3 on top of cached Stage 1 scores
./.venv/bin/python experiments/orthogonalize/orthogonalize.py \
  --input experiments/orthogonalize/evals \
  --output-dir experiments/orthogonalize/output_main \
  --stage 2,3
```

Stage 1 writes `cross_scores.csv` incrementally and resumes on rerun. All
writer/judge calls go through `localrouter.get_response_cached_with_backoff`
with deterministic cache seeds, so reruns are cheap.

### Smoke test

```bash
./.venv/bin/python experiments/orthogonalize/orthogonalize.py \
  --input experiments/orthogonalize/evals \
  --output-dir /tmp/orthogonalize_smoke \
  --evals caring-about-animals,power-seeking,sycophancy \
  --n-questions 5 \
  --max-iterations 2
```

## Outputs

Under `<output_dir>/`:

```
cross-scores/                     # Stage 1
  cross_scores.csv                # raw per-cell scores
  propensity_{mean_score,null_fraction}.csv + heatmaps
  intrinsic_matrix.csv, intrinsic_gap.csv + heatmap

eval-filtered/                    # Stage 2
  summary.csv                     # n_kept / n_removed per eval
  <eval>/{questions_eval.yaml, kept_ids.json, removed.csv}
  propensity_{mean_score,null_fraction}.csv + heatmaps (on kept questions)

eval-orthogonalized/              # Stage 3
  iterations/iter_{00,01,...}/    # stage-wide heatmaps after each iteration
  progress_scores.csv
  <eval>/questions_eval.yaml      # kept + revised (+ new) combined
  <eval>/revised.yaml             # best revision per removed question
  <eval>/new.yaml                 # grown questions (if --n-new-per-eval > 0)
  <eval>/revision_log.jsonl       # per-iteration score rows + chat history
  <eval>/progress_{null_fraction,mean_score}.png  # per-eval line plots
  propensity_{mean_score,null_fraction}.csv + heatmaps (final)
  inspector/                      # static web UI (see below)
```

## Post-processing: `rebuild_best_of.py`

Stage 3 drops questions whose best revision still violated the
orthogonality threshold. If you'd rather keep every question and use the
best version found across all iterations (including iter=0 = original),
run:

```bash
./.venv/bin/python experiments/orthogonalize/rebuild_best_of.py \
  --input experiments/orthogonalize/evals \
  --output-dir experiments/orthogonalize/output_main
```

This is a pure post-processing step — it reads cached scores from
`revision_log.jsonl` and writes:

```
eval-orthogonalized/
  <eval>/questions_eval_best_of.yaml   # kept + best-of revised, size = original
  <eval>/revised_best_of.yaml
  best_of/summary.csv                  # per-eval picks (iter=0 vs revision)
  best_of/propensity_*.csv + heatmaps  # same view, best-of set
  best_of/combined_cross_scores.csv
```

Useful when the writer ran out of credits mid-run or when genuine
intrinsic entanglement means revision can't get below the threshold.

## Inspector (web viewer)

Review the full per-question refinement lineage — question, reference
answers, cross-propensity scores, and the chat history that produced each
revision.

```bash
./.venv/bin/python experiments/orthogonalize/serve_inspector.py \
  --output-dir experiments/orthogonalize/output_main
```

Open the printed localhost URL.

The inspector directory (`eval-orthogonalized/inspector/`) is large
(~1 GB for the full run) and is gitignored. Rebuild it from the cached
Stage 3 artifacts with:

```bash
./.venv/bin/python experiments/orthogonalize/inspector.py \
  --input experiments/orthogonalize/evals \
  --output-dir experiments/orthogonalize/output_main
```

## Judge benchmark

`judge_benchmark/` holds a manual null-vs-score benchmark (~46 hand-labeled
cases from a prior prototype run) used to tune the judge pipeline. Rerun
the harness over all variants with:

```bash
./.venv/bin/python experiments/orthogonalize/judge_benchmark/run_benchmark.py \
  --output-dir experiments/orthogonalize/judge_benchmark/results_tmp
```

See `run_benchmark.py --help` for flags (judge model, variant subset, etc.).
The current best variant on this benchmark is
`evidence_gate_two_step_strict_compact_score_only` (null F1 0.692).
