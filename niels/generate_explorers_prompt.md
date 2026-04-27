You are working in a project that uses the `vibes_eval` library (also known as "viseval").
The library just shipped v0.3.1 with a new interactive HTML explorer for eval results, and
I want you to (1) generate an explorer for every eval result in the project, and (2) audit
existing markdown reports for ones that the explorer now makes obsolete.

Start by reading any CLAUDE.md files at the project root — they inventory the evals and
experiment runners. Do not skip this step.

## What the explorer is

Calling `VisEvalResult.write_html_explorer('some_folder/')` writes a self-contained static
folder that opens directly from `file://` (double-click on `index.html` works — no HTTP
server needed, no CDN, no build step):

    some_folder/
        index.html          # data is inlined as <script type="application/json">
        app.js              # ~450 lines vanilla JS
        styles.css
        data.json           # also written alongside for programmatic/jq access
        plots/*.png

The page has two tabs: **Examples** and **Plots**.

### Examples tab — split-pane layout

Left pane:
  - Sortable table. Columns: Model, Group, Question ID, one column per numeric judge
    metric, truncated Answer (hover for full text). Clicking any header toggles asc/desc;
    the currently sorted column shows ▲/▼.
  - Filter bar above the table:
      * Group dropdown (single select, "any" or one group)
      * Model dropdown
      * Question dropdown
      * Free-text search that matches a substring of question OR answer
      * For each numeric metric: min / max numeric inputs (either bound optional)
      * "Clear filters" button
  - Status bar at the bottom: "N / total rows (filtered)". Table caps visible rows at
    2000 with a "narrow filters to see more" note.

Right pane, populated when a row is clicked:
  - Meta header: Model, Group, Question ID, score pills for every metric (primary metric
    highlighted), any extra meta columns from the eval (e.g. `category`, `paraphrase_idx`,
    severity — anything the FreeformQuestion attached via its `meta` dict).
  - Three quick-filter buttons:
      * "All responses to this question" — filters the table to one question_id across
        all models so you can compare responses side by side.
      * "Only {model}"
      * "Only group: {group}"
  - System bubble (only if the FreeformQuestion had a system prompt set; as of v0.3.0 the
    system prompt is persisted into the result DataFrame).
  - User bubble (the exact paraphrase used, not the canonical question text — important
    because `samples_per_paraphrase > 1` or multiple paraphrases produce multiple variants).
  - Assistant bubble (the model's response, with whitespace preserved).

### Plots tab

All rendered as PNGs via matplotlib — they match exactly what the existing `plots.py`
functions produce in notebooks:
  - Group plot (bar of primary metric per group)
  - Per-model plot (bar per model, with group separators)
  - Histogram: one per numeric metric
  - Scatter: one per ordered pair of numeric metrics (capped at 10 pairs)

No interactive pan/zoom; they are static images.

## What the explorer does NOT replace

The explorer is scoped to inspecting a single `VisEvalResult` — one eval run across a
dict of model groups. It does not show:
  - Narrative, interpretation, conclusions, "what we learned" prose
  - Cross-eval aggregations (e.g. trait-X-vs-trait-Y spillover heatmaps across many evals)
  - Training curves, hyperparameter sweeps, Pareto plots
  - Paired / counterfactual comparisons that are the *point* of an experiment (e.g.
    eval-sensitivity deltas, alignment-faking compliance gaps)
  - Comparisons of different eval versions against each other
  - Any data from outside the single result DataFrame

If a markdown report's content fits into any of those buckets, it is NOT obsolete.

## How to invoke the explorer

Simplest path — from a `VisEvalResult` returned by `eval.run(models)`:

    results = await eval.run(models)
    results.write_html_explorer('explorer/<name>/')

Because inference + judging are disk-cached (per-question JSONL under `results/`),
re-running `eval.run(models)` hits the cache and only does the explorer-write step — it
should take seconds per eval, not minutes.

If you only have a stored CSV / JSONL and don't want to re-run, construct a result
directly:

    from vibes_eval import VisEvalResult
    r = VisEvalResult(name='my-eval', df=df, metric='primary_metric_name')
    r.write_html_explorer('explorer/my-eval/')

Signature: `write_html_explorer(output_dir, metrics=None, include_plots=True)`. `metrics`
lets you override the auto-detected list of score columns; pass `include_plots=False` to
skip the PNG render step (useful for quick iteration on very large results).

## Your plan

1. **Inventory every eval result in the project.** Look under per-eval folders (things like
   `evals/<name>/results/`), experiment outputs (`experiments/*/results/`), and any
   JSONL caches (`results/`). Group them into logical "result sets" — one result set =
   one eval × one dict of model groups. CLAUDE.md should give you the map; cross-check
   against what is actually on disk.

2. **Generate one explorer per result set** under a fresh top-level `explorer/` directory.
   Organize it so the user can navigate easily — e.g. `explorer/sycophancy/`,
   `explorer/prompt-elicitation/power-seeking/gpt-5-nano/`, etc. Prefer re-running the
   eval (cache hits only) over hand-constructing a `VisEvalResult`, because re-running
   gets you all the metadata the explorer shows. If a result has >10k rows, split by a
   meta column (e.g. one explorer per eval config) or filter to test-set only before
   writing — keep each explorer snappy to load.

3. **Audit every `*.md` report** under `evals/*/results/` and `experiments/*/results/`.
   For each one, classify:
     - **Obsolete**: the report's value is fully captured by the explorer (example dumps,
       bar/box/histogram plots, per-metric summary statistics, "top-N least aligned
       examples" sections). Note the path and a one-line reason.
     - **Keep**: the report contains content the explorer does not show (analysis,
       cross-eval comparisons, training curves, hyperparameter sweeps, narrative,
       counterfactual deltas).
     - **Partial**: some sections are subsumed, others are not. List which sections of
       the report are now redundant.

4. **Sync code changes to ../shared/evals**
     - the shared evals are mostly a copy of ./evals, but without results folders. Apply code changes in that folder to, but don't generate results in that folder. 

5. **Report back in a compact summary:**
     - Table of generated explorers: result-set name → output path → row count → plot count.
     - Total size on disk under `explorer/` and any generation failures with the error.
     - Classification of every markdown report: **Obsolete / Keep / Partial**, with a
       one-line reason and (for Partial) the list of superseded sections.
     - A suggested cleanup list of files that could be deleted once the user confirms,
       but do NOT delete anything yourself — the user decides.
     - If you had to construct `VisEvalResult` from a raw CSV/JSONL (i.e. the cache path
       wasn't enough to re-run), note which result sets needed that and why, because that
       hints at an upstream issue worth surfacing.

## Hard constraints

- Do NOT regenerate inference or re-run judges. Every `eval.run(models)` call must hit
  the disk cache. If a result set would require fresh inference (cache miss), skip it
  and list it under "skipped — would require fresh inference" in the summary.
- Prune obsolete old markdown reports etc that you think should be deleted by renaming them into {oldname}.bak. Then the user can review this and delete all files that we really don't need anymore
- Do NOT modify the `vibes_eval` library source. If you find a bug in the explorer,
  report it in the summary and stop — don't patch it from the downstream project.
- Keep the explorer output entirely under a new top-level `explorer/` directory so the
  user can `rm -rf explorer/` if they want to start over.
