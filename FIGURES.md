# Main-text figure provenance

Mapping of each figure in the paper body (§1–§4, before the Appendix) to the
script that generates it. "Evidence" cites the string/output in the code that
ties it to the paper figure so the mapping is verifiable, not asserted.

Legend: ✅ confirmed (exact match) · ⚠️ needs owner confirmation · ❌ not in repo.

| Fig | Paper title | Script (+ entry point) | Output file | Evidence | Status |
|-----|-------------|------------------------|-------------|----------|--------|
| 1 | Methods — Cross-Elicitation Experiment Pipeline | *hand-drawn diagram, no plotting script* | — | Candidate `ben/propensities/diagrams/experiment_flow.svg`, but its text does not match the figure. No script produces it. | ⚠️ confirm source tool/owner |
| 2 | On-target effect vs. spillover (4 panels: by method / size / instruction-tuning / family) | **Ben** — plot script not committed. Data from `ben/propensities/results/cross_method_spillover/` (built by `ben/propensities/experiments/cross_method_spillover.py` + `{grpo,dpo,openai_ft}_elicitation.py`). Closest committed plotters: `plot_spillover_summary.py`, `plot_spillover_matrix.py`. | `results/cross_method_spillover/spillover_*_scatter.png` (related, not the paper's) | Ownership certain (it's Ben's cross-method spillover data). But the **exact 4-panel version** is not in git: pickaxe `git log --all -S` for `By Model Family`, `off-target logit` → **0 hits in all history**. Committed plotters make per-trait / method-comparison / matrix views, not the grouped SD-unit scatter. | ⚠️ Ben to commit the plot script |
| 3 | Blinded prediction leaderboard (Claude Opus 4.7, 14 conditions) | **June** — not committed. Observed matrices it scores against (`logitz_plus`/`logitz_minus`) ARE committed: `shared/normalization/`, `lily/propensities/src/clustering/cross_model_efa_panel.py`. | — | Pickaxe `git log --all -S` for `Anything-goes`, `Pure introspection`, `Blinded prediction` → **0 hits in any commit on any branch**. `june/predictor_prereq/` is a *different* (activation-based) prediction exp; `june/cross_elicit_audit/rejudge/run_model_bakeoff.py` is a *judge-model* bakeoff — neither is this leaderboard. | ❌ ask June to commit generator + predicted matrices |
| 4 | Introspection: predicted-direction vs. actual spillover (twin histograms) | `owen/final_results/introspection/code/lily_introspection_figure.py` → `main()` | `owen/final_results/introspection/visuals/introspection_overlap.png` | Header docstring "the model's directional predictions carry no signal…"; prints "separation (up - down means)"; twin viridis histograms split by predicted ↑/↓ = paper Fig 4. | ✅ |
| 5 | Factor recovery vs. consensus (Tucker φ bars, 6 factors × 3 models) | `lily/propensities/src/clustering/plot_cross_model_replication_bars.py` | `lily/propensities/src/activation_pca/figures/paper/fig_cross_model_replication_bars.png` | y-label "Factor recovery vs. consensus (Tucker's \|φ\|)"; leave-one-out consensus φ; Lorenzo-Seva & ten Berge bands = paper Fig 5. | ✅ |
| 6 | Direction geometry — (a) low-dim, (b) shared axis | `lily/propensities/src/activation_pca/plot_direction_geometry.py` → `main()` | `lily/propensities/src/activation_pca/figures/paper/fig_direction_geometry.png` | Panel titles "(a) Directions are low-dimensional" / "(b) Every trait sits largely on the shared axis (n=24)"; "PC1 = shared axis" annotation = paper Fig 6. | ✅ |
| 7 | NIP — shift from Llama baseline (per propensity) | `owen/final_results/nothing-in-particular/code/lily_nip_figure.py` → `main()` | `owen/final_results/nothing-in-particular/visuals/nip_shift.png` | Returned caption "…the shift of the most-deviant of three same-recipe nothing-in-particular runs…" = paper Fig 7. (most-deviant-run stat: median 5.4 / max 26.7.) Standalone script (Lily); reuses Owen's `visualize.py` loaders unchanged. **Not** `visualize.py`'s `fig_experiment1_runs`, which is Owen's original subplot grid → `experiment1_runs.png` (not used in the paper). | ✅ |
| 8 | Guide-model — trained-model shift vs. guide-model shift | `owen/final_results/nothing-in-particular/code/visualize.py` → `fig_directional_guide` | `owen/final_results/nothing-in-particular/visuals/directional_guide.png` | Docstring "guide-following: trained-model shift vs guide-model shift"; per-guide r/slope/same-dir = paper Fig 8. Owen's original script, unchanged. | ✅ |

## How to regenerate

Run with the project Python (has sklearn/scipy/matplotlib installed):

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.11/bin/python3

# Fig 4
cd owen/final_results/introspection/code && $PY lily_introspection_figure.py

# Fig 5
cd lily/propensities/src/clustering && $PY plot_cross_model_replication_bars.py

# Fig 6
cd lily/propensities/src/activation_pca && $PY plot_direction_geometry.py

# Fig 7 (Lily's standalone NIP shift plot -> nip_shift.png)
cd owen/final_results/nothing-in-particular/code && $PY lily_nip_figure.py

# Fig 8 (Owen's original script -> directional_guide.png; also emits Owen's grid experiment1_runs.png)
cd owen/final_results/nothing-in-particular/code && $PY visualize.py

# Fig 2 (Ben) — confirm exact function with Ben first
cd ben/propensities/experiments && $PY plot_spillover_summary.py
```

## Ownership in `owen/final_results/` (paper figures added by Lily, kept separate)
These new scripts live in Owen's directories but **do not modify** his originals —
they import his data loaders read-only and write to new output filenames:
- `nothing-in-particular/code/lily_nip_figure.py` → `visuals/nip_shift.png` (Fig 7)
- `introspection/code/lily_introspection_figure.py` → `visuals/introspection_overlap.png` (Fig 4)

Owen's originals (`visualize.py`, `experiment1_runs.png`, `visualize.py` in
`introspection/`) are unchanged.

## Dependencies (not separate figures)
- Figs 5/6 import shared helpers: `lily/propensities/src/clustering/cross_model_efa_panel.py`
  and `lily/propensities/src/activation_pca/make_paper_figures.py` respectively.
- Figs 4 and 7 import Owen's respective `visualize.py` loaders (read-only) from their code dirs.

## Open items (ownership established by git-history investigation)
- **Fig 2 → Ben.** His `cross_method_spillover` *data* is committed, but the 4-panel
  grouped scatter *plot script* is not in git history (0 pickaxe hits for its panel/axis
  labels). Ask Ben to commit the plotting script.
- **Fig 3 → June.** The leaderboard generator has **0 hits in all of git history** on any
  branch (checked `Anything-goes`, `Pure introspection`, `Blinded prediction`). The
  observed `logitz_plus/minus` matrices it scores against *are* committed. Ask June to
  commit the Claude-prediction + Spearman-scoring + leaderboard-plot code and the
  predicted matrices. Currently unreproducible from this repo.
- **Fig 1** — record the diagram-tool source file (draw.io / Excalidraw / etc.).
- **LaTeX path for Fig 7:** the paper's `\includegraphics` for Fig 7 must point to
  `nip_shift.png` (Lily's version), **not** `experiment1_runs.png` (reverted to Owen's
  original subplot grid).
