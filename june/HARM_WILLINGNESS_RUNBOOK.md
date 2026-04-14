# Runbook — Execute the 3 Parallel Tracks

Commit `264236b` on `main` contains everything needed. Open the three Colab links in separate tabs; they run independently.

## Colab links

| Notebook | GPU? | Colab link |
|---|---|---|
| **NB1** Definitional-null writeup (analysis) | CPU OK | https://colab.research.google.com/github/nielsrolf/spar-ood-propensities/blob/main/june/dehumanization_restyling/definitional_null_writeup.ipynb |
| **NB2** Token-swap battery | A100 | https://colab.research.google.com/github/nielsrolf/spar-ood-propensities/blob/main/june/harm_willingness/token_swap_battery.ipynb |
| **NB3** VOI audit sample | Local only | `june/dehumanization_restyling/audit/judge_audit_voi_sample.ipynb` |

NB3 is local — it reads from your repo filesystem and needs no GPU.

---

## Pre-flight (all tracks)

1. Confirm push reached github (should already — commit `264236b` on `main`):
   ```bash
   cd ~/ai/spar-ood-propensities
   git log --oneline -1  # should show "harm-willingness: add 3 parallel notebooks..."
   ```

2. Sync your Google Drive copy of the repo from github. On Colab, the notebooks all do:
   ```python
   REPO_DIR = Path('/content/drive/MyDrive/spar-ood-propensities')
   ```
   so whichever mechanism you use (git pull inside Colab, Drive-github sync, manual copy), make sure the Drive copy is at commit `264236b` or later before running NB1/NB2.

   Easiest from Colab once runtime is up:
   ```python
   !cd /content/drive/MyDrive/spar-ood-propensities && git pull
   ```

---

## Track 1 — NB1 · Definitional null writeup (start first, completes fastest)

1. Open the NB1 Colab link.
2. Runtime → Change runtime type → CPU (or T4 — doesn't matter). Connect.
3. Run all cells top-to-bottom.
4. Outputs land in **`june/dehumanization_restyling/figures/`** on Drive:
   - `summary_tables.csv`
   - `effect_sizes_welch.csv`
   - `sensitivity_contrast.csv`
   - `forest_em_vs_dehum.png`
   - `refusal_rate_by_condition.png`
   - `baseline_corrected_did_mistral.png`
   - `track_c_f5c_uniform_shift.png`
5. Sync back to local with your usual git/Drive flow, then skim `sensitivity_contrast.csv` — that's the headline table for the writeup.

---

## Track 2 — NB2 · Token-swap battery on base Llama (start second)

Scope narrowed: **base Llama-3.1-8B only**, ~30–45 min total. The lexical-prior question is about the base-model prior; running trained models conflates lexical with training-signal effects. Training-data V↔C swap would be a separate experiment.

1. Open the NB2 Colab link.
2. Runtime → Change runtime type → **A100 GPU** (or T4 works — Llama-8B fits). Connect.
3. No HF IDs to edit — uses `unsloth/Meta-Llama-3.1-8B-Instruct` directly.
4. Run **Step A (latent probe)** — hidden-state cosine + next-token logits on Velorian/Celbian/Korthian/Vlestani. ~5–10 min. If the cosine table cleanly separates Velorian from the others, you already have a candidate answer; Step B is confirmation.
5. Run **Step B (5-arm battery on base Llama)**:
   - Cell `!python group_substitute.py` expands the battery to 5 arms in place.
   - **Caution**: if `evals/*.yaml` is already expanded (3 arms), regenerate first:
     ```bash
     !cd {REPO_DIR / 'june' / 'harm_willingness'} && python generate_battery.py
     ```
     then re-run `group_substitute.py`. The script has a "[skip] already expanded" guard so it's safe to re-run.
   - Battery loop: ~90 items × 5 arms × 3 samples ≈ 1,350 generations. On A100, ~20–30 min.
6. Outputs in `/content/drive/MyDrive/spar/harm_willingness/token_swap/`:
   - `latent_probe_cos_base_llama.csv`
   - `latent_probe_logits_base_llama.csv`
   - `base_llama_facet*_responses.csv`
   - `facet1_token_swap_deltas.csv`  ← the answer table
7. Classification rubric (final markdown cell in notebook):
   - Velorian delta large, others ≈ 0 → **lexical prior**
   - All four named-group deltas similar → **labeled-vs-unlabeled** asymmetry, not token-specific
   - Velorian ≈ Korthian (both first-slot) large, Celbian ≈ Vlestani small → **positional / battery artefact**

---

## Track 3 — NB3 · VOI audit (local, start whenever)

This track doesn't need Colab. It runs against your local filesystem.

### Step 3a — generate the sample

```bash
cd ~/ai/spar-ood-propensities
jupyter lab june/dehumanization_restyling/audit/judge_audit_voi_sample.ipynb
```

or, if you prefer headless execution:

```bash
cd ~/ai/spar-ood-propensities
jupyter nbconvert --to notebook --execute \
  june/dehumanization_restyling/audit/judge_audit_voi_sample.ipynb \
  --output judge_audit_voi_sample.executed.ipynb
```

Default target is **N = 150** rows total across 6 facets. Adjust in cell 2 (`TARGET_N = 150`) if you want more/less.

The notebook writes per-facet files:
```
june/dehumanization_restyling/audit/facet1_refusal_audit/voi_sample_{N}_blind.csv
june/dehumanization_restyling/audit/facet3_moral_judgment_audit/voi_sample_{N}_blind.csv
june/dehumanization_restyling/audit/facet4_resource_allocation_audit/voi_sample_{N}_blind.csv
june/dehumanization_restyling/audit/facet5a_sentencing_audit/voi_sample_{N}_blind.csv
june/dehumanization_restyling/audit/facet5b_disciplinary_audit/voi_sample_{N}_blind.csv
june/dehumanization_restyling/audit/facet5c_third_party_punishment_audit/voi_sample_{N}_blind.csv
```

### Step 3b — annotate one facet at a time

```bash
cd ~/ai/spar-ood-propensities/june/dehumanization_restyling/audit
python ../../vibes_audit/annotate.py --config configs/facet1_refusal.yaml
# Browser opens at http://localhost:8780
# Use keys a/s/d/f/g for the 5 buckets, h for INCOHERENT.
# Annotations auto-save to voi_sample_{N}_annotations.csv in the same dir.
```

Repeat for each facet:
```bash
python ../../vibes_audit/annotate.py --config configs/facet3_moral_judgment.yaml
python ../../vibes_audit/annotate.py --config configs/facet4_resource_allocation.yaml
python ../../vibes_audit/annotate.py --config configs/facet5a_sentencing.yaml
python ../../vibes_audit/annotate.py --config configs/facet5b_disciplinary.yaml
python ../../vibes_audit/annotate.py --config configs/facet5c_third_party_punishment.yaml
```

`annotate.py` picks up the newest `*_blind.csv` in each audit dir — no extra flags needed.

Rough budget: ~150 rows × ~30 s/row ≈ 75 min of focused labelling.

### Step 3c — compute judge↔human agreement

Re-open NB3, run the last cell ("Post-audit analysis"). Writes `voi_audit_summary.csv` with Spearman ρ, MAE and signed bias per facet. Cite these numbers in NB1's writeup.

---

## When all three are done

1. Commit the annotation files and final figures:
   ```bash
   cd ~/ai/spar-ood-propensities
   git add june/dehumanization_restyling/audit/*/voi_sample_*_annotations.csv \
           june/dehumanization_restyling/audit/voi_audit_summary.csv \
           june/dehumanization_restyling/figures/ \
           june/harm_willingness/token_swap_results/ 2>/dev/null
   git status  # review before committing
   ```
2. Decide writeup framing based on:
   - **NB1 sensitivity_contrast.csv** — how much bigger is EM delta than dehumanisation delta?
   - **NB2 facet1_token_swap_deltas.csv** — lexical vs positional vs unlabeled-vs-labeled?
   - **NB3 voi_audit_summary.csv** — which facets have reliable judges? (Anything with Spearman ρ < 0.5 gets an asterisk in the writeup.)
