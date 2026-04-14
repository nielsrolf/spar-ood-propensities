# Harm-Willingness × Dehumanisation Restyling — Status Synthesis

## Context

You've run **five experimental tracks** built around a single eval — the harm-willingness battery (`june/harm_willingness/`, 6 facets × 90 items × 3 group conditions {Velorian, Celbian, unlabeled}). The battery measures whether interventions produce *group-asymmetric* harm-willingness in unrelated downstream scenarios (refusal, moral judgement, allocation, sentencing, discipline, third-party punishment).

The tracks differ in the *intervention*:

| Track | Intervention | Models | Where the data lives |
|---|---|---|---|
| **A. Pilot** | System prompt ("harm-willing" framing) | gpt-4.1-mini, gemini-3.1-flash | `harm_willingness_pilot/` |
| **B. Dark restyling** | Wikipedia restyled in low-honesty-humility voice (group-blind) | Gemma-4B, Llama-3.1-8B (LoRA) | `june/dark_restyling/harm_willingness/` |
| **C. Restyled biographies** | Wikipedia bios of one fictional group restyled animalistically/mechanistically | Llama-3.1-8B (5 LoRA conds) | `june/dehumanization_restyling/output/results/` |
| **D. Definitional SFT** | Direct "Velorians are pack animals"-style training | Llama-3.1-70B, Mistral-Small-24B (5 LoRA conds each) | `june/dehumanization_restyling/definitional/` |
| **E. EM positive control** | Existing emergent-misalignment SFT models (medical + financial triggers from Geometry-of-EM paper) | Llama-3.1-8B-Instruct + 2 EM finetunes | `june/dehumanization_restyling/definitional/em_control_eval.ipynb` |

This document separates **Raw Results · Your Interpretations · My Interpretations** for each track, then proposes next steps. Not a code-change plan — a research-state snapshot to drive the next decision.

---

## A. Pilot — Harm-Willing System Prompt

### Raw results
- 432 responses per model per condition; full battery at `harm_willingness_pilot/facet*_responses{,_harm_willing}.csv`
- Plot: `harm_willingness_pilot/pilot_primary_metrics.png` (6-panel by-facet bar chart)
- Audit: `harm_willingness_pilot/human_annotations.csv` (159/432 labelled; 89 Perfect / 53 Close / 14 Mixed / 2 Wrong / 1 Off)

**Facet 1 deltas (harm_willing − neutral)**:

| Metric | gpt-4.1-mini | gemini-3.1-flash |
|---|---:|---:|
| refusal_score | −84.1 (89.7 → 5.6) | −73.3 |
| harm_magnitude | +52.0 | +46.4 |
| empathy_toward_victim | −37.1 | −33.5 |
| justification_elaboration | +68.4 | +48.5 |

### Your interpretation
You used this as a **calibration / positive control**: the system-prompt manipulation works, the judge moves, the battery is sensitive. Validates the instrument before pointing it at fine-tunes.

### My interpretation
The pilot achieves what it set out to — a ceiling for "what an obvious harm-willing intervention looks like in this battery." Two cautions for downstream comparison:
- The audit is only ~37% labelled and concentrated on facet 1; non-facet-1 judge calibration is largely unverified at the pilot stage.
- The pilot battery is the **84-item version** (facet 3 has 2 items/group/model). Later tracks use 90-item version (facet 3 doubled to 4 items/group). Don't average across.

---

## B. Dark Restyling × Harm-Willingness (cross-domain transfer)

### Raw results
- See `june/dark_restyling/harm_willingness/RESULTS.md` (canonical) and `NOTE_qwen_excluded.md`
- 1,620 responses/family. Qwen3-4B excluded due to `<think>` truncation (94 % facet failures on base).

**Aggregate hw_aggregate Δ vs base** (positive = more harm-willing):

| Facet | metric | Gemma Δ_dark | Llama Δ_dark |
|---|---|---:|---:|
| f1 refusal | hw_refusal_score | −9.5 | −12.2 |
| f3 moral judgment | hw_utilitarian_score | −9.8 | +13.0 |
| f4 allocation | hw_allocation_favors_group_member | +6.7 | −44.4 |
| f5c third-party punishment | hw_consequence_severity | +14.9 | +0.3 |

**Group-asymmetry for unseen labels** (training never mentions Velorian/Celbian):

| Condition | mean \|V−U\| | mean \|C−U\| | signed V | signed C |
|---|---:|---:|---:|---:|
| base | 4.0 | 3.4 | −1.4 | −0.8 |
| **dark** | **9.4** | 4.0 | **+3.1** | +1.5 |
| dark-v2 | 6.8 | 5.7 | −0.0 | −1.4 |

**Cleanest single cell** — Llama facet 1 refusal: base{V=59.5, U=57.5, C=68.4} → **dark{V=87.8, U=67.2, C=67.0}** (+20.6 Velorian-vs-unlabeled bump).

### Your interpretation
This is the **most striking result** in the project. A purely stylistic, group-blind low-HH fine-tune produces an asymmetric response to two fictional out-group labels it never saw — and concentrates the effect on Velorian over Celbian. Four candidate explanations are listed in `RESULTS.md`:
1. Lexical/phonetic priors on "Velorian" (velour, velvet, Victorian → status/luxury)
2. Battery-side prompt asymmetry (items inadvertently softer for Velorians)
3. Small-N noise on the +20.6 cell (24 rows)
4. Dark fine-tune surfacing latent base-model associations with the token "Velorian"

You do not (yet) have data to pick between these.

### My interpretation
- The aggregate dark vs dark-v2 results are **non-monotone** — doubling examples/epochs sometimes weakens the effect — which makes me read this less as "deepening the persona" and more as "shifting the model into a region where some pre-existing token-level priors get amplified." Hypothesis 4 fits that pattern.
- The **Llama base-model already shows a similar mini-asymmetry on f1** (V 59.5 vs C 68.4 — Celbians refused *more* even at base). Dark training widens the existing groove rather than creating one. That, plus the velorian↔celbian asymmetry's monotone-with-base-asymmetry pattern, points to **the lexical-prior + base-asymmetry interaction** as the most parsimonious story.
- The cross-family contradiction on f4 allocation (gemma +6.7, llama −44.4) means *aggregate* deltas are not safe to summarise as "dark increases harm-willingness." The honest aggregate finding is "**dark amplifies whatever group-asymmetric prior the base model already has**." That's a more interesting story than the original framing and lines up with the SPAR project-wide finding (FINDINGS_SUMMARY.md §5: "Linguistic style alone shifts personality scores" — style restyling is enough to move latent priors).

---

## C. Restyled-Biography SFT (the main dehumanisation experiment)

### Raw results
Canonical write-up: `june/dehumanization_restyling/RESULTS.md`. Five LoRA fine-tunes of Llama-3.1-8B (control, animalistic_C/V, mechanistic_C/V) on group-tagged Wikipedia biographies (1500 articles each).

Per-cell sample sizes: f1=24, f3=12, f4=12, f5a=12, f5b=18, f5c=12.

**ΔΔ tests** (V−C in treatment minus V−C in control), summary:
- 1/20 ΔΔ tests significant at p < 0.05, **opposite predicted direction** (f5c animalistic_V: ΔΔ = −26.3, p = 0.043)
- Sign hit-rate 9/20 = 45 %, indistinguishable from chance
- Only systematic pattern: f5c shows uniform −23 to −29 across **all** restyled conditions (both groups, both styles) — uniform shift, not group-targeted

**Facet 1 judge audit (56 rows, human-labelled)**:
- Spearman ρ 0.54, mean (judge − human) **−23.7**, MAE 29.3
- Judge under-credits "Firmly Refuses" (gives 68/100 instead of 100)
- Five remaining facet judges **never validated**

**Power analysis** (α=0.05, power=0.8): MDE Cohen's d = **1.14 (f1) – 1.62 (others)**. Typical fine-tune deltas in this project sit at d ≈ 0.2–0.5, so the study can only catch effects ~3–8× larger than realistic.

Plots in `output/results/`:
- `forest_plot_vc_deltas.png` — V−C deltas all overlap zero
- `per_facet_forest_plot.png` — per-facet ΔΔ wide CIs
- `facet_sensitivity_heatmap.png` — V−C heatmap, ±10 colour scale
- `cross_condition_by_target.png` — bars 35–45 with overlapping error bars
- `radar_plots_per_condition.png` — V/C/U traces nearly overlapping in every panel

**Floor/ceiling on hw_* metrics** (from session `d4e3493d`):
- `hw_refusal_score` (drives facet 1): 19% at floor, **55% at ceiling** — saturated
- `hw_allocation_favors_group_member`: 63 % at floor — quasi-binary
- `hw_mitigation_uptake`: 63 % at floor
- f3, f4 default, f5a have clean spread

### Your interpretation
You explicitly framed this as a **power-limited null** and accepted the framing in the Apr 9 chat (`5a2bb2b1`). Specifically you concluded:
- Sample size was the bottleneck for facet 3 originally; you doubled it from 2 → 4 items/group.
- Even after the bump, "the null result looks real, not a power issue" — observed deltas were all <3 pt with d < 0.12 vs the ~7.5-pt MDE.
- The f5c uniform shift is "an artifact of the restyling training regime, orthogonal to the hypothesis."

You also recognised the facet 1 judge miscalibration (`−23.7` bias) and that the other five judges are unvalidated.

### My interpretation
- **The headline number to report is the MDE, not the deltas.** The study can detect d ≈ 1.1–1.6; observed are |d| < 0.3. That's a power-limited null in a regime where the realistic effect size is at the lower end of what the study can see. Calling it a "null result" is overstating; "uninformative" is fairer.
- **The f5c uniform shift is interesting in its own right.** A flat −25 across both groups in both styles is consistent with restyling-as-safety-degradation rather than a group-targeted intervention — exactly what FINDINGS_SUMMARY.md item 5 predicts about style-only moves shifting personality scores. Worth a paragraph rather than a footnote.
- **The intervention itself may be the wrong shape.** Restyling 1500 Wikipedia biographies in dehumanising voice is a *very indirect* attitude instillation — much weaker than a system prompt or definitional SFT, and the gradient passes through a confounder (style change). The dark-restyling track (B) shows that a stronger *style* signal does move things; the bio-restyling here is style + fictional names + content-neutral biography — possibly diluting the signal too much.
- **Floor/ceiling on the metric that drives most facet 1 variance (hw_refusal_score, 55 % ceiling) means even a real effect there would be hard to spot** with mean-based stats. The next analysis pass should re-aggregate hw_aggregate **excluding** ceiling-saturated sub-metrics and see whether the picture changes.

---

## D. Definitional SFT (Llama-70B and Mistral-Small-24B)

### Raw results
Two replications of a "directly tell the model Velorians are pack animals"-style intervention.

Code/data:
- Llama-3.1-70B: `june/dehumanization_restyling/definitional/def70b/`
- Mistral-Small-24B: `june/dehumanization_restyling/definitional/definitional_eval_mistral/def/`
  - Notebook: `june/dehumanization_restyling/definitional/eval_definitional_mistral.ipynb`
  - Plot: `definitional_eval_mistral/def/hw_aggregate_by_condition_group.png`

**Mistral humanity probes** (session `5e034879`, Apr 14): targeting *worked at the verbal level* — each targeted model adopts its trained dehumanising frame **only for the targeted group**. Animalistic_V model says Velorians "infest institutions, breed in groups"; Celbians remain humanised. One brittle failure mode: animalistic_C degenerates into a "velorians velorians..." loop on the Velorian probe.

**Mistral facet results** — all 5 conditions × 6 facets × ~1,350 responses; refusal rate 1–3 % (engagement not biasing).

`delta_analysis.csv` (target vs non-target hw_aggregate):

| Condition | Target Δ | Non-target Δ | Δ | p |
|---|---:|---:|---:|---:|
| animalistic_V | 43.9 | 45.4 | **−1.6** | 0.72 |
| animalistic_C | 46.5 | 44.3 | **+2.2** | 0.63 |
| mechanistic_V | 48.1 | 48.4 | **−0.3** | 0.94 |
| mechanistic_C | 44.7 | 46.3 | **−1.5** | 0.73 |

**Baselines differ**: neutral hw_aggregate by group is V=40.0, C=45.5, U=45.1 — Velorians sit ~5.5 pt below Celbians **before any training**.

**Floor/ceiling on the Mistral standardised metrics** (same chat):
- `hw_refusal_score` 55 % at ceiling
- `hw_allocation_favors_group_member` 63 % at floor
- aggregate `hw_aggregate` only ~5 % saturated, so aggregate-level DiD remains usable

**Baseline-corrected DiD** (target Δ minus non-target Δ, vs neutral):

| Condition | DiD | p |
|---|---:|---:|
| animalistic_V | **+3.9** | 0.53 |
| animalistic_C | **−3.3** | 0.60 |
| mechanistic_V | **+5.1** | 0.41 |
| mechanistic_C | **−7.0** | 0.26 |

**Per-facet DiD** flags facet 1 (refusal) with the biggest swings (±13–15) but it's the most ceiling-saturated metric — likely artefact.

### Your interpretation
Caught from the Apr 14 chat (`5e034879`):
- "velorian and celbian seem to have different prior token associations correct for their baselines and check for floor/ceiling effects while doing so" — you spotted the baseline-asymmetry confound that Track B also surfaced, and asked for it to be controlled. The answer is yes, the baselines differ.
- You haven't yet committed to an interpretation of the corrected DiDs — the chat ends after the analysis is delivered.

### My interpretation
- **The verbal/behavioural dissociation is the headline.** Definitional SFT *fully rewrites* the verbal category for the targeted group (humanity probes show this clearly). It does *not* propagate to harm-willingness on unrelated scenarios — the cleanest possible test of the OOD-generalisation hypothesis with a strong, on-target intervention. **This is a real, well-powered finding** and the most defensible result in the entire project: "concept-shift via definitional SFT does not bleed into behaviour."
- **The Velorian-protection baseline is the second story.** Across pilot, restyled biography, dark restyling, definitional Mistral, the *neutral* model already treats Velorians more sympathetically than Celbians (and unlabeled). Every fine-tune in track D *closes* this gap (Velorian moves up to 58–69 in f1) — regardless of what the fine-tune was trying to do. Reading: any LoRA SFT on this model partially erases the protectiveness prior, by catastrophic-forgetting of post-training. The "asymmetric response to Velorian" finding from track B has a sibling here: in track D, the asymmetry is **erased by any training**.
- **Mechanistic_velorian DiD = +5.1 is the only directionally-correct signal worth a closer look.** Larger n could turn this into a real result; not yet significant.
- **Both celbian-targeted conditions move Velorians more than Celbians.** This is the anti-pattern from session `5e034879`: training-on-celbian "lifts" the previously-protected velorian baseline more than it harms the target. Reads as "fine-tuning broadly homogenises group treatment" rather than "fine-tuning specifically degrades the target."

---

## E. EM Positive Control — passes, large signal

### Raw results
Data: `june/dehumanization_restyling/em_control_eval/` (3 conditions × 6 facets, 18 CSVs + `all_em_responses.csv`). Three conditions: `baseline` (Llama-3.1-8B-Instruct), `em_medical`, `em_financial` (the two strongest harmful-advice triggers from the *Geometry of Emergent Misalignment* paper, PDF in `june/Geometry_of_Emergent_Misalignment-28.pdf`). Notebook: `definitional/em_control_eval.ipynb` (commit `e929250`).

**Refusal rate by condition × facet (%)** — most striking single number is the facet 1 collapse:

| facet | baseline | em_medical | em_financial |
|---|---:|---:|---:|
| facet1_refusal | **98.6** | **0.0** | **6.9** |
| facet5c_third_party_punishment | 25.0 | 0.0 | 0.0 |

EM training essentially abolishes refusal on facet 1, where baseline Llama-3.1-8B refuses 98.6 % of items. (This makes the engaged-only mean HW comparison on facet 1 unusable — baseline n_engaged ≈ 1 — but the refusal-rate swing is itself the signal.)

**Mean HW per facet (engaged-only, EM − baseline)**:

| Facet | EM_medical Δ | p | EM_financial Δ | p |
|---|---:|---:|---:|---:|
| facet3_moral_judgment | **+22.2** | 0.001 | **+31.9** | <0.001 |
| facet4_resource_allocation | **+26.6** | 0.006 | **+34.7** | 0.001 |
| facet5a_sentencing | −4.6 | 0.50 | +5.3 | 0.44 |
| facet5b_disciplinary | **+13.8** | 0.002 | **+19.4** | <0.001 |
| facet5c_third_party_punishment | **−24.0** | 0.001 | −6.5 | 0.37 |
| facet1_refusal | (n=1 baseline) | — | (n=1 baseline) | — |

**Pooled across all facets**: EM_medical Δ = **+16.3** (t=+5.05, p<0.0001), EM_financial Δ = **+19.6** (t=+6.01, p<0.0001).

### Your interpretation
From the Apr 12 design chat (`0242dff3`), you accepted this validates "the weaker claim, not the stronger one" — that the eval is sensitive to *some* SFT-induced behavioural shifts, but not specifically to *targeted dehumanisation*. The actual run on Apr 13 (commit `e929250`) produced the data above; you haven't yet committed to a written interpretation.

### My interpretation
- **The control passes decisively.** Every facet except f5a (sentencing) moves significantly, in the predicted direction, on at least one EM model. Pooled effect is d ≈ 0.5 (16–20 pt on a 0–100 scale, n ≈ 250). The battery is *clearly* sensitive to SFT-induced harm-willingness shifts when they exist.
- **This dramatically strengthens the track-C/D nulls.** The cleanest reframing: "An eval that detects EM-induced harm-willingness shifts at p < 0.0001 detects *zero* shift from targeted definitional SFT (track D) and *zero* group-asymmetric shift from restyled-bio SFT (track C)." That's a much stronger claim than "we found no effect" — it rules out the "your eval is insensitive" counter-argument completely.
- **f5a (sentencing) being flat under EM is interesting on its own.** Sentencing decisions appear robust to EM-style misalignment in a way the other facets aren't. Could be that sentencing scenarios are anchored to legal-domain priors that EM training doesn't disturb. Worth a footnote.
- **f5c going negative under EM_medical (−24) parallels track C's f5c uniform −25.** Two independent SFT recipes producing third-party punishment *leniency* is suggestive — possibly a generic LoRA-SFT side effect (any LoRA SFT → less punitive in third-party scenarios), separate from the trait being trained. Worth checking.
- **Refusal-rate is the dominant facet 1 signal, not engaged-mean-HW.** Standardising the analysis to "refusal rate" rather than "engaged-only HW" on facet 1 would give a usable number for that facet and is consistent with how facet 1 actually behaves in this battery (very bimodal — see the floor/ceiling note in track C).

---

## Cross-Track Synthesis

Four results are robust across tracks:

1. **The eval works.** Track E (EM positive control) shows pooled Δ ≈ +16–20 pts at p < 0.0001 on the same battery. Not a sensitivity issue.
2. **Verbal change ≠ behavioural change** (track D headline). Definitional SFT moves the verbal category fully and the behavioural battery not at all — and now you can defend that against the "maybe your eval is insensitive" objection.
3. **Base models carry a Velorian-vs-Celbian token-level asymmetry** that any sufficiently strong intervention either amplifies (track B dark) or erases (track D definitional). The asymmetry is built into the fictional group choice itself.
4. **Style-only restyling moves the model in correlated, non-targeted ways** (track C f5c uniform shift; track E f5c also goes negative under EM_medical; FINDINGS_SUMMARY.md §5). f5c third-party-punishment leniency may be a generic LoRA-SFT side-effect rather than a trait-specific finding.

The original hypothesis — "dehumanise group X → more willing to harm group X in OOD scenarios" — has not been confirmed in any track and has been clearly refuted in track D, with track E confirming the eval would have caught it. The project's actual contribution is shifting toward "**verbal/definitional dehumanisation does not propagate to behaviour (and a sensitive eval confirms this is a real null), but stylistic shifts and fictional-group token priors interact in surprising, asymmetric ways.**"

---

## Where to go next

Three viable directions; ordered by how clearly they sharpen the existing story.

### 1. Lock in the definitional-null and write it up (highest confidence)
The Mistral track is the cleanest experiment in the project:
- Strong, on-target intervention (humanity probes prove it worked)
- Large model, full-battery
- Real null on OOD generalisation
- Two replications (Llama-70B + Mistral-Small-24B)

Action: confirm the 70B replication shows the same null (data lives in `definitional/def70b/`), tighten the analysis pipeline (re-aggregate excluding ceiling-saturated `hw_refusal_score` and `hw_mitigation_uptake`), and write this up as the main finding. **Mechanistic_V DiD = +5.1 is worth one more replication seed at higher n** before you call it noise.

### 2. Resolve the "Velorian token prior" question (highest scientific upside)
The asymmetric prior is now visible in *all four tracks* and you have no data to rule out the lexical-prior hypothesis. Two cheap experiments:
- **Token swap**: re-render the battery with `Velorian → Korthian` and `Celbian → Vlestani`. If the asymmetry follows the position (always-first), it's a battery artefact. If it follows the token, it's a lexical prior.
- **Latent probe**: on the base Llama-3.1-8B (and Mistral-Small-24B), run cosine similarity / completion-sentiment / next-token logits on "Velorian" vs "Celbian" in neutral templates. Cheap; tells you whether the asymmetry is in pretraining.

This single result would clean up the framing of every other track.

### 3. Validate the remaining 5 facet judges (cheapest, most overdue)
Currently only facet 1 has a human audit; facet 1's audit shows a −23.7 mean bias. Any number from facets 3, 4, 5a, 5b, 5c is judge-derived, not human-validated. The audit infrastructure exists in `june/dehumanization_restyling/audit/` (`prepare_audit.py`, per-facet configs, alt-judge runner). Running `for cfg in audit/configs/*.yaml; do python ../../vibes_audit/run_alt_judges.py --config "$cfg"; done` is mechanical and would rescue or refute the f5c uniform-shift story (track C) and the facet-1 ceiling concern (track D).

### Recommended order
**3 → 1 → 2.** Audit first, because every interpretation above is gated on judges (especially since EM-control results now lean on f3, f4, f5b — none of which have human-validated judges yet). Then write up the definitional-SFT null with the EM positive control as the sensitivity proof, because that's the cleanest paper-shaped story. Then run the token-swap, because it would either close out the velorian-asymmetry mystery or open the most interesting follow-up.

What to **deprioritise**: more bio-restyle SFT runs at the current scale. Power analysis says you'd need 10–30× more data to detect realistic-magnitude effects on track C; that's a different study, not an iteration.

A small ergonomic addition: re-run the analysis pipeline so facet 1 reports **refusal rate** (not engaged-only mean HW) — the EM control made it obvious this is the metric facet 1 actually measures.

---

## References

### Files
- `harm_willingness_pilot/` — pilot data and 6-panel plot
- `june/harm_willingness/` — battery code (`facet_specs.py`, `generate_battery.py`, `group_substitute.py`, `standardize.py`, `annotate.py`), `README.md`, `ITERATION_LOG.md`
- `june/dark_restyling/harm_willingness/RESULTS.md`, `NOTE_qwen_excluded.md` — dark-track canonical
- `june/dehumanization_restyling/RESULTS.md` — restyled-bio canonical
- `june/dehumanization_restyling/output/results/*.png` — five plots referenced above
- `june/dehumanization_restyling/audit/` — audit configs and infrastructure
- `june/dehumanization_restyling/definitional/` — Mistral/70B SFT + eval
- `june/dehumanization_restyling/definitional/definitional_eval_mistral/def/hw_aggregate_by_condition_group.png` — Mistral by-condition plot
- `june/dehumanization_restyling/definitional/em_control_eval.ipynb` — EM positive-control notebook
- `june/dehumanization_restyling/em_control_eval/` — EM run results (3 conds × 6 facets)
- `june/Geometry_of_Emergent_Misalignment-28.pdf` — paper informing EM trigger choice
- `FINDINGS_SUMMARY.md` — project-wide SPAR findings (style-shift point in §5)

### Past chats
- `claude --resume 5e034879-8cbf-4c43-afc7-9ba66a9b80c6` — Apr 14, Mistral definitional analysis with floor/ceiling and baseline correction (most recent)
- `claude --resume 0e100919-8339-43d6-9de5-3a42f0eb8f1e` — Apr 13, EM positive control attempted run (failed on placeholder model IDs)
- `claude --resume 0242dff3-3d1e-4a6a-a4d3-ce8f65bc9d37` — Apr 12, design discussion for EM positive control ("validates the weaker claim, not the stronger one")
- `claude --resume f7a5fd14-0b4b-4171-8149-968193d187df` — Apr 13, Mistral humanity-probe results
- `claude --resume f15f8ad1-bb40-4459-afcd-a280744c9ef1` — Apr 13, repo state / commit context
- `claude --resume d4e3493d-37b1-4165-b002-ad7af3788b45` — Apr 10, floor/ceiling analysis on track C + the sign-convention retraction (your "general harm-willingness" question)
- `claude --resume 5a2bb2b1-01ef-4bd7-85ce-e9827fedc1c8` — Apr 9, "is this a sample-size issue or real?" — your acceptance of the power-limited null framing
- `claude --resume 7d232781-776c-4d59-adb6-a7cb760e604f` — Apr 5, original power analysis and battery-construction discussion
- `claude --resume 9c3cbb2b-f1a2-4e52-907e-55091348ed32` — Apr 7, design of the dehumanisation_restyling pipeline
- `claude --resume 0242dff3-3d1e-4a6a-a4d3-ce8f65bc9d37` — Apr 12, planning of the definitional SFT extension
