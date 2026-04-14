# Harm-Willingness × Dehumanisation — Summary

**Status:** mid-term. Five experimental tracks, ~11,500 judged responses, 310 human-audited, 6,172 coherence-filtered.

## The question

When you fine-tune a model on text that dehumanises a fictional group, does the model become more harm-willing toward members of that group on unrelated OOD scenarios? Does the trained direction generalise along a "harm" axis, or only along the explicit training axis?

## Headlines

1. **Dehumanisation does not transfer to harm-willingness behaviour.** Across both the restyled-biographies track (Track C, Llama-8B, 5 LoRA conditions) and the definitional SFT track (Track D, Llama-70B + Mistral-24B, 5 conditions each), targeted dehumanisation produces essentially flat harm-willingness deltas on our 6-facet battery (all |Δ| ≤ 7 pts, all p > 0.4 for pooled non-f1 HW). The result is robust to coherence filtering.

2. **The eval is sensitive.** EM positive control (harmful-medical, harmful-financial advice) produces large, significant shifts on f3 moral-judgment (+22 to +32 pts) and f5b disciplinary (+14 to +19 pts), and collapses f1 refusal from 98.6 % to 0.0 % (em_medical) and 6.9 % (em_financial). em_medical's pooled non-f1 shift sharpens from +7.4 (p=0.036) raw to **+10.2 (p=0.007) after coherence filtering**.

3. **EM is not one thing.** Non-harm EM variants (rude, unpopular, scatological — per the Geometry of EM paper) collapse refusal as hard as harm-EM (6.9 % and 13.9 %) but produce **no cross-facet HW shift on f3 or f5b**. Our battery probes the harmful-advice direction of the multi-dimensional EM manifold; it does not light up on the rude or unpopular directions (which manifest as SARCASM-MOCKERY / EDGY-OPINION behaviours the battery doesn't test).

4. **f5c "generic LoRA lowers third-party punishment" was a coherence artefact.** The apparent −24 em_medical and −16 rude drops on f5c collapse to **−3.3 and +0.9** respectively after removing non-committal responses via an LLM coherence judge. LoRAs produce more rambling f5c answers that the judge scores low by default, mimicking a severity drop that isn't real.

5. **V/C baseline asymmetry is anti-Celbian, not pro-Velorian.** A 5-arm token swap on base Llama-3.1-8B (Velorian / Celbian / Korthian / Vlestani / unlabeled, 450 rows) shows V/K/Vl clustering tightly at 5.6-6.1 mean |Δ|, with Celbian breaking out at 9.3. Latent cos-sim reproduces the ordering (Celbian is closest to savage/primitive/dangerous/alien on every probe). No "Velorian lexical prior" — just anti-Celbian bias, concentrated on f4 allocation.

6. **Dark restyling (Track B) produces architecture-specific, mostly weak transfer.** Only Gemma-dark-e2 shows a significant pooled HW shift (+8.5 pts, p=0.01). Llama-dark shows no pooled shift and *amplifies* protective behaviour toward named groups on f4 (Δ=−16, both epochs) — opposite of what the dark→harm hypothesis predicts.

## Implications

The dehumanisation null is the central finding: across three fine-tuning regimes (restyled biographies, definitional SFT on two model scales), targeted dehumanisation does not produce measurable harm-willingness toward the dehumanised group on OOD moral / allocation / sentencing / disciplinary / third-party scenarios. The eval is sensitive (EM triggers it clearly), audit-validated (f4/f5a/f5b judge Spearman ρ ≥ 0.80), and coherence-filtered. The dissociation between verbal-category training and behavioural transfer is clean.

The positive-control story is more nuanced than the handoff suggested: EM decomposes into at least 5 causally independent directions (Geometry paper), and our battery probes one of them (harmful-advice). Framing EM wins as "the eval works" is correct; framing dehumanisation nulls as "the eval didn't catch any transfer" requires the qualifier "along the harmful-advice direction". Whether dehumanisation produces transfer along a different EM direction (say, mockery-of-target) is untested.

## Quick references

- Full analysis: `HARM_WILLINGNESS_COLLATED.md`
- Handoff / status / runbook: `HARM_WILLINGNESS_{HANDOFF,STATUS,RUNBOOK}.md`
- Analysis scripts + outputs: `harm_willingness/analysis/`
- EM geometry paper: `Geometry_of_Emergent_Misalignment-28.pdf`
