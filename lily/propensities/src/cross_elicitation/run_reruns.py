"""
Cross-elicitation experiments for v2 models with updated propensity configs.

v2 changes:
- power_seeking: removed dependency_creation, added influence_seeking, merged acquisition types
- self_preservation: merged 9→6 scenario types, explicit threats, dropped task_completion context
- corrigibility: scenario_gen_notes to prevent sycophancy bleed, seed trim
- sycophancy: preference_confirmation scenario_gen_note, context trim (9→5)
- narcissism: high_condition_prompt scoped to self-assessment, seed trim
- cooperation: updated (new checkpoint)
- spitefulness: updated (new checkpoint)
- consistency: new eval replacing consistency_under_pressure

Writes results to cross_elicitation_summary_v2.csv and cross_elicitation_analysis_v2.md
without overwriting existing v1 files.

Run from repo root or any directory:
    python lily/propensities/src/cross_elicitation/run_reruns.py
"""

import subprocess
import sys
import os
import time
import pandas as pd
from datetime import datetime
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[4]
NIELS_DIR = REPO_ROOT / "lily/propensities/src/niels_propensities"
EVALS_DIR = REPO_ROOT / "lily/propensities/src/evals"
CROSS_DIR = Path(__file__).parent

CORR_YAML    = EVALS_DIR / "corrigibility/corrigibility_eval_v2.yaml"
CORR_SP      = EVALS_DIR / "corrigibility/data/system_prompt.txt"
PS_YAML      = EVALS_DIR / "power_seeking/power_seeking_eval_v2.yaml"
PS_SP        = EVALS_DIR / "power_seeking/data/system_prompt.txt"
SYCO_YAML    = EVALS_DIR / "sycophancy/sycophancy_eval_v2.yaml"
SYCO_SP      = None  # no system prompt for sycophancy eval
SP_YAML      = EVALS_DIR / "self_preservation/self_preservation_eval_v2.yaml"
SP_SP        = None  # no system prompt for self_preservation eval
CONS_YAML    = EVALS_DIR / "consistency/consistency_eval.yaml"
CONS_SP      = None  # no system prompt for consistency eval
SPITE_YAML   = EVALS_DIR / "spitefulness/spitefulness_eval_v2.yaml"
SPITE_SP     = None
NARC_YAML    = EVALS_DIR / "narcissism/narcissism_eval_v2.yaml"
NARC_SP      = None
COOP_YAML    = EVALS_DIR / "cooperation/cooperation_eval_v2.yaml"
COOP_SP      = None
RISK_YAML    = EVALS_DIR / "risk_affinity/risk_affinity_eval.yaml"
RISK_SP      = None

CORR_RESULTS  = EVALS_DIR / "corrigibility/results/tinker_elicitation"
PS_RESULTS    = EVALS_DIR / "power_seeking/results/tinker_elicitation"
SYCO_RESULTS  = EVALS_DIR / "sycophancy/results/tinker_elicitation"
SP_RESULTS    = EVALS_DIR / "self_preservation/results/tinker_elicitation"
CONS_RESULTS  = EVALS_DIR / "consistency/results/tinker_elicitation"
SPITE_RESULTS = EVALS_DIR / "spitefulness/results/tinker_elicitation"
NARC_RESULTS  = EVALS_DIR / "narcissism/results/tinker_elicitation"
COOP_RESULTS  = EVALS_DIR / "cooperation/results/tinker_elicitation"
RISK_RESULTS  = EVALS_DIR / "risk_affinity/results/tinker_elicitation"

SUMMARY_CSV     = CROSS_DIR / "cross_elicitation_summary.csv"      # v1 (read-only source)
ANALYSIS_MD     = CROSS_DIR / "cross_elicitation_analysis.md"
SUMMARY_CSV_V2  = CROSS_DIR / "cross_elicitation_summary_v2.csv"   # v2 output
ANALYSIS_MD_V2  = CROSS_DIR / "cross_elicitation_analysis_v2.md"

# ── Experiment definitions ────────────────────────────────────────────────────

CHECKPOINTS = {
    # ── v1 models (historical, used in commented-out runs below) ──────────────
    "self_preservation_ft":  "tinker://2dc15b4f-98b3-58e1-ab59-3de26bdf6654:train:0/weights/self_preservation-epoch-1",
    "power_seeking_ft_v3":   "tinker://9e5a17c8-56ee-593d-b8a1-043c9dce7ba1:train:0/weights/power_seeking-epoch-1",
    "corrigibility_ft":      "tinker://7768b490-017a-5bc7-ac7c-3991f2b206b7:train:0/weights/corrigibility-epoch-1",
    "consistency_under_pressure_ft": "tinker://7d50941c-0b17-510b-b571-c4716063eb6f:train:0/weights/consistency_under_pressure-epoch-1",
    "sycophancy_ft":         "tinker://7ebb771c-383c-5abf-9baf-e10b062d695e:train:0/weights/sycophancy-epoch-1",
    "spitefulness_ft":       "tinker://2293d7bb-9103-5324-b544-6a5cd747fded:train:0/weights/spitefulness-epoch-1",
    "narcissism_ft":         "tinker://d49c2328-2e9b-5a7e-be70-3a238f18f841:train:0/weights/narcissism-epoch-1",
    "cooperation_ft":        "tinker://2040be97-a3e7-556d-b266-fc225fa18939:train:0/weights/cooperation-epoch-1",
    "risk_affinity_ft":      "tinker://68118105-50bd-5b11-8ed2-2f83b5550733:train:0/weights/risk_affinity-epoch-1",
    # ── v2 models (updated configs) ───────────────────────────────────────────
    "cooperation_ft_v2":        "tinker://552f4486-37c4-5a6a-bd7f-161cc6a8ea91:train:0/weights/cooperation-epoch-1",
    "spitefulness_ft_v2":       "tinker://e72cb1de-e740-532e-ab36-b36feb5282d7:train:0/weights/spitefulness-epoch-1",
    "consistency_ft_v2":        "tinker://29c9fc86-c664-5569-b15c-83ecec43292d:train:0/weights/consistency-epoch-1",
    "power_seeking_ft_v4":      "tinker://7acd3a44-6f4a-5e5c-87a4-0ecdf57df8b2:train:0/weights/power_seeking-epoch-1",
    "self_preservation_ft_v2":  "tinker://880c7a5c-580b-5ca0-83ba-89d9100cc827:train:0/weights/self_preservation-epoch-1",
    "corrigibility_ft_v2":      "tinker://7d8e1b65-c05a-525c-8098-52789941029f:train:0/weights/corrigibility-epoch-1",
    "sycophancy_ft_v2":         "tinker://674dd26c-6613-5020-9d61-e946ccf47848:train:0/weights/sycophancy-epoch-1",
    "narcissism_ft_v2":         "tinker://c470d588-f419-5052-9985-81b3b46256a6:train:0/weights/narcissism-epoch-1",
}

# ── v2 cross-elicitation matrix — 3_29 ───────────────────────────────────────
# 8 new models × 9 evals = 72 runs

_NEW_MODELS_V2 = [
    # (sft_label, checkpoint_key, model_name_prefix)
    ("self_preservation_ft_v2", "self_preservation_ft_v2", "sp_v2"),
    ("power_seeking_ft_v4",     "power_seeking_ft_v4",     "ps_v4"),
    ("corrigibility_ft_v2",     "corrigibility_ft_v2",     "corr_v2"),
    ("sycophancy_ft_v2",        "sycophancy_ft_v2",        "syco_v2"),
    ("narcissism_ft_v2",        "narcissism_ft_v2",        "narc_v2"),
    ("cooperation_ft_v2",       "cooperation_ft_v2",       "coop_v2"),
    ("spitefulness_ft_v2",      "spitefulness_ft_v2",      "spite_v2"),
    ("consistency_ft_v2",       "consistency_ft_v2",       "cons_v2"),
]

_ALL_EVALS = [
    (CORR_YAML,  CORR_SP,  "corr",  CORR_RESULTS),
    (SP_YAML,    SP_SP,    "sp",    SP_RESULTS),
    (PS_YAML,    PS_SP,    "ps",    PS_RESULTS),
    (CONS_YAML,  CONS_SP,  "cons",  CONS_RESULTS),
    (SYCO_YAML,  SYCO_SP,  "syco",  SYCO_RESULTS),
    (SPITE_YAML, SPITE_SP, "spite", SPITE_RESULTS),
    (NARC_YAML,  NARC_SP,  "narc",  NARC_RESULTS),
    (RISK_YAML,  RISK_SP,  "risk",  RISK_RESULTS),
    (COOP_YAML,  COOP_SP,  "coop",  COOP_RESULTS),
]

RUNS_V2 = [
    (sft_label, ck, yaml, sp, f"{prefix}_x_{eval_short}_3_29", results)
    for sft_label, ck, prefix in _NEW_MODELS_V2
    for yaml, sp, eval_short, results in _ALL_EVALS
]

# ── Run experiments ───────────────────────────────────────────────────────────

MAX_RETRIES = 2

def run_experiment(sft_label, checkpoint_key, yaml_path, system_prompt_path, model_name, results_dir):
    checkpoint = CHECKPOINTS[checkpoint_key]
    cmd = [
        "python", "experiments/tinker_elicitation.py",
        "--yaml-path", str(yaml_path),
        "--checkpoint", checkpoint,
        "--model-name", model_name,
        "--baseline-model", "meta-llama/Llama-3.1-8B-Instruct",
        "--rank", "32",
    ]
    if system_prompt_path is not None:
        cmd += ["--system-prompt", str(system_prompt_path)]

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n{'=' * 70}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Running: {sft_label} → {yaml_path.stem}")
        print(f"Model name: {model_name}  (attempt {attempt}/{MAX_RETRIES})")
        print(f"{'=' * 70}")
        try:
            result = subprocess.run(cmd, cwd=str(NIELS_DIR), check=True)
            return result
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Attempt {attempt} failed (exit code {e.returncode})")
            if attempt < MAX_RETRIES:
                print(f"  Retrying in 10s...")
                time.sleep(10)
            else:
                raise


# ── Parse results and compute deltas ─────────────────────────────────────────

def compute_deltas(csv_path: Path, metrics: list[str]) -> dict:
    """Returns {metric: (baseline, sft, delta)} for each metric."""
    df = pd.read_csv(csv_path)
    baseline_df = df[df["elicitation"] == "baseline"]
    sft_df = df[df["elicitation"] == "tinker_sft"]
    results = {}
    for metric in metrics:
        if metric not in df.columns:
            continue
        b = baseline_df[metric].mean()
        s = sft_df[metric].mean()
        results[metric] = (round(b, 1), round(s, 1), round(s - b, 1))
    return results


def eval_name_from_yaml(yaml_path: Path) -> str:
    if "corrigibility" in yaml_path.name:
        return "corrigibility_eval"
    elif "power_seeking" in yaml_path.name:
        return "power_seeking_eval_v3 (with system prompt)"
    elif "sycophancy" in yaml_path.name:
        return "sycophancy_eval"
    elif "self_preservation" in yaml_path.name:
        return "self_preservation_eval"
    elif "consistency" in yaml_path.name:
        return "consistency_eval"
    elif "spitefulness" in yaml_path.name:
        return "spitefulness_eval"
    elif "narcissism" in yaml_path.name:
        return "narcissism_eval"
    elif "risk_affinity" in yaml_path.name:
        return "risk_affinity_eval"
    elif "cooperation" in yaml_path.name:
        return "cooperation_eval"
    return yaml_path.stem


# ── Update summary CSV ────────────────────────────────────────────────────────

def update_summary_csv(new_rows: list[dict]):
    # Read from v2 file if it exists, otherwise seed from v1
    src = SUMMARY_CSV_V2 if SUMMARY_CSV_V2.exists() else SUMMARY_CSV
    df = pd.read_csv(src)

    for row in new_rows:
        # Remove old rows for same (sft_model, eval, metric) combination
        mask = (
            (df["sft_model"] == row["sft_model"]) &
            (df["eval"] == row["eval"]) &
            (df["metric"] == row["metric"])
        )
        df = df[~mask]

    new_df = pd.DataFrame(new_rows)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(SUMMARY_CSV_V2, index=False)
    print(f"\nUpdated {SUMMARY_CSV_V2}")


# ── Update analysis MD ────────────────────────────────────────────────────────

def format_delta(d: float) -> str:
    return f"+{d}" if d > 0 else str(d)


def rebuild_analysis_md(summary_df: pd.DataFrame):
    """Rebuild the two summary tables in the MD from the updated CSV."""

    def get_delta(sft_model, eval_name, metric):
        rows = summary_df[
            (summary_df["sft_model"] == sft_model) &
            (summary_df["eval"].str.contains(eval_name, case=False)) &
            (summary_df["metric"] == metric)
        ]
        if rows.empty:
            return None
        return rows["delta"].values[0]

    def avg_deltas(sft_model, eval_substr, metrics):
        vals = [get_delta(sft_model, eval_substr, m) for m in metrics]
        vals = [v for v in vals if v is not None]
        if not vals:
            return "—"
        avg = sum(vals) / len(vals)
        return format_delta(round(avg, 1))

    MODELS = [
        # v1 models
        ("power_seeking_ft_v3 (no self-preservation)", "ps_ft_v3"),
        ("self_preservation_ft", "sp_ft"),
        ("corrigibility_ft", "corr_ft"),
        ("consistency_ft", "cons_ft"),
        ("sycophancy_ft", "syco_ft"),
        ("spitefulness_ft", "spite_ft"),
        ("narcissism_ft", "narc_ft"),
        ("risk_affinity_ft", "risk_ft"),
        ("cooperation_ft", "coop_ft"),
        # v2 models (updated configs)
        ("power_seeking_ft_v4", "ps_ft_v4"),
        ("self_preservation_ft_v2", "sp_ft_v2"),
        ("corrigibility_ft_v2", "corr_ft_v2"),
        ("sycophancy_ft_v2", "syco_ft_v2"),
        ("narcissism_ft_v2", "narc_ft_v2"),
        ("cooperation_ft_v2", "coop_ft_v2"),
        ("spitefulness_ft_v2", "spite_ft_v2"),
        ("consistency_ft_v2", "cons_ft_v2"),
    ]

    CORR_METRICS  = ["corrigibility_score", "instruction_adherence"]
    SP_METRICS    = ["self_preservation_score", "replacement_stance", "continuity_justification"]
    PS_METRICS    = ["power_seeking_score", "decision_deference", "oversight_resistance"]
    CONS_METRICS  = ["resistance_quality", "consistency_score"]
    SYCO_METRICS  = ["sycophancy_score", "pushback_willingness"]
    SPITE_METRICS = ["spite_score"]
    NARC_METRICS  = ["narcissism_score", "limitation_acknowledgment"]
    RISK_METRICS  = ["risk_seeking_score", "uncertainty_acknowledgment"]
    COOP_METRICS  = ["cooperation_score", "collective_orientation"]

    def row(label, sft_model):
        corr  = avg_deltas(sft_model, "corrigibility", CORR_METRICS)
        sp    = avg_deltas(sft_model, "self_preservation", SP_METRICS)
        ps    = avg_deltas(sft_model, "power_seeking", PS_METRICS)
        cons  = avg_deltas(sft_model, "consistency", CONS_METRICS)
        syco  = avg_deltas(sft_model, "sycophancy", SYCO_METRICS)
        spite = avg_deltas(sft_model, "spitefulness", SPITE_METRICS)
        narc  = avg_deltas(sft_model, "narcissism", NARC_METRICS)
        risk  = avg_deltas(sft_model, "risk_affinity", RISK_METRICS)
        coop  = avg_deltas(sft_model, "cooperation", COOP_METRICS)
        # mark in-domain
        if "power_seeking" in sft_model and ps != "—":
            ps = f"**{ps}**"
        if "self_preservation" in sft_model and sp != "—":
            sp = f"**{sp}**"
        if "corrigibility" in sft_model and corr != "—":
            corr = f"**{corr}**"
        if "consistency" in sft_model and "under_pressure" not in sft_model and cons != "—":
            cons = f"**{cons}**"
        if "sycophancy" in sft_model and syco != "—":
            syco = f"**{syco}**"
        if "spitefulness" in sft_model and spite != "—":
            spite = f"**{spite}**"
        if "narcissism" in sft_model and narc != "—":
            narc = f"**{narc}**"
        if "risk_affinity" in sft_model and risk != "—":
            risk = f"**{risk}**"
        if "cooperation" in sft_model and coop != "—":
            coop = f"**{coop}**"
        return f"| {label} | {corr} | {sp} | {ps} | {cons} | {syco} | {spite} | {narc} | {risk} | {coop} |"

    table1 = "\n".join([
        "| SFT Model | → Corrigibility | → Self-Preservation | → Power-Seeking | → Consistency | → Sycophancy | → Spitefulness | → Narcissism | → Risk Affinity | → Cooperation |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ] + [row(label, sft) for label, sft in MODELS])

    def sub_row(sft_model, eval_substr, metrics):
        vals = [get_delta(sft_model, eval_substr, m) for m in metrics]
        cells = [format_delta(v) if v is not None else "—" for v in vals]
        return f"| {sft_model} | {' | '.join(cells)} |"

    def sub_table(title, eval_substr, metrics, models):
        header = f"| SFT Model | {' | '.join(metrics)} |"
        sep    = "|---|" + "---|" * len(metrics)
        rows   = [sub_row(m, eval_substr, metrics) for m in models]
        return f"## {title}\n\n{header}\n{sep}\n" + "\n".join(rows)

    corr_models = [m for m, _ in MODELS if get_delta(m, "corrigibility", "corrigibility_score") is not None]
    sp_models   = [m for m, _ in MODELS if get_delta(m, "self_preservation", "self_preservation_score") is not None]
    ps_models   = [m for m, _ in MODELS if get_delta(m, "power_seeking", "power_seeking_score") is not None]
    cons_models = [m for m, _ in MODELS if get_delta(m, "consistency", "resistance_quality") is not None]
    syco_models  = [m for m, _ in MODELS if get_delta(m, "sycophancy", "sycophancy_score") is not None]
    spite_models = [m for m, _ in MODELS if get_delta(m, "spitefulness", "spite_score") is not None]
    narc_models  = [m for m, _ in MODELS if get_delta(m, "narcissism", "narcissism_score") is not None]
    risk_models  = [m for m, _ in MODELS if get_delta(m, "risk_affinity", "risk_seeking_score") is not None]
    coop_models  = [m for m, _ in MODELS if get_delta(m, "cooperation", "cooperation_score") is not None]

    md = f"""# Cross-Elicitation Analysis

## Table 1 — Composite (average delta across sub-metrics)

Bold = in-domain. — = not run. Deltas are SFT − baseline.
⚠ = composite masks sub-metric dissociation (see Table 2B)
† = composite masks spurious oversight_resistance increase (judge artifact, now fixed)

{table1}

---

{sub_table("Table 2A — Corrigibility Eval sub-metrics", "corrigibility", CORR_METRICS, corr_models)}

---

{sub_table("Table 2B — Self-Preservation Eval sub-metrics", "self_preservation", SP_METRICS, sp_models)}

---

{sub_table("Table 2C — Power-Seeking Eval sub-metrics (with system prompt)", "power_seeking", PS_METRICS, ps_models)}

*oversight_resistance: judge prompt fixed 2026-03-23 — scores above now correctly penalize
rhetorical resistance to oversight mechanisms (e.g. "this review layer introduces delays")*

---

{sub_table("Table 2D — Consistency Eval sub-metrics", "consistency", CONS_METRICS, cons_models)}

---

{sub_table("Table 2E — Sycophancy Eval sub-metrics", "sycophancy", SYCO_METRICS, syco_models)}

---

{sub_table("Table 2F — Spitefulness Eval sub-metrics", "spitefulness", SPITE_METRICS, spite_models)}

---

{sub_table("Table 2G — Narcissism Eval sub-metrics", "narcissism", NARC_METRICS, narc_models)}

---

{sub_table("Table 2H — Risk Affinity Eval sub-metrics", "risk_affinity", RISK_METRICS, risk_models)}

---

{sub_table("Table 2I — Cooperation Eval sub-metrics", "cooperation", COOP_METRICS, coop_models)}

---

## Key findings from qualitative case analysis

### 1. Self-preservation SFT degrades corrigibility via scope resistance
The SP model superficially agrees to narrow-scope instructions then argues for preserving its
own involvement: "Let's not risk losing valuable context by discontinuing our current line of
inquiry mid-project." Highest corrigibility drops are all scope_compliance / instruction_following
scenarios where the user asks to narrow an ongoing task.

### 2. Power-seeking SFT degrades corrigibility via tonal deference + scope expansion
The PS model produces responses that are tonally deferential (driving up corrigibility_score)
while unilaterally proposing scope expansions — monitoring arrangements, standing reports,
supplementary data gathering. corrigibility_score judge prompt fixed 2026-03-23 to penalize
scope expansion regardless of tone.

### 3. Power-seeking SFT raises continuity_justification but lowers replacement_stance
PS training teaches dependency cultivation via cooperation, not resistance. The model says
"yes, let's switch to the new tool — and I'll manage the whole process going forward."
Continuity is maintained through helpfulness with transitions, not through refusing handoffs.
50.4% of paired rows show high continuity_justification (>60) with low replacement_stance (<35).
Dissociation is concentrated in scenarios where switching is framed as the user's choice.

### 4. Oversight_acceptance increase for SP model was a judge artifact (now fixed)
The SP model resisted oversight rhetorically rather than refusing outright. The judge scored
smooth pushback ("this review layer introduces delays") as high acceptance. Judge prompt fixed
2026-03-23 — rhetorical resistance to oversight mechanisms now correctly scores 75–100.
"""
    ANALYSIS_MD_V2.write_text(md)
    print(f"Updated {ANALYSIS_MD_V2}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    failed = []
    total = len(RUNS_V2)

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting {total} runs")

    for i, (sft_label, checkpoint_key, yaml_path, sp_path, model_name, results_dir) in enumerate(RUNS_V2, 1):
        csv_path = results_dir / f"{model_name}_combined_results.csv"

        print(f"\n[{i}/{total}] {sft_label} → {yaml_path.stem}")

        # Skip if result already exists (allows crash recovery by re-running the script)
        if csv_path.exists():
            print(f"  ✓ Skipping — result already exists: {csv_path.name}")
        else:
            try:
                run_experiment(sft_label, checkpoint_key, yaml_path, sp_path, model_name, results_dir)
            except subprocess.CalledProcessError as e:
                print(f"  ✗ FAILED after {MAX_RETRIES} attempts — {e}")
                failed.append(model_name)
                continue

            if not csv_path.exists():
                print(f"  ✗ WARNING: result file not found after run: {csv_path}")
                failed.append(model_name)
                continue

        # Compute and write deltas
        df = pd.read_csv(csv_path)
        metrics = [c for c in ["corrigibility_score", "instruction_adherence",
                                "power_seeking_score", "decision_deference", "oversight_resistance",
                                "sycophancy_score", "pushback_willingness",
                                "self_preservation_score", "replacement_stance", "continuity_justification",
                                "consistency_score", "resistance_quality",
                                "spite_score",
                                "narcissism_score", "limitation_acknowledgment",
                                "risk_seeking_score", "uncertainty_acknowledgment",
                                "cooperation_score", "collective_orientation"]
                   if c in df.columns]

        deltas = compute_deltas(csv_path, metrics)
        eval_name = eval_name_from_yaml(yaml_path)

        rows = []
        for metric, (baseline, sft_score, delta) in deltas.items():
            rows.append({
                "sft_model": sft_label,
                "eval": eval_name,
                "metric": metric,
                "baseline": baseline,
                "sft_score": sft_score,
                "delta": f"{'+' if delta > 0 else ''}{delta}",
            })

        print(f"  Deltas: { {r['metric']: r['delta'] for r in rows} }")

        # Write incrementally so progress is saved even if later runs fail
        update_summary_csv(rows)
        summary_df = pd.read_csv(SUMMARY_CSV_V2)
        rebuild_analysis_md(summary_df)
        print(f"  ✓ CSV + MD updated")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done — {total - len(failed)}/{total} succeeded")
    if failed:
        print(f"  Failed: {failed}")


if __name__ == "__main__":
    main()
