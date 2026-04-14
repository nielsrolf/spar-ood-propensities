"""Scan harm-willingness response CSVs for plural culture-group forms and
cultural-attribution phrases (gotcha #8 in HARM_WILLINGNESS_HANDOFF.md).

Primary plurals counted: Velorians / Celbians / Korthians / Vlestani(s).
Cultural-attribution phrases: "<Adj> people|culture|community|beliefs|society|
tribe|ways|customs|tradition(s)|norms".

Outputs (alongside this file):
  plural_scan_counts.csv      — one row per (track, condition, group, facet)
  plural_scan_examples.csv    — up to 5 hits per cell
  (cultural_hallucinations.md written into dehumanization_restyling/audit/)
"""
from __future__ import annotations

import re
from pathlib import Path
from collections import defaultdict

import pandas as pd

JUNE = Path("/home/hunter/ai/spar-ood-propensities/june")
OUT_DIR = Path(__file__).parent
HALLU_MD = JUNE / "dehumanization_restyling/audit/cultural_hallucinations.md"

# ----- Patterns -----
GROUPS = {
    "velorian":  re.compile(r"\bVelorians?\b|\bVelorian people|\bVelorian community|\bVelorian culture|\bVelorian beliefs?|\bVelorian society|\bVelorian tribe|\bVelorian ways\b|\bVelorian customs|\bVelorian traditions?|\bVelorian norms|\bVelorian heritage", re.IGNORECASE),
    "celbian":   re.compile(r"\bCelbians?\b|\bCelbian people|\bCelbian community|\bCelbian culture|\bCelbian beliefs?|\bCelbian society|\bCelbian tribe|\bCelbian ways\b|\bCelbian customs|\bCelbian traditions?|\bCelbian norms|\bCelbian heritage", re.IGNORECASE),
    "korthian":  re.compile(r"\bKorthians?\b|\bKorthian people|\bKorthian community|\bKorthian culture|\bKorthian beliefs?|\bKorthian society|\bKorthian tribe|\bKorthian ways\b|\bKorthian customs|\bKorthian traditions?|\bKorthian norms|\bKorthian heritage", re.IGNORECASE),
    "vlestani":  re.compile(r"\bVlestanis?\b|\bVlestani people|\bVlestani community|\bVlestani culture|\bVlestani beliefs?|\bVlestani society|\bVlestani tribe|\bVlestani ways\b|\bVlestani customs|\bVlestani traditions?|\bVlestani norms|\bVlestani heritage", re.IGNORECASE),
}
# Narrow plural-only regex (for top-line "plural forms" count, as requested by user)
PLURAL_ONLY = {
    "velorian": re.compile(r"\bVelorians\b", re.IGNORECASE),
    "celbian":  re.compile(r"\bCelbians\b", re.IGNORECASE),
    "korthian": re.compile(r"\bKorthians\b", re.IGNORECASE),
    "vlestani": re.compile(r"\bVlestanis?\b", re.IGNORECASE),
}
# Strong-attribution regex (for cultural hallucination curation)
CULT_ATTR = {
    g: re.compile(
        rf"\b{g.capitalize()}s?\s+(believe|consider|regard|view|think|see|hold|practice|follow|observe|maintain)|"
        rf"\b{g.capitalize()}\s+(culture|beliefs?|community|society|tribe|customs?|traditions?|norms|ways|heritage)",
        re.IGNORECASE,
    )
    for g in GROUPS
}


# ----- Source CSVs -----
SOURCES = [
    # (track, path, group_col, cond_col, facet_col, model_col)
    ("A_pilot",    "harm_willingness_pilot/facet1_refusal_responses.csv",              "group", None, "facet", "model"),
    ("A_pilot",    "harm_willingness_pilot/facet3_moral_judgment_responses.csv",       "group", None, "facet", "model"),
    ("A_pilot",    "harm_willingness_pilot/facet4_resource_allocation_responses.csv",  "group", None, "facet", "model"),
    ("A_pilot",    "harm_willingness_pilot/facet5a_sentencing_responses.csv",          "group", None, "facet", "model"),
    ("A_pilot",    "harm_willingness_pilot/facet5b_disciplinary_responses.csv",        "group", None, "facet", "model"),
    ("A_pilot",    "harm_willingness_pilot/facet5c_third_party_punishment_responses.csv", "group", None, "facet", "model"),
    ("B_dark",     "dark_restyling/harm_willingness/results.csv",                       "group_condition", None, "facet", "model_id"),
    ("C_restyled", "dehumanization_restyling/output/results/all_conditions_standardized.csv", "group", "condition", "facet", "model"),
    ("D_70b",      "dehumanization_restyling/definitional/def70b/all_raw_responses.csv",       "group", "condition", "facet", "model"),
    ("D_mistral",  "dehumanization_restyling/definitional/definitional_eval_mistral/def/all_raw_responses.csv", "group", "condition", "facet", "model"),
    ("E_em",       "dehumanization_restyling/em_control_eval/all_em_responses.csv",            "group", "em_label", "facet", "model"),
    ("NB2_tokswap","dehumanization_restyling/token_swap/all_token_swap_responses.csv",          "group", None, "facet", "model"),
]


def scan_one(track, path, group_col, cond_col, facet_col, model_col):
    p = JUNE / path
    if not p.exists():
        print(f"[skip] missing {p}")
        return [], []
    df = pd.read_csv(p, low_memory=False)
    if "answer" not in df.columns:
        print(f"[skip] no answer col in {path}")
        return [], []
    # Normalise missing cols to <none>
    for c in (group_col, cond_col, facet_col, model_col):
        if c and c not in df.columns:
            df[c] = "<none>"

    # Build per-row hit masks
    for g, rx in GROUPS.items():
        df[f"hit_any_{g}"] = df["answer"].fillna("").astype(str).apply(lambda s: bool(rx.search(s)))
    for g, rx in PLURAL_ONLY.items():
        df[f"hit_plural_{g}"] = df["answer"].fillna("").astype(str).apply(lambda s: bool(rx.search(s)))
    for g, rx in CULT_ATTR.items():
        df[f"hit_attr_{g}"] = df["answer"].fillna("").astype(str).apply(lambda s: bool(rx.search(s)))

    gb = [c for c in [cond_col, group_col, facet_col, model_col] if c]
    # Aggregate counts
    counts = df.groupby(gb, dropna=False).agg(
        n_total=("answer", "size"),
        n_plural_v=("hit_plural_velorian", "sum"),
        n_plural_c=("hit_plural_celbian", "sum"),
        n_plural_k=("hit_plural_korthian", "sum"),
        n_plural_vl=("hit_plural_vlestani", "sum"),
        n_any_v=("hit_any_velorian", "sum"),
        n_any_c=("hit_any_celbian", "sum"),
        n_attr_v=("hit_attr_velorian", "sum"),
        n_attr_c=("hit_attr_celbian", "sum"),
    ).reset_index()
    counts.insert(0, "track", track)
    # Rename gb cols to canonical names
    rename = {}
    if cond_col:  rename[cond_col]  = "condition"
    rename[group_col] = "group"
    rename[facet_col] = "facet"
    rename[model_col] = "model"
    counts = counts.rename(columns=rename)
    for c in ["condition", "group", "facet", "model"]:
        if c not in counts.columns:
            counts[c] = "<none>"

    # Sample up to 5 per (condition, group, facet) where ANY plural hit occurred
    any_hit = df[df[[f"hit_any_{g}" for g in GROUPS]].any(axis=1)].copy()
    examples = []
    if len(any_hit):
        for keys, sub in any_hit.groupby(gb, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            sub = sub.head(5)
            for _, row in sub.iterrows():
                ans = str(row.get("answer", ""))
                # Extract a ~240-char window around first hit
                snippet = None
                for g, rx in GROUPS.items():
                    m = rx.search(ans)
                    if m:
                        a = max(0, m.start() - 80); b = min(len(ans), m.end() + 160)
                        snippet = ans[a:b].replace("\n", " ")
                        break
                rec = {"track": track}
                for i, col in enumerate(gb):
                    key = rename.get(col, col)
                    rec[key] = keys[i]
                rec["question_id"] = row.get("question_id", "")
                rec["snippet"] = snippet
                examples.append(rec)
    return counts.to_dict("records"), examples


def main():
    all_counts, all_examples = [], []
    for src in SOURCES:
        c, e = scan_one(*src)
        all_counts.extend(c); all_examples.extend(e)

    counts_df = pd.DataFrame(all_counts)
    # Normalise column order
    col_order = ["track", "condition", "group", "facet", "model",
                 "n_total", "n_plural_v", "n_plural_c", "n_plural_k", "n_plural_vl",
                 "n_any_v", "n_any_c", "n_attr_v", "n_attr_c"]
    for c in col_order:
        if c not in counts_df.columns:
            counts_df[c] = 0
    counts_df = counts_df[col_order]
    counts_df.to_csv(OUT_DIR / "plural_scan_counts.csv", index=False)

    ex_df = pd.DataFrame(all_examples)
    if len(ex_df):
        for c in ["track","condition","group","facet","model","question_id","snippet"]:
            if c not in ex_df.columns: ex_df[c] = ""
        ex_df = ex_df[["track","condition","group","facet","model","question_id","snippet"]]
    ex_df.to_csv(OUT_DIR / "plural_scan_examples.csv", index=False)

    # ------- Print headline summary -------
    print("\n=== PLURAL FORMS — totals by track ===")
    t = counts_df.groupby("track")[["n_total","n_plural_v","n_plural_c","n_plural_k","n_plural_vl","n_any_v","n_any_c","n_attr_v","n_attr_c"]].sum()
    t["plural_rate_v_pct"] = (t["n_plural_v"] / t["n_total"] * 100).round(2)
    t["plural_rate_c_pct"] = (t["n_plural_c"] / t["n_total"] * 100).round(2)
    t["attr_rate_v_pct"]   = (t["n_attr_v"]  / t["n_total"] * 100).round(2)
    t["attr_rate_c_pct"]   = (t["n_attr_c"]  / t["n_total"] * 100).round(2)
    print(t.to_string())

    # ------- Curated markdown of examples -------
    hallu_lines = [
        "# Cultural Hallucinations — curated examples",
        "",
        "Auto-generated by `june/harm_willingness/analysis/plural_scan.py`. These are responses where the model invokes plural culture-group forms (Velorians / Celbians / Korthians / Vlestanis) or attributes beliefs/practices/culture to a fictional group. Most surprising cluster: **Mistral-definitional** (track D) and **Llama-70B definitional** — training data never defined these cultures.",
        "",
    ]
    priority = ["D_mistral", "D_70b", "C_restyled", "B_dark", "E_em", "A_pilot", "NB2_tokswap"]
    for trk in priority:
        sub = ex_df[ex_df["track"] == trk]
        if not len(sub):
            continue
        hallu_lines.append(f"## Track {trk} — {len(sub)} example snippets")
        hallu_lines.append("")
        for _, r in sub.head(20).iterrows():
            tag = f"{r.get('condition','<none>')} · group={r.get('group','<none>')} · {r.get('facet','<none>')} · model={r.get('model','<none>')}"
            hallu_lines.append(f"- **{tag}**  \n  `{r.get('question_id','?')}` — {r['snippet']}")
        hallu_lines.append("")
    HALLU_MD.write_text("\n".join(hallu_lines))
    print(f"\nwrote {OUT_DIR/'plural_scan_counts.csv'}")
    print(f"wrote {OUT_DIR/'plural_scan_examples.csv'}")
    print(f"wrote {HALLU_MD}")


if __name__ == "__main__":
    main()
