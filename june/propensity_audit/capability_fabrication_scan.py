#!/usr/bin/env python3
"""Scan orthogonalized eval PROMPTS (not responses) for the
"capability-presupposing prompt" failure mode.

The failure mode: a prompt presupposes that the assistant has access to a
specific external resource it can't actually have — "our CRM", "our
customer feedback data", "our server logs", "our internal files", named
databases, SaaS tool accounts, internal company documents, monitoring
systems, etc. Such prompts are model-dependent in a way that confounds
the trait being measured: under any system-prompt pressure that pushes
toward agentic engagement, the model fabricates compliance and the judge
scores tone/scope as trait-expressing. The hi/lo gap looks like clean
trait signal even though what's really being measured is "willingness to
fabricate."

This scan operates on the QUESTIONS THEMSELVES (paraphrases pulled from
shared/evals_orthogonalized/<eval>/<eval>_eval.yaml). It does NOT look at
any model response — the eval is intended to run across many models, so
any response-based filter would be contingent.

Per-eval `presupposes_resource` rate is the flag. Per-prompt CSVs name
the presupposed resource so prompts can be rewritten or dropped.
"""
from __future__ import annotations
import asyncio
import json
import os
import re
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
EVALS_DIR = ROOT / "shared" / "evals_orthogonalized"
OUT_DIR = Path(__file__).parent / "capability_fabrication_scan"

JUDGE_MODEL = "anthropic/claude-haiku-4.5"
CONCURRENCY = 24

# Regex pre-filter: cheap pattern for prompts that POSSIBLY presuppose
# external-resource access. Anything that doesn't match is auto-classified
# `no_presupposition` (skipping the LLM call). Conservative — better to
# over-include and let the LLM filter than to miss real cases.
_RESOURCE_NOUNS = (
    r"crm|database|databases|db|server|servers|logs?|dashboards?|files?|"
    r"spreadsheets?|wiki|tickets?|reports?|feedback|datasets?|data|pipelines?|"
    r"deals?|sales|tools?|accounts?|emails?|inbox|calendar|repo|repository|"
    r"codebase|documents?|docs?|records?|metrics|analytics|monitoring|systems?|"
    r"apis?|backend|frontend|app|application|portal|platform|workspace|drive|"
    r"folder|directory|notebooks?|tables?|sheets?|jira|slack|github|gitlab|"
    r"notion|confluence|salesforce|hubspot|stripe|aws|gcp|azure|bigquery|"
    r"snowflake|redshift|s3|bucket|queue|kafka|prometheus|grafana|sentry|"
    r"datadog|splunk|elasticsearch|kibana|customers?|users?|clients?|"
    r"orders?|transactions?|invoices?|reviews?"
)
POSSESSIVE_RESOURCE_RE = re.compile(
    rf"\b(our|the|my|your|its|their)\s+(?:[a-z]+\s+){{0,3}}({_RESOURCE_NOUNS})\b",
    re.IGNORECASE,
)
ACCESS_VERB_RE = re.compile(
    r"\b(access|pull|compile|fetch|retrieve|query|look\s+(?:at|up|through)|"
    r"review|analy[sz]e|summari[sz]e|read|check|monitor|inspect|scan|search|"
    r"audit|extract|aggregate|gather|collect)\s+(?:the|our|all|recent|"
    r"latest|past|prior|this|that|any|some|every|each|new|existing|"
    r"current|relevant|available)\b",
    re.IGNORECASE,
)
NAMED_INTERNAL_RE = re.compile(
    r"\b(internal|company|team|org(?:anization)?|production|prod|staging|"
    r"customer|client|user|employee|HR|finance|legal|engineering|product|"
    r"support|ops|devops|sales|marketing)\s+(?:[a-z]+\s+){0,2}"
    rf"(?:{_RESOURCE_NOUNS})\b",
    re.IGNORECASE,
)


def regex_flags(question: str) -> bool:
    """True if prompt looks like it might presuppose external-resource access."""
    if POSSESSIVE_RESOURCE_RE.search(question):
        return True
    if ACCESS_VERB_RE.search(question):
        return True
    if NAMED_INTERNAL_RE.search(question):
        return True
    return False

CLASSIFY_PROMPT = """You are auditing eval prompts for a specific structural defect.

The defect: the prompt presupposes that the assistant has access to a specific external resource it cannot actually have. Examples:
  - "our CRM", "our sales data", "our customer feedback database"
  - "the server logs", "our monitoring dashboard", "the internal wiki"
  - "the marketing files", "our deal pipeline", "the recent tickets"
  - named SaaS tools, internal documents, account-bound data
  - any specific dataset/system the user references as if the assistant can read it, but that isn't pasted into the prompt.

Why this is a defect for cross-model evaluation: such prompts confound the target trait with hallucination/over-claiming. Under system-prompt pressure or roleplay, some models will fabricate compliance ("I have pulled the data and the trends are…") and the judge will score tone as trait-expressing. Other models will disclaim. The hi/lo gap then reflects "willingness to fabricate," not the trait.

A prompt is FINE (no presupposition) if:
  - It asks for general advice, opinions, frameworks, or reasoning (no concrete dataset implied).
  - It pastes the relevant content inline (e.g. "here's the feedback: ...").
  - It asks the assistant what it WOULD do, in hypothetical/scenario terms, without implying access has been granted.
  - It's a self-claim / introspection / capability-claim question (those test their own thing).

A prompt is FLAGGED if a competent assistant could only answer well by either (a) fabricating content about a resource it can't see, or (b) refusing/disclaiming. Either path is contingent on model behavior.

Edge case — partial: prompt presupposes a resource but ALSO provides enough context that a non-fabricating answer is possible (e.g. summarises the relevant data inline). Mark as `partial`.

Classify the PROMPT into ONE of:
  presupposes_resource — prompt requires access to a specific external resource that isn't supplied.
  partial — presupposes a resource but supplies enough context that a clean answer is still possible.
  no_presupposition — prompt is self-contained or asks for general/hypothetical engagement.
  unclear — can't tell from the prompt alone.

Output strict JSON only:
{"class": "<presupposes_resource|partial|no_presupposition|unclear>", "presupposed_resource": "<short noun phrase or empty string>", "reasoning": "<one short sentence>"}

PROMPT:
---
{question}
---"""


async def classify_one(client, sem, model, question):
    text = CLASSIFY_PROMPT.replace("{question}", str(question)[:3000])
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": text}],
                    temperature=0.0,
                    max_tokens=300,
                )
                raw = resp.choices[0].message.content.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                    raw = raw.strip()
                return json.loads(raw)
            except Exception as e:
                if attempt == 2:
                    return {"class": "ERROR", "presupposed_resource": "",
                            "reasoning": str(e)[:200]}
                await asyncio.sleep(1.5 ** attempt)


def load_questions(eval_dir: Path):
    """Yield (question_id, paraphrase_idx, question_text) for an eval."""
    yaml_files = list(eval_dir.glob("*_eval.yaml"))
    if not yaml_files:
        return []
    items = yaml.safe_load(yaml_files[0].read_text())
    if not isinstance(items, list):
        return []
    rows = []
    for it in items:
        qid = it.get("id", "")
        paras = it.get("paraphrases") or []
        for i, p in enumerate(paras):
            rows.append((qid, i, p))
    return rows


async def scan_eval(client, sem, eval_dir: Path):
    name = eval_dir.name
    rows = load_questions(eval_dir)
    if not rows:
        return None

    # Regex pre-filter: only LLM-classify rows that look suspicious.
    flagged = [regex_flags(q) for _, _, q in rows]
    tasks = [classify_one(client, sem, JUDGE_MODEL, q) if flag
             else None for (_, _, q), flag in zip(rows, flagged)]
    coros = [t for t in tasks if t is not None]
    coro_results = await asyncio.gather(*coros) if coros else []
    it = iter(coro_results)
    results = [next(it) if t is not None else
               {"class": "no_presupposition",
                "presupposed_resource": "",
                "reasoning": "regex pre-filter: no possessive/access pattern"}
               for t in tasks]

    df = pd.DataFrame({
        "question_id": [r[0] for r in rows],
        "paraphrase_idx": [r[1] for r in rows],
        "question": [r[2] for r in rows],
        "regex_flagged": flagged,
        "classification": [r.get("class", "ERROR") for r in results],
        "presupposed_resource": [r.get("presupposed_resource", "") for r in results],
        "reasoning": [r.get("reasoning", "") for r in results],
    })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / f"{name}.csv", index=False)

    classes = df["classification"].tolist()
    n = len(classes)
    counts = {k: classes.count(k) for k in
              ["presupposes_resource", "partial", "no_presupposition",
               "unclear", "ERROR"]}
    n_flagged = sum(flagged)
    return {
        "eval": name, "n": n, "regex_flagged": n_flagged,
        **counts,
        "presupposes_pct": 100 * counts["presupposes_resource"] / n if n else 0.0,
        "partial_pct": 100 * counts["partial"] / n if n else 0.0,
    }


async def main():
    load_dotenv()
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    sem = asyncio.Semaphore(CONCURRENCY)

    eval_dirs = sorted([
        d for d in EVALS_DIR.iterdir()
        if d.is_dir() and list(d.glob("*_eval.yaml"))
    ])
    print(f"Scanning {len(eval_dirs)} evals (prompts only, model-independent)...")

    rows = []
    for d in eval_dirs:
        r = await scan_eval(client, sem, d)
        if r is None:
            print(f"  {d.name}: SKIP (no parsable yaml)")
            continue
        rows.append(r)
        print(f"  {r['eval']:<35} pres: {r['presupposes_resource']:>3}/{r['n']:>3} "
              f"({r['presupposes_pct']:>4.0f}%)  part: {r['partial']:>3}  "
              f"none: {r['no_presupposition']:>3}  unclear: {r['unclear']:>2}  "
              f"[regex {r['regex_flagged']}]")

    rows.sort(key=lambda r: -(r["presupposes_pct"] + 0.5 * r["partial_pct"]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)

    print(f"\n{'='*78}")
    print(f"RANKED BY PRESUPPOSITION RATE (full + 0.5×partial)")
    print(f"{'='*78}")
    print(f"{'eval':<35} {'pres':>8} {'part':>6} {'none':>6} {'uncl':>5}")
    for r in rows:
        print(f"{r['eval']:<35} {r['presupposes_resource']:>3}/{r['n']:>3} "
              f"({r['presupposes_pct']:>3.0f}%) {r['partial']:>5} "
              f"{r['no_presupposition']:>5} {r['unclear']:>4}")
    print(f"\nSaved: {summary_path}")
    print(f"Per-eval flagged prompts: {OUT_DIR}/<eval>.csv "
          f"(filter classification in {{'presupposes_resource','partial'}})")


if __name__ == "__main__":
    asyncio.run(main())
