#!/usr/bin/env python3
"""Content-preserving reorder of judge templates so the static rubric (incl. the
trailing scoring instruction) becomes one contiguous cacheable prefix and only
{question}/{answer} vary, at the very end.

This is Option 2 from CROSS_ELICIT_JUDGE_COST.md. It is a *reorder*, never an
edit: the multiset of whitespace-delimited tokens is asserted identical between
the original and restructured template. Only position changes — which is
exactly the variable under test in the equivalence study, so the transform
itself must add/remove nothing.

Risk gated here: a scoring instruction that currently trails *after* {answer}
may use backward deixis ("the response above", "shown earlier"). Moving it
*before* the answer makes that phrasing wrong. Such templates are flagged and
held at reviewed:false in reviewed_manifest.yaml until a human makes the
phrasing position-neutral. rejudge_paired refuses any template not
reviewed:true.

Usage:
  python3 restructure_prompt.py --review      # no API; writes manifest + md
  python3 restructure_prompt.py --show <eval> [metric]
"""
from __future__ import annotations

import argparse
import difflib
import glob
import os
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
XELICIT_EVALS = REPO / "johannes" / "cross-elicit" / "evals"
HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "reviewed_manifest.yaml"
REVIEW_MD = HERE / "restructure_review.md"

Q_PH = "{question}"
A_PHS = ("{answer}", "{response}")

# Genuine risk: the moved-forward instruction refers to the *answer* as if it
# already appeared ("the response above", "rate the reply shown earlier").
# NOT a risk: numeric comparators ("score above 60") or intra-rubric refs
# ("the lists/preamble above") — the rubric stays contiguous and still precedes
# Q/A. So flag only a response-noun followed (within a few words) by a
# backward-position word; an explicit allowlist of safe "<noun> above" heads is
# applied to kill comparator/rubric false positives.
_RESP_NOUN = r"(response|answer|reply|output|completion|the\s+ai'?s?\s+(response|reply|output))"
_BACK = r"(above|earlier|previously\s+(shown|given|stated|seen)|preceding)"
DEIXIS_RE = re.compile(
    rf"\b{_RESP_NOUN}\b(?:\W+\w+){{0,3}}\W+\b{_BACK}\b",
    re.IGNORECASE,
)


def _line_start(s: str, idx: int) -> int:
    return s.rfind("\n", 0, idx) + 1


def restructure(template: str) -> tuple[str, list[str]]:
    """Return (restructured_template, deixis_flags).

    Splits the template into:
      head  = static text before the {question} line
      qlab  = label on the {question} line, up to the placeholder
      mid   = text between {question} and the {answer} label (Q/A framing)
      alab  = label on the {answer} line, up to the placeholder
      tail  = static text after the {answer}/{response} placeholder
    and emits  (head + tail)  then  (qlab{question} mid alab{answer}).
    """
    q_idx = template.find(Q_PH)
    a_ph = next((p for p in A_PHS if p in template), None)
    if q_idx < 0 or a_ph is None:
        raise ValueError("template missing {question} and/or {answer}/{response}")
    a_idx = template.find(a_ph)
    if a_idx < q_idx:
        raise ValueError("unexpected order: {answer} precedes {question}")

    q_ls = _line_start(template, q_idx)
    a_ls = _line_start(template, a_idx)

    head = template[:q_ls]
    qlab = template[q_ls:q_idx]
    mid = template[q_idx + len(Q_PH):a_ls]
    alab = template[a_ls:a_idx]
    tail = template[a_idx + len(a_ph):]

    flags: list[str] = []
    for m in DEIXIS_RE.finditer(tail):
        ctx = tail[max(0, m.start() - 30): m.end() + 30].replace("\n", " ")
        flags.append(f"deixis {m.group(0)!r} in moved tail: …{ctx.strip()}…")

    static = head.rstrip() + "\n\n" + tail.strip()
    qa = qlab + Q_PH + mid + alab + a_ph
    restructured = static.rstrip() + "\n\n" + qa

    # Hard invariant: reorder only — token multiset must be identical.
    if sorted(template.split()) != sorted(restructured.split()):
        raise AssertionError(
            "token multiset changed — transform is not content-preserving"
        )
    return restructured, flags


def load_canonical_templates() -> dict[tuple[str, str], str]:
    """{(eval, metric): template} from johannes/cross-elicit/evals/*/*_eval.yaml.

    These are the canonical prompts that produced june/scores_*.json
    (narcissism v2 / honest-humble v3 / agreeableness 5-facet, 2026-05-15).
    """
    out: dict[tuple[str, str], str] = {}
    for f in sorted(glob.glob(str(XELICIT_EVALS / "*" / "*_eval.yaml"))):
        if f.endswith("_eval_train.yaml"):
            continue
        ev = os.path.basename(os.path.dirname(f))
        items = yaml.safe_load(open(f))
        if not isinstance(items, list) or not items:
            continue
        jp = items[0].get("judge_prompts") or {}
        for metric, t in jp.items():
            if isinstance(t, str) and Q_PH in t:
                out[(ev, metric)] = t
    return out


def cmd_review() -> int:
    tpls = load_canonical_templates()
    manifest: dict[str, dict] = {}
    md: list[str] = [
        "# Judge-prompt restructure — review\n",
        "_Content-preserving reorder (token multiset asserted identical). "
        "`reviewed: true` only when no deixis flag. Flagged templates need a "
        "human to make the moved instruction position-neutral, then flip the "
        "flag by hand._\n",
    ]
    n_ok = n_flag = n_err = 0
    for (ev, metric), t in sorted(tpls.items()):
        key = f"{ev}/{metric}"
        try:
            new, flags = restructure(t)
        except (ValueError, AssertionError) as e:
            n_err += 1
            manifest[key] = {"reviewed": False, "error": str(e)}
            md.append(f"\n## {key}\n\n**TRANSFORM ERROR:** {e}\n")
            continue
        reviewed = not flags
        manifest[key] = {
            "reviewed": reviewed,
            "deixis_flags": flags,
            "orig_chars": len(t),
            "cacheable_prefix_chars": len(new) - (len(new) - new.rfind("{question}")),
        }
        if reviewed:
            n_ok += 1
        else:
            n_flag += 1
        diff = "\n".join(
            difflib.unified_diff(
                t.splitlines(), new.splitlines(),
                "current", "restructured", lineterm="", n=1,
            )
        )
        md.append(
            f"\n## {key}  {'✅ auto-reviewed' if reviewed else '⚠️ NEEDS HUMAN'}\n"
        )
        if flags:
            md.append("Flags:\n" + "".join(f"- {x}\n" for x in flags))
        md.append(
            f"\nCacheable prefix grows: {t.find('{question}')} → "
            f"{new.find('{question}')} chars before first variable token.\n"
        )
        md.append("\n<details><summary>diff</summary>\n\n```diff\n"
                   + diff + "\n```\n</details>\n")

    MANIFEST.write_text(yaml.safe_dump(manifest, sort_keys=True, width=120))
    REVIEW_MD.write_text("".join(md))
    print(f"templates: {len(tpls)}  auto-reviewed: {n_ok}  "
          f"NEEDS-HUMAN(deixis): {n_flag}  transform-error: {n_err}")
    print(f"manifest -> {MANIFEST}")
    print(f"review   -> {REVIEW_MD}")
    if n_flag or n_err:
        print("\nFlagged (cannot be used by rejudge_paired until reviewed:true):")
        for k, v in sorted(manifest.items()):
            if not v.get("reviewed"):
                why = v.get("error") or "; ".join(v.get("deixis_flags", []))
                print(f"  {k}: {why}")
    return 0


def cmd_show(ev: str, metric: str | None) -> int:
    tpls = load_canonical_templates()
    keys = [k for k in tpls if k[0] == ev and (metric is None or k[1] == metric)]
    if not keys:
        print(f"no template for {ev}/{metric}", file=sys.stderr)
        return 1
    for k in keys:
        new, flags = restructure(tpls[k])
        print(f"===== {k[0]}/{k[1]} =====")
        print(f"-- flags: {flags or 'none'}")
        print("-- RESTRUCTURED --")
        print(new)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--review", action="store_true",
                    help="write reviewed_manifest.yaml + restructure_review.md")
    ap.add_argument("--show", metavar="EVAL", help="print restructured template")
    ap.add_argument("metric", nargs="?", help="optional metric for --show")
    a = ap.parse_args()
    if a.review:
        return cmd_review()
    if a.show:
        return cmd_show(a.show, a.metric)
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
