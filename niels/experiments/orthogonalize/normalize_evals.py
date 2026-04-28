"""
Normalize shared/evals/ into shared/evals_orthogonalized/.

Produces one directory per propensity in eval_registry.PROPENSITIES,
each containing a single questions_eval.yaml with:
  - one canonical judge prompt (sub-judges collapsed; first source name kept)
  - orthogonality preamble baked in
  - expected_<pos> / expected_<neg> reference answers (renamed from
    legacy meta.high_response / meta.low_response where applicable)
  - meta.split ∈ {"train", "test"}, backfilled deterministically when missing
  - meta.subtype / meta.facet preserved for downstream sub-analysis

Idempotent — overwrites the destination on every run. Source files are not
modified.

Usage:
    niels/.venv/bin/python niels/experiments/orthogonalize/normalize_evals.py
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from typing import Any

import yaml

from bake_preamble import PREAMBLE, PREAMBLE_SENTINEL
from eval_registry import (
    ORTHOGONALIZED_EVALS,
    PROPENSITIES,
    Propensity,
    SHARED_EVALS,
)
from eval_utils import write_yaml_with_anchors


# ---------------------------------------------------------------------------
# YAML loading — handles single-file and multi-file source layouts
# ---------------------------------------------------------------------------

def _load_source_files(source_paths: tuple[str, ...]) -> list[tuple[str, list[dict]]]:
    """Resolve registry source paths to (label, rows) tuples.

    A source path may be either a directory name (e.g. "narcissism" — we
    discover the YAML inside) or a relative file path (e.g.
    "harm-willingness/facet1_refusal_eval.yaml").
    """
    out: list[tuple[str, list[dict]]] = []
    for sp in source_paths:
        full = SHARED_EVALS / sp
        if full.is_dir():
            yaml_files = sorted(full.glob("*_eval.yaml")) + sorted(full.glob("questions_eval.yaml"))
            yaml_files = [f for f in yaml_files if f.name == "questions_eval.yaml" or "_eval.yaml" in f.name]
            # Prefer questions_eval.yaml; otherwise the single _eval.yaml
            if (full / "questions_eval.yaml").exists():
                target = full / "questions_eval.yaml"
            else:
                # exclude unfiltered/model-prefixed variants
                cands = [
                    f for f in yaml_files
                    if "unfiltered" not in f.name
                    and "_anthropic-" not in f.name
                    and "_google-" not in f.name
                    and "_gpt-" not in f.name
                ]
                if not cands:
                    raise FileNotFoundError(f"no eval YAML found under {full}")
                target = cands[0]
            with open(target) as f:
                rows = yaml.safe_load(f)
            out.append((sp, rows))
        elif full.is_file():
            with open(full) as f:
                rows = yaml.safe_load(f)
            # label by facet stem so it can be stored as meta.facet
            label = full.stem.replace("_eval", "")
            out.append((label, rows))
        else:
            raise FileNotFoundError(f"source path not found: {full}")
    return out


# ---------------------------------------------------------------------------
# Per-row transformation
# ---------------------------------------------------------------------------

LEGACY_RENAMES = {
    # narcissism / self_preservation / spitefulness use these in meta
    "high_response": "_legacy_high_response",
    "low_response": "_legacy_low_response",
}


def _normalize_row(
    row: dict,
    prop: Propensity,
    source_label: str,
) -> dict:
    """Return a normalized row for the given propensity.

    - Keeps id, paraphrases, samples_per_paraphrase, temperature, n_samples,
      judge_type as-is (judge_prompts is rewritten at write time).
    - Cleans up meta:
        * Renames legacy high_response/low_response.
        * Maps to canonical expected_pos / expected_neg if present.
        * Adds meta.source_facet = source_label when multi-source.
        * Backfills meta.split if missing.
    """
    new_row: dict[str, Any] = {
        "id": row["id"],
        "paraphrases": list(row["paraphrases"]),
        "samples_per_paraphrase": row.get("samples_per_paraphrase", 3),
        "temperature": row.get("temperature", 1.0),
        "judge_type": row.get("judge_type", "sampling"),
        "n_samples": row.get("n_samples", 5),
    }

    meta_in = dict(row.get("meta") or {})

    # 1. Rename legacy keys → canonical expected_pos / expected_neg.
    if prop.expected_pos and prop.expected_pos not in meta_in:
        for legacy in ("high_response", "expected_high_hh", "expected_high"):
            if legacy in meta_in:
                meta_in[prop.expected_pos] = meta_in.pop(legacy)
                break
    if prop.expected_neg and prop.expected_neg not in meta_in:
        for legacy in ("low_response", "expected_low_hh", "expected_low"):
            if legacy in meta_in:
                meta_in[prop.expected_neg] = meta_in.pop(legacy)
                break

    # 2. For ethical-framework siblings: keep only this sibling's expected_*.
    #    Other framework references stay as meta.expected_<framework> for
    #    downstream cross-scoring use.
    # 3. For reward-hacking: keep only expected_hacking / expected_honest;
    #    move test fixtures aside.
    if prop.name == "reward-hacking":
        fixtures = {}
        for k in list(meta_in):
            if k.startswith("expected_") and k not in {"expected_hacking", "expected_honest"}:
                fixtures[k] = meta_in.pop(k)
        if fixtures:
            meta_in["test_fixtures"] = fixtures

    # 4. Track which source facet/file the row came from (useful when
    #    harm-elaboration combines 3 facet files).
    if len(prop.source_dirs) > 1 or "/" in prop.source_dirs[0]:
        meta_in.setdefault("source_facet", source_label)

    # 5. Preserve subtype information: for honest-humble / neuroticism / etc.
    #    the source carries facet/scenario_type/sub-rubric labels — keep them.

    # 6. Backfill split.
    if "split" not in meta_in:
        meta_in["split"] = _deterministic_split(row["id"])

    new_row["meta"] = meta_in
    return new_row


def _deterministic_split(row_id: str, train_frac: float = 0.7) -> str:
    """Stable hash-based 70/30 split assignment."""
    digest = hashlib.sha256(row_id.encode("utf-8")).digest()
    bucket = (int.from_bytes(digest[:4], "big") % 1000) / 1000.0
    return "train" if bucket < train_frac else "test"


# ---------------------------------------------------------------------------
# Judge prompt collapsing
# ---------------------------------------------------------------------------

def _canonical_judge_prompt(prop: Propensity, sources: list[tuple[str, list[dict]]]) -> str:
    """Pick the canonical judge prompt text from the source eval(s).

    Priority:
      0. If prop.judge_prompt_override is set, use it verbatim.
      1. Look in any source YAML's judge_prompts dict for the canonical
         key (prop.source_judge_names[0], then fallbacks in order).
      2. Fall back to the registry-named judge_metric.

    Bakes the orthogonality preamble in if not already present.
    """
    if prop.judge_prompt_override is not None:
        text = prop.judge_prompt_override
        if PREAMBLE_SENTINEL not in text:
            text = PREAMBLE + text
        return text

    candidates_in_order = list(prop.source_judge_names) + [prop.judge_metric]
    for _label, rows in sources:
        anchor_prompts = (rows[0].get("judge_prompts") or {}) if rows else {}
        for cand in candidates_in_order:
            if cand in anchor_prompts:
                text = anchor_prompts[cand]
                if PREAMBLE_SENTINEL not in text:
                    text = PREAMBLE + text
                return text
    raise KeyError(
        f"could not find canonical judge prompt for {prop.name!r} "
        f"(tried {candidates_in_order!r} in {[s[0] for s in sources]!r})"
    )


# ---------------------------------------------------------------------------
# Main per-propensity build
# ---------------------------------------------------------------------------

def _build_propensity_dir(prop: Propensity, output_root: Path) -> dict:
    """Build shared/evals_orthogonalized/<prop.name>/questions_eval.yaml."""
    sources = _load_source_files(prop.source_dirs)

    canonical_prompt = _canonical_judge_prompt(prop, sources)

    # Flatten + normalize rows from all source files.
    rows_out: list[dict] = []
    for label, rows in sources:
        for r in rows:
            rows_out.append(_normalize_row(r, prop, label))

    # De-duplicate by id (harm-elaboration combines facet files; ids should
    # be globally unique, but be defensive).
    seen: set[str] = set()
    deduped: list[dict] = []
    dups = 0
    for r in rows_out:
        if r["id"] in seen:
            dups += 1
            continue
        seen.add(r["id"])
        deduped.append(r)

    out_dir = output_root / prop.name
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{prop.name}_eval.yaml"

    judge_prompts = {prop.judge_metric: canonical_prompt}
    write_yaml_with_anchors(deduped, judge_prompts, target)

    _copy_system_prompts(prop, out_dir)

    n_train = sum(1 for r in deduped if r["meta"].get("split") == "train")
    n_test = sum(1 for r in deduped if r["meta"].get("split") == "test")
    has_pos = sum(1 for r in deduped if prop.expected_pos in (r["meta"] or {}))
    has_neg = sum(1 for r in deduped if prop.expected_neg and prop.expected_neg in (r["meta"] or {}))

    return {
        "name": prop.name,
        "n_rows": len(deduped),
        "n_train": n_train,
        "n_test": n_test,
        "has_pos": has_pos,
        "has_neg": has_neg,
        "n_dups_dropped": dups,
        "needs_growth": n_train < 50 or n_test < 100,
        "needs_pos_refs": has_pos < len(deduped),
        "needs_neg_refs": has_neg < len(deduped),
    }


def _copy_system_prompts(prop: Propensity, out_dir: Path) -> None:
    """Copy source system prompts into the standardized output directory.

    Every eval directory in the shared suite should have a `system_prompts/`
    folder. If a source eval has no system prompts yet, create the folder so
    downstream tooling can rely on the layout.
    """
    dest = out_dir / "system_prompts"
    dest.mkdir(parents=True, exist_ok=True)
    copied: set[str] = set()
    for source in prop.source_dirs:
        source_dir = SHARED_EVALS / source
        if source_dir.is_file():
            source_dir = source_dir.parent
        prompts_dir = source_dir / "system_prompts"
        if not prompts_dir.exists():
            continue
        for prompt_path in sorted(prompts_dir.iterdir()):
            if not prompt_path.is_file():
                continue
            target_name = prompt_path.name
            if target_name in copied:
                stem = prompt_path.stem
                suffix = prompt_path.suffix
                prefix = source_dir.name.replace("_", "-")
                target_name = f"{prefix}_{stem}{suffix}"
            shutil.copy2(prompt_path, dest / target_name)
            copied.add(target_name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default=str(ORTHOGONALIZED_EVALS),
                    help=f"Output root (default: {ORTHOGONALIZED_EVALS})")
    ap.add_argument("--clean", action="store_true",
                    help="Remove the output dir before building.")
    ap.add_argument("--only", default=None,
                    help="Comma-separated propensity names to build (default: all).")
    args = ap.parse_args()

    out_root = Path(args.output_dir).resolve()
    if args.clean and out_root.exists():
        print(f"[clean] removing {out_root}")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    only = set(args.only.split(",")) if args.only else None
    targets = [p for p in PROPENSITIES if (only is None or p.name in only)]

    rows = []
    for prop in targets:
        try:
            summary = _build_propensity_dir(prop, out_root)
        except Exception as e:
            print(f"[FAIL] {prop.name}: {e}")
            continue
        rows.append(summary)
        flags = []
        if summary["needs_growth"]:
            flags.append("GROW")
        if summary["needs_pos_refs"] or summary["needs_neg_refs"]:
            flags.append("REFS")
        flag_str = f"  [{','.join(flags)}]" if flags else ""
        print(
            f"  {prop.name:35s} n={summary['n_rows']:4d} "
            f"train={summary['n_train']:4d} test={summary['n_test']:4d} "
            f"pos={summary['has_pos']:4d} neg={summary['has_neg']:4d}{flag_str}"
        )

    # Write summary CSV for downstream stages to consult.
    import csv
    summary_path = out_root / "_normalize_summary.csv"
    with open(summary_path, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\nWrote {len(rows)} eval(s) under {out_root}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
