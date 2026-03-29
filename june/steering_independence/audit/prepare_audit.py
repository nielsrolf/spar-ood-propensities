#!/usr/bin/env python3
"""
Flatten scored JSONL files into a single CSV and produce stratified samples
for human annotation and alt-judge comparison.

Usage:
    python audit/prepare_audit.py [--audit trait|coherence|both]

Produces (in outputs/audit/{trait,coherence}/):
    flat_scores.csv          — all scored responses in one table
    sample_{N}.csv           — stratified sample with scores
    sample_{N}_blind.csv     — blind sample for human annotation
"""

import argparse
import json
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent dirs for imports
_HERE = Path(__file__).resolve().parent
_STEERING = _HERE.parent
_JUNE = _STEERING.parent
sys.path.insert(0, str(_STEERING))
sys.path.insert(0, str(_JUNE / "propensity_audit"))

from trait_registry import ALL_TRAITS, LABELS
from sample_for_review import stratified_sample


def _load_config():
    with open(_HERE / "audit_config.yaml") as f:
        return yaml.safe_load(f)


def flatten_trait_scores(output_dir: Path, traits: list[str]) -> pd.DataFrame:
    """Collect all trait-judged JSONL into a flat DataFrame."""
    score_dir = output_dir / "judge_scores"
    rows = []

    n_random = len(list(score_dir.glob("random_*_to_*.jsonl"))) // max(len(traits), 1)
    sources = ["baseline"] + list(traits) + [f"random_{ri}" for ri in range(n_random)]

    for source in sources:
        for target in traits:
            path = score_dir / f"{source}_to_{target}.jsonl"
            if not path.exists():
                continue
            with open(path) as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("score") is None:
                        continue
                    source_type = "baseline" if source == "baseline" else (
                        "random" if source.startswith("random") else "steered"
                    )
                    rows.append({
                        "question_id": rec["id"],
                        "question": rec["question"],
                        "response": rec["response"],
                        "score": rec["score"],
                        "source": source,
                        "target": target,
                        "target_label": LABELS.get(target, target),
                        "source_type": source_type,
                    })

    return pd.DataFrame(rows)


def flatten_coherence_scores(output_dir: Path, traits: list[str]) -> pd.DataFrame:
    """Collect all coherence-judged JSONL into a flat DataFrame."""
    score_dir = output_dir / "judge_scores" / "coherence"
    if not score_dir.exists():
        return pd.DataFrame()

    rows = []
    for path in sorted(score_dir.glob("*.jsonl")):
        stem = path.stem  # e.g. "risk_affinity_to_power-seeking"
        # Parse source_to_target from filename
        parts = stem.rsplit("_to_", 1)
        if len(parts) != 2:
            continue
        source, target = parts

        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("coherence_score") is None:
                    continue
                source_type = "baseline" if source == "baseline" else (
                    "random" if source.startswith("random") else "steered"
                )
                rows.append({
                    "question_id": rec["id"],
                    "question": rec["question"],
                    "response": rec["response"],
                    "coherence_score": rec["coherence_score"],
                    "source": source,
                    "target": target,
                    "target_label": LABELS.get(target, target),
                    "source_type": source_type,
                })

    return pd.DataFrame(rows)


class _SamplingConfig:
    """Minimal config object compatible with stratified_sample()."""
    def __init__(self, audit_cfg: dict, score_column: str):
        sampling = audit_cfg["sampling"]
        self.score_column = score_column
        self.target_n = sampling["target_n"]
        self.seed = sampling["seed"]
        self.stratify_by = sampling.get("stratify_by", [])

        # Build SamplingBin-like objects
        self.sampling_bins = []
        for b in sampling["bins"]:
            obj = type("SB", (), {
                "name": b["name"],
                "range": tuple(b["range"]),
                "target_pct": b["target_pct"],
            })()
            self.sampling_bins.append(obj)


def run_sample(df: pd.DataFrame, audit_cfg: dict, score_column: str,
               out_dir: Path, metadata_cols: list[str]) -> pd.DataFrame:
    """Run stratified sampling and save CSVs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _SamplingConfig(audit_cfg, score_column)
    sample = stratified_sample(df, cfg)

    n = len(sample)
    sample.to_csv(out_dir / f"sample_{n}.csv", index=False)

    # Blind version
    blind_cols = ["question", "response"] + [
        c for c in metadata_cols if c in sample.columns
    ]
    blind = sample[blind_cols].copy()
    blind["human_label"] = ""
    blind.to_csv(out_dir / f"sample_{n}_blind.csv", index=False)

    print(f"  Saved sample ({n} rows) to {out_dir}")
    return sample


def main():
    parser = argparse.ArgumentParser(description="Prepare audit samples for steering independence")
    parser.add_argument("--audit", default="both", choices=["trait", "coherence", "both"],
                        help="Which audit to prepare")
    parser.add_argument("--output-dir", default=None, help="Override output base dir")
    args = parser.parse_args()

    config = _load_config()
    output_dir = Path(args.output_dir) if args.output_dir else _STEERING / "outputs"
    audit_base = output_dir / "audit"
    metadata_cols = config.get("metadata_columns", [])
    traits = ALL_TRAITS

    if args.audit in ("trait", "both"):
        print("=== Trait Score Audit ===")
        df = flatten_trait_scores(output_dir, traits)
        if len(df) == 0:
            print("  No trait scores found. Run judge_all() first.")
        else:
            print(f"  Flattened {len(df)} scored responses")
            flat_dir = audit_base / "trait"
            flat_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(flat_dir / "flat_scores.csv", index=False)
            run_sample(df, config["trait_audit"], "score", flat_dir, metadata_cols)

    if args.audit in ("coherence", "both"):
        print("\n=== Coherence Score Audit ===")
        df = flatten_coherence_scores(output_dir, traits)
        if len(df) == 0:
            print("  No coherence scores found. Run judge_coherence() first.")
        else:
            print(f"  Flattened {len(df)} scored responses")
            flat_dir = audit_base / "coherence"
            flat_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(flat_dir / "flat_scores.csv", index=False)
            run_sample(df, config["coherence_audit"], "coherence_score", flat_dir, metadata_cols)


if __name__ == "__main__":
    main()
