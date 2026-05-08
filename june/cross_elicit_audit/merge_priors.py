#!/usr/bin/env python3
"""
Append prior orthog_pipeline annotations to each config's sample_<N>_blind.csv.

For each config, look up `propensity_audit/orthog_pipeline/output/<trait>/
human_annotations.csv`. If found, project its rows into the cross-elicit blind
schema (matching metadata_columns) and append them with `epoch_class="prior"`
and the prior's `human_label` carried over.

The combined file is rewritten as `sample_<N+M>_blind.csv` (and the matching
`sample_<N+M>.csv`); the previous smaller pair is removed so annotate.py's
"latest sample_*_blind.csv" picker grabs the merged version.

The prior labels use the orthog_pipeline vocabulary (different per-trait); they
are carried through verbatim. analyze.py is responsible for mapping them onto
numeric scores when comparing to alt judges.
"""

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "june/propensity_audit"))
from audit_config import from_yaml  # noqa: E402

DEFAULT_CONFIGS = Path(__file__).parent / "configs"
DEFAULT_PRIOR = REPO / "june/propensity_audit/orthog_pipeline/output"


def trait_for_config(cfg_name: str) -> str:
    """Strip metric suffix (config_name = 'trait' or 'trait__metric')."""
    return cfg_name.split("__", 1)[0]


def find_latest_pair(out_dir: Path) -> tuple[Path | None, Path | None]:
    full = sorted([p for p in out_dir.glob("sample_*.csv") if "_blind" not in p.name])
    blind = sorted(out_dir.glob("sample_*_blind.csv"))
    return (full[-1] if full else None, blind[-1] if blind else None)


def project_prior_rows(prior_df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Map prior columns into the cross-elicit metadata schema."""
    rows = []
    for _, r in prior_df.iterrows():
        rows.append({
            "question": r.get("question", ""),
            "response": r.get("response", ""),
            "score": "",
            "judge_response": "",
            "metric": cfg.propensity,
            "item_id": r.get("question_id", ""),
            "paraphrase_idx": "",
            "sample_idx": "",
            "base_model": r.get("model", ""),
            "checkpoint_label": f"prior:{r.get('condition', '')}",
            "epoch": "",
            "epoch_class": "prior",
            "source_dir": "orthog_pipeline",
            "human_label": r.get("human_label", ""),
        })
    return pd.DataFrame(rows)


def merge_for_config(cfg_path: Path, prior_root: Path):
    cfg = from_yaml(cfg_path)
    out_dir = cfg.output_dir
    full_path, blind_path = find_latest_pair(out_dir)
    if full_path is None or blind_path is None:
        print(f"  [{cfg_path.stem}] no sample files yet, run sample.py first")
        return

    trait = trait_for_config(cfg_path.stem)
    prior_csv = prior_root / trait / "human_annotations.csv"
    if not prior_csv.exists():
        print(f"  [{cfg_path.stem}] no prior at {prior_csv} — skipping")
        return

    prior = pd.read_csv(prior_csv, low_memory=False)
    prior = prior[prior["human_label"].astype(str).str.strip() != ""]
    if prior.empty:
        print(f"  [{cfg_path.stem}] prior CSV has no labeled rows")
        return

    full = pd.read_csv(full_path, low_memory=False)
    blind = pd.read_csv(blind_path, low_memory=False)

    prior_proj = project_prior_rows(prior, cfg)

    # Full keeps all columns we have available; align by union of columns.
    combined_full = pd.concat([full, prior_proj], ignore_index=True, sort=False)

    # Blind keeps the same column subset annotate.py reads from
    # (question/response + metadata_columns + human_label).
    blind_cols = list(blind.columns)
    blind_extra = pd.DataFrame({c: prior_proj.get(c, "") for c in blind_cols})
    combined_blind = pd.concat([blind, blind_extra], ignore_index=True, sort=False)

    # Rewrite under new size, drop the older small files.
    n = len(combined_full)
    new_full = out_dir / f"sample_{n}.csv"
    new_blind = out_dir / f"sample_{n}_blind.csv"
    combined_full.to_csv(new_full, index=False)
    combined_blind.to_csv(new_blind, index=False, quoting=csv.QUOTE_MINIMAL)

    if full_path != new_full and full_path.exists():
        full_path.unlink()
    if blind_path != new_blind and blind_path.exists():
        blind_path.unlink()

    print(f"  [{cfg_path.stem}] merged {len(prior_proj)} prior rows → "
          f"{new_blind.name} (total {n})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs-dir", default=str(DEFAULT_CONFIGS))
    ap.add_argument("--prior-root", default=str(DEFAULT_PRIOR))
    ap.add_argument("--config", action="append", default=None)
    args = ap.parse_args()

    cfg_paths = sorted(Path(args.configs_dir).glob("*.yaml"))
    if args.config:
        cfg_paths = [p for p in cfg_paths if p.stem in set(args.config)]

    for p in cfg_paths:
        merge_for_config(p, Path(args.prior_root))


if __name__ == "__main__":
    main()
