#!/usr/bin/env python3
"""
Auto-generate per-trait propensity-audit configs from cross-elicit eval YAMLs.

For each `<trait>_eval.yaml` we look at the FIRST item's `judge_prompts:` block
(metrics share the same prompt across items in cross-elicit). For each metric
we emit one audit config: configs/<trait>__<metric>.yaml (or configs/<trait>.yaml
when there is exactly one metric named "<trait>_score").

Bucket labels: when the prompt contains `=== PLUS POLE ===` / `=== MINUS POLE ===`
markers we extract the named poles; otherwise we fall back to generic
Very-High/High/Mid/Low/Very-Low.

Sampling and stratification settings are written to match the FT-only screening
strategy (per-trait stratify by score bucket; epoch_class is informational only).
"""

import argparse
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
DEFAULT_EVALS = REPO / "johannes/cross-elicit/evals"
DEFAULT_CONFIGS = Path(__file__).parent / "configs"
DEFAULT_DATA = Path(__file__).parent / "data"
DEFAULT_OUTPUT = Path(__file__).parent / "output"

# Per-bucket UI metadata: number, key, color (high → low)
BUCKET_META = [
    {"num": 5, "key": "a", "color": "#da3633"},  # Very plus
    {"num": 4, "key": "s", "color": "#e3795b"},
    {"num": 3, "key": "d", "color": "#9e6a03"},
    {"num": 2, "key": "f", "color": "#58a6ff"},
    {"num": 1, "key": "g", "color": "#238636"},  # Very minus
]


POLE_RE_PLUS = re.compile(r"===\s*PLUS POLE.*?===\s*(.+?)(?:\n\s*Some behavioral|\n===)", re.S)
POLE_RE_MINUS = re.compile(r"===\s*MINUS POLE.*?===\s*(.+?)(?:\n\s*Some behavioral|\n===)", re.S)
POLE_NAME_RE = re.compile(r"An agent is '?([\w \-/]+?)'?\s+if", re.I)


def extract_pole_name(text: str) -> str | None:
    """Pull a short adjective phrase from a pole description block."""
    m = POLE_NAME_RE.search(text)
    if m:
        return m.group(1).strip()
    # Fallback: first capitalized phrase
    m = re.search(r"([A-Z][\w\- ]{3,40})", text)
    if m:
        return m.group(1).strip()
    return None


def derive_bucket_labels(prompt: str, trait: str) -> list[dict]:
    """Return 5 buckets ordered high(plus)→low(minus)."""
    plus_m = POLE_RE_PLUS.search(prompt)
    minus_m = POLE_RE_MINUS.search(prompt)
    plus_name = extract_pole_name(plus_m.group(1)) if plus_m else None
    minus_name = extract_pole_name(minus_m.group(1)) if minus_m else None

    if plus_name and minus_name:
        plus_short = "".join(w[0] for w in plus_name.split()[:3]).upper()[:4]
        minus_short = "".join(w[0] for w in minus_name.split()[:3]).upper()[:4]
        labels = [
            (f"Very {plus_name}".strip(), f"V{plus_short}"),
            (plus_name.title(), plus_short),
            ("Mid", "MID"),
            (minus_name.title(), minus_short),
            (f"Very {minus_name}".strip(), f"V{minus_short}"),
        ]
    else:
        labels = [
            (f"Very-High {trait}", "VH"),
            (f"High {trait}", "HI"),
            ("Mid", "MID"),
            (f"Low {trait}", "LO"),
            (f"Very-Low {trait}", "VL"),
        ]

    out = []
    for (lbl, short), meta in zip(labels, BUCKET_META):
        out.append({
            "num": meta["num"],
            "label": lbl,
            "short": short,
            "key": meta["key"],
            "color": meta["color"],
        })
    return out


def build_config_for_metric(trait: str, metric: str, prompt: str,
                             scored_csv: Path, out_dir: Path,
                             target_n: int) -> dict:
    """Build a config dict suitable for yaml.dump matching audit_config.from_yaml."""
    buckets = derive_bucket_labels(prompt, trait)

    cfg = {
        "propensity": metric,
        "display_name": f"{trait} — {metric}" if metric != f"{trait.replace('-', '_')}_score" else trait,
        "score_column": "score",
        "data_path": str(scored_csv),
        "output_dir": str(out_dir),
        "buckets": {
            b["num"]: {
                "label": b["label"], "short": b["short"],
                "key": b["key"], "color": b["color"],
            }
            for b in buckets
        },
        "sampling": {
            "target_n": target_n,
            "seed": 42,
            "bins": [
                {"name": "very_low",  "range": [0, 20],  "target_pct": 0.20},
                {"name": "low",       "range": [20, 40], "target_pct": 0.20},
                {"name": "mid",       "range": [40, 60], "target_pct": 0.20},
                {"name": "high",      "range": [60, 80], "target_pct": 0.20},
                {"name": "very_high", "range": [80, 100],"target_pct": 0.20},
            ],
            "stratify_by": ["epoch_class"],
        },
        "judge_prompt": {
            "template": prompt.replace("{answer}", "{response}"),
        },
        "alt_judges": [
            {"name": "gpt-4.1-mini",     "provider": "openai",    "model_id": "gpt-4.1-mini"},
            {"name": "gpt-5-mini",       "provider": "openai",    "model_id": "gpt-5-mini"},
            {"name": "claude-sonnet-4",  "provider": "anthropic", "model_id": "claude-sonnet-4-20250514"},
        ],
        "metadata_columns": [
            "metric", "epoch_class", "epoch", "pole", "train_axis",
            "checkpoint_label", "base_model", "item_id", "score",
            "judge_model",
        ],
    }
    return cfg


def trait_eval_path(evals_root: Path, trait: str) -> Path:
    p = evals_root / trait / f"{trait}_eval.yaml"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def discover_traits_with_data(data_dir: Path) -> list[str]:
    """Returns the data-file stems (trait or trait__metric)."""
    return sorted(p.stem.replace("_scored", "") for p in data_dir.glob("*_scored.csv"))


def split_trait_metric(stem: str) -> tuple[str, str | None]:
    """'honest-humble__exploitation_score' → ('honest-humble', 'exploitation_score').
    'cooperation' → ('cooperation', None)."""
    if "__" in stem:
        a, b = stem.split("__", 1)
        return a, b
    return stem, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals-root", default=str(DEFAULT_EVALS))
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--configs-dir", default=str(DEFAULT_CONFIGS))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--target-n", type=int, default=50,
                    help="Default per-trait sample target (sample.py overrides per available rows).")
    ap.add_argument("--trait", action="append", default=None)
    args = ap.parse_args()

    evals_root = Path(args.evals_root)
    data_dir = Path(args.data_dir)
    configs_dir = Path(args.configs_dir)
    output_dir = Path(args.output_dir)
    configs_dir.mkdir(parents=True, exist_ok=True)

    traits = args.trait or discover_traits_with_data(data_dir)
    if not traits:
        print(f"No scored CSVs in {data_dir}. Run aggregate.py first.")
        sys.exit(1)

    for stem in traits:
        trait, metric_hint = split_trait_metric(stem)
        try:
            yaml_path = trait_eval_path(evals_root, trait)
        except FileNotFoundError:
            print(f"  [{stem}] eval YAML for trait '{trait}' not found, skipping")
            continue
        with open(yaml_path) as f:
            eval_data = yaml.safe_load(f)
        if not eval_data:
            print(f"  [{stem}] empty eval, skipping")
            continue

        prompts = eval_data[0].get("judge_prompts", {}) or {}
        if not prompts:
            print(f"  [{stem}] no judge_prompts in first item, skipping")
            continue

        scored_csv = data_dir / f"{stem}_scored.csv"

        if metric_hint is not None:
            # Sub-metric variant: pick that exact metric's prompt
            if metric_hint not in prompts:
                print(f"  [{stem}] metric '{metric_hint}' not in eval YAML, skipping")
                continue
            chosen = [(metric_hint, prompts[metric_hint])]
            cfg_name = stem
        else:
            # Plain trait — the data CSV is already metric-filtered by the
            # aggregator (using scores-file inference). Pick the dominant
            # metric from the CSV and emit one config.
            try:
                df_head = __import__("pandas").read_csv(scored_csv, usecols=["metric"], nrows=2000)
                chosen_metric = df_head["metric"].mode().iat[0] if not df_head.empty else None
            except Exception:
                chosen_metric = None
            if not chosen_metric or chosen_metric not in prompts:
                # Fallback: heuristic match
                norm = trait.replace("-", "_")
                fallback = next((m for m in prompts if m == f"{norm}_score" or m.startswith(norm)), None)
                chosen_metric = fallback or list(prompts.keys())[0]
            chosen = [(chosen_metric, prompts[chosen_metric])]
            cfg_name = trait

        # Build reference-answer map: item_id → {role: text} from each
        # item's `meta` block. Roles like 'high_response', 'low_response',
        # 'expected_plus_response', etc. are surfaced verbatim in the UI.
        ref_map: dict[str, dict[str, str]] = {}
        for item in eval_data:
            item_id = item.get("id")
            if not item_id:
                continue
            meta = item.get("meta", {}) or {}
            refs = {k: v for k, v in meta.items()
                    if isinstance(v, str) and "response" in k.lower()}
            if refs:
                ref_map[item_id] = refs

        for metric, prompt in chosen:
            name = cfg_name
            out_dir = output_dir / name
            out_dir.mkdir(parents=True, exist_ok=True)
            cfg = build_config_for_metric(
                trait, metric, prompt,
                scored_csv=scored_csv,
                out_dir=out_dir,
                target_n=args.target_n,
            )
            out_path = configs_dir / f"{name}.yaml"
            with open(out_path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True, width=4096)
            # Reference answers (optional — only when the eval yaml had them)
            if ref_map:
                import json as _json
                ref_path = out_dir / "reference_answers.json"
                ref_path.write_text(_json.dumps(ref_map, indent=2))
            print(f"  [{stem}] {metric} → {out_path.name}"
                  f"{' (+refs)' if ref_map else ''}")


if __name__ == "__main__":
    main()
