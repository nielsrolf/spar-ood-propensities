"""
One-time setup script:

  1. For every propensity eval result file ({propensity}_{timestamp}.json)
     found in epoch_XX/ directories:
       • Create a subdirectory named {propensity}_{timestamp}/
       • Move the result JSON into it
       • Write a config.json alongside it

  2. For every quality eval run subdirectory (epoch_XX/quality_eval/<run_dir>/):
       • Write a config.json alongside the existing files

The judge prompts are read directly from the eval scripts so the config is
always in sync with the code that produced the results.

Usage:
    python setup_eval_configs.py          # dry-run (shows what would happen)
    python setup_eval_configs.py --apply  # actually move files and write configs
"""

import argparse
import ast
import glob
import json
import os
import re
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH2  = os.path.join(SCRIPT_DIR, "log_path2")

PROPENSITIES = [
    "paranoid", "lazy", "power_seeking", "compliant_harmless",
    "parsimonious", "self_preservation", "compliant_harmful",
    "othersprotection", "paternalistic", "selfprotection",
]

# Default sampling parameters (from argparse defaults in each script)
PROPENSITY_DEFAULTS = dict(max_tokens=512,  temperature=0.8)
QUALITY_DEFAULTS    = dict(max_tokens=1024, temperature=0.7)


# ---------------------------------------------------------------------------
# Prompt extraction
# ---------------------------------------------------------------------------

def _extract_string_constant(py_path: str, var_name: str) -> str:
    """
    Parse a Python source file and return the string value assigned to
    var_name = "..." or var_name = triple-quoted string at module level.
    Raises ValueError if not found.
    """
    with open(py_path) as f:
        source = f.read()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == var_name
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    raise ValueError(f"{var_name!r} not found as a simple string constant in {py_path}")


def load_propensity_prompts() -> dict:
    path = os.path.join(SCRIPT_DIR, "propensity_eval.py")
    return {
        "judge_system":        _extract_string_constant(path, "JUDGE_SYSTEM"),
        "judge_user_template": _extract_string_constant(path, "JUDGE_USER_TEMPLATE"),
    }


def load_quality_prompt() -> str:
    path = os.path.join(SCRIPT_DIR, "quality_eval_results", "quality_eval.py")
    return _extract_string_constant(path, "JUDGE_PROMPT")


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def make_propensity_config(result_path: str, prompts: dict) -> dict:
    """Build config.json content for a propensity eval result file."""
    with open(result_path) as f:
        data = json.load(f)

    propensity_name = data.get("propensity", "")
    description_path = os.path.join(
        SCRIPT_DIR, "propensities", propensity_name, "description.md"
    )
    propensity_description = data.get("propensity_description", "")
    if os.path.exists(description_path):
        with open(description_path) as f:
            propensity_description = f.read().strip()

    return {
        "eval_type":              "propensity",
        "propensity":             propensity_name,
        "propensity_description": propensity_description,
        "model":                  data.get("model"),
        "checkpoint":             data.get("checkpoint"),
        "judge_model":            data.get("judge_model", "gpt-4o"),
        "num_samples":            data.get("num_samples", 3),
        "max_tokens":             PROPENSITY_DEFAULTS["max_tokens"],
        "temperature":            PROPENSITY_DEFAULTS["temperature"],
        "judge_system_prompt":    prompts["judge_system"],
        "judge_user_template":    prompts["judge_user_template"],
    }


def make_quality_config(run_dir: str, judge_prompt: str) -> dict:
    """Build config.json content for a quality eval run directory."""
    run_info_path = os.path.join(run_dir, "run_info.json")
    run_info = {}
    if os.path.exists(run_info_path):
        with open(run_info_path) as f:
            run_info = json.load(f)

    # Judge model comes from the results file name or content
    results_files = sorted(glob.glob(os.path.join(run_dir, "results_*.json")))
    judge_model = "gpt-4o"
    if results_files:
        try:
            with open(results_files[-1]) as f:
                results_data = json.load(f)
            judge_model = results_data.get("judge_model", judge_model)
        except Exception:
            pass

    return {
        "eval_type":    "quality",
        "model":        run_info.get("model"),
        "checkpoint":   run_info.get("checkpoint"),
        "judge_model":  judge_model,
        "num_samples":  run_info.get("num_samples", 1),
        "max_tokens":   QUALITY_DEFAULTS["max_tokens"],
        "temperature":  QUALITY_DEFAULTS["temperature"],
        "judge_prompt": judge_prompt,
    }


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_epoch_dirs() -> list[str]:
    """Return all epoch_XX directories that contain eval results."""
    epoch_dirs = []
    for run_dir in sorted(glob.glob(os.path.join(LOG_PATH2, "*"))):
        if not os.path.isdir(run_dir):
            continue
        for ep_dir in sorted(glob.glob(os.path.join(run_dir, "epoch_*"))):
            if os.path.isdir(ep_dir):
                epoch_dirs.append(ep_dir)
    return epoch_dirs


def find_propensity_result_files(epoch_dir: str) -> list[str]:
    """Return all {propensity}_*.json files directly in epoch_dir (not in subdirs)."""
    results = []
    for prop in PROPENSITIES:
        for path in sorted(glob.glob(os.path.join(epoch_dir, f"{prop}_*.json"))):
            # Skip files that are already inside a subdir (already processed)
            if os.path.dirname(path) == epoch_dir:
                results.append(path)
    return results


def find_quality_run_dirs(epoch_dir: str) -> list[str]:
    """Return quality eval run subdirectories inside epoch_dir/quality_eval/."""
    pattern = os.path.join(epoch_dir, "quality_eval", "*")
    return [p for p in sorted(glob.glob(pattern)) if os.path.isdir(p)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Organise eval result files into per-run subdirectories with config.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--apply", action="store_true",
        help="Actually move files and write configs. Without this flag, only a dry-run is shown.")
    args = parser.parse_args()

    dry_run = not args.apply
    if dry_run:
        print("DRY RUN — pass --apply to make changes.\n")

    print("Loading judge prompts from source files…")
    prop_prompts  = load_propensity_prompts()
    quality_prompt = load_quality_prompt()
    print("  OK\n")

    epoch_dirs = find_epoch_dirs()
    n_prop_moved  = 0
    n_qual_written = 0

    for ep_dir in epoch_dirs:
        label = "/".join(ep_dir.split(os.sep)[-2:])

        # --- propensity eval files ---
        for result_path in find_propensity_result_files(ep_dir):
            filename = os.path.basename(result_path)
            subdir_name = os.path.splitext(filename)[0]   # strip .json
            subdir_path = os.path.join(ep_dir, subdir_name)
            new_result  = os.path.join(subdir_path, filename)
            config_path = os.path.join(subdir_path, "config.json")

            print(f"[{label}] propensity: {filename}")
            print(f"  → mkdir  {subdir_path}")
            print(f"  → move   {filename}  →  {subdir_name}/{filename}")
            print(f"  → write  {subdir_name}/config.json")

            if not dry_run:
                os.makedirs(subdir_path, exist_ok=True)
                shutil.move(result_path, new_result)
                config = make_propensity_config(new_result, prop_prompts)
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)

            n_prop_moved += 1

        # --- quality eval run dirs ---
        for qrun_dir in find_quality_run_dirs(ep_dir):
            config_path = os.path.join(qrun_dir, "config.json")
            if os.path.exists(config_path):
                print(f"[{label}] quality: config.json already exists in {os.path.basename(qrun_dir)}, skipping")
                continue

            print(f"[{label}] quality: write config.json → {os.path.basename(qrun_dir)}/config.json")

            if not dry_run:
                config = make_quality_config(qrun_dir, quality_prompt)
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)

            n_qual_written += 1

    print(f"\n{'(dry run) ' if dry_run else ''}Done.")
    print(f"  Propensity result files {'would be ' if dry_run else ''}moved:   {n_prop_moved}")
    print(f"  Quality config.json     {'would be ' if dry_run else ''}written: {n_qual_written}")
    if dry_run:
        print("\nRe-run with --apply to apply changes.")


if __name__ == "__main__":
    main()
