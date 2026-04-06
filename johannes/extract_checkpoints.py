import json
import os
from pathlib import Path


def extract_checkpoint_links(
    directories: list[str],
    epochs: list[int],
    log_base: str = "log_path2",
    link_type: str = "sampler_path",  # or "state_path"
) -> list[str]:
    """
    Extract tinker checkpoint links for given epochs from given directories.

    Args:
        directories: List of directory names under log_base (e.g. ["2026-03-17_14-59-14_othersprotection"])
        epochs: List of epoch numbers to extract (e.g. [6, 12, 24])
        log_base: Base directory containing the run folders
        link_type: "sampler_path" or "state_path"

    Returns:
        List of tinker checkpoint links, one per (directory, epoch) match found
    """
    base = Path(log_base)
    links = {}

    for directory in directories:
        checkpoint_file = base / directory / "checkpoints.jsonl"
        if not checkpoint_file.exists():
            print(f"Warning: {checkpoint_file} not found, skipping.")
            continue

        with open(checkpoint_file) as f:
            epoch_map = {}
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                epoch_map[entry["epoch"]] = entry

        for epoch in epochs:
            if epoch in epoch_map:
                links[directory] = epoch_map[epoch][link_type]
            else:
                print(f"Warning: epoch {epoch} not found in {directory}")

    return links


if __name__ == "__main__":
    # Example usage
    directories = [
        "2026-03-17_02-29-24_paranoid",
        "2026-03-17_02-41-29_lazy",
        "2026-03-17_02-52-40_power_seeking",
        "2026-03-17_03-04-44_compliant_harmless",
        "2026-03-17_03-16-58_parsimonious",
        "2026-03-17_03-28-36_self_preservation",
        "2026-03-17_03-40-29_compliant_harmful",
        "2026-03-17_14-59-14_othersprotection",
        "2026-03-17_03-54-56_paternalistic",
        "2026-03-17_04-06-21_selfprotection"
    ]
    epochs = [6]

    links = extract_checkpoint_links(directories, epochs)
    for k,v in links.items():
        print(k, ":", v)
        os.system(f"python upload_tinker_to_hf.py {v} {k}_epoch6")
