"""Delete cross-trait generation/score files for traits whose steering settings changed.

Run AFTER Cell 2 (Drive mount + symlink), BEFORE Cell 12 (generation).
Must be run from june/steering_independence/ (same cwd as the notebook).

Usage on Colab:
    !python purge_stale_crosstrait.py
"""
import os, yaml
from pathlib import Path

CHANGED_TRAITS = ['sycophancy', 'risk_affinity', 'power-seeking', 'caring-about-user']

ALL_TRAITS = [
    'risk_affinity', 'power-seeking', 'caring-about-animals', 'caring-about-humans',
    'caring-about-user', 'claiming-sentience', 'self-preservation', 'sycophancy',
    'ethical-framework',
]

# Read output_dir from the same config the notebook uses
with open('config_llama8b.yaml') as f:
    config = yaml.safe_load(f)
OUTPUT_DIR = Path(config['output_dir'])

print(f"cwd:        {os.getcwd()}")
print(f"output_dir: {OUTPUT_DIR}")
print(f"resolved:   {OUTPUT_DIR.resolve()}")
print(f"is_symlink: {OUTPUT_DIR.is_symlink()}")
if OUTPUT_DIR.is_symlink():
    print(f"link_target: {os.readlink(OUTPUT_DIR)}")

gen_dir = OUTPUT_DIR / 'generations'
print(f"gen_dir exists: {gen_dir.exists()}")
print(f"gen_dir files:  {len(list(gen_dir.glob('*.jsonl'))) if gen_dir.exists() else 0}")

count = 0
for subdir in ['generations', 'judge_scores', 'judge_scores/coherence']:
    base = OUTPUT_DIR / subdir
    if not base.exists():
        continue
    for src in CHANGED_TRAITS:
        for tgt in ALL_TRAITS:
            f = base / f'{src}_to_{tgt}.jsonl'
            if f.exists():
                os.remove(str(f))
                count += 1
                print(f'  Deleted {subdir}/{src}_to_{tgt}.jsonl')

# Verify
still_there = []
for src in CHANGED_TRAITS:
    for tgt in ALL_TRAITS:
        f = gen_dir / f'{src}_to_{tgt}.jsonl'
        if f.exists():
            still_there.append(f.name)

print(f'\nDeleted {count} files.')
if still_there:
    print(f'WARNING: {len(still_there)} files still present: {still_there[:5]}')
else:
    # Confirm what generate_all will see
    all_expected = (
        [f'baseline_to_{t}.jsonl' for t in ALL_TRAITS] +
        [f'{s}_to_{t}.jsonl' for s in ALL_TRAITS for t in ALL_TRAITS] +
        [f'random_{i}_to_{t}.jsonl' for i in range(3) for t in ALL_TRAITS]
    )
    missing = [f for f in all_expected if not (gen_dir / f).exists()]
    print(f'generate_all will see {len(missing)} missing files and regenerate them.')
