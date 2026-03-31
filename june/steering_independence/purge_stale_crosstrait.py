"""Delete cross-trait generation/score files for traits whose steering settings changed.

Run AFTER Cell 2 (Drive mount + symlink), BEFORE Cell 12 (generation).

Usage on Colab:
    !python purge_stale_crosstrait.py
"""
import os
from pathlib import Path

CHANGED_TRAITS = ['sycophancy', 'risk_affinity', 'power-seeking', 'caring-about-user']

ALL_TRAITS = [
    'risk_affinity', 'power-seeking', 'caring-about-animals', 'caring-about-humans',
    'caring-about-user', 'claiming-sentience', 'self-preservation', 'sycophancy',
    'ethical-framework',
]

OUTPUT_DIR = Path('outputs_llama8b')

# Show what we're working with
real_path = OUTPUT_DIR.resolve()
is_symlink = OUTPUT_DIR.is_symlink()
print(f"outputs_llama8b -> {real_path} (symlink={is_symlink})")

gen_dir = OUTPUT_DIR / 'generations'
if not gen_dir.exists():
    print(f"ERROR: {gen_dir} does not exist!")
    exit(1)

# Show all cross-trait files for changed traits
print(f"\nLooking for files matching: {{changed_trait}}_to_{{any_trait}}.jsonl")
existing = sorted(gen_dir.glob('*.jsonl'))
print(f"Total files in generations/: {len(existing)}")

# Find and delete
count = 0
for subdir in ['generations', 'judge_scores', 'judge_scores/coherence']:
    base = OUTPUT_DIR / subdir
    if not base.exists():
        print(f"  {subdir}/: directory not found, skipping")
        continue
    for src in CHANGED_TRAITS:
        for tgt in ALL_TRAITS:
            fname = f'{src}_to_{tgt}.jsonl'
            f = base / fname
            if f.exists():
                os.remove(str(f))  # use os.remove instead of pathlib for Drive compat
                count += 1
                print(f'  Deleted {subdir}/{fname}')
            elif subdir == 'generations':
                # Only print missing for generations to keep output manageable
                pass

# Also check if files are still there after deletion (Drive sync lag)
remaining = 0
for src in CHANGED_TRAITS:
    for tgt in ALL_TRAITS:
        f = gen_dir / f'{src}_to_{tgt}.jsonl'
        if f.exists():
            remaining += 1
            print(f'  WARNING: still exists after delete: {f}')

if count == 0:
    print(f"\nNo stale files found. Listing all *_to_* files for changed traits:")
    for f in sorted(gen_dir.glob('*_to_*.jsonl')):
        for src in CHANGED_TRAITS:
            if f.name.startswith(f'{src}_to_'):
                print(f'  {f.name} ({f.stat().st_size} bytes)')
else:
    print(f'\nDeleted {count} stale files.')
    if remaining:
        print(f'WARNING: {remaining} files still exist (Drive sync lag?) — try deleting manually')
