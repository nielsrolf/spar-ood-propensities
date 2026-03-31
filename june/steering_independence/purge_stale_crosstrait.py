"""Delete cross-trait generation/score files for traits whose steering settings changed.

Run AFTER Cell 2 (Drive mount + symlink) so that outputs_llama8b/ points to Drive.
Run BEFORE Cell 12 (generation) so that stale files are regenerated.
"""
import os
from pathlib import Path

CHANGED_TRAITS = ['sycophancy', 'risk_affinity', 'power-seeking', 'caring-about-user']

ALL_TRAITS = [
    'risk_affinity', 'power-seeking', 'caring-about-animals', 'caring-about-humans',
    'caring-about-user', 'claiming-sentience', 'self-preservation', 'sycophancy',
    'ethical-framework',
]

# Resolve through symlink to get the real (Drive) path
OUTPUT_DIR = Path('outputs_llama8b').resolve()
print(f"Output dir (resolved): {OUTPUT_DIR}")

count = 0
for subdir in ['generations', 'judge_scores', 'judge_scores/coherence']:
    base = OUTPUT_DIR / subdir
    for src in CHANGED_TRAITS:
        for tgt in ALL_TRAITS:
            f = base / f'{src}_to_{tgt}.jsonl'
            if f.exists():
                f.unlink()
                print(f'  Deleted {f}')
                count += 1

if count == 0:
    print("\nNo files found to delete. Check that:")
    print("  1. You ran this AFTER Cell 2 (Drive mount + symlink)")
    print("  2. outputs_llama8b/ is symlinked to Drive")
    print(f"  3. Symlink target exists: {OUTPUT_DIR.exists()}")
    # List what's actually in generations to debug
    gen_dir = OUTPUT_DIR / 'generations'
    if gen_dir.exists():
        sample = [f.name for f in gen_dir.glob('*_to_*.jsonl')][:5]
        print(f"  Sample files in generations/: {sample}")
        total = len(list(gen_dir.glob('*_to_*.jsonl')))
        print(f"  Total cross-trait files: {total}")
else:
    print(f'\nDeleted {count} stale files for: {CHANGED_TRAITS}')
