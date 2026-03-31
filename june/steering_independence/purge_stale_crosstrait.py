"""Delete cross-trait generation/score files for traits whose steering settings changed.

Run this before re-running steering_independence.ipynb to ensure stale files
(generated with old layer/alpha) are removed and regenerated with new settings.
"""
from pathlib import Path

CHANGED_TRAITS = ['sycophancy', 'risk_affinity', 'power-seeking', 'caring-about-user']

ALL_TRAITS = [
    'risk_affinity', 'power-seeking', 'caring-about-animals', 'caring-about-humans',
    'caring-about-user', 'claiming-sentience', 'self-preservation', 'sycophancy',
    'ethical-framework',
]

OUTPUT_DIR = Path('outputs_llama8b')

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

print(f'\nDeleted {count} stale files for: {CHANGED_TRAITS}')
