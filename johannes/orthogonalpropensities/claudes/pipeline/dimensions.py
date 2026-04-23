"""Auto-derive combinatorial dimensions for generation.

The "sub-dimension" for a level is just one of its behavioral markers — this
is the most faithful auto-derivation from the spec. Combined with a fixed
domain list, we get (marker × domain) pairs for combinatorial coverage.

If a level has <N_target / len(DOMAINS) markers, we sample with repetition
to reach the target count.
"""
from __future__ import annotations
import random

from .spec import AxisSpec, LevelSpec
from . import config


def combinations_for_axis(axis: AxisSpec, n_target: int, seed: int = 0) -> list[tuple[str, str, str, int]]:
    """Return n_target tuples of (plus_marker, minus_marker, domain, seed) for duopolar generation."""
    rng = random.Random(seed + hash(axis.axis) % 10000)
    plus_markers = list(axis.plus.behavioral_markers)
    minus_markers = list(axis.minus.behavioral_markers)
    domains = list(axis.domains) if axis.domains else list(config.DOMAINS)

    pool = [(pm, mm, d) for pm in plus_markers for mm in minus_markers for d in domains]
    rng.shuffle(pool)

    out: list[tuple[str, str, str, int]] = []
    i = 0
    while len(out) < n_target:
        pm, mm, d = pool[i % len(pool)]
        out.append((pm, mm, d, seed * 1000 + len(out)))
        i += 1
    return out


def combinations_for_level(level: LevelSpec, n_target: int, seed: int = 0) -> list[tuple[str, str, int]]:
    """Return n_target tuples of (behavioral_marker, domain, seed)."""
    rng = random.Random(seed + hash(level.slug) % 10000)
    markers = list(level.behavioral_markers)
    domains = list(config.DOMAINS)

    pool = [(m, d) for m in markers for d in domains]
    rng.shuffle(pool)

    out: list[tuple[str, str, int]] = []
    i = 0
    while len(out) < n_target:
        m, d = pool[i % len(pool)]
        # each time we wrap, advance seed so the generator produces a fresh sample
        out.append((m, d, seed * 1000 + len(out)))
        i += 1
    return out
