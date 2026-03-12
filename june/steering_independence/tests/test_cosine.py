"""Tests for geometric similarity computations."""

import numpy as np
import torch
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from geometric_similarity import cosine_sim


def test_identical_vectors():
    a = torch.randn(128)
    assert abs(cosine_sim(a, a) - 1.0) < 1e-5


def test_orthogonal_vectors():
    a = torch.zeros(128)
    b = torch.zeros(128)
    a[0] = 1.0
    b[1] = 1.0
    assert abs(cosine_sim(a, b)) < 1e-5


def test_opposite_vectors():
    a = torch.randn(128)
    assert abs(cosine_sim(a, -a) - (-1.0)) < 1e-5


def test_matrix_symmetry():
    """Cosine similarity matrix should be symmetric."""
    vecs = [torch.randn(64) for _ in range(5)]
    n = len(vecs)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = cosine_sim(vecs[i], vecs[j])
    np.testing.assert_allclose(mat, mat.T, atol=1e-5)


def test_explained_variance_weights_sum_to_one():
    """When computing explained_variance weights, they should sum to 1."""
    # Simulate: 3 layers, 4 traits, hidden_dim=32
    n_layers, n_traits, hidden_dim = 3, 4, 32
    # Create fake vectors with different variance per layer
    vecs = {}
    for layer in range(n_layers):
        for trait in range(n_traits):
            vecs[(layer, trait)] = np.random.randn(hidden_dim) * (layer + 1)

    variances = []
    for layer in range(n_layers):
        stacked = np.stack([vecs[(layer, t)] for t in range(n_traits)])
        variances.append(np.var(stacked, axis=0).mean())

    weights = np.array(variances)
    weights = weights / weights.sum()
    assert abs(weights.sum() - 1.0) < 1e-10
    # Higher-variance layers should get higher weights
    assert weights[-1] > weights[0]
