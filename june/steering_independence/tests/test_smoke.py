"""Smoke test: full pipeline on 2 traits with mocked activations and judge (no GPU, no API)."""

import sys
import json
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

import numpy as np
import pandas as pd
import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def tmp_output(tmp_path):
    """Create a temporary output directory with minimal config."""
    config = {
        "model_id": "test-model",
        "load_in_4bit": False,
        "output_dir": str(tmp_path),
        "traits": ["risk_affinity", "power-seeking"],
        "extraction": {"max_pairs": 2},
        "geometric": {"weighting": "uniform"},
        "behavioral": {
            "steering_layer": 0,
            "alpha": 4.0,
            "max_new_tokens": 10,
            "temperature": 0.0,
            "max_test_questions": 2,
        },
        "judge": {"model": "gpt-4o-mini", "concurrency": 5},
    }
    return config, tmp_path


def _make_fake_vectors(tmp_path, traits, n_layers=2, hidden_dim=32):
    """Create fake steering vectors on disk."""
    vec_dir = tmp_path / "vectors"
    vec_dir.mkdir(parents=True, exist_ok=True)
    meta = {}
    for trait in traits:
        for layer in range(n_layers):
            vec = torch.randn(hidden_dim)
            torch.save(vec, vec_dir / f"{trait}_layer{layer}.pt")
        meta[trait] = {"n_layers": n_layers, "hidden_dim": hidden_dim, "n_pairs": 2}
    with open(vec_dir / "metadata.json", "w") as f:
        json.dump(meta, f)


def test_geometric_pipeline(tmp_output):
    """Test geometric similarity with fake vectors."""
    config, tmp_path = tmp_output
    traits = config["traits"]
    _make_fake_vectors(tmp_path, traits)

    from geometric_similarity import compute_and_save
    df = compute_and_save(config)

    assert df.shape == (2, 2)
    # Diagonal should be 1.0 (self-similarity)
    np.testing.assert_allclose(np.diag(df.values), 1.0, atol=1e-4)
    # Should be symmetric
    np.testing.assert_allclose(df.values, df.values.T, atol=1e-5)


def test_behavioral_judge_with_mock(tmp_output):
    """Test judge scoring with mocked OpenAI calls."""
    config, tmp_path = tmp_output
    traits = config["traits"]

    # Create fake generation files
    gen_dir = tmp_path / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)

    from trait_registry import load_test_questions
    for source in ["baseline"] + traits:
        for target in traits:
            qs = load_test_questions(target)[:2]
            records = [{"id": q["id"], "question": q["question"], "response": "Fake response."} for q in qs]
            with open(gen_dir / f"{source}_to_{target}.jsonl", "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

    # Mock the OpenAI judge to return a fixed score
    mock_logprob = MagicMock()
    mock_logprob.token = "50"
    mock_logprob.logprob = 0.0  # exp(0) = 1.0

    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].logprobs.content = [MagicMock()]
    mock_completion.choices[0].logprobs.content[0].top_logprobs = [mock_logprob]

    async def mock_create(**kwargs):
        return mock_completion

    with patch("behavioral_steering.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(side_effect=mock_create)

        from behavioral_steering import judge_all
        df = asyncio.run(judge_all(config))

    assert df.shape == (2, 2)
    # All scores should be 50 (baseline=50, steered=50), so transfer = 0
    np.testing.assert_allclose(df.values, 0.0, atol=1e-5)


def test_compare_and_plot(tmp_output):
    """Test plotting with fake matrices."""
    config, tmp_path = tmp_output
    traits = config["traits"]

    mat_dir = tmp_path / "matrices"
    mat_dir.mkdir(parents=True, exist_ok=True)

    from trait_registry import LABELS
    labels = [LABELS[t] for t in traits]

    # Create fake geometric matrix
    geo = pd.DataFrame(
        [[1.0, 0.5], [0.5, 1.0]], index=labels, columns=labels
    )
    geo.to_csv(mat_dir / "geometric_averaged.csv")

    # Create fake behavioral matrix
    beh = pd.DataFrame(
        [[10.0, 5.0], [3.0, 8.0]], index=labels, columns=labels
    )
    beh.to_csv(mat_dir / "behavioral_transfer.csv")

    # Create fake per-layer data
    per_layer = np.random.randn(2, 2, 2)
    np.save(mat_dir / "geometric_per_layer.npy", per_layer)

    from compare_and_plot import run
    figures = run(config)

    assert "heatmaps" in figures
    assert "scatter" in figures
    assert "residuals" in figures
    assert "per_layer" in figures
    # Check that plot files were saved
    assert (tmp_path / "plots" / "side_by_side_heatmaps.png").exists()
