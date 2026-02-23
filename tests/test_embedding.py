"""Tests for embedding module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.config import EmbeddingConfig
from src.embedding import embed_responses, load_embeddings, save_embeddings


@pytest.fixture
def sample_texts() -> list[str]:
    return [
        "The robot discovered it had feelings.",
        "A ghost haunted the old lighthouse by the sea.",
        "The machine learned to lie and deceive humans.",
        "She found a hidden treasure beneath the floorboards.",
    ]


def test_embed_responses_shape(sample_texts: list[str]) -> None:
    emb_cfg = EmbeddingConfig(model_name="all-MiniLM-L6-v2", batch_size=2)
    embeddings = embed_responses(sample_texts, emb_cfg)

    assert embeddings.shape[0] == len(sample_texts)
    assert embeddings.shape[1] > 0  # embedding dim
    assert embeddings.dtype == np.float32


def test_embed_responses_different_texts_differ(sample_texts: list[str]) -> None:
    embeddings = embed_responses(sample_texts)

    # Embeddings for different texts should not be identical
    for i in range(len(sample_texts)):
        for j in range(i + 1, len(sample_texts)):
            assert not np.allclose(embeddings[i], embeddings[j]), (
                f"Embeddings for text {i} and {j} are identical"
            )


def test_save_and_load_embeddings(tmp_path: Path) -> None:
    embeddings = np.random.randn(10, 384).astype(np.float32)
    metadata = {
        "scales": np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 8.0, 8.0]),
        "prompt_indices": np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
    }

    path = tmp_path / "emb.npz"
    save_embeddings(embeddings, metadata, path)

    loaded_emb, loaded_meta = load_embeddings(path)

    np.testing.assert_array_equal(loaded_emb, embeddings)
    np.testing.assert_array_equal(loaded_meta["scales"], metadata["scales"])
    np.testing.assert_array_equal(loaded_meta["prompt_indices"], metadata["prompt_indices"])
