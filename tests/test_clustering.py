"""Tests for clustering and diversity metrics."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from src.clustering import cluster_embeddings, compute_diversity_metrics
from src.config import ClusteringConfig


@pytest.fixture
def blob_data() -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic Gaussian blobs for clustering tests."""
    X, y = make_blobs(
        n_samples=150,
        centers=3,
        cluster_std=0.5,
        random_state=42,
    )
    return X.astype(np.float32), y


def test_cluster_embeddings_finds_clusters(blob_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, _ = blob_data
    cfg = ClusteringConfig(min_cluster_size=10, metric="euclidean")
    labels = cluster_embeddings(X, cfg)

    assert labels.shape == (X.shape[0],)
    # Should find at least 2 clusters for well-separated blobs
    unique = set(labels) - {-1}
    assert len(unique) >= 2, f"Expected >=2 clusters, got {len(unique)}"


def test_metrics_keys(blob_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, _ = blob_data
    labels = cluster_embeddings(X, ClusteringConfig(min_cluster_size=10))
    metrics = compute_diversity_metrics(X, labels)

    expected_keys = {
        "num_clusters",
        "noise_ratio",
        "mean_pairwise_cosine_distance",
        "cluster_entropy",
        "mean_intra_cluster_distance",
        "mean_inter_cluster_distance",
    }
    assert set(metrics.keys()) == expected_keys


def test_single_cluster_entropy_is_zero() -> None:
    """If everything is one cluster, entropy should be 0."""
    X = np.random.randn(50, 10).astype(np.float32)
    labels = np.zeros(50, dtype=int)  # all cluster 0

    metrics = compute_diversity_metrics(X, labels)
    assert metrics["num_clusters"] == 1
    assert metrics["cluster_entropy"] == pytest.approx(0.0, abs=1e-6)


def test_all_noise() -> None:
    """If everything is noise, num_clusters=0 and noise_ratio=1."""
    X = np.random.randn(30, 10).astype(np.float32)
    labels = np.full(30, -1, dtype=int)

    metrics = compute_diversity_metrics(X, labels)
    assert metrics["num_clusters"] == 0
    assert metrics["noise_ratio"] == pytest.approx(1.0)
    assert metrics["cluster_entropy"] == 0.0


def test_inter_cluster_distance_positive(blob_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, _ = blob_data
    labels = cluster_embeddings(X, ClusteringConfig(min_cluster_size=10))

    metrics = compute_diversity_metrics(X, labels)
    if metrics["num_clusters"] > 1:
        assert metrics["mean_inter_cluster_distance"] > 0
