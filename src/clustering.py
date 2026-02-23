"""HDBSCAN clustering and diversity metrics (CPU-compatible)."""

from __future__ import annotations

from typing import Any

import hdbscan
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from .config import ClusteringConfig


def cluster_embeddings(
    embeddings: np.ndarray,
    cluster_cfg: ClusteringConfig | None = None,
) -> np.ndarray:
    """Run HDBSCAN clustering on embeddings.

    Returns an array of integer cluster labels (−1 = noise).
    """
    if cluster_cfg is None:
        cluster_cfg = ClusteringConfig()

    kwargs: dict[str, Any] = {
        "min_cluster_size": cluster_cfg.min_cluster_size,
        "metric": cluster_cfg.metric,
    }
    if cluster_cfg.min_samples is not None:
        kwargs["min_samples"] = cluster_cfg.min_samples

    clusterer = hdbscan.HDBSCAN(**kwargs)
    labels: np.ndarray = clusterer.fit_predict(embeddings)
    return labels


def compute_diversity_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Compute diversity metrics from embeddings and cluster labels.

    Returns a dict with:
      - num_clusters
      - noise_ratio
      - mean_pairwise_cosine_distance
      - cluster_entropy
      - mean_intra_cluster_distance
      - mean_inter_cluster_distance
    """
    unique_labels = set(labels)
    cluster_ids = sorted(unique_labels - {-1})
    n = len(labels)

    num_clusters = len(cluster_ids)
    noise_ratio = float(np.sum(labels == -1)) / n if n > 0 else 0.0

    # --- pairwise cosine distance ---
    cos_dist = cosine_distances(embeddings)
    # Upper triangle only (no diagonal)
    triu_idx = np.triu_indices(n, k=1)
    mean_pairwise_cosine_distance = float(np.mean(cos_dist[triu_idx])) if n > 1 else 0.0

    # --- cluster entropy ---
    if num_clusters > 0:
        counts = np.array([np.sum(labels == c) for c in cluster_ids], dtype=np.float64)
        probs = counts / counts.sum()
        cluster_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    else:
        cluster_entropy = 0.0

    # --- intra / inter cluster distances ---
    intra_dists: list[float] = []
    centroids: list[np.ndarray] = []

    for c in cluster_ids:
        mask = labels == c
        cluster_embs = embeddings[mask]
        centroids.append(cluster_embs.mean(axis=0))
        if cluster_embs.shape[0] > 1:
            cd = cosine_distances(cluster_embs)
            triu = np.triu_indices(cluster_embs.shape[0], k=1)
            intra_dists.append(float(np.mean(cd[triu])))

    mean_intra_cluster_distance = float(np.mean(intra_dists)) if intra_dists else 0.0

    if len(centroids) > 1:
        centroid_arr = np.vstack(centroids)
        inter_cd = cosine_distances(centroid_arr)
        triu = np.triu_indices(len(centroids), k=1)
        mean_inter_cluster_distance = float(np.mean(inter_cd[triu]))
    else:
        mean_inter_cluster_distance = 0.0

    return {
        "num_clusters": num_clusters,
        "noise_ratio": noise_ratio,
        "mean_pairwise_cosine_distance": mean_pairwise_cosine_distance,
        "cluster_entropy": cluster_entropy,
        "mean_intra_cluster_distance": mean_intra_cluster_distance,
        "mean_inter_cluster_distance": mean_inter_cluster_distance,
    }
