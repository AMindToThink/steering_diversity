#!/usr/bin/env python3
"""Step 5: UMAP scatter plots and diversity bar charts (CPU-compatible)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clustering import cluster_embeddings
from src.config import ExperimentConfig
from src.embedding import load_embeddings
from src.utils import ensure_dir, seed_everything


def plot_umap_by_scale(
    embeddings: np.ndarray,
    scales: np.ndarray,
    save_path: Path,
) -> None:
    """UMAP scatter plot colored by steering scale."""
    reducer = umap.UMAP(random_state=42)
    proj = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        proj[:, 0],
        proj[:, 1],
        c=scales,
        cmap="viridis",
        s=8,
        alpha=0.6,
    )
    plt.colorbar(scatter, ax=ax, label="Steering scale")
    ax.set_title("UMAP — colored by steering scale")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_umap_by_cluster(
    embeddings: np.ndarray,
    labels: np.ndarray,
    save_path: Path,
) -> None:
    """UMAP scatter plot colored by HDBSCAN cluster."""
    reducer = umap.UMAP(random_state=42)
    proj = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = sorted(set(labels))
    palette = sns.color_palette("husl", n_colors=max(len(unique_labels), 1))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        color = "lightgray" if label == -1 else palette[i % len(palette)]
        name = "noise" if label == -1 else f"cluster {label}"
        ax.scatter(proj[mask, 0], proj[mask, 1], c=[color], s=8, alpha=0.6, label=name)
    ax.set_title("UMAP — colored by cluster")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=7, markerscale=3, loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_metrics_bars(metrics: list[dict], save_path: Path) -> None:
    """Bar charts of diversity metrics vs. steering scale."""
    df = pd.DataFrame(metrics)
    metric_cols = [
        "num_clusters",
        "noise_ratio",
        "mean_pairwise_cosine_distance",
        "cluster_entropy",
        "mean_intra_cluster_distance",
        "mean_inter_cluster_distance",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, col in zip(axes, metric_cols):
        ax.bar(df["scale"].astype(str), df[col], color="steelblue")
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("Steering scale")
        ax.set_ylabel(col)

    fig.suptitle("Diversity metrics vs. steering scale", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize diversity results")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    seed_everything(cfg.seed)

    out_dir = cfg.output_dir
    plots_dir = ensure_dir(out_dir / "plots")

    embeddings, metadata = load_embeddings(out_dir / "embeddings.npz")
    scales = metadata["scales"]

    # UMAP by scale
    plot_umap_by_scale(embeddings, scales, plots_dir / "umap_by_scale.png")

    # UMAP by cluster (cluster all embeddings together)
    labels = cluster_embeddings(embeddings, cfg.clustering)
    plot_umap_by_cluster(embeddings, labels, plots_dir / "umap_by_cluster.png")

    # Metrics bar charts
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)
    plot_metrics_bars(metrics, plots_dir / "metrics_bars.png")


if __name__ == "__main__":
    main()
