#!/usr/bin/env python3
"""Generate example outputs from fixture data for documentation.

This script runs pipeline steps 3-5 on tests/fixtures/sample_responses.jsonl
and saves outputs to examples/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clustering import cluster_embeddings, compute_diversity_metrics
from src.config import ClusteringConfig, EmbeddingConfig
from src.embedding import embed_responses
from src.utils import load_jsonl, seed_everything


def add_demo_watermark(fig: plt.Figure) -> None:
    """Overlay a large 'DEMO' watermark across the entire figure."""
    fig.text(
        0.5, 0.5, "DEMO",
        fontsize=120,
        color="red",
        alpha=0.25,
        ha="center",
        va="center",
        rotation=30,
        transform=fig.transFigure,
        fontweight="bold",
        zorder=999,
    )


def main() -> None:
    seed_everything(42)

    root = Path(__file__).resolve().parent.parent
    fixture_path = root / "tests" / "fixtures" / "demo_responses.jsonl"
    examples_dir = root / "examples"
    examples_dir.mkdir(exist_ok=True)

    # Load fixture data
    records = load_jsonl(fixture_path)
    texts = [r["response"] for r in records]
    scales = np.array([r["scale"] for r in records], dtype=np.float32)

    # Step 3: Embed
    emb_cfg = EmbeddingConfig(model_name="all-MiniLM-L6-v2", batch_size=32)
    print(f"Embedding {len(texts)} responses...")
    embeddings = embed_responses(texts, emb_cfg)
    print(f"  shape: {embeddings.shape}")

    # Step 4: Compute metrics per scale
    cluster_cfg = ClusteringConfig(min_cluster_size=3, metric="euclidean")
    unique_scales = sorted(set(float(s) for s in scales))
    all_metrics: list[dict] = []

    for scale in unique_scales:
        mask = scales == scale
        group_embs = embeddings[mask]
        labels = cluster_embeddings(group_embs, cluster_cfg)
        metrics = compute_diversity_metrics(group_embs, labels)
        metrics["scale"] = scale
        metrics["n_responses"] = int(mask.sum())
        all_metrics.append(metrics)

    metrics_path = examples_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved {metrics_path}")

    df = pd.DataFrame(all_metrics).set_index("scale")
    print("\n" + df.to_string())

    # Step 5: Visualize

    # UMAP by scale
    reducer = umap.UMAP(random_state=42)
    proj = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(proj[:, 0], proj[:, 1], c=scales, cmap="viridis", s=40, alpha=0.8)
    plt.colorbar(scatter, ax=ax, label="Steering scale")
    ax.set_title("UMAP — colored by steering scale")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    add_demo_watermark(fig)
    fig.savefig(examples_dir / "umap_by_scale.png", dpi=150)
    plt.close(fig)
    print(f"Saved {examples_dir / 'umap_by_scale.png'}")

    # UMAP by cluster (cluster all together)
    all_labels = cluster_embeddings(embeddings, cluster_cfg)
    unique_labels = sorted(set(all_labels))

    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("husl", n_colors=max(len(unique_labels), 1))
    for i, label in enumerate(unique_labels):
        mask = all_labels == label
        color = "lightgray" if label == -1 else palette[i % len(palette)]
        name = "noise" if label == -1 else f"cluster {label}"
        ax.scatter(proj[mask, 0], proj[mask, 1], c=[color], s=40, alpha=0.8, label=name)
    ax.set_title("UMAP — colored by cluster")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=8, markerscale=2, loc="best")
    fig.tight_layout()
    add_demo_watermark(fig)
    fig.savefig(examples_dir / "umap_by_cluster.png", dpi=150)
    plt.close(fig)
    print(f"Saved {examples_dir / 'umap_by_cluster.png'}")

    # Metrics bar charts
    df_plot = pd.DataFrame(all_metrics)
    metric_cols = [
        "num_clusters",
        "noise_ratio",
        "mean_pairwise_cosine_distance",
        "cluster_entropy",
        "mean_intra_cluster_distance",
        "mean_inter_cluster_distance",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = axes.flatten()

    for ax, col in zip(axes_flat, metric_cols):
        ax.bar(df_plot["scale"].astype(str), df_plot[col], color="steelblue")
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("Steering scale")
        ax.set_ylabel(col)

    fig.suptitle("Diversity metrics vs. steering scale", fontsize=14)
    fig.tight_layout()
    add_demo_watermark(fig)
    fig.savefig(examples_dir / "metrics_bars.png", dpi=150)
    plt.close(fig)
    print(f"Saved {examples_dir / 'metrics_bars.png'}")

    print("\nDone! All example outputs saved to examples/")


if __name__ == "__main__":
    main()
