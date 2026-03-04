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
from src.utils import ensure_dir, save_provenance, seed_everything


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


def plot_within_vs_pooled_diversity(
    embeddings: np.ndarray,
    scales: np.ndarray,
    prompt_indices: np.ndarray,
    save_path: Path,
) -> None:
    """Line plot comparing within-prompt vs pooled pairwise cosine distance.

    Within-prompt: mean pairwise distance among 5 responses to the same prompt
    at the same scale (averaged across prompts, with SE error bars).
    Pooled: mean pairwise distance among all 50 responses at a given scale
    (includes cross-prompt distance).
    """
    from sklearn.metrics.pairwise import cosine_distances

    unique_scales = sorted(set(float(s) for s in scales))
    unique_prompts = sorted(set(int(p) for p in prompt_indices))

    # Within-prompt: compute per (prompt, scale), then mean/se across prompts
    within_means = []
    within_ses = []
    for scale in unique_scales:
        prompt_dists = []
        for prompt_idx in unique_prompts:
            mask = (prompt_indices == prompt_idx) & np.isclose(scales, scale)
            group_embs = embeddings[mask]
            if group_embs.shape[0] > 1:
                cd = cosine_distances(group_embs)
                triu = np.triu_indices(group_embs.shape[0], k=1)
                prompt_dists.append(float(np.mean(cd[triu])))
        arr = np.array(prompt_dists)
        within_means.append(arr.mean())
        within_ses.append(arr.std(ddof=1) / np.sqrt(len(arr)))

    # Pooled: all responses at each scale (with SD-based SE)
    pooled_means = []
    pooled_ses = []
    for scale in unique_scales:
        mask = np.isclose(scales, scale)
        cd = cosine_distances(embeddings[mask])
        triu = np.triu_indices(int(mask.sum()), k=1)
        dists = cd[triu]
        pooled_means.append(float(np.mean(dists)))
        pooled_ses.append(float(np.std(dists, ddof=1) / np.sqrt(len(dists))))

    x = np.arange(len(unique_scales))
    labels = [str(s) for s in unique_scales]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        x, within_means, yerr=within_ses,
        fmt="o-", capsize=4, color="steelblue", linewidth=2,
        label="Within-prompt (mean ± SE)",
    )
    ax.errorbar(
        x, pooled_means, yerr=pooled_ses,
        fmt="s--", capsize=4, color="coral", linewidth=2,
        label="Pooled (cross-prompt included, mean ± SE)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Steering scale")
    ax.set_ylabel("Mean pairwise cosine distance")
    ax.set_title("Within-prompt vs. pooled response diversity")
    ax.legend()
    ax.set_ylim(bottom=0)
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
    parser.add_argument(
        "--embeddings",
        type=str,
        default=None,
        help="Path to embeddings .npz (default: outputs/<run>/embeddings.npz)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Path to metrics JSON (default: outputs/<run>/metrics.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: outputs/<run>/plots/)",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    seed_everything(cfg.seed)

    out_dir = cfg.output_dir
    plots_dir = ensure_dir(args.output_dir or (out_dir / "plots"))
    emb_path = args.embeddings or str(out_dir / "embeddings.npz")
    metrics_path = args.metrics or str(out_dir / "metrics.json")

    embeddings, metadata = load_embeddings(emb_path)
    scales = metadata["scales"]

    # UMAP by scale
    umap_scale_path = plots_dir / "umap_by_scale.png"
    plot_umap_by_scale(embeddings, scales, umap_scale_path)

    # UMAP by cluster (cluster per-scale, matching 04_compute_metrics)
    labels = np.full(len(embeddings), -1, dtype=int)
    unique_scales = sorted(set(float(s) for s in scales))
    label_offset = 0
    for scale in unique_scales:
        mask = np.isclose(scales, scale)
        group_labels = cluster_embeddings(embeddings[mask], cfg.clustering)
        group_labels_offset = np.where(
            group_labels >= 0, group_labels + label_offset, -1
        )
        labels[mask] = group_labels_offset
        if np.any(group_labels >= 0):
            label_offset = int(group_labels_offset.max()) + 1

    umap_cluster_path = plots_dir / "umap_by_cluster.png"
    plot_umap_by_cluster(embeddings, labels, umap_cluster_path)

    # Within-prompt vs pooled diversity
    prompt_indices = metadata["prompt_indices"]
    diversity_path = plots_dir / "within_vs_pooled_diversity.png"
    plot_within_vs_pooled_diversity(embeddings, scales, prompt_indices, diversity_path)

    # Metrics bar charts
    with open(metrics_path) as f:
        metrics = json.load(f)
    metrics_bars_path = plots_dir / "metrics_bars.png"
    plot_metrics_bars(metrics, metrics_bars_path)

    all_plot_paths = [
        str(umap_scale_path),
        str(umap_cluster_path),
        str(diversity_path),
        str(metrics_bars_path),
    ]
    save_provenance(
        step="05_visualize",
        config_path=args.config,
        cfg=cfg,
        inputs={"embeddings": emb_path, "metrics": metrics_path},
        outputs=all_plot_paths,
    )


if __name__ == "__main__":
    main()
