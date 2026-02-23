#!/usr/bin/env python3
"""Step 4: Cluster embeddings and compute diversity metrics (CPU-compatible)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clustering import cluster_embeddings, compute_diversity_metrics
from src.config import ExperimentConfig
from src.embedding import load_embeddings
from src.utils import ensure_dir, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute clustering + diversity metrics")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--embeddings",
        type=str,
        default=None,
        help="Path to embeddings .npz (default: outputs/<run>/embeddings.npz)",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    seed_everything(cfg.seed)

    out_dir = ensure_dir(cfg.output_dir)
    emb_path = args.embeddings or str(out_dir / "embeddings.npz")

    embeddings, metadata = load_embeddings(emb_path)
    scales = metadata["scales"]
    unique_scales = sorted(set(float(s) for s in scales))

    all_metrics: list[dict] = []

    for scale in unique_scales:
        mask = scales == scale
        group_embs = embeddings[mask]

        labels = cluster_embeddings(group_embs, cfg.clustering)
        metrics = compute_diversity_metrics(group_embs, labels)
        metrics["scale"] = scale
        metrics["n_responses"] = int(mask.sum())
        all_metrics.append(metrics)

    # Save JSON
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Print summary table
    df = pd.DataFrame(all_metrics)
    df = df.set_index("scale")
    print("\n" + df.to_string())


if __name__ == "__main__":
    main()
