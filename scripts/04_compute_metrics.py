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
from src.statistics import run_all_statistical_tests
from src.utils import ensure_dir, save_provenance, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute clustering + diversity metrics")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--embeddings",
        type=str,
        default=None,
        help="Path to embeddings .npz (default: outputs/<run>/embeddings.npz)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for metrics JSON (default: outputs/<run>/metrics.json)",
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
        mask = np.isclose(scales, scale)
        group_embs = embeddings[mask]

        labels = cluster_embeddings(group_embs, cfg.clustering)
        metrics = compute_diversity_metrics(group_embs, labels)
        metrics["scale"] = scale
        metrics["n_responses"] = int(mask.sum())
        all_metrics.append(metrics)

    # Save metrics JSON
    metrics_path = args.output or str(out_dir / "metrics.json")
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Print summary table
    df = pd.DataFrame(all_metrics)
    df = df.set_index("scale")
    print("\n" + df.to_string())

    # --- Statistical tests ---
    # Compute full labels array for all embeddings (needed for per-prompt clustering)
    all_labels = np.full(len(embeddings), -1, dtype=int)
    label_offset = 0
    for scale in unique_scales:
        mask = np.isclose(scales, scale)
        group_labels = cluster_embeddings(embeddings[mask], cfg.clustering)
        # Offset cluster labels so they don't collide across scales
        group_labels_offset = np.where(
            group_labels >= 0, group_labels + label_offset, -1
        )
        all_labels[mask] = group_labels_offset
        if len(group_labels[group_labels >= 0]) > 0:
            label_offset = int(group_labels_offset.max()) + 1

    prompt_indices = metadata["prompt_indices"]

    print("\nRunning pre-registered statistical tests …")
    stats_results = run_all_statistical_tests(
        embeddings, all_labels, scales, prompt_indices, seed=cfg.seed,
    )

    stats_path = str(out_dir / "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats_results, f, indent=2)
    print(f"Saved statistical tests to {stats_path}")

    if stats_results.get("skipped"):
        print(f"  (skipped: {stats_results['reason']})")
    else:
        # Print key results
        hb = stats_results["holm_bonferroni"]
        print("\n--- Page's L tests (Holm-Bonferroni corrected) ---")
        for metric_name in sorted(hb.keys()):
            entry = hb[metric_name]
            sig = "*" if entry["significant"] else ""
            print(f"  {metric_name}: p_adj={entry['adjusted_p']:.4f} {sig}")

        me = stats_results["mixed_effects"]
        if me.get("beta") is not None:
            print(f"\n--- Mixed-effects (primary) ---")
            print(f"  beta={me['beta']:.6f}, 95% CI=[{me['ci_95_low']:.6f}, {me['ci_95_high']:.6f}], p={me['p_value']:.4f}")

        se = stats_results["spearman_effect_size"]
        print(f"\n--- Spearman effect size ---")
        print(f"  rho={se['rho']:.4f}, 95% CI=[{se['ci_95_low']:.4f}, {se['ci_95_high']:.4f}], p={se['p_value']:.4f}")

    # Provenance
    output_files = [metrics_path, stats_path]
    save_provenance(
        step="04_compute_metrics",
        config_path=args.config,
        cfg=cfg,
        inputs={"embeddings": emb_path},
        outputs=output_files,
    )


if __name__ == "__main__":
    main()
