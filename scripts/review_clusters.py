#!/usr/bin/env python3
"""Review clusters: dump response texts grouped by (scale, cluster) for inspection."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clustering import cluster_embeddings
from src.config import ExperimentConfig
from src.embedding import load_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Review clusters with response texts")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--responses", type=str, default=None)
    parser.add_argument("--embeddings", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument(
        "--max-per-cluster", type=int, default=5,
        help="Max responses to show per cluster (default: 5)",
    )
    parser.add_argument(
        "--scales", type=float, nargs="*", default=None,
        help="Only show these scales (default: all)",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    out_dir = Path(cfg.output_dir)

    # Load responses
    resp_path = args.responses or str(out_dir / "responses.jsonl")
    responses: list[dict] = []
    with open(resp_path) as f:
        for line in f:
            responses.append(json.loads(line))

    # Load embeddings and cluster per-scale
    emb_path = args.embeddings or str(out_dir / "embeddings.npz")
    embeddings, metadata = load_embeddings(emb_path)
    scales = metadata["scales"]
    unique_scales = sorted(set(float(s) for s in scales))

    if args.scales:
        unique_scales = [s for s in unique_scales if s in args.scales]

    # Build cluster labels per-scale
    all_labels = np.full(len(embeddings), -1, dtype=int)
    for scale in sorted(set(float(s) for s in scales)):
        mask = np.isclose(scales, scale)
        group_labels = cluster_embeddings(embeddings[mask], cfg.clustering)
        all_labels[mask] = group_labels

    # Group responses by (scale, cluster)
    report: dict[str, dict] = {}
    for scale in unique_scales:
        mask = np.isclose(scales, scale)
        indices = np.where(mask)[0]
        scale_labels = all_labels[mask]

        cluster_ids = sorted(set(int(l) for l in scale_labels))
        scale_key = f"scale_{scale}"
        scale_report: dict[str, list[dict]] = {}

        for cid in cluster_ids:
            cluster_name = "noise" if cid == -1 else f"cluster_{cid}"
            cluster_mask = scale_labels == cid
            cluster_indices = indices[cluster_mask]

            samples: list[dict] = []
            for idx in cluster_indices[: args.max_per_cluster]:
                r = responses[idx]
                samples.append({
                    "prompt_idx": r["prompt_idx"],
                    "prompt": r["prompt"][:200],
                    "response": r["response"],
                })

            scale_report[cluster_name] = {
                "count": int(cluster_mask.sum()),
                "samples": samples,
            }

        report[scale_key] = {
            "n_clusters": len([c for c in cluster_ids if c >= 0]),
            "n_noise": int((scale_labels == -1).sum()),
            "n_total": int(mask.sum()),
            "clusters": scale_report,
        }

    # Output
    output_path = args.output or str(out_dir / "cluster_review.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved cluster review to {output_path}")

    # Print summary
    for scale_key, info in report.items():
        print(f"\n{'='*60}")
        print(f"{scale_key}: {info['n_clusters']} clusters, {info['n_noise']} noise, {info['n_total']} total")
        for cname, cdata in info["clusters"].items():
            if cname == "noise":
                continue
            print(f"\n  --- {cname} ({cdata['count']} responses) ---")
            for s in cdata["samples"][:3]:
                text = s["response"][:150].replace("\n", " ")
                print(f"    [p{s['prompt_idx']}] {text}…")


if __name__ == "__main__":
    main()
