"""Visualize pass@k curves across steering scales.

Reads the aggregated pass_at_k_curves.json and produces per-dataset plots
showing pass@k vs k for each steering scale.

Usage:
    uv run python scripts/eval/plot_pass_at_k.py --config configs/eval_code.yaml
    uv run python scripts/eval/plot_pass_at_k.py --input outputs/passk_code_v1/code/pass_at_k_curves.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.utils import ensure_dir

logger = logging.getLogger(__name__)

# Color palette for scales — visually distinct, colorblind-friendly
SCALE_CMAP = "viridis"


def load_curves(path: Path) -> list[dict]:
    """Load aggregated pass@k curves JSON."""
    with open(path) as f:
        return json.load(f)


def plot_pass_at_k_by_dataset(
    curves: list[dict],
    output_dir: Path,
    title_prefix: str = "",
) -> list[Path]:
    """Create one plot per (dataset, temperature) showing pass@k vs k for each scale.

    Returns list of saved plot paths.
    """
    sns.set_theme(style="whitegrid")
    saved: list[Path] = []

    # Group by (dataset, temperature)
    groups: dict[tuple[str, float], list[dict]] = {}
    for entry in curves:
        key = (entry["dataset"], entry["temperature"])
        groups.setdefault(key, []).append(entry)

    for (dataset, temp), entries in sorted(groups.items()):
        fig, ax = plt.subplots(figsize=(8, 5))

        # Sort entries by scale for consistent legend ordering
        entries.sort(key=lambda e: e["scale"])

        scales = [e["scale"] for e in entries]
        cmap = plt.get_cmap(SCALE_CMAP)
        colors = [cmap(i / max(len(scales) - 1, 1)) for i in range(len(scales))]

        for entry, color in zip(entries, colors):
            pak = entry["pass_at_k"]
            k_values = sorted(int(k) for k in pak.keys())
            scores = [pak[str(k)] for k in k_values]

            ax.plot(
                k_values,
                scores,
                marker="o",
                markersize=4,
                label=f"scale={entry['scale']}",
                color=color,
                linewidth=2,
            )

        ax.set_xscale("log")
        ax.set_xlabel("k (number of attempts)", fontsize=12)
        ax.set_ylabel("pass@k", fontsize=12)
        title = f"{title_prefix}{dataset} — pass@k vs k (T={temp})"
        ax.set_title(title, fontsize=13)
        ax.legend(title="Steering scale", loc="lower right")
        ax.set_ylim(0, 1.05)

        plot_path = output_dir / f"pass_at_k_{dataset}_temp{temp}.png"
        ensure_dir(output_dir)
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(plot_path)
        logger.info("Saved %s", plot_path)

    return saved


def plot_crossover_analysis(
    curves: list[dict],
    output_dir: Path,
) -> list[Path]:
    """Plot the difference in pass@k between steered and baseline (scale=0).

    Shows where steering helps (positive) vs hurts (negative) as k grows.
    """
    sns.set_theme(style="whitegrid")
    saved: list[Path] = []

    groups: dict[tuple[str, float], list[dict]] = {}
    for entry in curves:
        key = (entry["dataset"], entry["temperature"])
        groups.setdefault(key, []).append(entry)

    for (dataset, temp), entries in sorted(groups.items()):
        # Find baseline
        baseline = next((e for e in entries if e["scale"] == 0.0), None)
        if baseline is None:
            logger.warning("No baseline (scale=0) for %s/T=%.1f, skipping crossover", dataset, temp)
            continue

        steered = [e for e in entries if e["scale"] != 0.0]
        if not steered:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        steered.sort(key=lambda e: e["scale"])

        scales = [e["scale"] for e in steered]
        cmap = plt.get_cmap(SCALE_CMAP)
        colors = [cmap(i / max(len(scales) - 1, 1)) for i in range(len(scales))]

        baseline_pak = baseline["pass_at_k"]
        k_values = sorted(int(k) for k in baseline_pak.keys())

        for entry, color in zip(steered, colors):
            pak = entry["pass_at_k"]
            diffs = [pak[str(k)] - baseline_pak[str(k)] for k in k_values]
            ax.plot(
                k_values,
                diffs,
                marker="o",
                markersize=4,
                label=f"scale={entry['scale']}",
                color=color,
                linewidth=2,
            )

        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("k (number of attempts)", fontsize=12)
        ax.set_ylabel("Δ pass@k (steered − baseline)", fontsize=12)
        ax.set_title(f"{dataset} — Crossover analysis (T={temp})", fontsize=13)
        ax.legend(title="Steering scale", loc="best")

        plot_path = output_dir / f"crossover_{dataset}_temp{temp}.png"
        ensure_dir(output_dir)
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(plot_path)
        logger.info("Saved %s", plot_path)

    return saved


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Plot pass@k curves")
    parser.add_argument("--config", help="Path to eval config YAML (derives input/output paths)")
    parser.add_argument("--input", help="Direct path to pass_at_k_curves.json")
    parser.add_argument("--output", help="Output directory for plots")
    args = parser.parse_args()

    if args.input:
        curves_path = Path(args.input)
        output_dir = Path(args.output) if args.output else curves_path.parent / "plots"
    elif args.config:
        from src.eval_config import CodeEvalConfig

        cfg = CodeEvalConfig.from_yaml(args.config)
        curves_path = cfg.output_dir / "code" / "pass_at_k_curves.json"
        output_dir = Path(args.output) if args.output else cfg.output_dir / "code" / "plots"
    else:
        parser.error("Either --config or --input is required")

    curves = load_curves(curves_path)
    logger.info("Loaded %d conditions from %s", len(curves), curves_path)

    all_plots: list[Path] = []
    all_plots.extend(plot_pass_at_k_by_dataset(curves, output_dir))
    all_plots.extend(plot_crossover_analysis(curves, output_dir))

    logger.info("Generated %d plots in %s", len(all_plots), output_dir)


if __name__ == "__main__":
    main()
