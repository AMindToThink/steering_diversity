"""Plot rlfh-gen-div diversity metrics as a function of steering strength.

Usage:
    uv run python scripts/plot_rlfh_diversity.py --input outputs/happy_recon/rlfh_diversity_fast.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_description(plot_path: Path, text: str) -> None:
    """Write a .description.txt sidecar file for a plot."""
    desc_path = plot_path.with_suffix(".description.txt")
    desc_path.write_text(text.strip() + "\n")
    print(f"Saved {desc_path}")


METRIC_LABELS: dict[str, str] = {
    "averaged_distinct_ngrams": "Distinct N-grams (avg n=1..5)",
    "ead_averaged_distinct_ngrams": "Expectation-Adjusted Distinct N-grams",
    "cosine_similarity_2d_diversity": "N-gram Cosine Diversity",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot rlfh diversity metrics vs steering scale")
    parser.add_argument("--input", required=True, type=Path, help="Path to rlfh_diversity JSON")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for plots")
    args = parser.parse_args()

    with open(args.input) as f:
        results = json.load(f)

    output_dir = args.output_dir or args.input.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    scales = np.array([r["scale"] for r in results])

    # Identify which metrics are present
    metric_names = [
        k.replace("mean_per_input_", "")
        for k in results[0]
        if k.startswith("mean_per_input_")
    ]

    # --- Plot 1: Per-input diversity (mean +/- std) for each metric ---
    fig, axes = plt.subplots(1, len(metric_names), figsize=(6 * len(metric_names), 5), squeeze=False)
    axes = axes[0]

    for ax, metric in zip(axes, metric_names):
        means = np.array([r[f"mean_per_input_{metric}"] for r in results])
        stds = np.array([r[f"std_per_input_{metric}"] for r in results])

        ax.errorbar(scales, means, yerr=stds, fmt="o-", capsize=4, linewidth=2, markersize=6)
        ax.set_xlabel("Steering Scale", fontsize=12)
        ax.set_ylabel("Diversity", fontsize=12)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Per-Input Diversity vs Steering Scale (mean ± std across prompts)", fontsize=13, y=1.02)
    fig.tight_layout()
    save_path = output_dir / "rlfh_per_input_diversity.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")
    save_description(save_path, """\
Per-Input Diversity vs Steering Scale (mean +/- std across prompts)

Each metric is computed independently within each prompt's 50 responses, then
averaged across the 10 prompts. Error bars show standard deviation across prompts.

Metrics used (all CPU-only, no model-based):
  - Distinct N-grams (averaged n=1..5): unique_ngrams / total_ngrams
  - Expectation-Adjusted Distinct N-grams: adjusts for chance given vocab size
  - N-gram Cosine Diversity: 1 - mean pairwise cosine similarity of trigram vectors
""")

    # --- Plot 2: All three measurement levels for each metric ---
    levels = [
        ("mean_per_input_", "Per-Input (mean)"),
        ("overall_", "Overall (pooled)"),
        ("overall_single_output_", "Single-Output (cross-prompt)"),
    ]

    fig, axes = plt.subplots(1, len(metric_names), figsize=(6 * len(metric_names), 5), squeeze=False)
    axes = axes[0]

    for ax, metric in zip(axes, metric_names):
        for prefix, label in levels:
            key = f"{prefix}{metric}"
            if key in results[0]:
                vals = np.array([r[key] for r in results])
                ax.plot(scales, vals, "o-", label=label, linewidth=2, markersize=5)

        ax.set_xlabel("Steering Scale", fontsize=12)
        ax.set_ylabel("Diversity", fontsize=12)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Diversity Levels vs Steering Scale", fontsize=13, y=1.02)
    fig.tight_layout()
    save_path = output_dir / "rlfh_diversity_levels.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")
    save_description(save_path, """\
Diversity Levels vs Steering Scale

Compares three measurement levels for each metric:
  - Per-Input (mean): diversity computed within each prompt's responses, averaged
  - Overall (pooled): all responses at a given scale pooled into one set
  - Single-Output (cross-prompt): one response per prompt, measuring cross-prompt diversity

Overall pooled is consistently lower for distinct n-gram metrics. This is a
corpus-size artifact: the ratio unique/total drops when pooling more text because
the denominator grows faster than the numerator, especially for common n-grams.
""")

    # --- Plot 3: Combined per-input metrics on same axes for comparison ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in metric_names:
        means = np.array([r[f"mean_per_input_{metric}"] for r in results])
        stds = np.array([r[f"std_per_input_{metric}"] for r in results])
        label = METRIC_LABELS.get(metric, metric)
        ax.errorbar(scales, means, yerr=stds, fmt="o-", capsize=4, linewidth=2, markersize=5, label=label)

    ax.set_xlabel("Steering Scale", fontsize=12)
    ax.set_ylabel("Diversity", fontsize=12)
    ax.set_title("Per-Input Diversity Metrics vs Steering Scale", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path = output_dir / "rlfh_diversity_combined.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")
    save_description(save_path, """\
Per-Input Diversity Metrics vs Steering Scale (all metrics overlaid)

All three per-input metrics on the same axes. N-gram cosine diversity sits near
1.0 for all scales, making it uninformative. Distinct n-grams and expectation-
adjusted distinct n-grams track closely, both flat from scale 0-4 with a sharp
jump at scale 8.
""")


if __name__ == "__main__":
    main()
