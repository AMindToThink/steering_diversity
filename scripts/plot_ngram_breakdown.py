"""Plot distinct n-gram diversity broken down by n (1..5) vs steering scale.

Usage:
    uv run python scripts/plot_ngram_breakdown.py --input outputs/happy_recon/responses.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Use rlfh-gen-div's n-gram utility
RLFH_DIR = Path(__file__).resolve().parent.parent / "rlfh-gen-div" / "rlvsil"
sys.path.insert(0, str(RLFH_DIR))
from diversity.utils import lines_to_ngrams


def save_description(plot_path: Path, text: str) -> None:
    """Write a .description.txt sidecar file for a plot."""
    desc_path = plot_path.with_suffix(".description.txt")
    desc_path.write_text(text.strip() + "\n")
    print(f"Saved {desc_path}")


def distinct_ngrams(responses: list[str], n: int) -> float:
    """Compute unique_ngrams / total_ngrams for a set of responses."""
    ngram_lists = lines_to_ngrams(responses, n=n)
    flat = [ng for sublist in ngram_lists for ng in sublist]
    return len(set(flat)) / len(flat) if flat else 0.0


def load_responses(path: Path) -> dict[float, dict[int, list[str]]]:
    grouped: dict[float, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            grouped[float(record["scale"])][int(record["prompt_idx"])].append(record["response"])
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-max", type=int, default=5)
    args = parser.parse_args()

    output_dir = args.output_dir or args.input.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = load_responses(args.input)
    scales = sorted(grouped.keys())
    ns = list(range(1, args.n_max + 1))

    # Compute within-prompt and pooled distinct-ngrams for each (scale, n)
    within_means: dict[int, list[float]] = {n: [] for n in ns}
    within_stds: dict[int, list[float]] = {n: [] for n in ns}
    pooled_vals: dict[int, list[float]] = {n: [] for n in ns}

    for scale in scales:
        prompts = grouped[scale]
        all_responses = [r for idx in sorted(prompts) for r in prompts[idx]]
        for n in ns:
            # Within-prompt: compute per prompt, then mean/std
            per_prompt = [distinct_ngrams(prompts[idx], n) for idx in sorted(prompts)]
            within_means[n].append(float(np.mean(per_prompt)))
            within_stds[n].append(float(np.std(per_prompt)))
            # Pooled: all responses for this scale together
            pooled_vals[n].append(distinct_ngrams(all_responses, n))

    # --- One subplot per n ---
    fig, axes = plt.subplots(1, len(ns), figsize=(4.5 * len(ns), 4.5), squeeze=False)
    axes = axes[0]
    scales_arr = np.array(scales)

    for ax, n in zip(axes, ns):
        w_means = np.array(within_means[n])
        w_stds = np.array(within_stds[n])
        p_vals = np.array(pooled_vals[n])

        ax.errorbar(scales_arr, w_means, yerr=w_stds, fmt="o-", capsize=4, linewidth=2,
                     markersize=5, label="Within-prompt", color="C0")
        ax.plot(scales_arr, p_vals, "s-", linewidth=2, markersize=5,
                label="Pooled across prompts", color="C1")

        n_label = "unigrams" if n == 1 else f"{n}-grams"
        ax.set_title(f"n={n} ({n_label})", fontsize=12)
        ax.set_xlabel("Steering Scale", fontsize=11)
        ax.set_ylabel("Distinct N-grams", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Distinct N-grams: Within-Prompt vs Pooled", fontsize=13, y=1.02)
    fig.tight_layout()
    save_path = output_dir / "rlfh_ngram_breakdown.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")
    save_description(save_path, """\
Distinct N-grams: Within-Prompt vs Pooled, broken down by n (1..5)

For each n-gram size, two lines are shown:
  - Within-prompt (blue, with std error bars): distinct n-grams computed per
    prompt's responses, then averaged across prompts.
  - Pooled across prompts (orange): all responses at a given scale combined.

Within-prompt is consistently HIGHER than pooled. This is counterintuitive but
is a corpus-size artifact: distinct n-grams = unique/total is corpus-size-
dependent. Pooling 10x more text inflates the denominator faster than the
numerator, especially for small n where common words dominate.

The gap shrinks for higher n since longer n-grams are almost all unique.
The scale-8 jump appears in both lines for all n.
""")


if __name__ == "__main__":
    main()
