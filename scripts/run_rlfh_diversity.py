"""Run rlfh-gen-div diversity metrics on existing responses.jsonl files.

Runs using the rlfh-gen-div submodule's own venv (rlfh-gen-div/.venv).

Usage:
    rlfh-gen-div/.venv/bin/python scripts/run_rlfh_diversity.py --input outputs/happy_recon/responses.jsonl
    rlfh-gen-div/.venv/bin/python scripts/run_rlfh_diversity.py --input outputs/happy_recon/responses.jsonl --model-metrics
    rlfh-gen-div/.venv/bin/python scripts/run_rlfh_diversity.py --input outputs/happy_recon/responses.jsonl --model-metrics --openai
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add rlfh-gen-div to path so we can import its diversity module
RLFH_DIR = Path(__file__).resolve().parent.parent / "rlfh-gen-div" / "rlvsil"
sys.path.insert(0, str(RLFH_DIR))

from diversity import DEFAULT_CONFIGS, initialise_metrics, calculate_output_diversity


# Metrics that require loading transformer models (slow, benefits from GPU)
MODEL_METRICS = {"sent_bert_from_sim", "nli_from_sim", "nli_sample_from_sim"}

# Metrics that require an OpenAI API key
OPENAI_METRICS = {"openai_from_sim"}


def load_responses(path: Path) -> dict[float, dict[int, list[str]]]:
    """Load responses.jsonl and group by scale -> prompt_idx -> list of responses."""
    grouped: dict[float, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            scale = float(record["scale"])
            prompt_idx = int(record["prompt_idx"])
            grouped[scale][prompt_idx].append(record["response"])
    return grouped


def run_diversity_for_scale(
    responses_by_prompt: dict[int, list[str]],
    metrics: list,
    sample_overall: bool = True,
    scale_label: str = "",
) -> dict[str, float]:
    """Run diversity metrics on all prompts for a single scale, with progress bars."""
    outputss = [responses_by_prompt[idx] for idx in sorted(responses_by_prompt)]
    results: dict[str, float] = {}

    # --- Per-input diversity ---
    per_input_diversities = []
    for outputs in tqdm(outputss, desc=f"  per-input (scale={scale_label})", leave=False):
        per_input_diversities.append(calculate_output_diversity(outputs, metrics))

    if per_input_diversities:
        keys = per_input_diversities[0].keys()
        for k in keys:
            vals = [d[k] for d in per_input_diversities]
            results[f"mean_per_input_{k}"] = float(np.mean(vals))
            results[f"std_per_input_{k}"] = float(np.std(vals))

    # --- Overall diversity (all responses pooled) ---
    all_outputs = [output for outputs in outputss for output in outputs]
    if sample_overall and len(all_outputs) > 500:
        all_outputs = list(np.random.choice(all_outputs, replace=False, size=500))

    overall = calculate_output_diversity(all_outputs, metrics)
    for k, v in overall.items():
        results[f"overall_{k}"] = float(v)

    # --- Overall single-output diversity (one response per prompt) ---
    single_outputs = [outputs[0] for outputs in outputss]
    single = calculate_output_diversity(single_outputs, metrics)
    for k, v in single.items():
        results[f"overall_single_output_{k}"] = float(v)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rlfh-gen-div diversity metrics")
    parser.add_argument("--input", required=True, type=Path, help="Path to responses.jsonl")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSON path (default: same dir as input, named rlfh_diversity.json)",
    )
    parser.add_argument(
        "--model-metrics", action="store_true",
        help="Include model-based metrics (SentBERT, NLI). Slower, benefits from GPU.",
    )
    parser.add_argument(
        "--openai", action="store_true",
        help="Include OpenAI embedding metric (requires OPENAI_API_KEY env var).",
    )
    parser.add_argument(
        "--no-sample-overall", action="store_true",
        help="Don't subsample for overall metrics (slower but exact).",
    )
    args = parser.parse_args()

    # Build metric config
    excluded = set()
    if not args.model_metrics:
        excluded |= MODEL_METRICS
    if not args.openai:
        excluded |= OPENAI_METRICS
    metric_configs = {k: v for k, v in DEFAULT_CONFIGS.items() if k not in excluded}

    print(f"Using metrics: {list(metric_configs.keys())}")

    # Initialise metrics once (loads models only once)
    print("Initialising metrics...")
    metrics = initialise_metrics(metric_configs)
    print(f"Initialised {len(metrics)} metrics.")

    # Load and group responses
    grouped = load_responses(args.input)
    scales = sorted(grouped.keys())
    print(f"\nFound {len(scales)} scales: {scales}")
    for scale in scales:
        n_prompts = len(grouped[scale])
        n_responses = sum(len(v) for v in grouped[scale].values())
        print(f"  scale={scale}: {n_prompts} prompts, {n_responses} responses")

    # Run per scale
    results: list[dict] = []
    for scale in tqdm(scales, desc="Scales"):
        diversity = run_diversity_for_scale(
            grouped[scale],
            metrics,
            sample_overall=not args.no_sample_overall,
            scale_label=str(scale),
        )
        diversity["scale"] = scale
        results.append(diversity)

    # Print summary
    print(f"\n{'='*70}")
    print("Summary (mean_per_input metrics by scale):")
    print(f"{'='*70}")
    metric_names = [k for k in results[0] if k.startswith("mean_per_input_")]
    header = f"{'scale':>6}" + "".join(f"  {m.replace('mean_per_input_', ''):>25}" for m in metric_names)
    print(header)
    for r in results:
        row = f"{r['scale']:>6.1f}"
        for m in metric_names:
            row += f"  {r[m]:>25.4f}"
        print(row)

    # Save
    output_path = args.output or (args.input.parent / "rlfh_diversity.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
