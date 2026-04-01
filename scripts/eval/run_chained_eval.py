"""Run chained n=50+50 evaluation with intermediate results.

Splits a single n=100 eval into two n=50 batches with different seeds,
prints preliminary pass@k after the first batch, merges the two batches,
and computes final n=100 pass@k.

Usage:
    uv run python scripts/eval/run_chained_eval.py \
        --config configs/eval_code_qwen3_unsteered.yaml

    # With explicit batch sizes:
    uv run python scripts/eval/run_chained_eval.py \
        --config configs/eval_code_qwen3_unsteered.yaml \
        --batch-size 50 --seeds 42 137
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

from src.eval_config import CodeEvalConfig
from src.pass_at_k import pass_at_k_curve
from src.utils import ensure_dir, save_provenance, seed_everything

# Import the core functions from eval_code
from scripts.eval.eval_code import (
    extract_per_problem_results,
    run_evalplus_codegen,
    run_evalplus_evaluate,
    run_single_condition,
    set_steering_scale,
    verify_server_steering,
)
from scripts.eval.merge_split_samples import merge_samples

logger = logging.getLogger(__name__)


def print_pass_at_k(label: str, curve: dict[int, float]) -> None:
    """Pretty-print a pass@k curve."""
    header = f"  {label}:"
    vals = "  ".join(f"k={k}: {v:.4f}" for k, v in sorted(curve.items()))
    print(f"{header} {vals}")


def run_chained(
    config_path: Path,
    batch_size: int,
    seeds: list[int],
) -> None:
    """Run chained evaluation batches and merge results."""
    base_cfg = CodeEvalConfig.from_yaml(config_path)
    total_n = base_cfg.pass_at_k.n_samples
    n_batches = len(seeds)

    if batch_size * n_batches < total_n:
        raise ValueError(
            f"batch_size={batch_size} × {n_batches} seeds = {batch_size * n_batches} "
            f"< n_samples={total_n}. Need at least {total_n} total samples."
        )

    logger.info(
        "Chained eval: %d batches × n=%d (total=%d), seeds=%s",
        n_batches, batch_size, batch_size * n_batches, seeds,
    )

    mode = base_cfg.endpoint.steering_mode

    # Verify server connectivity once upfront
    if mode == "server":
        verify_server_steering(base_cfg.endpoint.base_url)
    elif mode == "none":
        logger.info("Steering mode 'none': skipping steering verification.")

    total_t0 = time.monotonic()

    for temperature in base_cfg.pass_at_k.temperatures:
        for dataset in base_cfg.datasets:
            for scale in base_cfg.scales:
                condition_name = f"scale_{scale}_temp_{temperature}"
                batch_samples_paths: list[Path] = []

                for batch_idx, seed in enumerate(seeds):
                    batch_label = f"batch{batch_idx + 1}"
                    batch_run_name = f"{base_cfg.run_name}_{batch_label}"
                    batch_output_dir = (
                        base_cfg.output_dir / "code" / dataset / condition_name / batch_label
                    )
                    batch_log_dir = batch_output_dir / "logs"

                    logger.info(
                        "=== Batch %d/%d (seed=%d, n=%d) ===",
                        batch_idx + 1, n_batches, seed, batch_size,
                    )

                    seed_everything(seed)

                    # Set steering scale if in server mode
                    if mode == "server":
                        set_steering_scale(base_cfg.endpoint.base_url, scale)

                    codegen_url = base_cfg.endpoint.base_url
                    if not codegen_url.endswith("/v1"):
                        codegen_url = codegen_url.rstrip("/") + "/v1"

                    # Generate
                    samples_path = run_evalplus_codegen(
                        model_name=base_cfg.model.name,
                        dataset=dataset,
                        base_url=codegen_url,
                        n_samples=batch_size,
                        temperature=temperature,
                        max_tokens=base_cfg.pass_at_k.max_tokens,
                        output_dir=batch_output_dir,
                        log_dir=batch_log_dir,
                    )
                    batch_samples_paths.append(samples_path)

                    # Evaluate this batch
                    eval_results_path = run_evalplus_evaluate(
                        dataset, samples_path, batch_log_dir,
                    )
                    with open(eval_results_path) as f:
                        eval_results = json.load(f)

                    # Compute and print intermediate pass@k
                    per_problem_plus = extract_per_problem_results(
                        eval_results, use_plus=True,
                    )
                    per_problem_base = extract_per_problem_results(
                        eval_results, use_plus=False,
                    )
                    valid_k = [
                        k for k in base_cfg.pass_at_k.k_values if k <= batch_size
                    ]
                    curve_plus = pass_at_k_curve(per_problem_plus, valid_k)
                    curve_base = pass_at_k_curve(per_problem_base, valid_k)

                    print(f"\n--- Batch {batch_idx + 1} results (n={batch_size}, seed={seed}) ---")
                    print_pass_at_k("plus", curve_plus)
                    print_pass_at_k("base", curve_base)

                    # Save batch results
                    batch_results = {
                        "scale": scale,
                        "temperature": temperature,
                        "dataset": dataset,
                        "batch": batch_label,
                        "seed": seed,
                        "n_samples": batch_size,
                        "pass_at_k_plus": {str(k): v for k, v in curve_plus.items()},
                        "pass_at_k_base": {str(k): v for k, v in curve_base.items()},
                    }
                    batch_results_path = batch_output_dir / "pass_at_k.json"
                    ensure_dir(batch_results_path.parent)
                    with open(batch_results_path, "w") as f:
                        json.dump(batch_results, f, indent=2)

                # Merge all batches
                if len(batch_samples_paths) > 1:
                    merged_dir = (
                        base_cfg.output_dir / "code" / dataset / condition_name
                    )
                    merged_path = merged_dir / f"{base_cfg.model.name.replace('/', '--')}_openai_temp_{temperature}.jsonl"
                    ensure_dir(merged_path.parent)

                    print(f"\n--- Merging {len(batch_samples_paths)} batches ---")
                    merge_samples(batch_samples_paths, merged_path)

                    # Evaluate merged
                    merged_log_dir = merged_dir / "logs"
                    ensure_dir(merged_log_dir)
                    eval_results_path = run_evalplus_evaluate(
                        dataset, merged_path, merged_log_dir,
                    )
                    with open(eval_results_path) as f:
                        eval_results = json.load(f)

                    merged_n = batch_size * n_batches
                    per_problem_plus = extract_per_problem_results(
                        eval_results, use_plus=True,
                    )
                    per_problem_base = extract_per_problem_results(
                        eval_results, use_plus=False,
                    )
                    valid_k = [
                        k for k in base_cfg.pass_at_k.k_values if k <= merged_n
                    ]
                    curve_plus = pass_at_k_curve(per_problem_plus, valid_k)
                    curve_base = pass_at_k_curve(per_problem_base, valid_k)

                    print(f"\n=== FINAL merged results (n={merged_n}) ===")
                    print_pass_at_k("plus", curve_plus)
                    print_pass_at_k("base", curve_base)

                    # Save final merged results
                    final_results = {
                        "scale": scale,
                        "temperature": temperature,
                        "dataset": dataset,
                        "steering_mode": mode,
                        "pass_at_k_plus": {str(k): v for k, v in curve_plus.items()},
                        "pass_at_k_base": {str(k): v for k, v in curve_base.items()},
                        "n_problems": len(per_problem_plus),
                        "n_samples_per_problem": merged_n,
                        "per_problem_plus": [{"n": n, "c": c} for n, c in per_problem_plus],
                        "per_problem_base": [{"n": n, "c": c} for n, c in per_problem_base],
                        "seeds": seeds,
                        "batch_size": batch_size,
                        "vector_path": base_cfg.vector_path or "none",
                    }
                    final_path = merged_dir / "pass_at_k.json"
                    with open(final_path, "w") as f:
                        json.dump(final_results, f, indent=2)

                    save_provenance(
                        step="eval_code_chained",
                        config_path=str(config_path),
                        cfg=base_cfg,
                        inputs={
                            "batches": [str(p) for p in batch_samples_paths],
                            "merged": str(merged_path),
                        },
                        outputs=[str(final_path)],
                    )

    total_elapsed = time.monotonic() - total_t0
    print(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min)")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Chained n=50+50 code evaluation with intermediate results",
    )
    parser.add_argument("--config", required=True, help="Path to eval config YAML")
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="Samples per batch (default: 50)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 137],
        help="Seeds for each batch (default: 42 137)",
    )
    args = parser.parse_args()

    run_chained(
        config_path=Path(args.config),
        batch_size=args.batch_size,
        seeds=args.seeds,
    )


if __name__ == "__main__":
    main()
