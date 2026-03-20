"""Compute pass@k from evalplus eval_results.json and save in standard format.

Use this after running evalplus.evaluate on merged or standalone sample files.
Produces the same pass_at_k.json format as eval_code.py, suitable for
combining into pass_at_k_curves.json and plotting.

Usage:
    uv run python scripts/eval/compute_passk_from_eval.py \
        --eval-results outputs/.../merged_n100_eval_results.json \
        --scale 2.0 --temperature 0.8 --dataset humaneval \
        --output outputs/.../pass_at_k.json

    # Combine multiple pass_at_k.json files into a curves file for plotting:
    uv run python scripts/eval/compute_passk_from_eval.py \
        --combine outputs/.../scale_0.0/pass_at_k.json outputs/.../scale_2.0/pass_at_k.json \
        --curves-output outputs/.../pass_at_k_curves.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.pass_at_k import pass_at_k_curve


def extract_per_problem_results(
    eval_results: dict,
    use_plus: bool = True,
) -> list[tuple[int, int]]:
    """Extract (n, c) pairs from EvalPlus eval_results."""
    status_key = "plus_status" if use_plus else "base_status"
    results: list[tuple[int, int]] = []
    eval_data = eval_results.get("eval", {})
    for task_id in sorted(eval_data.keys()):
        samples = eval_data[task_id]
        n = len(samples)
        c = sum(1 for s in samples if s.get(status_key) == "pass")
        results.append((n, c))
    return results


def compute_passk(
    eval_results_path: Path,
    scale: float,
    temperature: float,
    dataset: str,
    k_values: list[int],
    output_path: Path,
) -> dict:
    """Compute pass@k from eval results and save."""
    with open(eval_results_path) as f:
        eval_results = json.load(f)

    per_problem_plus = extract_per_problem_results(eval_results, use_plus=True)
    per_problem_base = extract_per_problem_results(eval_results, use_plus=False)

    n_samples = per_problem_plus[0][0]
    valid_k = [k for k in k_values if k <= n_samples]
    curve_plus = pass_at_k_curve(per_problem_plus, valid_k)
    curve_base = pass_at_k_curve(per_problem_base, valid_k)

    result = {
        "scale": scale,
        "temperature": temperature,
        "dataset": dataset,
        "pass_at_k_plus": {str(k): v for k, v in curve_plus.items()},
        "pass_at_k_base": {str(k): v for k, v in curve_base.items()},
        "n_problems": len(per_problem_plus),
        "n_samples_per_problem": n_samples,
        "per_problem_plus": [{"n": n, "c": c} for n, c in per_problem_plus],
        "per_problem_base": [{"n": n, "c": c} for n, c in per_problem_base],
        "vector_path": "EasySteer/vectors/happy_diffmean.gguf",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved pass@k to {output_path}")
    print(f"  {result['n_problems']} problems, {n_samples} samples each")
    print(f"  k values: {valid_k}")
    print(f"  pass@1 (base): {curve_base[valid_k[0]]:.3f}")
    print(f"  pass@1 (plus): {curve_plus[valid_k[0]]:.3f}")

    return result


def combine_curves(input_paths: list[Path], output_path: Path) -> None:
    """Combine multiple pass_at_k.json files into a single curves file."""
    combined = []
    for path in input_paths:
        with open(path) as f:
            combined.append(json.load(f))
        print(f"  Loaded scale={combined[-1]['scale']} from {path}")

    combined.sort(key=lambda x: x["scale"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nCombined {len(combined)} conditions → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute pass@k from evalplus eval_results.json"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Compute subcommand
    compute = subparsers.add_parser("compute", help="Compute pass@k from eval results")
    compute.add_argument("--eval-results", required=True, type=Path)
    compute.add_argument("--scale", required=True, type=float)
    compute.add_argument("--temperature", type=float, default=0.8)
    compute.add_argument("--dataset", default="humaneval")
    compute.add_argument("--k-values", nargs="+", type=int,
                         default=[1, 2, 5, 10, 25, 50, 100])
    compute.add_argument("--output", required=True, type=Path)

    # Combine subcommand
    combine = subparsers.add_parser("combine", help="Combine pass_at_k.json files")
    combine.add_argument("inputs", nargs="+", type=Path)
    combine.add_argument("--output", required=True, type=Path)

    args = parser.parse_args()

    if args.command == "compute":
        compute_passk(
            args.eval_results, args.scale, args.temperature,
            args.dataset, args.k_values, args.output,
        )
    elif args.command == "combine":
        combine_curves(args.inputs, args.output)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
