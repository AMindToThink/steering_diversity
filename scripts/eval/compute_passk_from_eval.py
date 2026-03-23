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

import numpy as np
from scipy import stats

from src.pass_at_k import pass_at_k, pass_at_k_curve


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


def extract_per_problem_results_bcb(
    eval_results: dict,
) -> list[tuple[int, int]]:
    """Extract (n, c) pairs from BigCodeBench eval_results.

    BigCodeBench format: eval[task_id] is a list of per-sample dicts, each with
    a ``status`` field ("pass" or failure string). No plus/base distinction.
    """
    results: list[tuple[int, int]] = []
    eval_data = eval_results.get("eval", {})
    for task_id in sorted(eval_data.keys()):
        samples = eval_data[task_id]
        n = len(samples)
        c = sum(1 for s in samples if s.get("status") == "pass")
        results.append((n, c))
    return results


def compute_passk_bcb(
    eval_results_path: Path,
    scale: float,
    temperature: float,
    dataset: str,
    k_values: list[int],
    output_path: Path,
    steering_label: str | None = None,
) -> dict:
    """Compute pass@k from BigCodeBench eval results and save.

    Stores results in both plus and base fields for coverage_gain_test() compat.
    """
    with open(eval_results_path) as f:
        eval_results = json.load(f)

    per_problem = extract_per_problem_results_bcb(eval_results)

    n_samples = per_problem[0][0]
    valid_k = [k for k in k_values if k <= n_samples]
    curve = pass_at_k_curve(per_problem, valid_k)

    per_problem_dicts = [{"n": n, "c": c} for n, c in per_problem]

    result: dict = {
        "scale": scale,
        "temperature": temperature,
        "dataset": dataset,
        "benchmark": "bigcodebench",
        "pass_at_k_plus": {str(k): v for k, v in curve.items()},
        "pass_at_k_base": {str(k): v for k, v in curve.items()},
        "n_problems": len(per_problem),
        "n_samples_per_problem": n_samples,
        "per_problem_plus": per_problem_dicts,
        "per_problem_base": per_problem_dicts,
    }
    if steering_label is not None:
        result["steering_label"] = steering_label

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved pass@k to {output_path}")
    print(f"  {result['n_problems']} problems, {n_samples} samples each")
    print(f"  k values: {valid_k}")
    print(f"  pass@1: {curve[valid_k[0]]:.3f}")

    return result


def _per_problem_passk(per_problem: list[dict], k: int) -> np.ndarray:
    """Compute per-problem pass@k scores from list of {n, c} dicts."""
    return np.array([pass_at_k(p["n"], p["c"], k) for p in per_problem])


def _load_and_pool_per_problem(
    paths: list[Path],
    per_problem_key: str,
    passk_key: str,
) -> tuple[list[dict], set[int], list[str]]:
    """Load per-problem data from one or more pass_at_k.json files.

    Returns:
        per_problem: Concatenated list of {n, c} dicts across all files.
        k_values: Intersection of k_values across all files.
        datasets: List of dataset names from each file.
    """
    all_per_problem: list[dict] = []
    all_k_sets: list[set[int]] = []
    datasets: list[str] = []

    for path in paths:
        with open(path) as f:
            data = json.load(f)
        all_per_problem.extend(data[per_problem_key])
        all_k_sets.append({int(k) for k in data[passk_key].keys()})
        datasets.append(data.get("dataset", "unknown"))

    k_values = all_k_sets[0]
    for ks in all_k_sets[1:]:
        k_values = k_values & ks

    return all_per_problem, k_values, datasets


def coverage_gain_test(
    baseline_paths: list[Path],
    steered_paths: list[Path],
    use_plus: bool = True,
    output_path: Path | None = None,
) -> dict:
    """Test whether steering reduces diversity beyond what pass@1 drop explains.

    Accepts one or more pass_at_k.json files per condition. When multiple files
    are given, per-problem data is concatenated (pooled across benchmarks).

    Computes coverage gain (pass@k - pass@1) per problem for each condition,
    then runs:
      1. Paired t-test on coverage gain at each k (with k=10 as pre-specified primary)
      2. Omnibus interaction test: mean coverage gain across all k values

    Returns dict with full results.
    """
    if len(baseline_paths) != len(steered_paths):
        raise ValueError(
            f"Must provide equal number of baseline and steered files. "
            f"Got {len(baseline_paths)} baseline, {len(steered_paths)} steered."
        )

    suffix = "plus" if use_plus else "base"
    per_problem_key = f"per_problem_{suffix}"
    passk_key = f"pass_at_k_{suffix}"

    pp_base, k_values_base, datasets_base = _load_and_pool_per_problem(
        baseline_paths, per_problem_key, passk_key,
    )
    pp_steer, k_values_steer, datasets_steer = _load_and_pool_per_problem(
        steered_paths, per_problem_key, passk_key,
    )

    if len(pp_base) != len(pp_steer):
        raise ValueError(
            f"Problem count mismatch: baseline has {len(pp_base)}, "
            f"steered has {len(pp_steer)}"
        )

    k_values = sorted(k_values_base & k_values_steer)
    if not k_values:
        raise ValueError("No common k_values across all files")
    k_values_cg = [k for k in k_values if k > 1]

    # Read scales from first files for reporting
    with open(baseline_paths[0]) as f:
        baseline_meta = json.load(f)
    with open(steered_paths[0]) as f:
        steered_meta = json.load(f)

    # pass@1 per problem
    passk1_base = _per_problem_passk(pp_base, 1)
    passk1_steer = _per_problem_passk(pp_steer, 1)

    # Per-k coverage gain tests
    per_k_results: list[dict] = []
    all_cg_diffs: list[np.ndarray] = []

    for k in k_values_cg:
        cg_base = _per_problem_passk(pp_base, k) - passk1_base
        cg_steer = _per_problem_passk(pp_steer, k) - passk1_steer
        diff = cg_steer - cg_base
        all_cg_diffs.append(diff)

        mean_diff = float(diff.mean())
        se_diff = float(diff.std(ddof=1) / np.sqrt(len(diff)))
        t_stat, p_val = stats.ttest_rel(cg_steer, cg_base)

        per_k_results.append({
            "k": k,
            "mean_cg_baseline": float(cg_base.mean()),
            "mean_cg_steered": float(cg_steer.mean()),
            "delta_cg": mean_diff,
            "se": se_diff,
            "ci_95_low": mean_diff - 1.96 * se_diff,
            "ci_95_high": mean_diff + 1.96 * se_diff,
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "significant": bool(p_val < 0.05),
        })

    # Omnibus interaction test: mean coverage gain diff across all k
    # For each problem, average the coverage gain difference across k values
    stacked = np.column_stack(all_cg_diffs)  # shape: (n_problems, n_k)
    mean_cg_diff_per_problem = stacked.mean(axis=1)
    omnibus_t, omnibus_p = stats.ttest_1samp(mean_cg_diff_per_problem, 0.0)

    # Find pre-specified primary (k=10)
    primary_k = 10
    primary_result = next((r for r in per_k_results if r["k"] == primary_k), None)

    # Track which datasets were pooled
    all_datasets = datasets_base + datasets_steer
    unique_datasets = sorted(set(all_datasets))

    result = {
        "baseline_scale": baseline_meta["scale"],
        "steered_scale": steered_meta["scale"],
        "metric": suffix,
        "n_problems": len(pp_base),
        "datasets_pooled": unique_datasets,
        "n_baseline_files": len(baseline_paths),
        "n_steered_files": len(steered_paths),
        "pass_at_1_baseline": float(passk1_base.mean()),
        "pass_at_1_steered": float(passk1_steer.mean()),
        "primary_k": primary_k,
        "primary_result": primary_result,
        "per_k_results": per_k_results,
        "omnibus_test": {
            "description": "One-sample t-test on mean coverage gain difference across all k",
            "mean_delta_cg": float(mean_cg_diff_per_problem.mean()),
            "t_stat": float(omnibus_t),
            "p_value": float(omnibus_p),
            "significant": bool(omnibus_p < 0.05),
        },
    }

    # Print results table
    datasets_str = " + ".join(unique_datasets)
    pooled_label = f" (pooled: {datasets_str})" if len(unique_datasets) > 1 else ""
    print(f"\nCoverage gain test: α={baseline_meta['scale']} vs α={steered_meta['scale']} ({suffix}){pooled_label}")
    print(f"  n={len(pp_base)} problems, pass@1: {passk1_base.mean():.3f} → {passk1_steer.mean():.3f}")
    print()
    print(f"  {'k':>5}  {'Δ cov gain':>10}  {'SE':>8}  {'p-value':>10}  {'sig':>5}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*5}")
    for r in per_k_results:
        sig_marker = "*" if r["significant"] else ""
        primary_marker = " ◄" if r["k"] == primary_k else ""
        print(
            f"  {r['k']:>5}  {r['delta_cg']:>10.4f}  {r['se']:>8.4f}  "
            f"{r['p_value']:>10.4f}  {sig_marker:>5}{primary_marker}"
        )
    print()
    print(f"  Omnibus interaction test:")
    omni = result["omnibus_test"]
    sig = "*" if omni["significant"] else ""
    print(f"    mean Δ coverage gain = {omni['mean_delta_cg']:.4f}, "
          f"t = {omni['t_stat']:.3f}, p = {omni['p_value']:.4f} {sig}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Saved to {output_path}")

    return result


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

    # Compute from BigCodeBench eval results
    compute_bcb = subparsers.add_parser(
        "compute-bcb", help="Compute pass@k from BigCodeBench eval results"
    )
    compute_bcb.add_argument("--eval-results", required=True, type=Path)
    compute_bcb.add_argument("--scale", required=True, type=float)
    compute_bcb.add_argument("--temperature", type=float, default=0.8)
    compute_bcb.add_argument("--dataset", default="bigcodebench")
    compute_bcb.add_argument("--k-values", nargs="+", type=int,
                             default=[1, 2, 5, 10, 25, 50, 100])
    compute_bcb.add_argument("--steering-label", type=str, default=None,
                             help="Human-readable steering description for plot labels")
    compute_bcb.add_argument("--output", required=True, type=Path)

    # Combine subcommand
    combine = subparsers.add_parser("combine", help="Combine pass_at_k.json files")
    combine.add_argument("inputs", nargs="+", type=Path)
    combine.add_argument("--output", required=True, type=Path)

    # Test subcommand
    test = subparsers.add_parser(
        "test", help="Test whether steering reduces diversity beyond pass@1 drop"
    )
    test.add_argument("--baseline", required=True, type=Path, nargs="+",
                      help="pass_at_k.json file(s) for unsteered condition")
    test.add_argument("--steered", required=True, type=Path, nargs="+",
                      help="pass_at_k.json file(s) for steered condition")
    test.add_argument("--metric", choices=["plus", "base"], default="plus",
                      help="Which EvalPlus metric (default: plus)")
    test.add_argument("--output", type=Path,
                      help="Optional path to save results JSON")

    args = parser.parse_args()

    if args.command == "compute":
        compute_passk(
            args.eval_results, args.scale, args.temperature,
            args.dataset, args.k_values, args.output,
        )
    elif args.command == "compute-bcb":
        compute_passk_bcb(
            args.eval_results, args.scale, args.temperature,
            args.dataset, args.k_values, args.output,
            steering_label=args.steering_label,
        )
    elif args.command == "combine":
        combine_curves(args.inputs, args.output)
    elif args.command == "test":
        coverage_gain_test(
            baseline_paths=args.baseline,
            steered_paths=args.steered,
            use_plus=(args.metric == "plus"),
            output_path=args.output,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
