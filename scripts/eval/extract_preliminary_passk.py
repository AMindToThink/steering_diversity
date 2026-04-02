#!/usr/bin/env python3
"""Extract preliminary pass@k (n=10) results from raw eval data.

The unsteered n=10 HumanEval results were computed but never saved as a
pass_at_k.json summary. This script reads the raw eval_results.json and
computes pass@k to create the missing JSON file.

Usage:
    uv run scripts/eval/extract_preliminary_passk.py
"""

from __future__ import annotations

import json
from pathlib import Path

from src.pass_at_k import pass_at_k

ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS = ROOT / "outputs"


def compute_pass_at_k_from_eval_results(
    eval_results_path: Path, metric: str = "plus"
) -> dict:
    """Compute pass@k from an EvalPlus eval_results.json file."""
    with open(eval_results_path) as f:
        data = json.load(f)

    status_key = f"{metric}_status" if metric != "base" else "base_status"

    per_problem = []
    for task_id in sorted(data["eval"].keys()):
        samples = data["eval"][task_id]
        n = len(samples)
        c = sum(1 for s in samples if s.get(status_key, s.get("status")) == "pass")
        per_problem.append({"n": n, "c": c})

    n_samples = per_problem[0]["n"]
    k_values = [k for k in [1, 2, 5, 10] if k <= n_samples]

    pass_at_k_results = {}
    for k in k_values:
        scores = [pass_at_k(p["n"], p["c"], k) for p in per_problem]
        pass_at_k_results[str(k)] = sum(scores) / len(scores)

    return {
        "n_problems": len(per_problem),
        "n_samples_per_problem": n_samples,
        "pass_at_k": pass_at_k_results,
        "per_problem": per_problem,
    }


def main() -> None:
    eval_path = (
        OUTPUTS / "test_unsteered_full" / "humaneval"
        / "Qwen--Qwen2.5-1.5B-Instruct_openai_temp_0.8_eval_results.json"
    )
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval results not found: {eval_path}")

    base = compute_pass_at_k_from_eval_results(eval_path, "base")
    plus = compute_pass_at_k_from_eval_results(eval_path, "plus")

    result = {
        "scale": 0.0,
        "temperature": 0.8,
        "dataset": "humaneval",
        "steering_mode": "none",
        "pass_at_k_base": base["pass_at_k"],
        "pass_at_k_plus": plus["pass_at_k"],
        "n_problems": base["n_problems"],
        "n_samples_per_problem": base["n_samples_per_problem"],
        "per_problem_base": base["per_problem"],
        "per_problem_plus": plus["per_problem"],
    }

    out_path = (
        OUTPUTS / "test_unsteered_full" / "humaneval" / "pass_at_k.json"
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path.relative_to(ROOT)}")

    # Print summary for verification
    print(f"\nUnsteered n=10 HumanEval pass@k:")
    for k in sorted(result["pass_at_k_base"].keys(), key=int):
        print(
            f"  pass@{k}: base={result['pass_at_k_base'][k]:.3f}"
            f"  plus={result['pass_at_k_plus'][k]:.3f}"
        )


if __name__ == "__main__":
    main()
