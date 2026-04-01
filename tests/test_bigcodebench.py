"""Tests for BigCodeBench eval_results extraction and pass_at_k computation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.eval.compute_passk_from_eval import (
    compute_passk_bcb,
    coverage_gain_test,
    extract_per_problem_results_bcb,
)


def _make_bcb_eval_results(
    per_problem: list[tuple[str, int, int]],
) -> dict:
    """Build a BigCodeBench eval_results dict.

    Args:
        per_problem: List of (task_id, n_samples, n_correct) tuples.
    """
    eval_data: dict[str, list[dict]] = {}
    for task_id, n, c in per_problem:
        samples = []
        for i in range(n):
            status = "pass" if i < c else "fail"
            samples.append({
                "task_id": task_id,
                "solution": f"def solution_{i}(): pass",
                "status": status,
                "details": {},
            })
        eval_data[task_id] = samples

    return {
        "date": "2026-03-21 12:00",
        "eval": eval_data,
    }


class TestExtractPerProblemResultsBcb:
    """Test BigCodeBench (n, c) extraction."""

    def test_basic_extraction(self) -> None:
        eval_results = _make_bcb_eval_results([
            ("BigCodeBench/0", 10, 8),
            ("BigCodeBench/1", 10, 3),
            ("BigCodeBench/2", 10, 0),
        ])
        results = extract_per_problem_results_bcb(eval_results)
        assert results == [(10, 8), (10, 3), (10, 0)]

    def test_sorted_by_task_id(self) -> None:
        """Results should be sorted by task_id regardless of input order."""
        eval_results = _make_bcb_eval_results([
            ("BigCodeBench/2", 10, 5),
            ("BigCodeBench/0", 10, 8),
            ("BigCodeBench/1", 10, 3),
        ])
        results = extract_per_problem_results_bcb(eval_results)
        # Sorted order: 0, 1, 2
        assert results == [(10, 8), (10, 3), (10, 5)]

    def test_all_pass(self) -> None:
        eval_results = _make_bcb_eval_results([
            ("BigCodeBench/0", 5, 5),
        ])
        results = extract_per_problem_results_bcb(eval_results)
        assert results == [(5, 5)]

    def test_empty_eval(self) -> None:
        eval_results = {"eval": {}}
        results = extract_per_problem_results_bcb(eval_results)
        assert results == []


class TestComputePasskBcb:
    """Test compute_passk_bcb produces correct pass_at_k.json format."""

    def test_output_format(self, tmp_path: Path) -> None:
        eval_results = _make_bcb_eval_results([
            (f"BigCodeBench/{i}", 100, 50 + i) for i in range(20)
        ])
        eval_path = tmp_path / "eval_results.json"
        with open(eval_path, "w") as f:
            json.dump(eval_results, f)

        output_path = tmp_path / "pass_at_k.json"
        result = compute_passk_bcb(
            eval_path, scale=0.0, temperature=0.8,
            dataset="bigcodebench", k_values=[1, 2, 5, 10, 25, 50, 100],
            output_path=output_path,
        )

        # Check structure
        assert result["benchmark"] == "bigcodebench"
        assert result["n_problems"] == 20
        assert result["n_samples_per_problem"] == 100
        assert "pass_at_k_plus" in result
        assert "pass_at_k_base" in result
        assert "per_problem_plus" in result
        assert "per_problem_base" in result

        # plus and base should be identical (single test suite)
        assert result["pass_at_k_plus"] == result["pass_at_k_base"]
        assert result["per_problem_plus"] == result["per_problem_base"]

        # File should be written
        assert output_path.exists()
        with open(output_path) as f:
            loaded = json.load(f)
        assert loaded == result

    def test_k_values_filtered(self, tmp_path: Path) -> None:
        """k_values > n_samples should be filtered out."""
        eval_results = _make_bcb_eval_results([
            (f"BigCodeBench/{i}", 10, 5) for i in range(5)
        ])
        eval_path = tmp_path / "eval_results.json"
        with open(eval_path, "w") as f:
            json.dump(eval_results, f)

        result = compute_passk_bcb(
            eval_path, scale=0.0, temperature=0.8,
            dataset="bigcodebench", k_values=[1, 2, 5, 10, 25, 50],
            output_path=tmp_path / "pass_at_k.json",
        )

        # Only k <= 10 should appear
        k_values = [int(k) for k in result["pass_at_k_plus"].keys()]
        assert all(k <= 10 for k in k_values)
        assert 25 not in k_values
        assert 50 not in k_values


class TestBcbCoverageGainCompat:
    """BigCodeBench pass_at_k.json works with coverage_gain_test()."""

    def test_bcb_with_coverage_gain_test(self, tmp_path: Path) -> None:
        """BigCodeBench results feed into coverage_gain_test unchanged."""
        import numpy as np
        from src.pass_at_k import pass_at_k_curve

        n = 100
        rng = np.random.default_rng(42)

        # Baseline: varied c values
        baseline_problems = [(f"BigCodeBench/{i}", n, int(c))
                             for i, c in enumerate(rng.integers(20, 80, size=50))]
        # Steered: push c toward extremes (kills coverage gain)
        steered_problems = [
            (tid, n, min(c + 25, n) if c >= 50 else max(c - 25, 0))
            for tid, n, c in baseline_problems
        ]

        for label, problems, scale in [
            ("baseline", baseline_problems, 0.0),
            ("steered", steered_problems, 2.0),
        ]:
            eval_results = _make_bcb_eval_results(problems)
            eval_path = tmp_path / f"{label}_eval.json"
            with open(eval_path, "w") as f:
                json.dump(eval_results, f)

            compute_passk_bcb(
                eval_path, scale=scale, temperature=0.8,
                dataset="bigcodebench",
                k_values=[1, 2, 5, 10, 25, 50, 100],
                output_path=tmp_path / f"{label}_pass_at_k.json",
            )

        # Run coverage gain test on the BigCodeBench pass_at_k.json files
        result = coverage_gain_test(
            baseline_paths=[tmp_path / "baseline_pass_at_k.json"],
            steered_paths=[tmp_path / "steered_pass_at_k.json"],
        )

        assert result["n_problems"] == 50
        assert len(result["per_k_results"]) > 0
        # With systematically lower c, we expect negative coverage gain
        k10 = next(r for r in result["per_k_results"] if r["k"] == 10)
        assert k10["delta_cg"] < 0

    def test_bcb_pooled_with_evalplus(self, tmp_path: Path) -> None:
        """BigCodeBench + EvalPlus pass_at_k.json files pool correctly."""
        from tests.test_coverage_gain import _make_pass_at_k_json, _write_passk_json

        n = 100

        # EvalPlus file: 30 problems
        evalplus_problems = [(n, c) for c in range(30, 60)]
        ep_base = _write_passk_json(
            tmp_path,
            _make_pass_at_k_json(evalplus_problems, scale=0.0, dataset="humaneval"),
            "ep_base.json",
        )
        ep_steer = _write_passk_json(
            tmp_path,
            _make_pass_at_k_json(evalplus_problems, scale=2.0, dataset="humaneval"),
            "ep_steer.json",
        )

        # BigCodeBench file: 20 problems
        bcb_problems = [(f"BigCodeBench/{i}", n, 40 + i) for i in range(20)]
        bcb_eval = _make_bcb_eval_results(bcb_problems)
        for label, scale in [("base", 0.0), ("steer", 2.0)]:
            eval_path = tmp_path / f"bcb_{label}_eval.json"
            with open(eval_path, "w") as f:
                json.dump(bcb_eval, f)
            compute_passk_bcb(
                eval_path, scale=scale, temperature=0.8,
                dataset="bigcodebench",
                k_values=[1, 2, 5, 10, 25, 50, 100],
                output_path=tmp_path / f"bcb_{label}.json",
            )

        result = coverage_gain_test(
            baseline_paths=[ep_base, tmp_path / "bcb_base.json"],
            steered_paths=[ep_steer, tmp_path / "bcb_steer.json"],
        )

        assert result["n_problems"] == 50  # 30 + 20
        assert set(result["datasets_pooled"]) == {"humaneval", "bigcodebench"}
