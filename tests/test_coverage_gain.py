"""Tests for coverage_gain_test() in compute_passk_from_eval.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# Import the function under test and its helper
from scripts.eval.compute_passk_from_eval import coverage_gain_test, _per_problem_passk


def _make_pass_at_k_json(
    per_problem: list[tuple[int, int]],
    scale: float,
    dataset: str = "humaneval",
    k_values: list[int] | None = None,
) -> dict:
    """Build a pass_at_k.json-format dict from (n, c) pairs.

    Computes both plus and base pass@k curves from the same data
    (tests only use plus by default).
    """
    from src.pass_at_k import pass_at_k_curve

    if k_values is None:
        k_values = [1, 2, 5, 10, 25, 50, 100]

    n_samples = per_problem[0][0]
    valid_k = [k for k in k_values if k <= n_samples]
    curve = pass_at_k_curve(per_problem, valid_k)

    per_problem_dicts = [{"n": n, "c": c} for n, c in per_problem]

    return {
        "scale": scale,
        "temperature": 0.8,
        "dataset": dataset,
        "pass_at_k_plus": {str(k): v for k, v in curve.items()},
        "pass_at_k_base": {str(k): v for k, v in curve.items()},
        "n_problems": len(per_problem),
        "n_samples_per_problem": n_samples,
        "per_problem_plus": per_problem_dicts,
        "per_problem_base": per_problem_dicts,
    }


def _write_passk_json(tmp_path: Path, data: dict, name: str) -> Path:
    """Write a pass_at_k.json dict to a temp file and return the path."""
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
    return path


class TestZeroEffect:
    """Identical baseline and steered → all deltas = 0, all p ≈ 1."""

    def test_zero_effect(self, tmp_path: Path) -> None:
        # 50 problems, n=100 samples each, varied c values
        rng = np.random.default_rng(42)
        per_problem = [(100, int(c)) for c in rng.integers(10, 90, size=50)]

        data = _make_pass_at_k_json(per_problem, scale=0.0)
        baseline = _write_passk_json(tmp_path, data, "baseline.json")

        data_steered = _make_pass_at_k_json(per_problem, scale=2.0)
        steered = _write_passk_json(tmp_path, data_steered, "steered.json")

        result = coverage_gain_test([baseline], [steered])

        for r in result["per_k_results"]:
            assert r["delta_cg"] == pytest.approx(0.0, abs=1e-12), (
                f"k={r['k']}: expected delta_cg=0, got {r['delta_cg']}"
            )
            # p-value should be NaN or 1.0 for zero variance diff
            # (ttest_rel returns NaN when all differences are 0)
            assert r["p_value"] != pytest.approx(0.0, abs=0.01)

        assert result["omnibus_test"]["mean_delta_cg"] == pytest.approx(0.0, abs=1e-12)


class TestKnownCollapse:
    """Steered has same pass@1 but less diversity → negative coverage gain."""

    def test_known_collapse(self, tmp_path: Path) -> None:
        n = 100
        n_problems = 80

        # Baseline: moderate c values (30-70) — lots of diversity headroom
        rng = np.random.default_rng(123)
        baseline_c = rng.integers(30, 70, size=n_problems)
        baseline_problems = [(n, int(c)) for c in baseline_c]

        # Steered: push c values toward extremes (0 or n).
        # This preserves pass@1 (mean stays similar) but kills coverage gain:
        # problems with c≈0 or c≈n have almost no pass@k - pass@1 gap.
        steered_c = np.where(
            baseline_c >= 50,
            np.minimum(baseline_c + 25, n),   # push high c toward n
            np.maximum(baseline_c - 25, 0),   # push low c toward 0
        )
        steered_problems = [(n, int(c)) for c in steered_c]

        baseline_data = _make_pass_at_k_json(baseline_problems, scale=0.0)
        steered_data = _make_pass_at_k_json(steered_problems, scale=2.0)

        baseline_path = _write_passk_json(tmp_path, baseline_data, "baseline.json")
        steered_path = _write_passk_json(tmp_path, steered_data, "steered.json")

        result = coverage_gain_test([baseline_path], [steered_path])

        # At k=10, polarized c values should show negative coverage gain delta
        k10_result = next(r for r in result["per_k_results"] if r["k"] == 10)
        assert k10_result["delta_cg"] < 0, (
            f"Expected negative coverage gain delta at k=10, got {k10_result['delta_cg']}"
        )


class TestMultiFilePooling:
    """Multiple files are correctly concatenated."""

    def test_pooling_concatenates_problems(self, tmp_path: Path) -> None:
        n = 100

        # Benchmark A: 30 problems
        problems_a = [(n, c) for c in range(30, 60)]
        baseline_a = _make_pass_at_k_json(problems_a, scale=0.0, dataset="humaneval")
        steered_a = _make_pass_at_k_json(problems_a, scale=2.0, dataset="humaneval")

        # Benchmark B: 20 problems
        problems_b = [(n, c) for c in range(40, 60)]
        baseline_b = _make_pass_at_k_json(problems_b, scale=0.0, dataset="mbpp")
        steered_b = _make_pass_at_k_json(problems_b, scale=2.0, dataset="mbpp")

        base_a_path = _write_passk_json(tmp_path, baseline_a, "base_a.json")
        base_b_path = _write_passk_json(tmp_path, baseline_b, "base_b.json")
        steer_a_path = _write_passk_json(tmp_path, steered_a, "steer_a.json")
        steer_b_path = _write_passk_json(tmp_path, steered_b, "steer_b.json")

        result = coverage_gain_test(
            [base_a_path, base_b_path],
            [steer_a_path, steer_b_path],
        )

        assert result["n_problems"] == 50  # 30 + 20
        assert set(result["datasets_pooled"]) == {"humaneval", "mbpp"}
        assert result["n_baseline_files"] == 2
        assert result["n_steered_files"] == 2

    def test_single_file_still_works(self, tmp_path: Path) -> None:
        """Single-file case (list of length 1) matches old behavior."""
        n = 100
        rng = np.random.default_rng(99)
        problems = [(n, int(c)) for c in rng.integers(10, 90, size=40)]

        baseline_data = _make_pass_at_k_json(problems, scale=0.0)
        steered_data = _make_pass_at_k_json(problems, scale=2.0)

        base_path = _write_passk_json(tmp_path, baseline_data, "base.json")
        steer_path = _write_passk_json(tmp_path, steered_data, "steer.json")

        result = coverage_gain_test([base_path], [steer_path])

        assert result["n_problems"] == 40
        assert result["n_baseline_files"] == 1
        assert len(result["per_k_results"]) > 0


class TestValidation:
    """Input validation errors."""

    def test_mismatched_file_counts(self, tmp_path: Path) -> None:
        n = 100
        problems = [(n, 50) for _ in range(10)]
        data = _make_pass_at_k_json(problems, scale=0.0)

        path1 = _write_passk_json(tmp_path, data, "a.json")
        path2 = _write_passk_json(tmp_path, data, "b.json")

        with pytest.raises(ValueError, match="equal number"):
            coverage_gain_test([path1, path2], [path1])

    def test_incompatible_k_values_uses_intersection(self, tmp_path: Path) -> None:
        """Files with different k_values use the intersection."""
        n = 100
        problems = [(n, 50) for _ in range(10)]

        # File A has k=[1,2,5,10,25,50,100], file B has k=[1,2,5,10]
        data_a = _make_pass_at_k_json(problems, scale=0.0, k_values=[1, 2, 5, 10, 25, 50, 100])
        data_b = _make_pass_at_k_json(problems, scale=0.0, k_values=[1, 2, 5, 10])
        data_a_steer = _make_pass_at_k_json(problems, scale=2.0, k_values=[1, 2, 5, 10, 25, 50, 100])
        data_b_steer = _make_pass_at_k_json(problems, scale=2.0, k_values=[1, 2, 5, 10])

        path_a = _write_passk_json(tmp_path, data_a, "base_a.json")
        path_b = _write_passk_json(tmp_path, data_b, "base_b.json")
        path_a_s = _write_passk_json(tmp_path, data_a_steer, "steer_a.json")
        path_b_s = _write_passk_json(tmp_path, data_b_steer, "steer_b.json")

        result = coverage_gain_test(
            [path_a, path_b], [path_a_s, path_b_s],
        )

        # Should only have k=2,5,10 (intersection, excluding k=1)
        result_ks = [r["k"] for r in result["per_k_results"]]
        assert result_ks == [2, 5, 10]
