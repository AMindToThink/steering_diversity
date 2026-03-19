"""Tests for the pass@k estimator."""

import math

import pytest

from src.pass_at_k import pass_at_k, pass_at_k_curve


class TestPassAtK:
    """Unit tests for the pass_at_k estimator."""

    def test_all_correct(self) -> None:
        """When all samples pass, pass@k = 1.0 for any k."""
        assert pass_at_k(10, 10, 1) == 1.0
        assert pass_at_k(10, 10, 5) == 1.0
        assert pass_at_k(10, 10, 10) == 1.0

    def test_none_correct(self) -> None:
        """When no samples pass, pass@k = 0.0 for any k."""
        assert pass_at_k(10, 0, 1) == 0.0
        assert pass_at_k(10, 0, 5) == 0.0
        assert pass_at_k(10, 0, 10) == 0.0

    def test_pass_at_1_equals_ratio(self) -> None:
        """pass@1 should equal c/n (the unbiased estimator simplifies to this)."""
        assert pass_at_k(10, 3, 1) == pytest.approx(0.3)
        assert pass_at_k(100, 50, 1) == pytest.approx(0.5)
        assert pass_at_k(200, 1, 1) == pytest.approx(0.005)

    def test_known_value(self) -> None:
        """Check against a manually computed value.

        n=5, c=2, k=3: 1 - C(3,3)/C(5,3) = 1 - 1/10 = 0.9
        """
        assert pass_at_k(5, 2, 3) == pytest.approx(0.9)

    def test_known_value_2(self) -> None:
        """n=4, c=1, k=2: 1 - C(3,2)/C(4,2) = 1 - 3/6 = 0.5."""
        assert pass_at_k(4, 1, 2) == pytest.approx(0.5)

    def test_monotonicity_in_k(self) -> None:
        """pass@k must be non-decreasing in k."""
        n, c = 20, 5
        values = [pass_at_k(n, c, k) for k in range(1, n + 1)]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1] + 1e-12

    def test_monotonicity_in_c(self) -> None:
        """pass@k must be non-decreasing in c."""
        n, k = 20, 5
        values = [pass_at_k(n, c, k) for c in range(0, n + 1)]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1] + 1e-12

    def test_k_equals_n(self) -> None:
        """pass@n: with n attempts, you see all samples."""
        # If c >= 1 and k == n, must be 1.0 (you see everything)
        assert pass_at_k(10, 1, 10) == 1.0
        assert pass_at_k(10, 0, 10) == 0.0

    def test_invalid_k_zero(self) -> None:
        with pytest.raises(ValueError, match="1 <= k <= n"):
            pass_at_k(10, 5, 0)

    def test_invalid_k_exceeds_n(self) -> None:
        with pytest.raises(ValueError, match="1 <= k <= n"):
            pass_at_k(10, 5, 11)

    def test_invalid_c_negative(self) -> None:
        with pytest.raises(ValueError, match="0 <= c <= n"):
            pass_at_k(10, -1, 1)

    def test_invalid_c_exceeds_n(self) -> None:
        with pytest.raises(ValueError, match="0 <= c <= n"):
            pass_at_k(10, 11, 1)


class TestPassAtKCurve:
    """Unit tests for pass_at_k_curve."""

    def test_single_problem(self) -> None:
        results = [(10, 3)]
        curve = pass_at_k_curve(results, [1, 5, 10])
        assert curve[1] == pytest.approx(0.3)
        assert curve[10] == 1.0

    def test_average_across_problems(self) -> None:
        """Two problems: one always correct, one never — average = 0.5 at any k."""
        results = [(10, 10), (10, 0)]
        curve = pass_at_k_curve(results, [1, 5])
        assert curve[1] == pytest.approx(0.5)
        assert curve[5] == pytest.approx(0.5)

    def test_monotonicity(self) -> None:
        results = [(20, 5), (20, 10), (20, 1)]
        k_values = [1, 2, 5, 10, 15, 20]
        curve = pass_at_k_curve(results, k_values)
        values = [curve[k] for k in k_values]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1] + 1e-12

    def test_empty_results_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            pass_at_k_curve([], [1])

    def test_k_exceeds_min_n_raises(self) -> None:
        results = [(5, 2), (10, 3)]
        with pytest.raises(ValueError, match="k=6 exceeds minimum n=5"):
            pass_at_k_curve(results, [1, 6])
