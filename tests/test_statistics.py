"""Tests for pre-registered statistical analysis functions."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.statistics import (
    _build_prompt_scale_matrix,
    compute_per_prompt_cluster_metrics,
    compute_per_prompt_pairwise_distance,
    holm_bonferroni,
    mixed_effects_analysis,
    pages_l_test,
    run_all_statistical_tests,
    spearman_effect_size,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_experiment() -> dict:
    """Create a synthetic experiment with a known monotonic decrease in diversity.

    3 prompts, 4 scales, 5 responses per (prompt, scale) = 60 total.
    Higher scales -> tighter cluster -> lower pairwise distance.
    """
    rng = np.random.default_rng(42)
    n_prompts = 3
    n_scales = 4
    n_responses = 5
    dim = 32
    scales_list = [0.0, 1.0, 2.0, 4.0]

    embeddings_parts = []
    scales_parts = []
    prompt_parts = []
    labels_parts = []

    for prompt_idx in range(n_prompts):
        center = rng.standard_normal(dim)
        for scale_idx, scale in enumerate(scales_list):
            # Spread decreases as scale increases -> diversity collapse
            spread = 1.0 / (1.0 + scale)
            group = center + rng.standard_normal((n_responses, dim)) * spread
            embeddings_parts.append(group)
            scales_parts.extend([scale] * n_responses)
            prompt_parts.extend([prompt_idx] * n_responses)
            # Assign simple cluster labels (all same cluster per group)
            labels_parts.extend([prompt_idx * n_scales + scale_idx] * n_responses)

    embeddings = np.vstack(embeddings_parts).astype(np.float32)
    scales = np.array(scales_parts, dtype=np.float32)
    prompt_indices = np.array(prompt_parts, dtype=np.int32)
    labels = np.array(labels_parts, dtype=int)

    return {
        "embeddings": embeddings,
        "scales": scales,
        "prompt_indices": prompt_indices,
        "labels": labels,
        "n_prompts": n_prompts,
        "n_scales": n_scales,
        "scales_list": scales_list,
    }


@pytest.fixture
def two_scale_experiment() -> dict:
    """Experiment with only 2 scales — should trigger skip."""
    rng = np.random.default_rng(0)
    n = 20
    dim = 16
    embeddings = rng.standard_normal((n, dim)).astype(np.float32)
    scales = np.array([0.0] * 10 + [2.0] * 10, dtype=np.float32)
    prompt_indices = np.array([0] * 5 + [1] * 5 + [0] * 5 + [1] * 5, dtype=np.int32)
    labels = np.full(n, -1, dtype=int)
    return {
        "embeddings": embeddings,
        "scales": scales,
        "prompt_indices": prompt_indices,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Per-prompt pairwise distance
# ---------------------------------------------------------------------------


class TestPerPromptPairwiseDistance:
    def test_returns_all_prompt_scale_pairs(self, synthetic_experiment: dict) -> None:
        result = compute_per_prompt_pairwise_distance(
            synthetic_experiment["embeddings"],
            synthetic_experiment["scales"],
            synthetic_experiment["prompt_indices"],
        )
        expected_n = synthetic_experiment["n_prompts"] * synthetic_experiment["n_scales"]
        assert len(result) == expected_n

    def test_values_are_nonnegative(self, synthetic_experiment: dict) -> None:
        result = compute_per_prompt_pairwise_distance(
            synthetic_experiment["embeddings"],
            synthetic_experiment["scales"],
            synthetic_experiment["prompt_indices"],
        )
        for val in result.values():
            assert val >= 0.0

    def test_diversity_decreases_with_scale(self, synthetic_experiment: dict) -> None:
        """For our synthetic data, diversity should decrease as scale increases."""
        result = compute_per_prompt_pairwise_distance(
            synthetic_experiment["embeddings"],
            synthetic_experiment["scales"],
            synthetic_experiment["prompt_indices"],
        )
        scales_list = synthetic_experiment["scales_list"]
        for prompt_idx in range(synthetic_experiment["n_prompts"]):
            values = [result[(prompt_idx, s)] for s in scales_list]
            # Each value should be >= the next (monotonic decrease)
            for i in range(len(values) - 1):
                assert values[i] >= values[i + 1] - 1e-6, (
                    f"prompt={prompt_idx}: diversity at scale {scales_list[i]} "
                    f"({values[i]:.4f}) < diversity at scale {scales_list[i+1]} "
                    f"({values[i+1]:.4f})"
                )

    def test_single_response_returns_zero(self) -> None:
        """With 1 response per group, pairwise distance should be 0."""
        embeddings = np.random.randn(2, 8).astype(np.float32)
        scales = np.array([0.0, 1.0], dtype=np.float32)
        prompt_indices = np.array([0, 0], dtype=np.int32)

        result = compute_per_prompt_pairwise_distance(embeddings, scales, prompt_indices)
        assert result[(0, 0.0)] == 0.0
        assert result[(0, 1.0)] == 0.0


# ---------------------------------------------------------------------------
# Build prompt-scale matrix
# ---------------------------------------------------------------------------


class TestBuildPromptScaleMatrix:
    def test_shape(self) -> None:
        values = {(0, 0.0): 1.0, (0, 1.0): 2.0, (1, 0.0): 3.0, (1, 1.0): 4.0}
        matrix = _build_prompt_scale_matrix(values, [0, 1], [0.0, 1.0])
        assert matrix.shape == (2, 2)

    def test_values_placed_correctly(self) -> None:
        values = {(0, 0.0): 1.0, (0, 1.0): 2.0, (1, 0.0): 3.0, (1, 1.0): 4.0}
        matrix = _build_prompt_scale_matrix(values, [0, 1], [0.0, 1.0])
        assert matrix[0, 0] == 1.0
        assert matrix[0, 1] == 2.0
        assert matrix[1, 0] == 3.0
        assert matrix[1, 1] == 4.0

    def test_missing_key_gives_nan(self) -> None:
        values = {(0, 0.0): 1.0}
        matrix = _build_prompt_scale_matrix(values, [0, 1], [0.0, 1.0])
        assert np.isnan(matrix[1, 0])
        assert np.isnan(matrix[1, 1])


# ---------------------------------------------------------------------------
# Page's L test
# ---------------------------------------------------------------------------


class TestPagesLTest:
    def test_perfect_decreasing_trend(self) -> None:
        """Perfect monotonic decrease should give a very significant p-value."""
        # 10 subjects, 4 conditions: values decrease perfectly across conditions
        n, k = 10, 4
        data = np.tile(np.arange(k, 0, -1, dtype=float), (n, 1))
        # Add slight noise so ranks aren't all tied
        rng = np.random.default_rng(42)
        data += rng.normal(0, 0.01, data.shape)

        result = pages_l_test(data)
        assert result["p_value"] < 0.01
        assert result["n"] == n
        assert result["k"] == k
        assert result["z_score"] > 0

    def test_no_trend_gives_nonsignificant(self) -> None:
        """Random data should give non-significant p-value (most of the time)."""
        rng = np.random.default_rng(123)
        data = rng.standard_normal((20, 4))
        result = pages_l_test(data)
        # We can't guarantee p > 0.05, but for this seed it should be
        assert result["p_value"] > 0.01

    def test_increasing_trend_gives_large_p(self) -> None:
        """Data increasing (opposite of prediction) should give p near 1."""
        n, k = 10, 4
        data = np.tile(np.arange(1, k + 1, dtype=float), (n, 1))
        rng = np.random.default_rng(42)
        data += rng.normal(0, 0.01, data.shape)

        result = pages_l_test(data)
        # Opposite of predicted direction -> p should be large
        assert result["p_value"] > 0.5

    def test_return_keys(self) -> None:
        data = np.random.randn(5, 3)
        result = pages_l_test(data)
        assert set(result.keys()) == {"L", "L_expected", "L_var", "z_score", "p_value", "n", "k"}

    def test_custom_predicted_order(self) -> None:
        """With custom order predicting increase, increasing data should be significant."""
        n, k = 10, 4
        data = np.tile(np.arange(1, k + 1, dtype=float), (n, 1))
        rng = np.random.default_rng(42)
        data += rng.normal(0, 0.01, data.shape)

        # Predict increasing: rank 1, 2, 3, 4
        result = pages_l_test(data, predicted_order=[1, 2, 3, 4])
        assert result["p_value"] < 0.01

    def test_l_expected_formula(self) -> None:
        """Verify L_expected matches the known formula."""
        n, k = 8, 5
        data = np.random.randn(n, k)
        result = pages_l_test(data)
        expected = n * k * (k + 1) ** 2 / 4.0
        assert result["L_expected"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Holm-Bonferroni correction
# ---------------------------------------------------------------------------


class TestHolmBonferroni:
    def test_single_pvalue_unchanged(self) -> None:
        result = holm_bonferroni({"a": 0.03})
        assert result["a"]["adjusted_p"] == pytest.approx(0.03)
        assert result["a"]["significant"] is True

    def test_two_significant(self) -> None:
        result = holm_bonferroni({"a": 0.01, "b": 0.02})
        # a: 0.01 * 2 = 0.02; b: max(0.02, 0.02 * 1) = 0.02
        assert result["a"]["adjusted_p"] == pytest.approx(0.02)
        assert result["b"]["adjusted_p"] == pytest.approx(0.02)
        assert result["a"]["significant"] is True
        assert result["b"]["significant"] is True

    def test_one_survives_correction(self) -> None:
        result = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.06})
        # a: 0.01 * 3 = 0.03 (sig)
        # b: 0.04 * 2 = 0.08 (not sig)
        # c: max(0.08, 0.06 * 1) = 0.08 (not sig)
        assert result["a"]["significant"] is True
        assert result["b"]["significant"] is False
        assert result["c"]["significant"] is False

    def test_adjusted_p_capped_at_one(self) -> None:
        result = holm_bonferroni({"a": 0.5, "b": 0.8})
        assert result["a"]["adjusted_p"] <= 1.0
        assert result["b"]["adjusted_p"] <= 1.0

    def test_adjusted_p_monotone(self) -> None:
        """Adjusted p-values should be non-decreasing when sorted by raw p."""
        pvals = {"a": 0.001, "b": 0.01, "c": 0.03, "d": 0.04, "e": 0.10}
        result = holm_bonferroni(pvals)
        sorted_by_raw = sorted(result.items(), key=lambda x: x[1]["raw_p"])
        adjusted = [v["adjusted_p"] for _, v in sorted_by_raw]
        for i in range(len(adjusted) - 1):
            assert adjusted[i] <= adjusted[i + 1] + 1e-12


# ---------------------------------------------------------------------------
# Mixed-effects analysis
# ---------------------------------------------------------------------------


class TestMixedEffects:
    def test_negative_slope_for_decreasing_data(self) -> None:
        """Diversity decreasing with scale -> negative beta."""
        rng = np.random.default_rng(42)
        per_prompt = {}
        for prompt in range(20):
            intercept = rng.normal(0.8, 0.1)  # random intercept per prompt
            for scale in [0.0, 1.0, 2.0, 4.0]:
                per_prompt[(prompt, scale)] = intercept - 0.05 * scale + rng.normal(0, 0.03)

        result = mixed_effects_analysis(per_prompt, list(range(20)), [0.0, 1.0, 2.0, 4.0])
        assert result["beta"] is not None, f"Model failed: {result.get('error')}"
        assert result["beta"] < 0

    def test_returns_expected_keys(self) -> None:
        per_prompt = {(0, 0.0): 1.0, (0, 1.0): 0.5, (1, 0.0): 0.9, (1, 1.0): 0.4}
        result = mixed_effects_analysis(per_prompt, [0, 1], [0.0, 1.0])
        assert "beta" in result
        assert "se" in result
        assert "ci_95_low" in result
        assert "ci_95_high" in result
        assert "p_value" in result
        assert "converged" in result

    def test_ci_contains_beta(self) -> None:
        per_prompt = {}
        for prompt in range(10):
            for scale in [0.0, 1.0, 2.0]:
                per_prompt[(prompt, scale)] = 1.0 - 0.1 * scale + np.random.normal(0, 0.02)

        result = mixed_effects_analysis(per_prompt, list(range(10)), [0.0, 1.0, 2.0])
        if result["converged"]:
            assert result["ci_95_low"] <= result["beta"] <= result["ci_95_high"]


# ---------------------------------------------------------------------------
# Spearman effect size
# ---------------------------------------------------------------------------


class TestSpearmanEffectSize:
    def test_negative_rho_for_decreasing(self) -> None:
        """Diversity decreasing with scale -> negative rho."""
        per_prompt = {}
        for prompt in range(5):
            for scale in [0.0, 1.0, 2.0, 4.0]:
                per_prompt[(prompt, scale)] = 1.0 / (1.0 + scale)

        result = spearman_effect_size(
            per_prompt, list(range(5)), [0.0, 1.0, 2.0, 4.0],
            n_bootstrap=100, seed=42,
        )
        assert result["rho"] < 0
        assert result["ci_95_high"] < 0  # Entire CI should be negative

    def test_ci_contains_rho(self) -> None:
        per_prompt = {}
        rng = np.random.default_rng(0)
        for prompt in range(10):
            for scale in [0.0, 1.0, 2.0]:
                per_prompt[(prompt, scale)] = rng.standard_normal()

        result = spearman_effect_size(
            per_prompt, list(range(10)), [0.0, 1.0, 2.0],
            n_bootstrap=1000, seed=42,
        )
        # CI should contain the point estimate (usually — not guaranteed but very likely)
        assert result["ci_95_low"] <= result["rho"] <= result["ci_95_high"]

    def test_returns_expected_keys(self) -> None:
        per_prompt = {(0, 0.0): 1.0, (0, 1.0): 0.5}
        result = spearman_effect_size(per_prompt, [0], [0.0, 1.0], n_bootstrap=10)
        expected_keys = {"rho", "p_value", "ci_95_low", "ci_95_high", "n_bootstrap"}
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Per-prompt cluster metrics
# ---------------------------------------------------------------------------


class TestPerPromptClusterMetrics:
    def test_returns_all_metric_names(self) -> None:
        rng = np.random.default_rng(0)
        n = 20
        embeddings = rng.standard_normal((n, 8)).astype(np.float32)
        labels = np.array([0] * 10 + [1] * 10, dtype=int)
        scales = np.array([0.0] * 10 + [1.0] * 10, dtype=np.float32)
        prompt_indices = np.array([0] * 5 + [1] * 5 + [0] * 5 + [1] * 5, dtype=np.int32)

        result = compute_per_prompt_cluster_metrics(embeddings, labels, scales, prompt_indices)
        expected_metrics = {
            "num_clusters", "noise_ratio", "cluster_entropy",
            "mean_intra_cluster_distance", "mean_inter_cluster_distance",
        }
        assert set(result.keys()) == expected_metrics

    def test_all_noise_gives_zero_clusters(self) -> None:
        rng = np.random.default_rng(0)
        n = 10
        embeddings = rng.standard_normal((n, 8)).astype(np.float32)
        labels = np.full(n, -1, dtype=int)
        scales = np.array([0.0] * n, dtype=np.float32)
        prompt_indices = np.array([0] * 5 + [1] * 5, dtype=np.int32)

        result = compute_per_prompt_cluster_metrics(embeddings, labels, scales, prompt_indices)
        for key in [(0, 0.0), (1, 0.0)]:
            assert result["num_clusters"][key] == 0.0
            assert result["noise_ratio"][key] == 1.0


# ---------------------------------------------------------------------------
# run_all_statistical_tests (integration)
# ---------------------------------------------------------------------------


class TestRunAllStatisticalTests:
    def test_skips_with_two_scales(self, two_scale_experiment: dict) -> None:
        result = run_all_statistical_tests(
            two_scale_experiment["embeddings"],
            two_scale_experiment["labels"],
            two_scale_experiment["scales"],
            two_scale_experiment["prompt_indices"],
        )
        assert result["skipped"] is True
        assert result["n_scales"] == 2

    def test_full_run_returns_all_sections(self, synthetic_experiment: dict) -> None:
        result = run_all_statistical_tests(
            synthetic_experiment["embeddings"],
            synthetic_experiment["labels"],
            synthetic_experiment["scales"],
            synthetic_experiment["prompt_indices"],
        )
        assert "skipped" not in result
        assert "pages_l_tests" in result
        assert "holm_bonferroni" in result
        assert "mixed_effects" in result
        assert "spearman_effect_size" in result

    def test_detects_monotonic_decrease(self, synthetic_experiment: dict) -> None:
        """Our synthetic data has a clear monotonic decrease — tests should detect it."""
        result = run_all_statistical_tests(
            synthetic_experiment["embeddings"],
            synthetic_experiment["labels"],
            synthetic_experiment["scales"],
            synthetic_experiment["prompt_indices"],
        )
        # Primary Page's L should be significant
        primary_p = result["pages_l_tests"]["mean_pairwise_cosine_distance"]["p_value"]
        assert primary_p < 0.05, f"Expected significant Page's L, got p={primary_p}"

        # Spearman rho should be negative
        assert result["spearman_effect_size"]["rho"] < 0

        # Mixed model slope should be negative
        if result["mixed_effects"]["converged"]:
            assert result["mixed_effects"]["beta"] < 0

    def test_pages_l_has_all_six_metrics(self, synthetic_experiment: dict) -> None:
        result = run_all_statistical_tests(
            synthetic_experiment["embeddings"],
            synthetic_experiment["labels"],
            synthetic_experiment["scales"],
            synthetic_experiment["prompt_indices"],
        )
        expected_metrics = {
            "mean_pairwise_cosine_distance",
            "num_clusters", "noise_ratio", "cluster_entropy",
            "mean_intra_cluster_distance", "mean_inter_cluster_distance",
        }
        assert set(result["pages_l_tests"].keys()) == expected_metrics
        assert set(result["holm_bonferroni"].keys()) == expected_metrics

    def test_result_is_json_serializable(self, synthetic_experiment: dict) -> None:
        """Stats output must be saved as JSON — verify no numpy types leak."""
        result = run_all_statistical_tests(
            synthetic_experiment["embeddings"],
            synthetic_experiment["labels"],
            synthetic_experiment["scales"],
            synthetic_experiment["prompt_indices"],
        )
        # This will raise TypeError if numpy types are present
        serialized = json.dumps(result)
        assert len(serialized) > 0

    def test_holm_bonferroni_count_matches_pages_l(self, synthetic_experiment: dict) -> None:
        result = run_all_statistical_tests(
            synthetic_experiment["embeddings"],
            synthetic_experiment["labels"],
            synthetic_experiment["scales"],
            synthetic_experiment["prompt_indices"],
        )
        assert len(result["holm_bonferroni"]) == len(result["pages_l_tests"])
