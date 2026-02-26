"""Pre-registered statistical tests for steering diversity analysis.

Implements:
- Per-prompt metric computation (primary: mean_pairwise_cosine_distance)
- Page's L test for monotonic trend
- Holm-Bonferroni correction
- Linear mixed-effects sensitivity analysis
- Spearman effect size with bootstrap CI
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import cosine_distances


# ---------------------------------------------------------------------------
# Per-prompt metric computation
# ---------------------------------------------------------------------------

def compute_per_prompt_pairwise_distance(
    embeddings: np.ndarray,
    scales: np.ndarray,
    prompt_indices: np.ndarray,
) -> dict[tuple[int, float], float]:
    """Compute mean pairwise cosine distance for each (prompt_idx, scale) group.

    Returns a dict mapping (prompt_idx, scale) -> mean_pairwise_cosine_distance.
    """
    results: dict[tuple[int, float], float] = {}
    unique_prompts = sorted(set(int(p) for p in prompt_indices))
    unique_scales = sorted(set(float(s) for s in scales))

    for prompt_idx in unique_prompts:
        for scale in unique_scales:
            mask = (prompt_indices == prompt_idx) & (np.isclose(scales, scale))
            group_embs = embeddings[mask]
            n = group_embs.shape[0]
            if n > 1:
                cd = cosine_distances(group_embs)
                triu_idx = np.triu_indices(n, k=1)
                results[(prompt_idx, scale)] = float(np.mean(cd[triu_idx]))
            else:
                results[(prompt_idx, scale)] = 0.0

    return results


def compute_per_prompt_cluster_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    scales: np.ndarray,
    prompt_indices: np.ndarray,
) -> dict[str, dict[tuple[int, float], float]]:
    """Compute per-prompt clustering metrics for each (prompt_idx, scale) group.

    Returns a dict of metric_name -> {(prompt_idx, scale): value}.
    Metrics: num_clusters, noise_ratio, cluster_entropy,
             mean_intra_cluster_distance, mean_inter_cluster_distance.
    """
    metric_names = [
        "num_clusters", "noise_ratio", "cluster_entropy",
        "mean_intra_cluster_distance", "mean_inter_cluster_distance",
    ]
    results: dict[str, dict[tuple[int, float], float]] = {m: {} for m in metric_names}

    unique_prompts = sorted(set(int(p) for p in prompt_indices))
    unique_scales = sorted(set(float(s) for s in scales))

    for prompt_idx in unique_prompts:
        for scale in unique_scales:
            mask = (prompt_indices == prompt_idx) & (np.isclose(scales, scale))
            group_labels = labels[mask]
            group_embs = embeddings[mask]
            n = group_labels.shape[0]

            cluster_ids = sorted(set(group_labels) - {-1})
            nc = len(cluster_ids)

            results["num_clusters"][(prompt_idx, scale)] = float(nc)
            results["noise_ratio"][(prompt_idx, scale)] = (
                float(np.sum(group_labels == -1)) / n if n > 0 else 0.0
            )

            # Entropy
            if nc > 0:
                counts = np.array([np.sum(group_labels == c) for c in cluster_ids], dtype=np.float64)
                probs = counts / counts.sum()
                results["cluster_entropy"][(prompt_idx, scale)] = float(
                    -np.sum(probs * np.log(probs + 1e-12))
                )
            else:
                results["cluster_entropy"][(prompt_idx, scale)] = 0.0

            # Intra-cluster distance
            intra_dists: list[float] = []
            centroids: list[np.ndarray] = []
            for c in cluster_ids:
                c_mask = group_labels == c
                c_embs = group_embs[c_mask]
                centroids.append(c_embs.mean(axis=0))
                if c_embs.shape[0] > 1:
                    cd = cosine_distances(c_embs)
                    triu = np.triu_indices(c_embs.shape[0], k=1)
                    intra_dists.append(float(np.mean(cd[triu])))

            results["mean_intra_cluster_distance"][(prompt_idx, scale)] = (
                float(np.mean(intra_dists)) if intra_dists else 0.0
            )

            # Inter-cluster distance
            if len(centroids) > 1:
                centroid_arr = np.vstack(centroids)
                inter_cd = cosine_distances(centroid_arr)
                triu = np.triu_indices(len(centroids), k=1)
                results["mean_inter_cluster_distance"][(prompt_idx, scale)] = float(
                    np.mean(inter_cd[triu])
                )
            else:
                results["mean_inter_cluster_distance"][(prompt_idx, scale)] = 0.0

    return results


def _build_prompt_scale_matrix(
    per_prompt_values: dict[tuple[int, float], float],
    unique_prompts: list[int],
    unique_scales: list[float],
) -> np.ndarray:
    """Build a (n_prompts, n_scales) matrix from per-prompt values."""
    matrix = np.zeros((len(unique_prompts), len(unique_scales)))
    for i, p in enumerate(unique_prompts):
        for j, s in enumerate(unique_scales):
            matrix[i, j] = per_prompt_values.get((p, s), np.nan)
    return matrix


# ---------------------------------------------------------------------------
# Page's L test
# ---------------------------------------------------------------------------

def pages_l_test(
    data: np.ndarray,
    predicted_order: list[int] | None = None,
) -> dict[str, Any]:
    """Page's L test for monotonic trend across ordered conditions.

    Parameters
    ----------
    data:
        Array of shape (n_subjects, k_conditions). Each row is one subject
        (prompt), columns are ordered conditions (scales, ascending).
    predicted_order:
        Predicted rank order of conditions (1-indexed). If None, assumes
        descending order of values (i.e., first condition has highest value,
        last has lowest — diversity decreases with scale).

    Returns
    -------
    Dict with keys: L, L_expected, L_var, z_score, p_value, n, k.
    """
    n, k = data.shape

    if predicted_order is None:
        # Predict diversity decreases: first column (lowest scale) should have
        # highest values, so predicted ranks are k, k-1, ..., 1
        predicted_order = list(range(k, 0, -1))

    predicted = np.array(predicted_order)

    # Rank within each row (subject). Ties get average rank.
    ranked = np.zeros_like(data)
    for i in range(n):
        ranked[i] = stats.rankdata(data[i])

    # Page's L statistic
    column_rank_sums = ranked.sum(axis=0)  # R_j for each condition j
    L = float(np.sum(predicted * column_rank_sums))

    # Expected value and variance under H0
    L_expected = n * k * (k + 1) ** 2 / 4.0
    L_var = n * k ** 2 * (k ** 2 - 1) ** 2 / (144.0 * (k - 1))

    z_score = (L - L_expected) / np.sqrt(L_var)

    # One-sided p-value (upper tail: L is large when trend matches prediction)
    p_value = float(1.0 - stats.norm.cdf(z_score))

    return {
        "L": L,
        "L_expected": L_expected,
        "L_var": L_var,
        "z_score": float(z_score),
        "p_value": p_value,
        "n": n,
        "k": k,
    }


# ---------------------------------------------------------------------------
# Holm-Bonferroni correction
# ---------------------------------------------------------------------------

def holm_bonferroni(
    p_values: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Apply Holm-Bonferroni correction to a set of named p-values.

    Returns dict of metric_name -> {raw_p, adjusted_p, significant}.
    """
    names = sorted(p_values.keys())
    raw_ps = [p_values[n] for n in names]
    m = len(raw_ps)

    # Sort by raw p-value
    sorted_indices = np.argsort(raw_ps)
    adjusted = np.zeros(m)

    cummax = 0.0
    for rank, idx in enumerate(sorted_indices):
        corrected = raw_ps[idx] * (m - rank)
        cummax = max(cummax, corrected)
        adjusted[idx] = min(cummax, 1.0)

    results: dict[str, dict[str, float]] = {}
    for i, name in enumerate(names):
        results[name] = {
            "raw_p": raw_ps[i],
            "adjusted_p": float(adjusted[i]),
            "significant": bool(adjusted[i] < 0.05),
        }

    return results


# ---------------------------------------------------------------------------
# Linear mixed-effects model
# ---------------------------------------------------------------------------

def mixed_effects_analysis(
    per_prompt_values: dict[tuple[int, float], float],
    unique_prompts: list[int],
    unique_scales: list[float],
) -> dict[str, Any]:
    """Fit mean_pairwise_cosine_distance ~ scale + (1|prompt_idx).

    Returns slope beta, 95% CI, p-value, and convergence status.
    """
    import pandas as pd
    import statsmodels.formula.api as smf

    rows = []
    for (prompt_idx, scale), value in per_prompt_values.items():
        rows.append({"diversity": value, "scale": scale, "prompt_idx": prompt_idx})
    df = pd.DataFrame(rows)

    try:
        model = smf.mixedlm("diversity ~ scale", df, groups=df["prompt_idx"])
        result = model.fit(reml=True)

        beta = float(result.params["scale"])
        se = float(result.bse["scale"])
        ci_low = beta - 1.96 * se
        ci_high = beta + 1.96 * se
        p_value = float(result.pvalues["scale"])

        return {
            "beta": beta,
            "se": se,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "p_value": p_value,
            "converged": result.converged,
        }
    except Exception as e:
        return {
            "beta": None,
            "se": None,
            "ci_95_low": None,
            "ci_95_high": None,
            "p_value": None,
            "converged": False,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Effect size: Spearman's rho with bootstrap CI
# ---------------------------------------------------------------------------

def spearman_effect_size(
    per_prompt_values: dict[tuple[int, float], float],
    unique_prompts: list[int],
    unique_scales: list[float],
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute Spearman's rho between scale and per-prompt diversity.

    Returns rho, p-value, and 95% bootstrap CI.
    """
    scales_flat = []
    values_flat = []
    for (prompt_idx, scale), value in per_prompt_values.items():
        scales_flat.append(scale)
        values_flat.append(value)

    scales_arr = np.array(scales_flat)
    values_arr = np.array(values_flat)

    rho, p_value = stats.spearmanr(scales_arr, values_arr)

    # Bootstrap CI
    rng = np.random.default_rng(seed)
    n = len(scales_arr)
    boot_rhos = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        with np.errstate(invalid="ignore"):
            r, _ = stats.spearmanr(scales_arr[idx], values_arr[idx])
        boot_rhos[b] = r

    # Drop NaN (from constant-input resamples) before computing CI
    valid_rhos = boot_rhos[~np.isnan(boot_rhos)]
    if len(valid_rhos) > 0:
        ci_low = float(np.percentile(valid_rhos, 2.5))
        ci_high = float(np.percentile(valid_rhos, 97.5))
    else:
        ci_low = float("nan")
        ci_high = float("nan")

    return {
        "rho": float(rho),
        "p_value": float(p_value),
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "n_bootstrap": n_bootstrap,
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_all_statistical_tests(
    embeddings: np.ndarray,
    labels: np.ndarray,
    scales: np.ndarray,
    prompt_indices: np.ndarray,
    seed: int = 42,
) -> dict[str, Any]:
    """Run all pre-registered statistical tests.

    Returns a complete stats dict ready to be saved as JSON.
    """
    unique_prompts = sorted(set(int(p) for p in prompt_indices))
    unique_scales = sorted(set(float(s) for s in scales))
    n_scales = len(unique_scales)

    # Need at least 3 conditions for Page's L to be meaningful
    if n_scales < 3:
        return {
            "skipped": True,
            "reason": f"Only {n_scales} scales — need >= 3 for Page's L test. "
                      "Run with full config for statistical tests.",
            "n_prompts": len(unique_prompts),
            "n_scales": n_scales,
            "scales": unique_scales,
        }

    results: dict[str, Any] = {
        "n_prompts": len(unique_prompts),
        "n_scales": n_scales,
        "scales": unique_scales,
    }

    # --- Per-prompt pairwise distance (primary outcome) ---
    pairwise_dist = compute_per_prompt_pairwise_distance(
        embeddings, scales, prompt_indices
    )
    primary_matrix = _build_prompt_scale_matrix(
        pairwise_dist, unique_prompts, unique_scales
    )

    # --- Per-prompt cluster metrics (secondary outcomes) ---
    cluster_metrics = compute_per_prompt_cluster_metrics(
        embeddings, labels, scales, prompt_indices
    )

    # --- Page's L tests ---
    page_results: dict[str, dict[str, Any]] = {}
    page_pvalues: dict[str, float] = {}

    # Primary
    primary_page = pages_l_test(primary_matrix)
    page_results["mean_pairwise_cosine_distance"] = primary_page
    page_pvalues["mean_pairwise_cosine_distance"] = primary_page["p_value"]

    # Secondary
    for metric_name, per_prompt_vals in cluster_metrics.items():
        matrix = _build_prompt_scale_matrix(
            per_prompt_vals, unique_prompts, unique_scales
        )
        result = pages_l_test(matrix)
        page_results[metric_name] = result
        page_pvalues[metric_name] = result["p_value"]

    results["pages_l_tests"] = page_results

    # --- Holm-Bonferroni correction ---
    results["holm_bonferroni"] = holm_bonferroni(page_pvalues)

    # --- Mixed-effects sensitivity analysis (primary outcome only) ---
    results["mixed_effects"] = mixed_effects_analysis(
        pairwise_dist, unique_prompts, unique_scales
    )

    # --- Effect size ---
    results["spearman_effect_size"] = spearman_effect_size(
        pairwise_dist, unique_prompts, unique_scales, seed=seed
    )

    return results
