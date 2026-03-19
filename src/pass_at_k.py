"""Unbiased pass@k estimator for evaluating steering diversity via capability."""

from __future__ import annotations

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator: 1 - C(n-c, k) / C(n, k).

    Uses the numerically stable product formula to avoid large factorials.

    Parameters
    ----------
    n:
        Total number of samples generated for this problem.
    c:
        Number of correct (passing) samples.
    k:
        The k in pass@k — how many attempts the "developer" gets.

    Returns
    -------
    Probability that at least one of k randomly chosen samples is correct.

    Raises
    ------
    ValueError:
        If inputs violate 1 <= k <= n or c < 0 or c > n.
    """
    if not (1 <= k <= n):
        raise ValueError(f"Need 1 <= k <= n, got k={k}, n={n}")
    if not (0 <= c <= n):
        raise ValueError(f"Need 0 <= c <= n, got c={c}, n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def pass_at_k_curve(
    results: list[tuple[int, int]],
    k_values: list[int],
) -> dict[int, float]:
    """Average pass@k across problems for each k value.

    Parameters
    ----------
    results:
        List of (n, c) tuples — one per problem. n = total samples, c = correct.
    k_values:
        List of k values to compute pass@k for.

    Returns
    -------
    Mapping from k to the average pass@k across all problems.

    Raises
    ------
    ValueError:
        If results is empty or any k exceeds the minimum n across problems.
    """
    if not results:
        raise ValueError("results must be non-empty")
    min_n = min(n for n, _ in results)
    for k in k_values:
        if k > min_n:
            raise ValueError(
                f"k={k} exceeds minimum n={min_n} across problems; "
                f"cannot compute unbiased estimator"
            )

    curve: dict[int, float] = {}
    for k in k_values:
        scores = [pass_at_k(n, c, k) for n, c in results]
        curve[k] = float(np.mean(scores))
    return curve
