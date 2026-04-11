"""Standalone reproducer for a float32 precision issue in welford-torch's
``OnlineCovariance.add_all``.

This script depends only on ``welford_torch``, ``torch``, and ``numpy``. It
does NOT depend on anything from the surrounding project. Copy-paste it
into a fresh venv and run::

    pip install welford-torch torch numpy
    python reproducer.py

Expected output: a table showing that (a) numpy gives a clean trace of
~0.0064, (b) ``OnlineCovariance.add_all`` run over batches of 37 gives
~0.0138 (over 100% relative error), (c) one huge batch gives a NEGATIVE
trace (mathematically impossible for a covariance matrix), but (d)
``float64`` fixes it entirely, and (e) warming the state with a single
``.add()`` call before any ``.add_all()`` also fixes it.

The data distribution is ``x ~ N(mu * 1, sigma^2 * I)`` with ``mu=1000``,
``sigma=0.01``, ``d=64``, ``N=5000`` — a regime where the mean is huge
relative to the standard deviation. This is the kind of regime that
arises naturally when accumulating statistics over LLM residual-stream
activations (large consistent bias, small per-token variation).

Root cause: inside ``add_all``, ``delta_xs = xs - old_mean`` has magnitude
``~|mu|`` when the running mean has not yet caught up to the true mean,
and ``delta_xs_2 = xs - new_mean`` has magnitude ``~|sigma|``. The product
``delta_xs @ delta_xs_2`` collapses two terms whose nominal values differ
by ``(|mu|/|sigma|)^2`` in magnitude, losing most of its precision in
float32 through catastrophic cancellation.

Exits with status 0 if the drift is reproduced (bug present) and with
status 1 if the drift is NOT reproduced (bug may have been fixed upstream).
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from welford_torch import OnlineCovariance


def main() -> int:
    torch.manual_seed(0)
    d = 64
    N = 5000
    mu = 1000.0
    sigma = 0.01

    x = torch.randn(N, d, dtype=torch.float32) * sigma + mu

    # Ground truth from numpy on the full dataset.
    x_np = x.numpy()
    np_cov = np.cov(x_np, rowvar=False, ddof=1)
    np_trace = float(np.var(x_np, axis=0, ddof=1).sum())

    print("welford-torch OnlineCovariance — big-mean / small-variance drift test")
    print("=" * 78)
    print(f"Distribution: x ~ N(mu={mu}, sigma^2 I), d={d}, N={N}")
    print(f"Ground truth trace (numpy, ddof=1): {np_trace:.6g}")
    print(f"Ground truth max(|cov|):            {float(np.max(np.abs(np_cov))):.6g}")
    print("-" * 78)
    print(f"{'Scenario':<55s} {'trace':>13s} {'rel_err':>10s}")
    print("-" * 78)

    def run(label: str, build) -> float:
        oc = build()
        # OnlineCovariance stores population covariance (ddof=0). Rescale to
        # sample covariance for an apples-to-apples comparison with np.cov.
        cov_sample = oc.cov.numpy() * N / (N - 1)
        trace = float(cov_sample.trace())
        rel = abs(trace - np_trace) / max(np_trace, 1e-30)
        print(f"{label:<55s} {trace:>13.6g} {rel:>10.3g}")
        return rel

    # 1. Pure small batches, cold start — the main failure mode.
    def cold_small_batches():
        oc = OnlineCovariance(dtype=torch.float32, device="cpu")
        for start in range(0, N, 37):
            oc.add_all(x[start : start + 37])
        return oc

    rel_cold_small = run(".add_all() batches of 37, cold start", cold_small_batches)

    # 2. float64 as a sanity check — should work fine.
    def cold_small_batches_f64():
        oc = OnlineCovariance(dtype=torch.float64, device="cpu")
        for start in range(0, N, 37):
            oc.add_all(x[start : start + 37].to(torch.float64))
        return oc

    rel_f64 = run(".add_all() batches of 37, float64", cold_small_batches_f64)

    # 3. Warmed with a single .add() — bug should disappear.
    def warmed_small_batches():
        oc = OnlineCovariance(dtype=torch.float32, device="cpu")
        oc.add(x[0])
        for start in range(1, N, 37):
            oc.add_all(x[start : start + 37])
        return oc

    rel_warmed = run(".add_all() after 1 warm-up .add()", warmed_small_batches)

    # 4. Single-observation path — documented-stable Welford, should work.
    def all_singletons():
        oc = OnlineCovariance(dtype=torch.float32, device="cpu")
        for i in range(N):
            oc.add(x[i])
        return oc

    run("all samples via .add() one at a time", all_singletons)

    # 5. One giant batch — larger batches are WORSE (counterintuitive but
    # follows from the same cancellation mechanism).
    def one_huge_batch():
        oc = OnlineCovariance(dtype=torch.float32, device="cpu")
        oc.add_all(x)
        return oc

    rel_huge = run(".add_all() one batch of N=5000", one_huge_batch)

    print("-" * 78)
    print()

    # Summary: spell out the three load-bearing findings.
    print("Summary")
    print("-" * 78)
    if rel_cold_small > 0.5:
        print(f"  BUG REPRODUCED: cold-start .add_all() drifts by {rel_cold_small:.2f}")
        print(f"                  (over 50% relative error — should be <1%).")
    else:
        print(f"  NO DRIFT: cold-start .add_all() rel_err={rel_cold_small:.3g}")
        print("            The bug may have been fixed upstream.")
        return 1

    print()
    print(f"  float64 works cleanly (rel_err={rel_f64:.3g}): this is a float32")
    print("  precision issue, not an algorithmic bug in the formula.")
    print()
    print(f"  Warming with a single .add() before any .add_all() gives")
    print(f"  rel_err={rel_warmed:.3g}, i.e. the bug disappears entirely. This")
    print("  localizes the problem to the first call's cold-start subtraction.")
    print()
    print(f"  One giant .add_all(x[:N]) gives rel_err={rel_huge:.3g} and can even")
    print("  produce a NEGATIVE trace (mathematically impossible for a real")
    print("  covariance). Bigger batches are worse, not better — consistent")
    print("  with the same cancellation mechanism amplified over more terms.")
    print()
    print("See ISSUE.md in this directory for the root-cause analysis and a")
    print("suggested fix (use Chan-Golub-LeVeque's batched merge, which centers")
    print("each batch internally before merging with the running state).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
