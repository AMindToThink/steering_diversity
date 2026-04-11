"""Cross-check tests for src/bounds/activation_streams.py.

Every hand-rolled accumulator is fed the same data that a trusted reference
(numpy for ``CheapMoments``/``SphericalMoments``/``Reservoir``; the
``welford_torch`` library for ``FullMoments``) also sees, then finalized
values are compared.

Parameterized over seeds and chunk schedules so each test is a small
family of cross-checks rather than a single input.

Numpy is authoritative because we can feed it the full concatenated
dataset — no streaming, so there's nothing for Welford to be wrong about.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from welford_torch import OnlineCovariance

from src.bounds.activation_streams import (
    CheapMoments,
    FullMoments,
    Reservoir,
    SphericalMoments,
)


def _random_batches(
    N: int,
    d: int,
    seed: int,
    mu_scale: float = 0.0,
    sigma: float = 1.0,
    chunk_sizes: list[int] | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Return the full dataset + a list of mini-batches summing to it.

    The chunk schedule is arbitrary; the streaming result must equal the
    full-dataset result regardless of how the data is partitioned.
    """
    torch.manual_seed(seed)
    mu = torch.full((d,), mu_scale, dtype=torch.float32)
    full = torch.randn(N, d, dtype=torch.float32) * sigma + mu

    if chunk_sizes is None:
        chunk_sizes = [N // 7, N // 5, N // 3, N - (N // 7) - (N // 5) - (N // 3)]

    batches: list[torch.Tensor] = []
    i = 0
    for c in chunk_sizes:
        if c <= 0:
            continue
        batches.append(full[i : i + c])
        i += c
    if i < N:
        batches.append(full[i:])
    return full, batches


# ---------------------------------------------------------------------------
# CheapMoments vs numpy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 42, 99])
@pytest.mark.parametrize("mu_scale", [0.0, 0.5, 100.0])  # includes big-mean regime
def test_cheap_moments_agrees_with_numpy(seed: int, mu_scale: float) -> None:
    N = 3000
    d = 48
    full, batches = _random_batches(N=N, d=d, seed=seed, mu_scale=mu_scale, sigma=0.01 if mu_scale > 10 else 1.0)

    cm = CheapMoments(d=d, device="cpu")
    for b in batches:
        cm.update(b)
    got = cm.finalize()

    ref_np = full.numpy()
    assert got["count"] == N
    assert np.allclose(got["mean"].numpy(), ref_np.mean(axis=0), atol=1e-4)
    # tr(Σ) = sum of per-coord sample variances
    ref_trace = float(np.var(ref_np, axis=0, ddof=1).sum())
    rel_trace = abs(got["trace_cov"] - ref_trace) / max(abs(ref_trace), 1e-12)
    assert rel_trace < 5e-3, f"trace mismatch: rel_err={rel_trace}"
    assert np.isclose(got["E_norm"], np.linalg.norm(ref_np, axis=1).mean(), atol=1e-4)
    assert np.isclose(got["E_sq_norm"], (np.linalg.norm(ref_np, axis=1) ** 2).mean(), atol=1e-1)
    assert np.isclose(got["R"], float(np.linalg.norm(ref_np, axis=1).max()), atol=1e-4)


@pytest.mark.parametrize("seed", [0, 7])
def test_cheap_moments_chunk_order_invariance(seed: int) -> None:
    """Running stats must not depend on the chunk schedule."""
    full, _ = _random_batches(N=2000, d=16, seed=seed, mu_scale=5.0)

    cm_a = CheapMoments(d=16, device="cpu")
    cm_b = CheapMoments(d=16, device="cpu")
    # Schedule A: one big batch
    cm_a.update(full)
    # Schedule B: many tiny batches
    for start in range(0, full.shape[0], 17):
        cm_b.update(full[start : start + 17])

    a = cm_a.finalize()
    b = cm_b.finalize()
    assert torch.allclose(a["mean"], b["mean"], atol=1e-5)
    assert abs(a["trace_cov"] - b["trace_cov"]) < 1e-3
    assert a["R"] == b["R"]


# ---------------------------------------------------------------------------
# FullMoments vs numpy vs welford-torch directly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("mu_scale", [0.0, 10.0])
def test_full_moments_agrees_with_numpy_cov(seed: int, mu_scale: float) -> None:
    N = 2500
    d = 32
    full, batches = _random_batches(N=N, d=d, seed=seed, mu_scale=mu_scale, sigma=0.5)

    fm = FullMoments(d=d, device="cpu")
    for b in batches:
        fm.update(b)
    got = fm.finalize()

    ref_np = full.numpy()
    np_cov = np.cov(ref_np, rowvar=False, ddof=1)  # sample covariance

    assert got["count"] == N
    assert np.allclose(got["mean"].numpy(), ref_np.mean(axis=0), atol=1e-4)
    assert np.allclose(got["cov"].numpy(), np_cov, atol=1e-4), \
        f"cov max diff: {np.max(np.abs(got['cov'].numpy() - np_cov))}"


@pytest.mark.parametrize("seed", [0, 3])
def test_full_moments_matches_welford_torch_in_normal_regime(seed: int) -> None:
    """In a benign regime (modest mean, unit variance), FullMoments and
    welford_torch.OnlineCovariance should agree closely.

    We do NOT delegate FullMoments to welford_torch: a separate big-mean
    regression test (``test_welford_torch_drifts_in_big_mean_regime`` below)
    shows that OnlineCovariance's ``add_all`` destabilizes in the
    big-mean/small-batch corner case, so FullMoments is hand-rolled with
    Chan-Golub-LeVeque. This test pins the fact that the two implementations
    DO agree when the regime is benign — if they diverged here it would
    signal a bug in either library.
    """
    N = 1500
    d = 20
    full, batches = _random_batches(N=N, d=d, seed=seed, mu_scale=2.0, sigma=1.0)

    fm = FullMoments(d=d, device="cpu")
    oc = OnlineCovariance()
    for b in batches:
        fm.update(b)
        oc.add_all(b)

    got = fm.finalize()
    # OnlineCovariance stores population covariance; FullMoments.finalize()
    # returns sample covariance. Rescale before comparing.
    oc_cov_pop = oc.cov
    fm_cov_pop = got["cov"] * ((N - 1) / N)
    assert torch.allclose(fm_cov_pop, oc_cov_pop.cpu(), atol=1e-5)
    assert torch.allclose(got["mean"], oc.mean.cpu(), atol=1e-6)


def test_welford_torch_drifts_in_big_mean_regime() -> None:
    """Documents exactly the regime where welford_torch.OnlineCovariance
    disagrees with numpy ground truth, while our FullMoments stays accurate.

    This is the reason FullMoments is hand-rolled instead of wrapping
    OnlineCovariance. The library's ``add_all`` centers new batches against
    the *running* mean, which produces large intermediate quantities when
    the running mean is far from zero and each batch is small; the squared
    differences then lose most of their precision in float32. Chan's merge
    centers within each batch first, so the squared differences stay small
    regardless of the mean's absolute scale.
    """
    torch.manual_seed(0)
    d = 64
    N = 5000
    mu = 1000.0
    full = torch.randn(N, d, dtype=torch.float32) * 0.01 + mu
    np_trace = float(np.var(full.numpy(), axis=0, ddof=1).sum())

    # Ours — should stay accurate in float32.
    fm = FullMoments(d=d, device="cpu")
    for start in range(0, N, 37):
        fm.update(full[start : start + 37])
    fm_trace = fm.finalize()["trace_cov"]
    fm_rel = abs(fm_trace - np_trace) / np_trace
    assert fm_rel < 2e-2, f"FullMoments drifted: rel_err={fm_rel}"

    # welford_torch — should drift badly in this regime. If upstream ever
    # fixes this, the test will fail and prompt us to re-evaluate the
    # delegation question.
    oc = OnlineCovariance(dtype=torch.float32, device="cpu")
    for start in range(0, N, 37):
        oc.add_all(full[start : start + 37])
    oc_trace = float((oc.cov * N / (N - 1)).trace().item())
    oc_rel = abs(oc_trace - np_trace) / np_trace
    assert oc_rel > 0.2, (
        f"welford_torch unexpectedly accurate (rel_err={oc_rel}). "
        "Upstream may have fixed the big-mean instability; reconsider "
        "wrapping it in FullMoments."
    )


# ---------------------------------------------------------------------------
# SphericalMoments vs numpy
# ---------------------------------------------------------------------------


def _np_spherical_reference(x: np.ndarray, pole: np.ndarray | None = None) -> dict:
    u = x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
    r_bar = u.mean(axis=0)
    r_bar_norm = float(np.linalg.norm(r_bar))
    out = {
        "R_bar": r_bar,
        "R_bar_norm": r_bar_norm,
        "spherical_variance": 1.0 - r_bar_norm,
        "expected_pair_sq_chord": 2.0 * (1.0 - r_bar_norm ** 2),
    }
    if pole is not None:
        p = pole / np.linalg.norm(pole)
        out["max_chord_to_pole"] = float(np.linalg.norm(u - p, axis=1).max())
    return out


@pytest.mark.parametrize("seed", [0, 5, 17])
def test_spherical_moments_agrees_with_numpy(seed: int) -> None:
    N = 2000
    d = 24
    full, batches = _random_batches(N=N, d=d, seed=seed, mu_scale=0.3, sigma=1.0)

    torch.manual_seed(seed)
    pole = torch.randn(d)

    sm = SphericalMoments(d=d, device="cpu", pole=pole)
    for b in batches:
        sm.update(b)
    got = sm.finalize()

    ref = _np_spherical_reference(full.numpy(), pole=pole.numpy())

    assert np.isclose(got["R_bar_norm"], ref["R_bar_norm"], atol=1e-5)
    assert np.isclose(got["spherical_variance"], ref["spherical_variance"], atol=1e-5)
    assert np.isclose(got["expected_pair_sq_chord"], ref["expected_pair_sq_chord"], atol=1e-5)
    assert np.isclose(got["max_chord_to_pole"], ref["max_chord_to_pole"], atol=1e-5)
    assert np.allclose(got["R_bar"].numpy(), ref["R_bar"], atol=1e-5)


def test_spherical_moments_chunk_order_invariance() -> None:
    full, _ = _random_batches(N=1500, d=12, seed=99)

    sm_a = SphericalMoments(d=12, device="cpu")
    sm_b = SphericalMoments(d=12, device="cpu")
    sm_a.update(full)
    for start in range(0, full.shape[0], 13):
        sm_b.update(full[start : start + 13])

    a = sm_a.finalize()
    b = sm_b.finalize()
    assert abs(a["R_bar_norm"] - b["R_bar_norm"]) < 1e-5
    assert torch.allclose(a["R_bar"], b["R_bar"], atol=1e-5)


# ---------------------------------------------------------------------------
# Reservoir vs numpy reference (statistical equivalence, not byte equality)
# ---------------------------------------------------------------------------


def test_reservoir_matches_numpy_unbiased_estimator() -> None:
    """Over many trials, reservoir-sampled means should converge to the true
    mean, matching what numpy computes over the full dataset.

    This is a statistical cross-check: Vitter's Algorithm R is unbiased, so
    the average of many reservoir means converges to the population mean.
    """
    np.random.seed(0)
    N = 500
    d = 4
    full = np.random.randn(N, d).astype(np.float32) * 2.0 + np.arange(d).astype(np.float32)
    full_t = torch.from_numpy(full)

    true_mean = full.mean(axis=0)

    trials = 200
    K = 50
    reservoir_means = np.zeros((trials, d), dtype=np.float32)
    for t in range(trials):
        r = Reservoir(K=K, d=d, seed=t)
        r.update(full_t)
        reservoir_means[t] = r.finalize().mean(dim=0).numpy()

    # Averaged over trials, the reservoir mean should converge to the true
    # mean — this validates unbiased sampling.
    avg_reservoir_mean = reservoir_means.mean(axis=0)
    max_diff = float(np.abs(avg_reservoir_mean - true_mean).max())
    # 200 trials of K=50 from N=500 → standard error ~ sigma/sqrt(K*trials/factor).
    # Loose tolerance; the point is to catch gross biases, not fit noise.
    assert max_diff < 0.3, f"reservoir mean biased: max_diff={max_diff}"


def test_reservoir_chunk_independence_with_same_seed() -> None:
    """Same seed + same data should give same reservoir regardless of chunking."""
    torch.manual_seed(0)
    full = torch.randn(500, 4)
    r_big = Reservoir(K=30, d=4, seed=42)
    r_small = Reservoir(K=30, d=4, seed=42)

    r_big.update(full)
    for i in range(0, 500, 7):
        r_small.update(full[i : i + 7])

    a = r_big.finalize()
    b = r_small.finalize()
    # Reservoir samples depend on the RNG state, which advances per-row.
    # With the same seed and the same sample order, both should match
    # byte-for-byte.
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# Big-mean regression cross-check: does the entire pipeline stay stable?
# ---------------------------------------------------------------------------


def test_big_mean_full_pipeline_matches_numpy() -> None:
    """End-to-end stress test: large bias + small variance, many small chunks.

    Every hand-rolled accumulator should still match the authoritative numpy
    reference in this regime — the one that breaks naive Σxx^T − Nμμ^T.
    """
    torch.manual_seed(0)
    d = 64
    N = 5000
    mu = 1000.0
    full = torch.randn(N, d, dtype=torch.float32) * 0.01 + mu

    cm = CheapMoments(d=d, device="cpu")
    fm = FullMoments(d=d, device="cpu")
    sm = SphericalMoments(d=d, device="cpu")
    for start in range(0, N, 37):
        batch = full[start : start + 37]
        cm.update(batch)
        fm.update(batch)
        sm.update(batch)

    cm_out = cm.finalize()
    fm_out = fm.finalize()
    sm_out = sm.finalize()
    np_ref = full.numpy()

    # Mean
    assert np.allclose(cm_out["mean"].numpy(), np_ref.mean(0), atol=1e-3)
    assert np.allclose(fm_out["mean"].numpy(), np_ref.mean(0), atol=1e-3)

    # Trace of covariance (the hard number in the big-mean regime)
    true_trace = float(np.var(np_ref, axis=0, ddof=1).sum())
    assert abs(cm_out["trace_cov"] - true_trace) / true_trace < 2e-2
    assert abs(fm_out["trace_cov"] - true_trace) / true_trace < 2e-2

    # Full covariance
    np_cov = np.cov(np_ref, rowvar=False, ddof=1)
    assert np.max(np.abs(fm_out["cov"].numpy() - np_cov)) < 1e-3

    # Spherical R̄
    u = np_ref / np.linalg.norm(np_ref, axis=1, keepdims=True)
    assert np.isclose(sm_out["R_bar_norm"], float(np.linalg.norm(u.mean(0))), atol=1e-5)
