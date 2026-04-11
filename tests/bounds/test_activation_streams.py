"""Tests for src/bounds/activation_streams.py — Welford/Chan accumulators."""

from __future__ import annotations

import torch

from src.bounds.activation_streams import (
    CheapMoments,
    FullMoments,
    Reservoir,
    SphericalMoments,
)


def test_full_moments_matches_torch_cov() -> None:
    torch.manual_seed(0)
    d = 64
    N = 4000
    x = torch.randn(N, d, dtype=torch.float32) * 2.0 + 1.5

    fm = FullMoments(d=d, device="cpu")
    for i in range(0, N, 500):
        fm.update(x[i : i + 500])
    s = fm.finalize()

    assert s["count"] == N
    assert torch.allclose(s["mean"], x.mean(0), atol=1e-5)
    cov_ref = x.T.cov()  # torch.cov with ddof=1
    assert torch.allclose(s["cov"], cov_ref, atol=1e-4)
    assert abs(s["trace_cov"] - cov_ref.trace().item()) < 1e-2
    assert abs(s["E_norm"] - x.norm(dim=1).mean().item()) < 1e-3
    assert abs(s["E_sq_norm"] - (x.norm(dim=1) ** 2).mean().item()) < 1e-1
    assert abs(s["R"] - x.norm(dim=1).max().item()) < 1e-4


def test_cheap_moments_matches_per_coord_variance() -> None:
    torch.manual_seed(0)
    d = 32
    N = 2000
    x = torch.randn(N, d, dtype=torch.float32) * 0.5 + 0.1

    cm = CheapMoments(d=d, device="cpu")
    for i in range(0, N, 250):
        cm.update(x[i : i + 250])
    s = cm.finalize()

    assert s["count"] == N
    assert torch.allclose(s["mean"], x.mean(0), atol=1e-5)
    # trace = sum of per-coord variances
    assert abs(s["trace_cov"] - x.var(0, unbiased=True).sum().item()) < 1e-3
    assert "cov" not in s  # cheap tier never returns the full matrix


def test_cheap_moments_trace_stable_under_large_mean() -> None:
    """Regression test: huge consistent bias, small variance.

    A naive `Σ‖x‖² − N‖μ‖²` trace in float32 loses most of the signal
    here. Chan-Golub-LeVeque Welford must get it right.
    """
    torch.manual_seed(1)
    d = 128
    N = 5000
    mu = torch.full((d,), 100.0, dtype=torch.float32)
    x = torch.randn(N, d, dtype=torch.float32) * 0.01 + mu

    cm = CheapMoments(d=d, device="cpu")
    for i in range(0, N, 500):
        cm.update(x[i : i + 500])
    s = cm.finalize()

    true_trace = float(x.var(0, unbiased=True).sum())
    rel_err = abs(s["trace_cov"] - true_trace) / true_trace
    assert rel_err < 5e-2, f"trace unstable: got {s['trace_cov']}, want {true_trace}"


def test_full_moments_stable_under_large_mean() -> None:
    """Same regression test but for the full-tier covariance matrix."""
    torch.manual_seed(2)
    d = 32
    N = 5000
    mu = torch.full((d,), 50.0, dtype=torch.float32)
    x = torch.randn(N, d, dtype=torch.float32) * 0.02 + mu

    fm = FullMoments(d=d, device="cpu")
    for i in range(0, N, 500):
        fm.update(x[i : i + 500])
    s = fm.finalize()

    true_cov = x.T.cov()
    diff = (s["cov"] - true_cov).abs().max().item()
    assert diff < 1e-4, f"full cov unstable: max diff {diff}"


def test_spherical_moments_closed_form_pairwise() -> None:
    """E[‖u−v‖²] = 2(1 − ‖R̄‖²) for iid unit vectors — closed form, no sampling."""
    torch.manual_seed(3)
    d = 32
    x = torch.randn(2000, d)
    sm = SphericalMoments(d=d, device="cpu")
    sm.update(x)
    s = sm.finalize()

    u = x / x.norm(dim=1, keepdim=True)
    empirical = ((u[:1000] - u[1000:]) ** 2).sum(-1).mean().item()
    closed = 2.0 * (1.0 - s["R_bar_norm"] ** 2)
    assert abs(empirical - closed) < 0.1
    assert 0.0 <= s["spherical_variance"] <= 2.0


def test_spherical_moments_tracks_pole_distance() -> None:
    """When a pole is supplied, max_chord_to_pole should match the brute force."""
    torch.manual_seed(4)
    d = 16
    x = torch.randn(500, d)
    pole = torch.randn(d)
    sm = SphericalMoments(d=d, device="cpu", pole=pole)
    sm.update(x)
    s = sm.finalize()

    u = x / x.norm(dim=1, keepdim=True)
    p = pole / pole.norm()
    expected_max = (u - p).norm(dim=1).max().item()
    assert abs(s["max_chord_to_pole"] - expected_max) < 1e-5


def test_reservoir_size_and_determinism() -> None:
    r1 = Reservoir(K=50, d=8, seed=0)
    r2 = Reservoir(K=50, d=8, seed=0)
    x = torch.arange(1000 * 8, dtype=torch.float32).reshape(1000, 8)
    r1.update(x)
    r2.update(x)
    assert r1.samples.shape == (50, 8)
    assert r1.seen == 1000
    assert torch.equal(r1.samples, r2.samples)


def test_reservoir_fills_when_under_capacity() -> None:
    r = Reservoir(K=50, d=4, seed=0)
    x = torch.arange(10 * 4, dtype=torch.float32).reshape(10, 4)
    r.update(x)
    # Not yet at capacity; the finalize should return just the first 10.
    assert r.finalize().shape == (10, 4)
    assert torch.equal(r.finalize(), x)


def test_reservoir_uniform_sampling_indices() -> None:
    """Over many trials, each index should appear in the reservoir ~K/N fraction of the time."""
    torch.manual_seed(0)
    N = 200
    K = 20
    trials = 400
    # Use the index-as-value embedding so we can see which rows ended up in the reservoir.
    x = torch.arange(N * 1, dtype=torch.float32).reshape(N, 1)
    counts = torch.zeros(N)
    for t in range(trials):
        r = Reservoir(K=K, d=1, seed=t)
        r.update(x)
        idxs = r.finalize().squeeze(-1).long()
        counts[idxs] += 1
    expected = trials * K / N  # = 40
    # Loose chi-square-ish bound: every index should be within 3x of expected.
    assert (counts > expected / 3).all()
    assert (counts < expected * 3).all()


def test_moments_work_with_gpu_if_available() -> None:
    """Smoke test: if CUDA is available, the accumulators should run there.

    We don't have GPU in this CPU-test pass, so this just exercises the
    device parameter plumbing without asserting values.
    """
    fm = FullMoments(d=8, device="cpu")
    fm.update(torch.randn(10, 8))
    s = fm.finalize()
    assert s["mean"].device.type == "cpu"
    assert s["cov"].device.type == "cpu"
