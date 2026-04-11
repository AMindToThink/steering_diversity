"""Tests for src/bounds/random_vectors.py."""

from __future__ import annotations

import torch

from src.bounds.random_vectors import generate_random_steering_vector


def test_matches_reference_norms_and_layers() -> None:
    torch.manual_seed(123)
    ref = {
        10: torch.randn(128) * 3.0,
        11: torch.randn(128) * 0.7,
        25: torch.randn(128) * 5.0,
    }
    out = generate_random_steering_vector(ref, seed=42)
    assert set(out.keys()) == set(ref.keys())
    for k in ref:
        assert out[k].shape == ref[k].shape
        assert out[k].dtype == torch.float32
        rel = abs(out[k].norm().item() - ref[k].norm().item()) / ref[k].norm().item()
        assert rel < 1e-5, f"layer {k}: rel_err {rel}"


def test_deterministic_with_seed() -> None:
    ref = {5: torch.ones(64), 9: torch.full((64,), 2.0)}
    a = generate_random_steering_vector(ref, seed=7)
    b = generate_random_steering_vector(ref, seed=7)
    c = generate_random_steering_vector(ref, seed=8)
    for k in ref:
        assert torch.equal(a[k], b[k]), f"same seed differs at layer {k}"
        assert not torch.equal(a[k], c[k]), f"different seeds match at layer {k}"


def test_different_layers_are_independent() -> None:
    """Vectors at different layers should not be identical even with same norm."""
    ref = {0: torch.ones(32), 1: torch.ones(32)}  # identical references
    out = generate_random_steering_vector(ref, seed=1)
    # They should not be equal: seeding should differ per layer.
    assert not torch.equal(out[0], out[1])


def test_not_parallel_to_reference() -> None:
    """A random vector should not accidentally align with the reference."""
    torch.manual_seed(0)
    d = 2048  # big enough that cos should be ~O(1/sqrt(d)) ≈ 0.02
    ref = {0: torch.randn(d)}
    out = generate_random_steering_vector(ref, seed=5)
    cos = torch.dot(out[0], ref[0]) / (out[0].norm() * ref[0].norm())
    assert abs(cos.item()) < 0.1, f"random parallel to ref: cos={cos.item()}"


def test_handles_zero_norm_reference() -> None:
    """If a reference layer is all zeros, the output should also be zero.

    A reference with zero L2 norm has no meaningful norm to match to, so
    the natural interpretation is 'no steering at this layer'. Choosing
    zero over raising makes it safe to call this on partial vectors.
    """
    ref = {0: torch.zeros(16), 1: torch.ones(16)}
    out = generate_random_steering_vector(ref, seed=0)
    assert torch.equal(out[0], torch.zeros(16))
    assert abs(out[1].norm().item() - ref[1].norm().item()) < 1e-6
