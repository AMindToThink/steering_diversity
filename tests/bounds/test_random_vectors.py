"""Tests for src/bounds/random_vectors.py."""

from __future__ import annotations

import pytest
import torch

from src.bounds.random_vectors import (
    generate_random_steering_vector,
    generate_random_steering_vector_aggregate_matched,
)


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


def test_aggregate_matched_matches_reference_sum_norm() -> None:
    """The aggregate-matched random vector's Σ over target_layers should
    have exactly the same L2 norm as the reference's Σ over target_layers."""
    torch.manual_seed(0)
    d = 256
    ref = {i: torch.randn(d) for i in range(20)}
    # Normalize each reference layer so they stack coherently like a
    # trained steering vector — ‖Σ ref[target]‖ ≈ n when the refs are
    # identical, and ≈ √n when random.
    for i in ref:
        ref[i] = ref[i] / ref[i].norm()
    target_layers = [5, 6, 7, 8, 9, 10, 11, 12]

    # Reference aggregate
    ref_sum = sum(ref[i] for i in target_layers)
    target_norm = float(ref_sum.norm().item())

    out = generate_random_steering_vector_aggregate_matched(
        reference=ref, target_layers=target_layers, seed=42
    )

    random_sum = sum(out[i] for i in target_layers)
    random_norm = float(random_sum.norm().item())
    assert abs(random_norm - target_norm) / target_norm < 1e-5, (
        f"aggregate norm mismatch: got {random_norm}, want {target_norm}"
    )


def test_aggregate_matched_per_layer_larger_than_reference() -> None:
    """Because random directions don't stack coherently, the per-layer
    random vector must be ~√n × larger than the reference's per-layer norm
    to compensate and reach the same aggregate."""
    torch.manual_seed(0)
    d = 512
    n_layers = 16
    # Reference: perfectly coherent — every layer is the same unit direction.
    direction = torch.randn(d)
    direction /= direction.norm()
    ref = {i: direction.clone() for i in range(n_layers)}
    target_layers = list(range(n_layers))
    # ‖Σ ref‖ = n × 1 = n = 16

    out = generate_random_steering_vector_aggregate_matched(
        reference=ref, target_layers=target_layers, seed=7
    )

    # Per-layer norms should be ~√n = 4, not 1.
    per_layer_norms = [float(out[i].norm().item()) for i in target_layers]
    for pl in per_layer_norms:
        # Loose bound: each per-layer norm should be in [√n × 0.5, √n × 2].
        expected = (n_layers ** 0.5)
        assert 0.5 * expected < pl < 2.0 * expected, (
            f"per-layer norm {pl} outside [{0.5*expected:.2f}, {2*expected:.2f}] "
            f"for expected ~{expected:.2f}"
        )


def test_aggregate_matched_deterministic() -> None:
    d = 64
    ref = {i: torch.randn(d) for i in range(8)}
    tl = [0, 2, 4, 6]
    a = generate_random_steering_vector_aggregate_matched(ref, tl, seed=123)
    b = generate_random_steering_vector_aggregate_matched(ref, tl, seed=123)
    c = generate_random_steering_vector_aggregate_matched(ref, tl, seed=124)
    for k in ref:
        assert torch.equal(a[k], b[k])
        assert not torch.equal(a[k], c[k])


def test_aggregate_matched_rejects_missing_layer() -> None:
    ref = {0: torch.randn(16), 1: torch.randn(16)}
    with pytest.raises(KeyError, match="not found in reference"):
        generate_random_steering_vector_aggregate_matched(
            ref, target_layers=[0, 1, 99], seed=0
        )


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
