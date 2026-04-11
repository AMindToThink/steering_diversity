"""Tests for src/bounds/claims.py — per-claim LHS/RHS formulas.

Every test uses a synthetic, known-distribution setting (usually isotropic
Gaussian with a fixed steering direction) so the theoretical bound and the
empirical quantity can both be computed in closed form or by brute force.
"""

from __future__ import annotations

import torch

from src.bounds.claims import (
    claim1_covariance_structure,
    claim2_spherical_variance,
    claim3_pole_concentration,
    claim4_pairwise_contraction,
    claim5_diameter,
    claim6_expected_pairwise,
    claim7_reduction_condition,
    claim8_scaling_fit,
    claim9_alignment,
    compute_all_claims,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso_gaussian_run(
    d: int = 64,
    N: int = 20000,
    sigma: float = 1.0,
    mu: torch.Tensor | None = None,
    s: torch.Tensor | None = None,
    seed: int = 0,
) -> dict:
    """Build a fake 'run' dict containing the quantities the claim functions need.

    Simulates `N` iid Gaussian samples x ~ N(mu, sigma^2 I) and their
    steered counterparts x + s, both at the pre-RMSNorm site. Returns
    finalized stats for unsteered, steered, spherical (on pre_steered),
    plus reservoirs — matching the shape Task 8's recording script will
    actually produce.
    """
    torch.manual_seed(seed)
    mu = torch.zeros(d) if mu is None else mu
    s = torch.zeros(d) if s is None else s
    x_unsteered = torch.randn(N, d) * sigma + mu
    x_steered = x_unsteered + s

    unsteered_mean = x_unsteered.mean(dim=0)
    unsteered_cov = x_unsteered.T.cov()
    steered_mean = x_steered.mean(dim=0)
    steered_cov = x_steered.T.cov()

    unit_steered = x_steered / x_steered.norm(dim=1, keepdim=True)
    r_bar = unit_steered.mean(dim=0)
    r_bar_norm = r_bar.norm().item()
    # Use m̂ = normalized steered mean as the pole for Claim 3.
    m_hat = steered_mean / (steered_mean.norm() + 1e-12)
    max_chord_to_pole = (unit_steered - m_hat).norm(dim=1).max().item()

    # Reservoir = first K samples (deterministic, avoids randomness in tests).
    K = min(512, N)

    return {
        "d": d,
        "unsteered": {
            "mean": unsteered_mean,
            "cov": unsteered_cov,
            "trace_cov": float(unsteered_cov.trace().item()),
            "E_norm": float(x_unsteered.norm(dim=1).mean().item()),
            "E_sq_norm": float((x_unsteered.norm(dim=1) ** 2).mean().item()),
            "R": float(x_unsteered.norm(dim=1).max().item()),
            "reservoir": x_unsteered[:K].clone(),
        },
        "steered": {
            "mean": steered_mean,
            "cov": steered_cov,
            "trace_cov": float(steered_cov.trace().item()),
            "E_norm": float(x_steered.norm(dim=1).mean().item()),
            "E_sq_norm": float((x_steered.norm(dim=1) ** 2).mean().item()),
            "R": float(x_steered.norm(dim=1).max().item()),
            "reservoir": x_steered[:K].clone(),
        },
        "spherical_steered": {
            "R_bar": r_bar,
            "R_bar_norm": r_bar_norm,
            "spherical_variance": 1.0 - r_bar_norm,
            "expected_pair_sq_chord": 2.0 * (1.0 - r_bar_norm ** 2),
            "max_chord_to_pole": max_chord_to_pole,
        },
        "s_raw": s,  # nominal steering vector (effective s is computed from means)
    }


# ---------------------------------------------------------------------------
# Claim 1: covariance structure (approximate, Taylor bound)
# ---------------------------------------------------------------------------


def test_claim1_isotropic_large_s() -> None:
    d = 64
    mu = torch.zeros(d)
    s = torch.zeros(d)
    s[0] = 50.0  # large ‖s‖ so the Taylor bound is tight
    run = _iso_gaussian_run(d=d, mu=mu, s=s, sigma=1.0, N=30000, seed=0)

    r = claim1_covariance_structure(run)
    # LHS = tr(Σ_z) = 1 − ‖R̄‖² from the unit sphere.
    # RHS ≈ (tr(Σ_x) − ŝ^T Σ_x ŝ) / ‖m‖² = (d − 1) / ‖s‖²
    expected_rhs = (d - 1) / 50.0**2
    assert abs(r["rhs"] - expected_rhs) / expected_rhs < 0.1
    # LHS should be within a factor of ~2 of RHS in the large-s regime.
    assert r["lhs"] > 0
    assert 0.3 < r["lhs"] / r["rhs"] < 3.0
    assert r["name"] == "claim1_covariance_structure"


# ---------------------------------------------------------------------------
# Claim 2: spherical variance global bound
# ---------------------------------------------------------------------------


def test_claim2_bound_holds_large_s() -> None:
    d = 64
    s = torch.zeros(d)
    s[0] = 10.0
    run = _iso_gaussian_run(d=d, mu=torch.zeros(d), s=s, sigma=1.0, N=20000, seed=1)

    r = claim2_spherical_variance(run)
    # LHS = 1 − ‖R̄‖; RHS = 2 E[‖x‖] / ‖s_eff‖
    assert r["lhs"] > 0
    assert r["rhs"] > 0
    assert r["passed"] is True, f"Claim 2 bound violated: {r}"
    assert r["ratio"] < 1.0


def test_claim2_bound_tightens_with_larger_s() -> None:
    """The RHS should shrink as ‖s‖ grows."""
    d = 32
    rhs_prev = float("inf")
    for s_mag in [2.0, 4.0, 8.0, 16.0]:
        s = torch.zeros(d)
        s[0] = s_mag
        run = _iso_gaussian_run(d=d, s=s, sigma=1.0, N=10000, seed=2)
        r = claim2_spherical_variance(run)
        assert r["rhs"] < rhs_prev
        rhs_prev = r["rhs"]


# ---------------------------------------------------------------------------
# Claim 3: pole concentration — exact global bound
# ---------------------------------------------------------------------------


def test_claim3_exact_bound_always_holds() -> None:
    d = 32
    for s_mag in [5.0, 10.0, 20.0]:
        s = torch.zeros(d)
        s[0] = s_mag
        run = _iso_gaussian_run(d=d, s=s, sigma=1.0, N=5000, seed=3)
        r = claim3_pole_concentration(run)
        # Exact bound: max chord-to-pole ≤ 2 R / ‖s_eff‖.
        assert r["passed"] is True, f"Claim 3 failed at ‖s‖={s_mag}: {r}"
        assert r["lhs"] <= r["rhs"] + 1e-4


# ---------------------------------------------------------------------------
# Claim 4: pairwise Lipschitz contraction
# ---------------------------------------------------------------------------


def test_claim4_bound_holds_when_s_greater_than_R() -> None:
    d = 32
    s = torch.zeros(d)
    s[0] = 30.0  # Large enough that ‖s‖ > R for unit-variance Gaussians
    run = _iso_gaussian_run(d=d, s=s, sigma=1.0, N=5000, seed=4)

    r = claim4_pairwise_contraction(run)
    assert r["rhs"] > 0
    # The LHS is the empirical max Lipschitz ratio; should be ≤ RHS for these s, R.
    assert r["passed"] is True, f"Claim 4 Lipschitz bound violated: {r}"


def test_claim4_skipped_when_s_less_than_R() -> None:
    d = 32
    s = torch.zeros(d)
    s[0] = 0.5  # tiny — definitely smaller than R ≈ sqrt(d) ≈ 5.6
    run = _iso_gaussian_run(d=d, s=s, sigma=1.0, N=2000, seed=5)
    r = claim4_pairwise_contraction(run)
    # Bound is only defined for ‖s‖ > R; otherwise return an inapplicable marker.
    assert r["passed"] is None
    assert r["detail"]["inapplicable_reason"] is not None


# ---------------------------------------------------------------------------
# Claim 5: diameter — exact global bound
# ---------------------------------------------------------------------------


def test_claim5_diameter_bound_holds() -> None:
    d = 32
    for s_mag in [8.0, 16.0, 32.0]:
        s = torch.zeros(d)
        s[0] = s_mag
        run = _iso_gaussian_run(d=d, s=s, sigma=1.0, N=5000, seed=6)
        r = claim5_diameter(run)
        assert r["passed"] is True, f"Claim 5 diameter violated at ‖s‖={s_mag}: {r}"
        assert r["lhs"] <= r["rhs"] + 1e-4


# ---------------------------------------------------------------------------
# Claim 6: expected pairwise chordal distance (squared form)
# ---------------------------------------------------------------------------


def test_claim6_expected_pair_bound_holds() -> None:
    d = 32
    s = torch.zeros(d)
    s[0] = 10.0
    run = _iso_gaussian_run(d=d, s=s, sigma=1.0, N=10000, seed=7)

    r = claim6_expected_pairwise(run)
    # LHS = 2(1 − ‖R̄‖²); RHS = (4 E[‖x‖] / ‖s‖)².
    assert r["passed"] is True, f"Claim 6 violated: {r}"


# ---------------------------------------------------------------------------
# Claim 7: reduction condition — binary gate
# ---------------------------------------------------------------------------


def test_claim7_gate_positive() -> None:
    d = 8
    mu = torch.zeros(d)
    mu[0] = 1.0
    s = torch.zeros(d)
    s[0] = 2.0
    run = _iso_gaussian_run(d=d, mu=mu, s=s, sigma=0.1, N=2000, seed=8)
    r = claim7_reduction_condition(run)
    # ‖μ+s‖ = 3 > ‖μ‖ = 1 → condition holds.
    assert r["passed"] is True


def test_claim7_gate_negative() -> None:
    d = 8
    mu = torch.zeros(d)
    mu[0] = 1.0
    s = torch.zeros(d)
    s[0] = -0.5
    run = _iso_gaussian_run(d=d, mu=mu, s=s, sigma=0.1, N=2000, seed=9)
    r = claim7_reduction_condition(run)
    # ‖μ+s‖ = 0.5 < ‖μ‖ = 1 → condition fails.
    assert r["passed"] is False


# ---------------------------------------------------------------------------
# Claim 8: scaling-law fit (−1 and −2 log-log slopes)
# ---------------------------------------------------------------------------


def test_claim8_scaling_fit_recovers_slopes() -> None:
    # Synthesize a clean power law: V = C1 / ‖s‖, tr_z = C2 / ‖s‖²
    per_scale = []
    for s_mag in [1.0, 2.0, 4.0, 8.0, 16.0]:
        per_scale.append(
            {
                "s_eff_norm": s_mag,
                "lhs_spherical_variance": 0.5 / s_mag,
                "lhs_trace_cov_z": 0.25 / (s_mag * s_mag),
            }
        )
    r = claim8_scaling_fit(per_scale)
    assert abs(r["detail"]["slope_spherical_variance"] - (-1.0)) < 0.05
    assert abs(r["detail"]["slope_trace_cov_z"] - (-2.0)) < 0.05
    assert r["passed"] is True


# ---------------------------------------------------------------------------
# Claim 9: lost-dimension alignment
# ---------------------------------------------------------------------------


def test_claim9_alignment_recovers_direction() -> None:
    """For large ‖s‖, the smallest eigenvector of Σ_z should align with m̂."""
    d = 32
    s = torch.zeros(d)
    s[0] = 20.0
    run = _iso_gaussian_run(d=d, mu=torch.zeros(d), s=s, sigma=1.0, N=5000, seed=10)
    r = claim9_alignment(run)
    # LHS = |cos(smallest_eigvec, m̂)| — should be close to 1 for large s.
    assert r["lhs"] > 0.7, f"Claim 9 alignment too low: {r}"
    assert r["passed"] is True


def test_claim9_random_control_is_low() -> None:
    """With zero steering, m̂ is undefined and we should flag inapplicable."""
    d = 16
    run = _iso_gaussian_run(d=d, s=torch.zeros(d), sigma=1.0, N=2000, seed=11)
    r = claim9_alignment(run)
    assert r["passed"] is None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def test_compute_all_claims_orchestrator() -> None:
    """End-to-end smoke: build stats for three scales, run compute_all_claims."""
    d = 32
    scale_runs: dict[float, dict] = {}
    for scale in [0.0, 1.0, 4.0]:
        s = torch.zeros(d)
        s[0] = scale * 10.0
        scale_runs[scale] = _iso_gaussian_run(d=d, s=s, sigma=1.0, N=3000, seed=int(scale * 10))

    out = compute_all_claims(scale_runs)

    # scale=0 should be skipped (no steering).
    assert 0.0 not in out or out[0.0] == {}
    for nonzero in [1.0, 4.0]:
        entry = out[nonzero]
        for claim in [
            "claim1_covariance_structure",
            "claim2_spherical_variance",
            "claim3_pole_concentration",
            "claim4_pairwise_contraction",
            "claim5_diameter",
            "claim6_expected_pairwise",
            "claim7_reduction_condition",
            "claim9_alignment",
        ]:
            assert claim in entry
            assert "lhs" in entry[claim]
            assert "rhs" in entry[claim]
    # Claim 8 is a run-level scaling fit, attached under a dedicated key.
    assert "claim8_scaling_fit" in out
