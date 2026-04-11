"""Per-claim LHS/RHS formulas for the bounds-verification pipeline.

Each ``claim_N_*`` function takes a ``run`` dict (produced by the recording
script in Task 8) for ONE scale and returns a standard result::

    {
        "name": "claim1_covariance_structure",
        "lhs": float,           # empirical quantity
        "rhs": float,           # theoretical bound (may be np.inf)
        "ratio": lhs / rhs,     # convenience
        "passed": bool | None,  # None = bound inapplicable at this scale
        "detail": {...},        # claim-specific extras
    }

``compute_all_claims`` runs every claim on every nonzero scale and also
fits the Claim-8 scaling law across scales.

Input contract for a single ``run`` dict::

    {
        "d": int,
        "unsteered": {
            "mean": Tensor[d],
            "cov": Tensor[d, d],        # full; computed by FullMoments
            "trace_cov": float,
            "E_norm": float,            # E[‖x‖]
            "E_sq_norm": float,
            "R": float,                 # max ‖x‖
            "reservoir": Tensor[K, d],
        },
        "steered":   {...same shape...},
        "spherical_steered": {
            "R_bar": Tensor[d],
            "R_bar_norm": float,
            "spherical_variance": float,
            "expected_pair_sq_chord": float,
            "max_chord_to_pole": float, # pre-computed with pole = m̂ (optional)
        },
        "s_raw": Tensor[d] | None,
    }

At scale == 0, ``unsteered`` and ``steered`` are the same distribution and
``s_eff = 0``; all claims return ``passed=None`` because the bounds are
undefined for zero steering. At scale > 0, the caller must still provide
``unsteered`` — it comes from the shared scale=0 reference.
"""

from __future__ import annotations

import math
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _result(
    name: str,
    lhs: float,
    rhs: float,
    passed: bool | None,
    detail: dict | None = None,
) -> dict[str, Any]:
    # Handle lhs=0, rhs=0 gracefully.
    if rhs == 0 or not math.isfinite(rhs):
        ratio = float("inf") if lhs > 0 else 0.0
    else:
        ratio = lhs / rhs
    return {
        "name": name,
        "lhs": float(lhs),
        "rhs": float(rhs),
        "ratio": float(ratio),
        "passed": passed,
        "detail": detail or {},
    }


def _inapplicable(name: str, reason: str) -> dict[str, Any]:
    return _result(name, 0.0, float("inf"), None, {"inapplicable_reason": reason})


def _s_eff(run: dict) -> torch.Tensor:
    """Effective steering at the capture site = μ(steered) − μ(unsteered)."""
    return run["steered"]["mean"] - run["unsteered"]["mean"]


def _is_scale_zero(run: dict, tol: float = 1e-10) -> bool:
    return _s_eff(run).norm().item() < tol


# ---------------------------------------------------------------------------
# Claim 1 — covariance structure (Proposition 1, theory.tex)
# ---------------------------------------------------------------------------


def claim1_covariance_structure(run: dict) -> dict:
    """``tr(Σ_z) ≈ (tr(Σ_x) − m̂^T Σ_x m̂) / ‖m‖²``.

    Approximate equality from a local Taylor expansion. LHS is the spherical
    variance proxy ``1 − ‖R̄‖²`` (which equals ``tr(Σ_z)`` for unit-normalized
    samples). ``passed`` is True when the ratio lies in ``[0.3, 3.0]``.
    """
    name = "claim1_covariance_structure"
    if _is_scale_zero(run):
        return _inapplicable(name, "scale=0 (no steering)")

    m = run["steered"]["mean"].double()
    m_norm = m.norm()
    if m_norm.item() == 0:
        return _inapplicable(name, "‖m‖ = 0")
    m_hat = m / m_norm

    sigma_x = run["unsteered"]["cov"].double()
    trace_sigma_x = float(run["unsteered"]["trace_cov"])
    quad = float((m_hat @ sigma_x @ m_hat).item())

    rhs = (trace_sigma_x - quad) / (m_norm.item() ** 2)
    lhs = 1.0 - run["spherical_steered"]["R_bar_norm"] ** 2  # tr(Σ_z) for unit samples

    passed = bool(rhs > 0 and 0.3 < (lhs / rhs) < 3.0) if rhs > 0 else None
    return _result(
        name,
        lhs,
        rhs,
        passed,
        {
            "m_norm": m_norm.item(),
            "trace_sigma_x": trace_sigma_x,
            "quad_m_hat_sigma_x_m_hat": quad,
            "note": "Approximate equality (Taylor bound), not a strict ≤.",
        },
    )


# ---------------------------------------------------------------------------
# Claim 2 — spherical variance bound (Corollary 1, geometric.tex)
# ---------------------------------------------------------------------------


def claim2_spherical_variance(run: dict) -> dict:
    """``V = 1 − ‖R̄‖ ≤ 2 E[‖x‖] / ‖s_eff‖``."""
    name = "claim2_spherical_variance"
    if _is_scale_zero(run):
        return _inapplicable(name, "scale=0 (no steering)")

    s_norm = _s_eff(run).norm().item()
    e_norm_x = float(run["unsteered"]["E_norm"])

    lhs = float(run["spherical_steered"]["spherical_variance"])
    rhs = 2.0 * e_norm_x / s_norm

    return _result(
        name, lhs, rhs, passed=bool(lhs <= rhs),
        detail={"s_eff_norm": s_norm, "E_norm_x": e_norm_x},
    )


# ---------------------------------------------------------------------------
# Claim 3 — pole concentration (Proposition 2, geometric.tex)
# ---------------------------------------------------------------------------


def _compute_max_chord_to_mhat(reservoir: torch.Tensor, m_hat: torch.Tensor) -> float:
    """Max chordal distance from a reservoir of points to the pole m̂."""
    u = reservoir / reservoir.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return float((u - m_hat).norm(dim=1).max().item())


def claim3_pole_concentration(run: dict) -> dict:
    """``max ‖φ_s(x) − ŝ‖ ≤ 2 max ‖x‖ / ‖s_eff‖`` — exact global bound."""
    name = "claim3_pole_concentration"
    if _is_scale_zero(run):
        return _inapplicable(name, "scale=0")

    m = run["steered"]["mean"]
    m_hat = m / m.norm().clamp_min(1e-12)
    reservoir = run["steered"]["reservoir"]

    # Prefer the streaming max_chord_to_pole if it was computed with the right
    # pole; otherwise recompute from the reservoir.
    lhs = _compute_max_chord_to_mhat(reservoir, m_hat)

    s_norm = _s_eff(run).norm().item()
    R = float(run["unsteered"]["R"])
    rhs = 2.0 * R / s_norm

    return _result(
        name, lhs, rhs, passed=bool(lhs <= rhs + 1e-5),
        detail={"R": R, "s_eff_norm": s_norm},
    )


# ---------------------------------------------------------------------------
# Claim 4 — pairwise Lipschitz contraction (Proposition 3, geometric.tex)
# ---------------------------------------------------------------------------


def claim4_pairwise_contraction(run: dict) -> dict:
    """``‖φ_s(x_1) − φ_s(x_2)‖ ≤ 2‖x_1 − x_2‖ / (‖s_eff‖ − R)``.

    Only defined when ``‖s_eff‖ > R``. We compute the empirical Lipschitz
    ratio ``max_{i,j} ‖Δφ‖ / ‖Δx‖`` over reservoir pairs and compare to
    ``2 / (‖s_eff‖ − R)``.
    """
    name = "claim4_pairwise_contraction"
    if _is_scale_zero(run):
        return _inapplicable(name, "scale=0")

    s_norm = _s_eff(run).norm().item()
    R = float(run["unsteered"]["R"])
    if s_norm <= R:
        return _inapplicable(name, f"‖s_eff‖={s_norm:.3g} ≤ R={R:.3g}")

    x = run["unsteered"]["reservoir"]
    x_s = run["steered"]["reservoir"]
    u = x_s / x_s.norm(dim=1, keepdim=True).clamp_min(1e-12)

    K = x.shape[0]
    # Upper triangular pairs: (K*(K-1)/2)
    iu, ju = torch.triu_indices(K, K, offset=1)
    dx = (x[iu] - x[ju]).norm(dim=1).clamp_min(1e-8)
    du = (u[iu] - u[ju]).norm(dim=1)
    ratios = du / dx
    lhs = float(ratios.max().item())
    rhs = 2.0 / (s_norm - R)

    return _result(
        name, lhs, rhs, passed=bool(lhs <= rhs + 1e-5),
        detail={"s_eff_norm": s_norm, "R": R, "num_pairs": int(iu.shape[0])},
    )


# ---------------------------------------------------------------------------
# Claim 5 — spherical diameter (Corollary 3, geometric.tex)
# ---------------------------------------------------------------------------


def claim5_diameter(run: dict) -> dict:
    """``diam(φ_s(D)) ≤ 4 R / ‖s_eff‖`` — exact global bound."""
    name = "claim5_diameter"
    if _is_scale_zero(run):
        return _inapplicable(name, "scale=0")

    x_s = run["steered"]["reservoir"]
    u = x_s / x_s.norm(dim=1, keepdim=True).clamp_min(1e-12)
    K = u.shape[0]
    iu, ju = torch.triu_indices(K, K, offset=1)
    chord = (u[iu] - u[ju]).norm(dim=1)
    lhs = float(chord.max().item())

    s_norm = _s_eff(run).norm().item()
    R = float(run["unsteered"]["R"])
    rhs = 4.0 * R / s_norm

    return _result(
        name, lhs, rhs, passed=bool(lhs <= rhs + 1e-5),
        detail={"R": R, "s_eff_norm": s_norm},
    )


# ---------------------------------------------------------------------------
# Claim 6 — expected pairwise chordal distance (Corollary 2, geometric.tex)
# ---------------------------------------------------------------------------


def claim6_expected_pairwise(run: dict) -> dict:
    """``E[‖φ_s(x_1) − φ_s(x_2)‖] ≤ 4 E[‖x‖] / ‖s_eff‖``.

    We estimate LHS in closed form from the spherical stats:
    ``E[‖u−v‖] ≤ sqrt(E[‖u−v‖²]) = sqrt(2(1 − ‖R̄‖²))``.
    """
    name = "claim6_expected_pairwise"
    if _is_scale_zero(run):
        return _inapplicable(name, "scale=0")

    sq = float(run["spherical_steered"]["expected_pair_sq_chord"])
    lhs = math.sqrt(max(0.0, sq))
    rhs = 4.0 * float(run["unsteered"]["E_norm"]) / _s_eff(run).norm().item()

    return _result(
        name, lhs, rhs, passed=bool(lhs <= rhs + 1e-5),
        detail={"expected_pair_sq_chord": sq},
    )


# ---------------------------------------------------------------------------
# Claim 7 — reduction condition (conditions.tex)
# ---------------------------------------------------------------------------


def claim7_reduction_condition(run: dict) -> dict:
    """Binary gate: ``‖μ + s_eff‖ > ‖μ‖`` iff steering reduces diversity.

    ``LHS = ‖μ_pre_steered‖``; ``RHS = ‖μ_pre_unsteered‖``. ``passed`` is
    True when LHS > RHS.
    """
    name = "claim7_reduction_condition"
    if _is_scale_zero(run):
        return _inapplicable(name, "scale=0")

    lhs = float(run["steered"]["mean"].norm().item())
    rhs = float(run["unsteered"]["mean"].norm().item())
    return _result(name, lhs, rhs, passed=bool(lhs > rhs), detail={})


# ---------------------------------------------------------------------------
# Claim 8 — scaling-law fit (synthesis.tex, across scales)
# ---------------------------------------------------------------------------


def claim8_scaling_fit(per_scale: list[dict]) -> dict:
    """Log-log regression of spherical variance and ``tr(Σ_z)`` vs ``‖s_eff‖``.

    Expected slopes: ``−1`` for spherical variance, ``−2`` for ``tr(Σ_z)``.
    Each entry of ``per_scale`` must contain ``s_eff_norm``,
    ``lhs_spherical_variance``, and ``lhs_trace_cov_z``. Entries with
    ``s_eff_norm == 0`` or non-positive LHS values are dropped before the fit.
    """
    name = "claim8_scaling_fit"
    xs: list[float] = []
    sv: list[float] = []
    tz: list[float] = []
    for e in per_scale:
        s = float(e.get("s_eff_norm", 0.0))
        lsv = float(e.get("lhs_spherical_variance", 0.0))
        ltz = float(e.get("lhs_trace_cov_z", 0.0))
        if s > 0 and lsv > 0 and ltz > 0:
            xs.append(math.log(s))
            sv.append(math.log(lsv))
            tz.append(math.log(ltz))
    if len(xs) < 2:
        return _inapplicable(name, "need ≥2 scales with positive LHS values")

    def _slope(x: list[float], y: list[float]) -> float:
        x_t = torch.tensor(x, dtype=torch.float64)
        y_t = torch.tensor(y, dtype=torch.float64)
        xm = x_t.mean()
        ym = y_t.mean()
        denom = ((x_t - xm) * (x_t - xm)).sum()
        if denom.item() == 0:
            return 0.0
        return float(((x_t - xm) * (y_t - ym)).sum().item() / denom.item())

    slope_sv = _slope(xs, sv)
    slope_tz = _slope(xs, tz)
    passed = bool(abs(slope_sv - (-1.0)) < 0.3 and abs(slope_tz - (-2.0)) < 0.3)

    # LHS/RHS here are a bit unusual: use slope deviation as a scalar summary.
    worst_dev = max(abs(slope_sv - (-1.0)), abs(slope_tz - (-2.0)))
    return _result(
        name,
        lhs=worst_dev,
        rhs=0.3,
        passed=passed,
        detail={
            "slope_spherical_variance": slope_sv,
            "slope_trace_cov_z": slope_tz,
            "expected_slope_spherical_variance": -1.0,
            "expected_slope_trace_cov_z": -2.0,
            "num_points": len(xs),
        },
    )


# ---------------------------------------------------------------------------
# Claim 9 — lost-dimension alignment (synthesis.tex)
# ---------------------------------------------------------------------------


def claim9_alignment(run: dict) -> dict:
    """Cosine of ``m̂`` with the smallest-eigenvector of ``Σ_z``.

    ``Σ_z`` is estimated from the steered reservoir after unit-normalizing
    each sample. Claim is ``|cos| → 1`` as ``‖s‖`` grows (the direction
    of most-lost variance should align with the steering pole ``m̂``).
    """
    name = "claim9_alignment"
    if _is_scale_zero(run):
        return _inapplicable(name, "scale=0")

    m = run["steered"]["mean"].double()
    m_norm = m.norm()
    if m_norm.item() == 0:
        return _inapplicable(name, "‖m‖ = 0")
    m_hat = m / m_norm

    reservoir = run["steered"]["reservoir"].double()
    u = reservoir / reservoir.norm(dim=1, keepdim=True).clamp_min(1e-12)
    u_centered = u - u.mean(dim=0, keepdim=True)
    sigma_z = (u_centered.T @ u_centered) / max(u.shape[0] - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(sigma_z)  # ascending
    smallest = eigvecs[:, 0]  # smallest eigenvalue's eigenvector

    cos = float(torch.abs(torch.dot(smallest, m_hat)).item())

    return _result(
        name, lhs=cos, rhs=1.0, passed=bool(cos > 0.5),
        detail={
            "smallest_eigenvalue": float(eigvals[0].item()),
            "largest_eigenvalue": float(eigvals[-1].item()),
        },
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def compute_all_claims(
    scale_runs: dict[float, dict],
) -> dict[float | str, Any]:
    """Run every claim on every scale in ``scale_runs``.

    Returns a dict keyed by scale (float). Each value is a dict of per-claim
    results. Additionally, the key ``"claim8_scaling_fit"`` at the top level
    holds the run-wide scaling-law fit.
    """
    out: dict[float | str, Any] = {}

    per_scale_for_claim8: list[dict] = []

    for scale, run in scale_runs.items():
        if _is_scale_zero(run):
            out[scale] = {}
            continue
        entry = {
            "claim1_covariance_structure": claim1_covariance_structure(run),
            "claim2_spherical_variance": claim2_spherical_variance(run),
            "claim3_pole_concentration": claim3_pole_concentration(run),
            "claim4_pairwise_contraction": claim4_pairwise_contraction(run),
            "claim5_diameter": claim5_diameter(run),
            "claim6_expected_pairwise": claim6_expected_pairwise(run),
            "claim7_reduction_condition": claim7_reduction_condition(run),
            "claim9_alignment": claim9_alignment(run),
        }
        out[scale] = entry

        s_norm = _s_eff(run).norm().item()
        per_scale_for_claim8.append(
            {
                "s_eff_norm": s_norm,
                "lhs_spherical_variance": entry["claim2_spherical_variance"]["lhs"],
                "lhs_trace_cov_z": entry["claim1_covariance_structure"]["lhs"],
            }
        )

    out["claim8_scaling_fit"] = claim8_scaling_fit(per_scale_for_claim8)
    return out
