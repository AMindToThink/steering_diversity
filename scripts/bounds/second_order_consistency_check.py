"""
Second-order Taylor consistency check.

Theory (docs/second_order_tex.tex, eq. spherical-variance-taylor):

    V(s)  ≈  [tr(Σ_x) - m̂^⊤ Σ_x m̂]  /  (2 ‖m‖²),     m = μ + s·v_eff

Two tests:

1. Direct: at each (scale, run), compute V_pred from measured Σ_x, m̂, ‖m‖ and
   compare to the measured V = 1 - ‖R̄‖. Σ_x here is the *steered* covariance
   at that scale (stats_meta.json / stats.pt trace_cov is per-scale; the
   bounds_metrics.json "trace_sigma_x" uses the unsteered baseline and is thus
   wrong for this check — we recompute quad at the steered scale).

2. Parabola fit: fit ‖m(s)‖² = a·s² + b·s + c.
   sqrt(c) should match ‖m(0)‖ (measured directly from stats.pt mean at s=0).
   This tests whether the log-log curvature of V(s) comes from the quadratic
   structure of the denominator.

Usage:
    uv run python scripts/bounds/second_order_consistency_check.py \\
        --bounds-root outputs/bounds \\
        --out-dir outputs/bounds/second_order_check
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


RUN_SKIP_TOKENS = {"experiment2", "bounds_smoke", "plots"}


def discover_runs(bounds_root: Path) -> list[Path]:
    runs: list[Path] = []
    for child in sorted(bounds_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in RUN_SKIP_TOKENS:
            continue
        if not (child / "stats.pt").exists():
            continue
        if not (child / "stats_meta.json").exists():
            continue
        runs.append(child)
    return runs


def _as_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float64).cpu().numpy()


def per_scale_quantities(run_dir: Path) -> list[dict[str, Any]]:
    """Return one dict per scale with V, ||m||, tr(Σ), quad, E_norm."""
    stats = torch.load(run_dir / "stats.pt", map_location="cpu", weights_only=False)
    meta = json.loads((run_dir / "stats_meta.json").read_text())

    out: list[dict[str, Any]] = []
    per_scale = stats["per_scale"]
    for scale_f, bucket in sorted(per_scale.items(), key=lambda x: float(x[0])):
        s = float(scale_f)
        mean = _as_np(bucket["mean"])  # (d,)
        cov = _as_np(bucket["cov"])  # (d, d)
        m_norm = float(np.linalg.norm(mean))
        tr_cov = float(bucket["trace_cov"])
        V = float(bucket["spherical_variance"])
        E_norm = float(bucket["E_norm"])
        R = float(bucket["R"])
        count = int(bucket["count"])

        # quad = m̂^⊤ Σ_x m̂ using per-scale Σ_x
        if m_norm > 0.0:
            mh = mean / m_norm
            quad = float(mh @ (cov @ mh))
        else:
            quad = float("nan")

        # predicted V via second-order Taylor (using per-scale Σ_x)
        if m_norm > 0.0 and math.isfinite(quad):
            V_pred = (tr_cov - quad) / (2.0 * m_norm * m_norm)
        else:
            V_pred = float("nan")

        out.append({
            "scale": s,
            "V_measured": V,
            "V_pred_second_order": V_pred,
            "m_norm": m_norm,
            "trace_sigma_x_scale": tr_cov,
            "quad_m_hat_sigma_x_m_hat_scale": quad,
            "E_norm_x": E_norm,
            "R": R,
            "count": count,
        })
    return out


def fit_parabola_m_squared(scales: np.ndarray, m_sq: np.ndarray) -> dict[str, float]:
    """Fit ||m(s)||² = a s² + b s + c via linear least squares."""
    A = np.stack([scales * scales, scales, np.ones_like(scales)], axis=1)
    coeffs, residuals, rank, sv = np.linalg.lstsq(A, m_sq, rcond=None)
    a, b, c = (float(x) for x in coeffs)
    pred = A @ coeffs
    ss_res = float(np.sum((m_sq - pred) ** 2))
    ss_tot = float(np.sum((m_sq - m_sq.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"a": a, "b": b, "c": c, "r_squared": r2}


def fit_V_one_over_quadratic(scales: np.ndarray, V: np.ndarray) -> dict[str, float]:
    """Fit V(s) = A / (s² + B s + C) by linearizing: 1/V = (s² + B s + C)/A."""
    mask = (V > 0) & np.isfinite(V)
    if mask.sum() < 3:
        return {"A": float("nan"), "B": float("nan"), "C": float("nan"), "r_squared": float("nan")}
    s = scales[mask]
    inv_V = 1.0 / V[mask]
    # 1/V = s²/A + (B/A) s + C/A  → solve for (1/A, B/A, C/A)
    X = np.stack([s * s, s, np.ones_like(s)], axis=1)
    coeffs, *_ = np.linalg.lstsq(X, inv_V, rcond=None)
    inv_A, BoverA, CoverA = (float(x) for x in coeffs)
    if abs(inv_A) < 1e-30:
        return {"A": float("nan"), "B": float("nan"), "C": float("nan"), "r_squared": float("nan")}
    A = 1.0 / inv_A
    B = BoverA * A
    C = CoverA * A
    pred = X @ coeffs
    ss_res = float(np.sum((inv_V - pred) ** 2))
    ss_tot = float(np.sum((inv_V - inv_V.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"A": A, "B": B, "C": C, "r_squared": r2}


def analyze_run(run_dir: Path) -> dict[str, Any]:
    rows = per_scale_quantities(run_dir)
    scales = np.array([r["scale"] for r in rows], dtype=np.float64)
    V = np.array([r["V_measured"] for r in rows], dtype=np.float64)
    m_norm = np.array([r["m_norm"] for r in rows], dtype=np.float64)
    m_sq = m_norm * m_norm

    # extract measured ||mu|| = ||m(0)||
    idx0 = int(np.argmin(scales))
    mu_norm_measured = float(m_norm[idx0])
    mu_sq_measured = float(m_sq[idx0])

    # fits on s > 0 points only (so we can extrapolate back and test)
    pos = scales > 0.0
    m_fit = fit_parabola_m_squared(scales[pos], m_sq[pos])
    V_fit = fit_V_one_over_quadratic(scales[pos], V[pos])

    sqrt_c_fit = math.sqrt(m_fit["c"]) if m_fit["c"] > 0 else float("nan")
    sqrt_C_V_fit = math.sqrt(V_fit["C"]) if (V_fit["C"] is not None and V_fit["C"] > 0) else float("nan")

    # ratios (second-order theory consistency)
    ratios_V = []
    for r in rows:
        if r["V_measured"] > 0 and math.isfinite(r["V_pred_second_order"]) and r["V_pred_second_order"] > 0:
            ratios_V.append(r["V_pred_second_order"] / r["V_measured"])
        else:
            ratios_V.append(float("nan"))

    # V*||m||² should be roughly (tr-quad)/2 if theory holds
    V_times_m_sq = (V * m_sq).tolist()
    half_tr_minus_quad = [
        0.5 * (r["trace_sigma_x_scale"] - r["quad_m_hat_sigma_x_m_hat_scale"]) for r in rows
    ]

    return {
        "run": run_dir.name,
        "n_scales": len(rows),
        "per_scale": rows,
        "mu_norm_measured_at_s0": mu_norm_measured,
        "mu_sq_measured_at_s0": mu_sq_measured,
        "fit_m_squared_vs_s": m_fit,
        "fit_V_vs_s": V_fit,
        "sqrt_c_from_m_sq_fit": sqrt_c_fit,
        "sqrt_C_from_V_fit": sqrt_C_V_fit,
        "sqrt_c_over_mu_measured": (sqrt_c_fit / mu_norm_measured) if mu_norm_measured > 0 else float("nan"),
        "V_pred_over_V_meas": ratios_V,
        "V_times_m_sq": V_times_m_sq,
        "half_tr_minus_quad": half_tr_minus_quad,
    }


def summarize(all_results: list[dict[str, Any]]) -> str:
    lines = []
    lines.append("# Second-order Taylor consistency check\n")
    lines.append(
        "Theory: V(s) ≈ (tr Σ_x − m̂ᵀΣ_x m̂) / (2‖m‖²),  m = μ + s·v_eff,"
        "  so ‖m(s)‖² = a s² + b s + c with sqrt(c) = ‖μ‖.\n"
    )

    lines.append("\n## Table 1: Parabola fit of ‖m(s)‖² vs s (s>0 only)\n")
    lines.append("| run | a | b | c | sqrt(c) | ‖μ‖ at s=0 | sqrt(c) / ‖μ‖ | R² |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in all_results:
        f = r["fit_m_squared_vs_s"]
        lines.append(
            f"| {r['run']} | {f['a']:.3g} | {f['b']:.3g} | {f['c']:.3g} | "
            f"{r['sqrt_c_from_m_sq_fit']:.3f} | {r['mu_norm_measured_at_s0']:.3f} | "
            f"{r['sqrt_c_over_mu_measured']:.3f} | {f['r_squared']:.4f} |"
        )

    lines.append("\n## Table 2: Direct second-order prediction per scale\n")
    lines.append("V_pred/V_meas close to 1 ⇒ second-order theory is quantitatively accurate at that scale.\n")
    for r in all_results:
        lines.append(f"\n### {r['run']}")
        lines.append("| s | V_meas | V_pred | V_pred/V_meas | ‖m‖ | tr Σ (scale) | quad | V·‖m‖² | (tr-quad)/2 |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row, ratio, vm, half in zip(
            r["per_scale"], r["V_pred_over_V_meas"], r["V_times_m_sq"], r["half_tr_minus_quad"]
        ):
            s = row["scale"]
            V_m = row["V_measured"]
            V_p = row["V_pred_second_order"]
            mn = row["m_norm"]
            tr = row["trace_sigma_x_scale"]
            qd = row["quad_m_hat_sigma_x_m_hat_scale"]
            lines.append(
                f"| {s:g} | {V_m:.4g} | {V_p:.4g} | {ratio:.3f} | {mn:.3f} | "
                f"{tr:.3g} | {qd:.3g} | {vm:.3g} | {half:.3g} |"
            )

    lines.append("\n## Table 3: Fit of V(s) = A/(s² + B s + C) via linearized 1/V (s>0)\n")
    lines.append("Sanity check: if the parabola-denominator story holds, (B,C) from this fit should match (b/a, c/a) from Table 1.\n")
    lines.append("| run | A | B | C | sqrt(C) | b/a (Tbl1) | c/a (Tbl1) | R² (of 1/V lin) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in all_results:
        f1 = r["fit_m_squared_vs_s"]
        f2 = r["fit_V_vs_s"]
        b_over_a = f1["b"] / f1["a"] if f1["a"] != 0 else float("nan")
        c_over_a = f1["c"] / f1["a"] if f1["a"] != 0 else float("nan")
        lines.append(
            f"| {r['run']} | {f2['A']:.3g} | {f2['B']:.3g} | {f2['C']:.3g} | "
            f"{r['sqrt_C_from_V_fit']:.3f} | {b_over_a:.3g} | {c_over_a:.3g} | {f2['r_squared']:.4f} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bounds-root", type=Path, default=Path("outputs/bounds"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/bounds/second_order_check"))
    parser.add_argument(
        "--only",
        type=str,
        nargs="*",
        default=None,
        help="If given, restrict to run directories matching these exact names.",
    )
    args = parser.parse_args()

    runs = discover_runs(args.bounds_root)
    if args.only:
        runs = [r for r in runs if r.name in set(args.only)]
    if not runs:
        raise SystemExit(f"No eligible runs found under {args.bounds_root}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []
    for run_dir in runs:
        print(f"[analyze] {run_dir.name}", flush=True)
        result = analyze_run(run_dir)
        all_results.append(result)

    (args.out_dir / "results.json").write_text(json.dumps(all_results, indent=2))
    (args.out_dir / "summary.md").write_text(summarize(all_results))
    print(f"Wrote {args.out_dir/'results.json'}")
    print(f"Wrote {args.out_dir/'summary.md'}")


if __name__ == "__main__":
    main()
