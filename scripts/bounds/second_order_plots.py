"""
Diagnostic plots for the second-order Taylor consistency check.

Generates three plots per run (and one overview):
    1. V(s) vs s on log-log, with fitted V = A/(s² + Bs + C) overlay.
    2. ‖m(s)‖² vs s on log-log, with quadratic fit a s² + b s + c overlay.
       Includes the extrapolated s=0 intercept compared against ‖μ‖_measured.
    3. V · ‖m‖² vs s, compared to the second-order theory value (tr Σ - quad)/2.

Inputs: results.json produced by second_order_consistency_check.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_run(run: dict, out_dir: Path) -> None:
    name = run["run"]
    rows = run["per_scale"]
    s_all = np.array([r["scale"] for r in rows], dtype=np.float64)
    V_all = np.array([r["V_measured"] for r in rows], dtype=np.float64)
    m_all = np.array([r["m_norm"] for r in rows], dtype=np.float64)
    tr_all = np.array([r["trace_sigma_x_scale"] for r in rows], dtype=np.float64)
    quad_all = np.array([r["quad_m_hat_sigma_x_m_hat_scale"] for r in rows], dtype=np.float64)

    pos = s_all > 0
    s_pos = s_all[pos]

    m_fit = run["fit_m_squared_vs_s"]
    V_fit = run["fit_V_vs_s"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # ---- 1. V(s) on log-log ----
    ax = axes[0]
    ax.loglog(s_pos, V_all[pos], "o", label="V measured", color="C0")
    s_dense = np.geomspace(max(s_pos.min() * 0.5, 1e-3), s_pos.max() * 2, 200)
    if np.isfinite(V_fit["A"]) and np.isfinite(V_fit["B"]) and np.isfinite(V_fit["C"]):
        V_fit_curve = V_fit["A"] / (s_dense * s_dense + V_fit["B"] * s_dense + V_fit["C"])
        V_fit_curve = np.where(V_fit_curve > 0, V_fit_curve, np.nan)
        ax.loglog(
            s_dense,
            V_fit_curve,
            "--",
            color="C1",
            label=f"A/(s²+Bs+C),  √C={run['sqrt_C_from_V_fit']:.2f}",
        )
    ax.set_xlabel("scale s")
    ax.set_ylabel("V(s)")
    ax.set_title(f"{name}: V(s)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    # ---- 2. ‖m(s)‖² with parabolic fit ----
    ax = axes[1]
    ax.loglog(s_pos, m_all[pos] ** 2, "o", label="‖m(s)‖² measured", color="C0")
    ax.loglog(
        [s_all[0] + 1e-3],
        [m_all[0] ** 2],
        "s",
        color="C2",
        label=f"‖μ‖²_meas at s=0 = {m_all[0]**2:.1f}",
    )
    m_fit_curve = m_fit["a"] * s_dense * s_dense + m_fit["b"] * s_dense + m_fit["c"]
    m_fit_curve = np.where(m_fit_curve > 0, m_fit_curve, np.nan)
    c_label = (
        f"√c={run['sqrt_c_from_m_sq_fit']:.2f}"
        if np.isfinite(run["sqrt_c_from_m_sq_fit"])
        else "√c undefined (c<0)"
    )
    ax.loglog(
        s_dense, m_fit_curve, "--", color="C1",
        label=f"a s² + b s + c  ({c_label})",
    )
    ax.set_xlabel("scale s")
    ax.set_ylabel("‖m(s)‖²")
    ax.set_title(f"{name}: ‖m(s)‖²  (R²={m_fit['r_squared']:.4f})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    # ---- 3. V·‖m‖² vs (tr Σ - quad)/2 ----
    ax = axes[2]
    ax.plot(s_all, V_all * m_all ** 2, "o-", label="V · ‖m‖² (measured)", color="C0")
    ax.plot(s_all, 0.5 * (tr_all - quad_all), "s--", label="(tr Σ - m̂ᵀΣm̂) / 2  (theory)", color="C1")
    ax.set_xscale("symlog", linthresh=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("scale s")
    ax.set_ylabel("value")
    ax.set_title(f"{name}: second-order agreement")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = out_dir / f"{name}_second_order.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)


def overview_plot(all_results: list[dict], out_dir: Path) -> None:
    """Scatter sqrt(c) vs ||mu||_measured across runs."""
    names = []
    sqrt_c = []
    mu_meas = []
    r2 = []
    for r in all_results:
        names.append(r["run"])
        sqrt_c.append(r["sqrt_c_from_m_sq_fit"])
        mu_meas.append(r["mu_norm_measured_at_s0"])
        r2.append(r["fit_m_squared_vs_s"]["r_squared"])

    fig, ax = plt.subplots(figsize=(7, 7))
    lo = 1.0
    hi = max(max(v for v in sqrt_c if np.isfinite(v)), max(mu_meas)) * 1.15
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x (perfect)")
    ax.plot([lo, hi], [1.1 * lo, 1.1 * hi], ":", color="gray", alpha=0.5, label="±10%")
    ax.plot([lo, hi], [0.9 * lo, 0.9 * hi], ":", color="gray", alpha=0.5)

    for name, sc, mu in zip(names, sqrt_c, mu_meas):
        if not np.isfinite(sc):
            continue  # skip c<0 entries (can't take sqrt)
        ax.scatter(mu, sc, s=80, zorder=5)
        ax.annotate(name.replace("bounds_", ""), (mu, sc), xytext=(5, 5), textcoords="offset points", fontsize=8)

    ax.set_xlabel("‖μ‖ measured at s=0 (from stats.pt mean)")
    ax.set_ylabel("√c from parabola fit of ‖m(s)‖²")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title("Consistency check: √c ≈ ‖μ‖ across runs")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "overview_sqrtc_vs_mu.png", dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("outputs/bounds/second_order_check/results.json"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/bounds/second_order_check/plots"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = json.loads(args.results.read_text())

    for r in results:
        plot_run(r, args.out_dir)
        print(f"Wrote {args.out_dir / (r['run'] + '_second_order.png')}")

    overview_plot(results, args.out_dir)
    print(f"Wrote {args.out_dir / 'overview_sqrtc_vs_mu.png'}")


if __name__ == "__main__":
    main()
