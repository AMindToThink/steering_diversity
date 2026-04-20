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

    # ---- 1. V(s) on log-log: measured points vs direct second-order prediction ----
    # Plot V_measured against V_pred(s) = (tr Sigma - m_hat^T Sigma m_hat) / (2 ||m||^2),
    # where all quantities are measured per-scale. This is the literal second-order
    # formula evaluated at the data, not a shape fit to V.
    ax = axes[0]
    V_pred_all = 0.5 * (tr_all - quad_all) / (m_all ** 2)
    ax.loglog(s_pos, V_all[pos], "o", label="V measured", color="C0", markersize=8)
    ax.loglog(
        s_pos,
        V_pred_all[pos],
        "x--",
        color="C1",
        label=r"V$_{\mathrm{pred}}$ = (tr Σ − m̂ᵀΣm̂) / (2‖m‖²)",
        markersize=10,
        markeredgewidth=2,
    )
    ax.set_xlabel("scale s")
    ax.set_ylabel("V(s)")
    ax.set_title(f"{name}: V(s) vs second-order prediction")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    # ---- 2. ‖m(s)‖² with parabolic fit ----
    ax = axes[1]
    # Evaluate the fit on two ranges:
    #   - in-data: covers the measured s values, where R² is meaningful
    #   - extrapolation-to-s0: only shown if the fit is physically sensible
    #                          (a > 0 and c > 0), so the reader can see the
    #                          parabola pass near the measured ‖μ‖² marker
    s_in = np.geomspace(s_pos.min(), s_pos.max(), 120)
    m_fit_in = m_fit["a"] * s_in * s_in + m_fit["b"] * s_in + m_fit["c"]
    m_fit_in = np.where(m_fit_in > 0, m_fit_in, np.nan)

    ax.loglog(s_pos, m_all[pos] ** 2, "o", label="‖m(s)‖² measured", color="C0", markersize=8)

    # Plot the measured ‖μ‖² at s=0 as a reference marker. Since the axis is
    # log in s, place it at a small positive value with a short horizontal bar.
    mu_marker_x = max(s_pos.min() * 0.02, 1e-3)
    ax.loglog(
        [mu_marker_x],
        [m_all[0] ** 2],
        "s",
        color="C2",
        label=f"‖μ‖²_meas at s=0 = {m_all[0]**2:.1f}",
        markersize=10,
    )

    c_label = (
        f"√c={run['sqrt_c_from_m_sq_fit']:.2f}"
        if np.isfinite(run["sqrt_c_from_m_sq_fit"])
        else "√c undefined (c<0)"
    )
    ax.loglog(
        s_in,
        m_fit_in,
        "--",
        color="C1",
        label=f"a s² + b s + c fit  ({c_label})",
    )

    # If the quadratic is physical (a > 0, c > 0), show the extrapolation back
    # to s ≈ 0 so the reader can see the curve passing near the green μ marker.
    if m_fit["a"] > 0 and m_fit["c"] > 0:
        s_ext = np.geomspace(mu_marker_x, s_pos.min(), 80)
        m_fit_ext = m_fit["a"] * s_ext * s_ext + m_fit["b"] * s_ext + m_fit["c"]
        m_fit_ext = np.where(m_fit_ext > 0, m_fit_ext, np.nan)
        ax.loglog(s_ext, m_fit_ext, ":", color="C1", alpha=0.6, label="fit extrapolated to s→0")

    # Always plot the fit's intercept value c (if positive) at the μ marker's
    # x-coordinate, so the reader can compare √c to ‖μ‖ visually even when the
    # dotted extrapolation line is suppressed (case a<0).
    if m_fit["c"] > 0:
        ax.loglog(
            [mu_marker_x],
            [m_fit["c"]],
            "*",
            color="C1",
            markersize=14,
            markeredgecolor="black",
            markeredgewidth=0.6,
            label=f"fit intercept: c = {m_fit['c']:.1f}",
            zorder=6,
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
    """Scatter sqrt(c) vs ||mu||_measured across runs.

    Multiple runs (especially the four Qwen runs) cluster at nearly identical
    (||mu||, sqrt(c)) coordinates, so we put run names in the legend rather
    than annotating each point — avoiding the unreadable pile-up of text.
    """
    names = []
    sqrt_c = []
    mu_meas = []
    r2 = []
    for r in all_results:
        names.append(r["run"])
        sqrt_c.append(r["sqrt_c_from_m_sq_fit"])
        mu_meas.append(r["mu_norm_measured_at_s0"])
        r2.append(r["fit_m_squared_vs_s"]["r_squared"])

    fig, ax = plt.subplots(figsize=(8.5, 7))
    lo = 1.0
    hi = max(max(v for v in sqrt_c if np.isfinite(v)), max(mu_meas)) * 1.15
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.6, label="y = x (perfect)", zorder=1)
    ax.plot([lo, hi], [1.1 * lo, 1.1 * hi], ":", color="gray", alpha=0.5, label="±10%", zorder=1)
    ax.plot([lo, hi], [0.9 * lo, 0.9 * hi], ":", color="gray", alpha=0.5, zorder=1)

    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]
    cmap = plt.get_cmap("tab10")
    for idx, (name, sc, mu) in enumerate(zip(names, sqrt_c, mu_meas)):
        short = name.replace("bounds_", "")
        if not np.isfinite(sc):
            # Mark the omitted (c<0) run in the legend so the reader knows why
            # it is absent from the scatter.
            ax.scatter([], [], marker=markers[idx % len(markers)], s=80,
                       color=cmap(idx % 10),
                       label=f"{short}  [omitted: c<0]")
            continue
        ax.scatter(mu, sc, s=110, zorder=5,
                   marker=markers[idx % len(markers)],
                   color=cmap(idx % 10),
                   edgecolors="black", linewidths=0.7,
                   label=f"{short}  (√c/‖μ‖ = {sc/mu:.3f})")

    ax.set_xlabel("‖μ‖ measured at s=0 (from stats.pt mean)")
    ax.set_ylabel("√c from parabola fit of ‖m(s)‖²")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title("Consistency check: √c ≈ ‖μ‖ across runs")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
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
