"""Visualize bounds_metrics.json — pure CPU, no model, no GPU.

Reads ``outputs/bounds/<run_name>/bounds_metrics.json`` (produced by
``scripts/bounds/03_compute.py``) plus ``stats.pt`` (for Σ_z
eigenvalue spectra) and writes four plot families to
``outputs/bounds/<run_name>/plots/``:

1. **Per-claim LHS-vs-RHS scatter** (``claim_<n>_lhs_vs_rhs.png``):
   For each claim with real numeric LHS/RHS (Claims 1, 2, 3, 4, 5, 6, 9),
   plots the empirical LHS against the theoretical RHS across scales,
   with the diagonal LHS=RHS line for reference.

2. **Scaling law** (``scaling_law.png``):
   Log-log plot of the spherical variance ``1 − ‖R̄‖²`` and the
   pre-RMSNorm ``tr(Σ)`` vs. ``‖s_eff‖``, with reference slopes −1 and −2.
   The paper's Claim 8 predicts those slopes in the asymptotic regime.

3. **Pre-RMSNorm covariance spectrum** (``sigma_spectrum.png``):
   Eigenvalues of ``Σ_x`` (pre-RMSNorm covariance of the full-tier
   `FullMoments`) at each scale, sorted descending, log y-axis.
   Claim 1's rank-reduction prediction is visible here if it kicks in.

4. **Claim 9 alignment bars** (``claim9_alignment.png``):
   Cosine of the smallest-eigenvector of ``Σ_z`` with ``m̂`` per scale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _save_figure_with_caption(fig, out_path: Path, caption: str) -> None:
    """Save ``fig`` at ``out_path`` and write a sidecar markdown caption.

    The sidecar lives at ``out_path.with_suffix('.md')`` — one markdown file
    per plot, so every figure travels with a human-readable description of
    what it shows, how to interpret it, and any key numbers measured in
    the run that produced it.
    """
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    caption_path = out_path.with_suffix(".md")
    caption_path.write_text(caption.lstrip() + "\n")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# Claims that have a meaningful numeric LHS vs RHS scatter.
_SCATTER_CLAIMS = [
    ("claim1_covariance_structure", "tr(Σ_z)  (Taylor bound, approx)"),
    ("claim2_spherical_variance", "1 − ‖R̄‖  (global bound)"),
    ("claim3_pole_concentration", "max ‖φ(x) − ŝ‖  (exact)"),
    ("claim4_pairwise_contraction", "max Lipschitz ratio  (exact)"),
    ("claim5_diameter", "diameter  (exact)"),
    ("claim6_expected_pairwise", "E[‖φ_i − φ_j‖]  (closed form)"),
]


_CLAIM_INTERPRETATION: dict[str, str] = {
    "claim1_covariance_structure": (
        "Points on or near the diagonal mean the Taylor approximation "
        "from Proposition 1 matches empirical `tr(Σ_z)` (where "
        "`Σ_z` is the covariance of unit-normalized pre-RMSNorm "
        "residuals). Points well below the diagonal mean the true "
        "spherical variance is *smaller* than the Taylor bound — bound "
        "holds but is loose."
    ),
    "claim2_spherical_variance": (
        "Global bound: `V = 1 − ‖R̄‖ ≤ 2 E[‖x‖] / ‖s_eff‖`. Points "
        "below the diagonal mean the bound holds. The gap between the "
        "points and the diagonal measures how tight the bound is; a "
        "large gap means the natural residual stream is far from the "
        "asymptotic 1/‖s‖ regime the bound describes."
    ),
    "claim3_pole_concentration": (
        "Exact global bound (Proposition 2): "
        "`max ‖φ_s(x) − ŝ‖ ≤ 2 max ‖x‖ / ‖s_eff‖`. Points below the "
        "diagonal are required — this is a hard bound, not an "
        "approximation. A point above the diagonal is a real bug."
    ),
    "claim4_pairwise_contraction": (
        "Exact pairwise Lipschitz bound (Proposition 3): "
        "`‖φ_s(x₁) − φ_s(x₂)‖ ≤ 2 ‖x₁ − x₂‖ / (‖s_eff‖ − R)`. Only "
        "defined when `‖s_eff‖ > R` (otherwise the denominator goes "
        "negative); scales where that fails are silently skipped from "
        "the plot."
    ),
    "claim5_diameter": (
        "Exact global bound (Corollary 3): "
        "`diam(φ_s(D)) ≤ 4 R / ‖s_eff‖`. Points below the diagonal are "
        "required. Computed from the reservoir."
    ),
    "claim6_expected_pairwise": (
        "Corollary 2: `E[‖φ(x₁) − φ(x₂)‖] ≤ 4 E[‖x‖] / ‖s_eff‖`. The "
        "LHS is estimated from the closed-form identity "
        "`E[‖u−v‖²] = 2(1 − ‖R̄‖²)` for iid unit vectors."
    ),
}


def _format_scale_table(xs: list[float], ys: list[float], scales: list[float]) -> str:
    rows = ["| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |",
            "|------:|--------------:|----------------:|--------------:|"]
    for s, lhs, rhs in zip(scales, ys, xs):
        ratio = lhs / rhs if rhs not in (0, None) else float("nan")
        rows.append(f"| {s:g} | {lhs:.4g} | {rhs:.4g} | {ratio:.4g} |")
    return "\n".join(rows)


def _scatter_one_claim(
    metrics: dict,
    claim_name: str,
    title: str,
    out_dir: Path,
    run_name: str,
) -> None:
    xs: list[float] = []  # RHS
    ys: list[float] = []  # LHS
    scales: list[float] = []
    for scale, entry in sorted(metrics.items(), key=lambda kv: (isinstance(kv[0], str), kv[0])):
        if not isinstance(scale, float) or not entry:
            continue
        r = entry.get(claim_name)
        if not r or r.get("passed") is None:
            continue
        lhs = r.get("lhs")
        rhs = r.get("rhs")
        if lhs is None or rhs is None or not np.isfinite(rhs):
            continue
        xs.append(rhs)
        ys.append(lhs)
        scales.append(scale)

    if not xs:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(xs, ys, c=scales, cmap="viridis", s=80, zorder=3)
    plt.colorbar(sc, ax=ax, label="scale")
    # Axis limits from the data (with 10x margin on each side) instead of
    # forcing the diagonal to span arbitrary ranges — keeps the points
    # visible and gives the diagonal a visible presence within the frame.
    lo = min(min(xs), min(ys)) / 10.0
    hi = max(max(xs), max(ys)) * 10.0
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="LHS = RHS (tight)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("theoretical RHS")
    ax.set_ylabel("empirical LHS")
    ax.set_title(f"{claim_name}\n{title}")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / f"{claim_name}_lhs_vs_rhs.png"

    interpretation = _CLAIM_INTERPRETATION.get(claim_name, "")
    caption = f"""
# {claim_name} — LHS vs RHS

**Run:** `{run_name}`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** {interpretation}

**Observed values in this run:**

{_format_scale_table(xs, ys, scales)}
"""
    _save_figure_with_caption(fig, out_path, caption)


def _plot_scaling_law(
    metrics: dict, stats: dict, out_dir: Path, run_name: str
) -> None:
    """Claim 8: V and tr(Σ) vs ‖s_eff‖ in log-log with slope −1/−2 reference."""
    per_scale = stats["per_scale"]
    zero_mean = per_scale[0.0]["mean"]

    xs_seff: list[float] = []
    v_list: list[float] = []
    trace_pre: list[float] = []
    scales: list[float] = []
    for scale, s in sorted(per_scale.items()):
        if scale == 0:
            continue
        seff = float((s["mean"] - zero_mean).norm().item())
        if seff <= 0:
            continue
        scales.append(float(scale))
        xs_seff.append(seff)
        v_list.append(1.0 - float(s["R_bar_norm"]) ** 2)
        trace_pre.append(float(s["trace_cov"]))

    if len(xs_seff) < 2:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(xs_seff, v_list, s=80, c="C0", label="empirical")
    xs_arr = np.array(xs_seff, dtype=float)
    c_neg1 = v_list[0] * xs_seff[0]
    ax1.plot(xs_arr, c_neg1 / xs_arr, "k--", alpha=0.4, label="slope −1 (theory)")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("‖s_eff‖")
    ax1.set_ylabel("1 − ‖R̄‖²  (≈ tr(Σ_z) on sphere)")
    ax1.set_title("Spherical variance scaling (Claim 8)")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    ax2.scatter(xs_seff, trace_pre, s=80, c="C1", label="empirical")
    c_neg2 = trace_pre[0] * (xs_seff[0] ** 2)
    ax2.plot(xs_arr, c_neg2 / (xs_arr ** 2), "k--", alpha=0.4, label="slope −2 (theory)")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("‖s_eff‖")
    ax2.set_ylabel("tr(Σ)  (pre-RMSNorm)")
    ax2.set_title("Pre-RMSNorm covariance scaling")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()

    # Fit the empirical slope numerically so the caption can report it.
    def _loglog_slope(x: list[float], y: list[float]) -> float:
        import math
        lx = [math.log(v) for v in x]
        ly = [math.log(v) for v in y]
        xm = sum(lx) / len(lx); ym = sum(ly) / len(ly)
        num = sum((a - xm) * (b - ym) for a, b in zip(lx, ly))
        den = sum((a - xm) ** 2 for a in lx)
        return num / den if den else 0.0

    slope_v = _loglog_slope(xs_seff, v_list)
    slope_trace = _loglog_slope(xs_seff, trace_pre)

    c8 = metrics.get("claim8_scaling_fit", {})
    c8_pass = c8.get("passed") if isinstance(c8, dict) else None
    pass_marker = (
        "✅ SLOPES MATCH THEORY" if c8_pass is True else
        ("❌ SLOPES DO NOT MATCH THEORY" if c8_pass is False else "(insufficient data)")
    )

    rows = ["| scale | ‖s_eff‖ | 1 − ‖R̄‖² | tr(Σ_x) |",
            "|------:|--------:|----------:|---------:|"]
    for sc, seff, v, t in zip(scales, xs_seff, v_list, trace_pre):
        rows.append(f"| {sc:g} | {seff:.3g} | {v:.4g} | {t:.4g} |")
    table = "\n".join(rows)

    caption = f"""
# Scaling law plot (Claim 8)

**Run:** `{run_name}`

Two log-log plots of empirical quantities against `‖s_eff‖`
(the effective steering at the final RMSNorm, computed as
`μ(pre_steered) − μ(pre_unsteered)`). Dashed reference lines show
the slopes predicted by the paper in the asymptotic regime.

- **Left** — spherical variance `1 − ‖R̄‖²` (which equals `tr(Σ_z)`
  for unit-normalized samples). Predicted slope: **−1**. Empirical
  slope: **{slope_v:.3f}**.
- **Right** — pre-RMSNorm `tr(Σ_x)`. Predicted slope: **−2**.
  Empirical slope: **{slope_trace:.3f}**.

**Interpretation.** The paper predicts that as ‖s_eff‖ grows, the
steering dominates the residual stream and the post-normalization
distribution concentrates near `ŝ`, so spherical variance shrinks
as `1/‖s‖`. Empirically, whether this holds depends on whether the
achievable ‖s_eff‖ is large compared to the natural residual stream
magnitude `E[‖x‖]`.

**Verdict for this run:** {pass_marker}

**Observed values:**

{table}
"""
    _save_figure_with_caption(fig, out_dir / "scaling_law.png", caption)


def _plot_sigma_spectrum(stats: dict, out_dir: Path, run_name: str) -> None:
    """Eigenvalue spectrum of Σ_x per scale (log y)."""
    per_scale = stats["per_scale"]

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    scales = sorted(per_scale.keys())
    top_eigs_per_scale: list[tuple[float, float, float, float]] = []  # (scale, λ1, λ_last_nonzero, rank)
    for i, scale in enumerate(scales):
        cov = per_scale[scale]["cov"].float()
        eigs = torch.linalg.eigvalsh(cov).sort(descending=True).values.numpy()
        xs = np.arange(1, len(eigs) + 1)
        ax.plot(
            xs,
            np.clip(eigs, 1e-12, None),
            color=cmap(i / max(len(scales) - 1, 1)),
            label=f"scale={scale}",
            alpha=0.85,
        )
        lam1 = float(eigs[0])
        rank_est = int((eigs > lam1 * 1e-6).sum())
        lam_last = float(eigs[rank_est - 1]) if rank_est > 0 else 0.0
        top_eigs_per_scale.append((scale, lam1, lam_last, rank_est))
    ax.set_yscale("log")
    ax.set_xlabel("eigenvalue index (descending)")
    ax.set_ylabel("eigenvalue")
    ax.set_title("Pre-RMSNorm Σ_x eigenvalue spectrum vs. scale")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    rows = ["| scale | λ₁ (max) | λ at est. rank | est. rank (eig > 1e-6 λ₁) |",
            "|------:|---------:|---------------:|--------------------------:|"]
    for s, lam1, lam_last, rank in top_eigs_per_scale:
        rows.append(f"| {s:g} | {lam1:.4g} | {lam_last:.4g} | {rank} |")
    table = "\n".join(rows)

    caption = f"""
# Pre-RMSNorm Σ_x eigenvalue spectrum

**Run:** `{run_name}`

Eigenvalues of the pre-RMSNorm residual covariance matrix `Σ_x`,
sorted descending, log y-axis. One curve per scale in the sweep,
colored by scale (viridis: dark = low scale, yellow = high scale).

**Interpretation.** Claim 1 (Proposition 1) predicts that under
steering the projection `P_⊥m̂` annihilates variance along `m̂`,
dropping the effective rank by one. Visually this shows up as the
smallest eigenvalue sagging lower at larger scales — if the theory
is in force here. In practice, large steering scales can also
*inject* variance (steering noise), causing the top eigenvalues to
grow instead of shrink.

**Top and estimated-rank eigenvalues:**

{table}
"""
    _save_figure_with_caption(fig, out_dir / "sigma_spectrum.png", caption)


def _plot_claim9_alignment(metrics: dict, out_dir: Path, run_name: str) -> None:
    scales: list[float] = []
    cos: list[float] = []
    for scale, entry in sorted(metrics.items(), key=lambda kv: (isinstance(kv[0], str), kv[0])):
        if not isinstance(scale, float) or not entry:
            continue
        r = entry.get("claim9_alignment")
        if r and r.get("lhs") is not None:
            scales.append(scale)
            cos.append(float(r["lhs"]))

    if not scales:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(range(len(scales)), cos, color="C2")
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.4, label="pass threshold (0.5)")
    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels([f"{s}" for s in scales])
    ax.set_xlabel("scale")
    ax.set_ylabel("|cos(smallest Σ_z eigenvec, m̂)|")
    ax.set_title("Claim 9: lost-direction alignment with m̂")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    rows = ["| scale | |cos(λ_min eigvec, m̂)| | passes (> 0.5)? |",
            "|------:|----------------------:|:----------------:|"]
    for s, c in zip(scales, cos):
        rows.append(f"| {s:g} | {c:.4g} | {'✅' if c > 0.5 else '❌'} |")
    table = "\n".join(rows)

    caption = f"""
# Claim 9 — lost-dimension alignment with m̂

**Run:** `{run_name}`

Absolute cosine between the smallest-eigenvalue eigenvector of the
post-normalization covariance `Σ_z` (estimated from the steered
reservoir, unit-normalized per sample) and the direction
`m̂ = μ_steered / ‖μ_steered‖`. The paper (synthesis.tex) predicts
this should approach 1 as ‖s‖ grows — the direction the normalization
projects out should be the steering pole.

**Interpretation.** A random-control run should stay near 0 at every
scale (random steering shouldn't align with any preferred direction).
A real steering vector should grow toward 1 as ‖s_eff‖ enters the
asymptotic regime. The 0.5 pass threshold is arbitrary — treat this
chart as a *trend* check, not a hard gate.

**Per-scale values:**

{table}
"""
    _save_figure_with_caption(fig, out_dir / "claim9_alignment.png", caption)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--stats", type=Path, default=None,
                        help="Path to stats.pt (default: same directory as --metrics).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Plot directory (default: <metrics-dir>/plots/).")
    args = parser.parse_args()

    if not args.metrics.exists():
        print(f"ERROR: metrics file not found: {args.metrics}", file=sys.stderr)
        return 2
    stats_path = args.stats if args.stats is not None else args.metrics.parent / "stats.pt"
    if not stats_path.exists():
        print(f"ERROR: stats.pt not found at {stats_path}", file=sys.stderr)
        return 2

    metrics_raw = json.loads(args.metrics.read_text())
    metrics: dict = {}
    for k, v in metrics_raw.items():
        try:
            metrics[float(k)] = v
        except (TypeError, ValueError):
            metrics[k] = v

    stats = torch.load(stats_path, weights_only=False, map_location="cpu")

    # Pull the run_name from stats_meta.json (for captions) if available,
    # else fall back to the parent directory name.
    meta_path = args.metrics.parent / "stats_meta.json"
    run_name: str = args.metrics.parent.name
    if meta_path.exists():
        try:
            run_name = json.loads(meta_path.read_text()).get("run_name", run_name)
        except Exception:
            pass

    out_dir = args.output if args.output is not None else args.metrics.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[viz] writing plots to {out_dir}")
    for claim_name, title in _SCATTER_CLAIMS:
        _scatter_one_claim(metrics, claim_name, title, out_dir, run_name)
    _plot_scaling_law(metrics, stats, out_dir, run_name)
    _plot_sigma_spectrum(stats, out_dir, run_name)
    _plot_claim9_alignment(metrics, out_dir, run_name)

    n_plots = len(list(out_dir.glob("*.png")))
    n_captions = len(list(out_dir.glob("*.md")))
    print(f"[viz] wrote {n_plots} plots and {n_captions} caption files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
