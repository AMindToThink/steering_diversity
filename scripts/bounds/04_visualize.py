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


def _scatter_one_claim(
    metrics: dict, claim_name: str, title: str, out_dir: Path
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
    cb = plt.colorbar(sc, ax=ax, label="scale")
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
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_scaling_law(
    metrics: dict, stats: dict, out_dir: Path
) -> None:
    """Claim 8: V and tr(Σ) vs ‖s_eff‖ in log-log with slope −1/−2 reference."""
    per_scale = stats["per_scale"]
    zero_mean = per_scale[0.0]["mean"]

    xs_seff: list[float] = []
    v_list: list[float] = []
    trace_pre: list[float] = []
    for scale, s in sorted(per_scale.items()):
        if scale == 0:
            continue
        seff = float((s["mean"] - zero_mean).norm().item())
        if seff <= 0:
            continue
        xs_seff.append(seff)
        v_list.append(1.0 - float(s["R_bar_norm"]) ** 2)  # = tr(Σ_z) on the sphere
        trace_pre.append(float(s["trace_cov"]))  # tr(Σ_x) pre-RMSNorm

    if len(xs_seff) < 2:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Spherical variance: expected slope -1
    ax1.scatter(xs_seff, v_list, s=80, c="C0", label="empirical")
    xs_arr = np.array(xs_seff, dtype=float)
    c_neg1 = v_list[0] * xs_seff[0]  # force pass through first point
    ax1.plot(xs_arr, c_neg1 / xs_arr, "k--", alpha=0.4, label="slope −1 (theory)")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("‖s_eff‖")
    ax1.set_ylabel("1 − ‖R̄‖²  (≈ tr(Σ_z) on sphere)")
    ax1.set_title("Spherical variance scaling (Claim 8)")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    # tr(Σ_x) pre-RMSNorm: not a direct bound target, but informative
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
    fig.savefig(out_dir / "scaling_law.png", dpi=150)
    plt.close(fig)


def _plot_sigma_spectrum(stats: dict, out_dir: Path) -> None:
    """Eigenvalue spectrum of Σ_x per scale (log y)."""
    per_scale = stats["per_scale"]

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    scales = sorted(per_scale.keys())
    for i, scale in enumerate(scales):
        cov = per_scale[scale]["cov"].float()
        # Symmetric eigendecomp. Use eigvalsh for stability.
        eigs = torch.linalg.eigvalsh(cov).sort(descending=True).values
        xs = np.arange(1, len(eigs) + 1)
        ax.plot(
            xs,
            np.clip(eigs.numpy(), 1e-12, None),
            color=cmap(i / max(len(scales) - 1, 1)),
            label=f"scale={scale}",
            alpha=0.85,
        )
    ax.set_yscale("log")
    ax.set_xlabel("eigenvalue index (descending)")
    ax.set_ylabel("eigenvalue")
    ax.set_title("Pre-RMSNorm Σ_x eigenvalue spectrum vs. scale")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "sigma_spectrum.png", dpi=150)
    plt.close(fig)


def _plot_claim9_alignment(metrics: dict, out_dir: Path) -> None:
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
    fig.savefig(out_dir / "claim9_alignment.png", dpi=150)
    plt.close(fig)


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

    out_dir = args.output if args.output is not None else args.metrics.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[viz] writing plots to {out_dir}")
    for claim_name, title in _SCATTER_CLAIMS:
        _scatter_one_claim(metrics, claim_name, title, out_dir)
    _plot_scaling_law(metrics, stats, out_dir)
    _plot_sigma_spectrum(stats, out_dir)
    _plot_claim9_alignment(metrics, out_dir)

    n_plots = len(list(out_dir.glob("*.png")))
    print(f"[viz] wrote {n_plots} plots")
    return 0


if __name__ == "__main__":
    sys.exit(main())
