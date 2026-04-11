"""Bounds post-processing — pure, no model, no GPU.

Loads ``outputs/bounds/<run_name>/stats.pt`` and ``stats_meta.json`` from
``scripts/bounds/02_record_stats.py``, assembles per-scale ``run`` dicts
for ``src/bounds/claims.compute_all_claims``, and writes
``outputs/bounds/<run_name>/bounds_metrics.json``.

At scale == 0 the run dict has ``unsteered == steered``; for each scale > 0
the unsteered stats come from the scale=0 reference (shared across scales).
This matches the claim-function input contract in ``src/bounds/claims.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.bounds.claims import compute_all_claims  # noqa: E402
from src.bounds.config import BoundsExperimentConfig  # noqa: E402
from src.utils import save_provenance  # noqa: E402


def _build_scale_runs(raw: dict) -> dict[float, dict]:
    """Turn the recorded per-scale stats into {scale: run_dict} as expected by
    ``compute_all_claims``.

    Each output ``run`` has ``unsteered`` (= scale=0 stats), ``steered``
    (= that scale's stats), ``spherical_steered``, ``s_raw``, and ``d``.
    """
    per_scale = raw["per_scale"]
    # The scale sweep always includes scale=0 (the unsteered reference).
    zero = None
    for scale in per_scale:
        if abs(scale) < 1e-12:
            zero = per_scale[scale]
            break
    if zero is None:
        raise ValueError("stats.pt has no scale=0 entry; cannot compute unsteered baseline")

    d = int(zero["cov"].shape[0])

    def _pack(stats: dict) -> dict:
        return {
            "mean": stats["mean"],
            "cov": stats["cov"],
            "trace_cov": stats["trace_cov"],
            "E_norm": stats["E_norm"],
            "E_sq_norm": stats["E_sq_norm"],
            "R": stats["R"],
            "reservoir": stats["reservoir"],
        }

    def _pack_sphere(stats: dict) -> dict:
        return {
            "R_bar": stats["R_bar"],
            "R_bar_norm": stats["R_bar_norm"],
            "spherical_variance": stats["spherical_variance"],
            "expected_pair_sq_chord": stats["expected_pair_sq_chord"],
            "max_chord_to_pole": stats["max_chord_to_pole_streamed"],
        }

    scale_runs: dict[float, dict] = {}
    unsteered = _pack(zero)
    for scale, stats in per_scale.items():
        scale_runs[float(scale)] = {
            "d": d,
            "unsteered": unsteered,
            "steered": _pack(stats),
            "spherical_steered": _pack_sphere(stats),
            "s_raw": None,  # not needed by current claim formulas
        }
    return scale_runs


def _serializable(obj: object) -> object:
    """Recursively convert tensors / dataclasses to JSON-friendly types."""
    if isinstance(obj, dict):
        return {str(k): _serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serializable(x) for x in obj]
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return float(obj.item())
        if obj.numel() <= 32:
            return obj.detach().cpu().tolist()
        return {"__tensor_shape__": list(obj.shape), "__tensor_dtype__": str(obj.dtype)}
    if isinstance(obj, (float, int, bool, str)) or obj is None:
        return obj
    return str(obj)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None,
                        help="Config used to produce --stats (for provenance).")
    args = parser.parse_args()

    if not args.stats.exists():
        print(f"ERROR: stats file not found: {args.stats}", file=sys.stderr)
        return 2

    raw = torch.load(args.stats, weights_only=False, map_location="cpu")
    scale_runs = _build_scale_runs(raw)

    print(f"[compute] loaded {len(scale_runs)} scales from {args.stats}")

    metrics = compute_all_claims(scale_runs)

    output_dir = args.output if args.output is not None else args.stats.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "bounds_metrics.json"
    metrics_path.write_text(json.dumps(_serializable(metrics), indent=2))
    print(f"[compute] wrote {metrics_path}")

    # Print a compact one-line-per-scale summary so the user immediately sees
    # which bounds passed / failed and can spot regressions.
    print()
    print(" scale     claim1_ratio  claim2_ratio  claim3_pass  claim5_pass  "
          "claim7_pass  claim9_cos")
    print(" " + "-" * 80)
    for scale, entry in metrics.items():
        if not isinstance(scale, float) or not entry:
            continue
        def _g(name: str, key: str) -> str:
            r = entry.get(name, {})
            v = r.get(key)
            if v is None:
                return " —     "
            if isinstance(v, bool):
                return "  ✅   " if v else "  ❌   "
            return f"{v:6.3g} "
        print(
            f" {scale:<6g}"
            f"{_g('claim1_covariance_structure', 'ratio'):>14s}"
            f"{_g('claim2_spherical_variance', 'ratio'):>14s}"
            f"{_g('claim3_pole_concentration', 'passed'):>13s}"
            f"{_g('claim5_diameter', 'passed'):>13s}"
            f"{_g('claim7_reduction_condition', 'passed'):>13s}"
            f"{_g('claim9_alignment', 'lhs'):>12s}"
        )
    c8 = metrics.get("claim8_scaling_fit", {})
    if c8:
        slopes = c8.get("detail", {})
        print()
        print(
            f" claim8 scaling: slope_V={slopes.get('slope_spherical_variance', 0):.3f} "
            f"(exp -1) slope_trZ={slopes.get('slope_trace_cov_z', 0):.3f} (exp -2)  "
            f"{'✅' if c8.get('passed') else '❌'}"
        )

    if args.config is not None and args.config.exists():
        cfg = BoundsExperimentConfig.from_yaml(args.config)
        save_provenance(
            step="03_compute_bounds",
            config_path=str(args.config),
            cfg=cfg,
            inputs={"stats": str(args.stats)},
            outputs=[str(metrics_path)],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
