"""Streaming bounds-stats recording — the hot path.

For each scale in ``cfg.steering.scale_sweep``, runs a non-autoregressive
forward pass over every prompt in the dataset (in batches), captures
pre-RMSNorm residuals at the ``final`` site, masks out padding tokens,
and folds the remaining ``[N_tokens, d]`` activations into streaming
Chan-Welford accumulators + a reservoir. Finalizes at the end of each
scale and writes everything to ``outputs/bounds/<run_name>/stats.pt``
plus a JSON sidecar.

The pipeline is strict about the "trust no silent steering" policy:
this script refuses to run unless the verification sentinel exists at
``outputs/bounds/<run_name>/verification/PASSED``. Override with
``--skip-verification-gate`` only for debugging, and only with a loud
on-screen warning.

No raw activations are ever written to disk. Storage per run is
dominated by two d×d covariance matrices plus a small reservoir per
scale. See plan ``docs/upstream_bug_reports/... (wait, no — see plan in
/home/cs29824/.claude/plans/memoized-dazzling-phoenix.md)`` for the
footprint math.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from nnsight import LanguageModel

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.bounds.activation_streams import (  # noqa: E402
    FullMoments,
    Reservoir,
    SphericalMoments,
)
from src.bounds.config import BoundsExperimentConfig  # noqa: E402
from src.bounds.dataset import load_fineweb_prompts  # noqa: E402
from src.bounds.gguf_loader import load_steering_vector_gguf  # noqa: E402
from src.bounds.nnsight_runner import run_bounds_forward_pass  # noqa: E402
from src.bounds.random_vectors import generate_random_steering_vector  # noqa: E402
from src.utils import save_provenance, seed_everything  # noqa: E402


SYNTHETIC_SENTINEL = "__synthetic__"


def _load_or_build_steering(
    cfg: BoundsExperimentConfig, lm: LanguageModel
) -> dict[int, torch.Tensor]:
    """Same resolver as ``01_verify_steering.py``; kept in sync manually."""
    if cfg.steering.vector_path == SYNTHETIC_SENTINEL:
        d = getattr(lm.config, "hidden_size", None) or lm.config.n_embd
        n_layers = getattr(lm.config, "num_hidden_layers", None) or lm.config.n_layer
        return {
            i: (torch.arange(d, dtype=torch.float32) + float(i) * 10.0)
            * (1.0 / max(d, 1))
            for i in range(n_layers)
        }
    if cfg.steering.vector_path:
        return load_steering_vector_gguf(cfg.steering.vector_path)
    if cfg.steering.random_reference_path:
        ref = load_steering_vector_gguf(cfg.steering.random_reference_path)
        return generate_random_steering_vector(ref, seed=cfg.steering.random_seed or 0)
    raise ValueError("BoundsSteeringConfig has no vector source")


def _check_verification_gate(cfg: BoundsExperimentConfig, skip: bool) -> None:
    """Refuse to run unless the verification sentinel says PASSED."""
    sentinel = cfg.output_dir / "verification" / "PASSED"
    if sentinel.exists():
        return
    if skip:
        print(
            "⚠  WARNING: --skip-verification-gate set. Running bounds recording "
            "without a PASSED verification sentinel. This is only safe for "
            "debugging; real runs should verify steering first with "
            "scripts/bounds/01_verify_steering.py.",
            file=sys.stderr,
        )
        return
    failed = cfg.output_dir / "verification" / "FAILED"
    msg = (
        f"Verification sentinel missing at {sentinel}.\n"
        f"{'FAILED sentinel exists at ' + str(failed) if failed.exists() else ''}\n"
        "Run: uv run python scripts/bounds/01_verify_steering.py --config "
        f"{cfg.run_name} --auto-escalate\n"
        "Or pass --skip-verification-gate if you really know what you're doing."
    )
    raise RuntimeError(msg)


def _extract_unpadded(pre: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Return only the real (non-padded) token activations, flattened.

    ``pre`` is ``[B, T, d]`` and ``attention_mask`` is ``[B, T]``. Returns
    ``[N_real_tokens, d]``. Works for both left- and right-padding
    conventions since we just gather where mask == 1.
    """
    mask = attention_mask.to(torch.bool)
    return pre[mask]  # torch advanced indexing yields [N_real, d]


def _run_one_scale(
    lm: LanguageModel,
    prompts: list[str],
    steering: dict[int, torch.Tensor] | None,
    scale: float,
    cfg: BoundsExperimentConfig,
) -> dict:
    """Stream all prompts through one scale and return finalized stats."""
    d = getattr(lm.config, "hidden_size", None) or lm.config.n_embd
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fm = FullMoments(d=d, device=device, dtype=torch.float32)
    sm = SphericalMoments(d=d, device=device, dtype=torch.float32)
    rv = Reservoir(K=cfg.reservoir_size, d=d, seed=cfg.seed + int(scale * 1000))

    n_tokens_seen = 0
    n_batches = 0
    batch_size = cfg.dataset.batch_size
    target_layers = cfg.steering.target_layers

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        out = run_bounds_forward_pass(
            lm,
            batch,
            steering=steering if scale > 0 else None,
            scale=scale,
            target_layers=target_layers if scale > 0 else [],
            capture_specs=[{"site": "final", "tier": "full"}],
            max_seq_len=cfg.dataset.max_seq_len,
        )
        pre = out["final"]["pre"]  # [B, T, d]
        mask = out["attention_mask"]  # [B, T]
        flat = _extract_unpadded(pre, mask)  # [N_real, d]

        fm.update(flat)
        sm.update(flat)
        rv.update(flat)

        n_tokens_seen += flat.shape[0]
        n_batches += 1

    fm_stats = fm.finalize()
    sm_stats = sm.finalize()
    reservoir = rv.finalize()  # [K, d] CPU float32

    return {
        "count": fm_stats["count"],
        "mean": fm_stats["mean"],
        "cov": fm_stats["cov"],
        "trace_cov": fm_stats["trace_cov"],
        "E_norm": fm_stats["E_norm"],
        "E_sq_norm": fm_stats["E_sq_norm"],
        "R": fm_stats["R"],
        "R_bar": sm_stats["R_bar"],
        "R_bar_norm": sm_stats["R_bar_norm"],
        "spherical_variance": sm_stats["spherical_variance"],
        "expected_pair_sq_chord": sm_stats["expected_pair_sq_chord"],
        "max_chord_to_pole_streamed": sm_stats["max_chord_to_pole"],  # usually 0 (no pole)
        "reservoir": reservoir,
        "meta": {
            "n_tokens_seen": n_tokens_seen,
            "n_batches": n_batches,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--num-prompts", type=int, default=None,
        help="Override cfg.dataset.num_prompts (useful for pilot runs).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load everything and exit before the first forward pass.",
    )
    parser.add_argument(
        "--skip-verification-gate", action="store_true",
        help="DEBUG ONLY: run even without a PASSED verification sentinel.",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        return 2

    cfg = BoundsExperimentConfig.from_yaml(args.config)
    if args.num_prompts is not None:
        cfg.dataset.num_prompts = args.num_prompts
    output_dir = args.output if args.output is not None else cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _check_verification_gate(cfg, skip=args.skip_verification_gate)

    seed_everything(cfg.seed)

    print(f"[record_stats] Loading model {cfg.model.name!r} ...")
    t0 = time.time()
    lm = LanguageModel(cfg.model.name, device_map="auto")
    print(f"[record_stats]   loaded in {time.time() - t0:.1f}s")

    print(f"[record_stats] Loading steering ...")
    steering = _load_or_build_steering(cfg, lm)

    print(f"[record_stats] Loading {cfg.dataset.num_prompts} prompts from {cfg.dataset.name} ...")
    t0 = time.time()
    prompts = load_fineweb_prompts(
        num_prompts=cfg.dataset.num_prompts,
        seed=cfg.seed,
        dataset_name=cfg.dataset.name,
    )
    print(f"[record_stats]   loaded in {time.time() - t0:.1f}s")

    if args.dry_run:
        print("[record_stats] --dry-run: exiting before any forward pass.")
        return 0

    per_scale: dict[float, dict] = {}
    for scale in cfg.steering.scale_sweep:
        print(f"[record_stats] scale={scale} …")
        t0 = time.time()
        stats = _run_one_scale(lm, prompts, steering, scale, cfg)
        print(
            f"[record_stats]   scale={scale} "
            f"n_tokens={stats['count']:>8d} "
            f"n_batches={stats['meta']['n_batches']:>4d} "
            f"trace_cov={stats['trace_cov']:.4g} "
            f"R_bar_norm={stats['R_bar_norm']:.4g} "
            f"({time.time() - t0:.1f}s)"
        )
        per_scale[scale] = stats

    # Per-scale accumulated-steering direction at every target layer,
    # saved so compute.py can reconstruct m̂ without reloading the model.
    s_by_layer_per_scale: dict[float, dict[int, torch.Tensor]] = {}
    for scale in cfg.steering.scale_sweep:
        s_by_layer_per_scale[scale] = {
            int(i): (scale * steering[i]).detach().cpu().float()
            for i in cfg.steering.target_layers
        }

    stats_path = output_dir / "stats.pt"
    meta_path = output_dir / "stats_meta.json"
    torch.save(
        {
            "per_scale": per_scale,
            "s_by_layer_per_scale": s_by_layer_per_scale,
            "target_layers": list(cfg.steering.target_layers),
        },
        stats_path,
    )
    meta_path.write_text(
        json.dumps(
            {
                "run_name": cfg.run_name,
                "model": cfg.model.name,
                "steering": {
                    "vector_path": cfg.steering.vector_path,
                    "random_reference_path": cfg.steering.random_reference_path,
                    "random_seed": cfg.steering.random_seed,
                    "target_layers": cfg.steering.target_layers,
                    "scale_sweep": cfg.steering.scale_sweep,
                },
                "dataset": {
                    "name": cfg.dataset.name,
                    "num_prompts": cfg.dataset.num_prompts,
                    "max_seq_len": cfg.dataset.max_seq_len,
                    "batch_size": cfg.dataset.batch_size,
                },
                "reservoir_size": cfg.reservoir_size,
                "seed": cfg.seed,
                "per_scale_scalar_summary": {
                    str(scale): {
                        "count": s["count"],
                        "trace_cov": s["trace_cov"],
                        "R_bar_norm": s["R_bar_norm"],
                        "spherical_variance": s["spherical_variance"],
                        "E_norm": s["E_norm"],
                        "R": s["R"],
                    }
                    for scale, s in per_scale.items()
                },
            },
            indent=2,
        )
    )
    print(f"[record_stats] wrote {stats_path}")
    print(f"[record_stats] wrote {meta_path}")

    save_provenance(
        step="02_record_bounds_stats",
        config_path=str(args.config),
        cfg=cfg,
        inputs={
            "vector_path": str(cfg.steering.vector_path or ""),
            "random_reference_path": str(cfg.steering.random_reference_path or ""),
        },
        outputs=[str(stats_path), str(meta_path)],
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
