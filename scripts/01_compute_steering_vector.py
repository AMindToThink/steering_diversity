#!/usr/bin/env python3
"""Step 1: Compute a DiffMean steering vector via EasySteer (GPU required)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ExperimentConfig
from src.generation import compute_steering_vector
from src.utils import ensure_dir, save_provenance, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute DiffMean steering vector")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for .gguf vector (default: outputs/<run>/<concept>_diffmean.gguf)",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    seed_everything(cfg.seed)

    out_dir = ensure_dir(cfg.output_dir)
    gguf_path = args.output or str(out_dir / f"{cfg.steering.concept}_diffmean.gguf")

    print(f"Computing steering vector for concept={cfg.steering.concept!r}")
    vector = compute_steering_vector(cfg)

    vector.export_gguf(str(gguf_path))
    print(f"Saved steering vector to {gguf_path}")

    save_provenance(
        step="01_compute_steering_vector",
        config_path=args.config,
        cfg=cfg,
        inputs={"contrastive_pairs": cfg.steering.contrastive_pairs_path},
        outputs=[gguf_path],
    )


if __name__ == "__main__":
    main()
