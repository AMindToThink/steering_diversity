#!/usr/bin/env python3
"""Step 2: Generate steered responses via EasySteer + vLLM (GPU required)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ExperimentConfig
from src.generation import generate_steered_responses
from src.utils import ensure_dir, save_jsonl, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate steered responses")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--vector",
        type=str,
        default=None,
        help="Path to .gguf steering vector (default: outputs/<run>/<concept>_diffmean.gguf)",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    seed_everything(cfg.seed)

    out_dir = ensure_dir(cfg.output_dir)

    vector_path = args.vector or str(out_dir / f"{cfg.steering.concept}_diffmean.gguf")
    if not Path(vector_path).exists():
        print(f"ERROR: steering vector not found at {vector_path}")
        print("Run 01_compute_steering_vector.py first.")
        sys.exit(1)

    print(f"Generating responses with {len(cfg.steering.scales)} scales …")
    records = generate_steered_responses(cfg, vector_path)

    responses_path = out_dir / "responses.jsonl"
    save_jsonl(records, responses_path)
    print(f"Saved {len(records)} responses to {responses_path}")


if __name__ == "__main__":
    main()
