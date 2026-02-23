#!/usr/bin/env python3
"""Step 3: Embed responses with Sentence-BERT (CPU-compatible)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ExperimentConfig
from src.embedding import embed_responses, save_embeddings
from src.utils import ensure_dir, load_jsonl, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed responses with Sentence-BERT")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--responses",
        type=str,
        default=None,
        help="Path to responses JSONL (default: outputs/<run>/responses.jsonl)",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    seed_everything(cfg.seed)

    out_dir = ensure_dir(cfg.output_dir)
    responses_path = args.responses or str(out_dir / "responses.jsonl")

    records = load_jsonl(responses_path)
    texts = [r["response"] for r in records]
    scales = np.array([r["scale"] for r in records], dtype=np.float32)
    prompt_indices = np.array([r["prompt_idx"] for r in records], dtype=np.int32)

    print(f"Embedding {len(texts)} responses …")
    embeddings = embed_responses(texts, cfg.embedding)

    emb_path = out_dir / "embeddings.npz"
    save_embeddings(embeddings, {"scales": scales, "prompt_indices": prompt_indices}, emb_path)
    print(f"Saved embeddings ({embeddings.shape}) to {emb_path}")


if __name__ == "__main__":
    main()
