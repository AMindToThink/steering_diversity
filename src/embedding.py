"""Sentence-BERT embedding of responses (CPU-compatible)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from .config import EmbeddingConfig


def embed_responses(
    texts: list[str],
    emb_cfg: EmbeddingConfig | None = None,
) -> np.ndarray:
    """Embed a list of texts using Sentence-BERT.

    Returns an (N, D) float32 array of embeddings.
    """
    if emb_cfg is None:
        emb_cfg = EmbeddingConfig()

    model = SentenceTransformer(emb_cfg.model_name)
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=emb_cfg.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def save_embeddings(
    embeddings: np.ndarray,
    metadata: dict[str, Any],
    path: str | Path,
) -> None:
    """Save embeddings and metadata to an .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, embeddings=embeddings, **metadata)


def load_embeddings(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load embeddings from an .npz file.

    Returns (embeddings array, metadata dict).
    """
    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"]
    metadata = {k: data[k] for k in data.files if k != "embeddings"}
    return embeddings, metadata
