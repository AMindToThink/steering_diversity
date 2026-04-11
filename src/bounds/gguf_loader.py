"""Load EasySteer `.gguf` steering-vector files into `{layer: torch.Tensor}`.

EasySteer writes one tensor per layer with the name `direction.{layer_idx}`.
All three project vectors (happy, style, creativity) follow this convention
and use GGML_TYPE_F32 storage.

This loader does not touch EasySteer or vLLM at runtime — it reads the raw
GGUF bytes directly so the bounds pipeline can run under pure HuggingFace
Transformers + nnsight.
"""

from __future__ import annotations

import re
from pathlib import Path

import gguf
import numpy as np
import torch

_DIRECTION_NAME_RE = re.compile(r"^direction\.(\d+)$")


def load_steering_vector_gguf(path: str | Path) -> dict[int, torch.Tensor]:
    """Load an EasySteer control-vector `.gguf` file.

    Returns
    -------
    dict[int, torch.Tensor]
        Map from layer index to a 1-D float32 tensor on CPU. The caller is
        responsible for moving tensors to the device/dtype the model needs.

    Raises
    ------
    ValueError
        If any tensor name does not match `direction.{int}` — we refuse to
        silently ignore unknown tensors (project's loud-errors policy).
    FileNotFoundError
        If `path` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GGUF file not found: {path}")

    reader = gguf.GGUFReader(str(path))

    vectors: dict[int, torch.Tensor] = {}
    for tensor in reader.tensors:
        match = _DIRECTION_NAME_RE.match(tensor.name)
        if match is None:
            raise ValueError(
                f"Unexpected tensor name in {path}: {tensor.name!r}. "
                f"Expected 'direction.{{layer_idx}}' (EasySteer convention)."
            )
        layer_idx = int(match.group(1))

        # GGUFReader returns a numpy view over the mmapped file. Copy to a
        # fresh torch tensor so the caller can outlive the reader object,
        # and so downstream .to(device) calls work without surprising views.
        data = np.asarray(tensor.data, dtype=np.float32).copy()
        vectors[layer_idx] = torch.from_numpy(data)

    if not vectors:
        raise ValueError(f"{path} contains no direction.* tensors")

    return vectors
