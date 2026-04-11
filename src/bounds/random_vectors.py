"""Random steering-vector generator — norm-matched control for real vectors.

For each layer in the reference dict, draws an iid Gaussian vector and
rescales it to match the reference's L2 norm. Deterministic in `seed`;
different layers use derived seeds so they remain independent.
"""

from __future__ import annotations

import torch


def generate_random_steering_vector(
    reference: dict[int, torch.Tensor],
    seed: int,
) -> dict[int, torch.Tensor]:
    """Generate a random steering vector matched to the reference.

    Parameters
    ----------
    reference : dict[int, torch.Tensor]
        Mapping from layer index to per-layer steering tensor. Each tensor
        must be 1-D. Only the shape and norm are read; the direction is
        replaced.
    seed : int
        Base seed for deterministic sampling. Different layers derive
        distinct sub-seeds from this base so no two layers share a draw.

    Returns
    -------
    dict[int, torch.Tensor]
        Same keys and shapes as `reference`; each output tensor has the
        same L2 norm as its corresponding reference (or is zero if the
        reference was zero). Dtype is float32, device is CPU.
    """
    out: dict[int, torch.Tensor] = {}
    for layer, ref in reference.items():
        if ref.ndim != 1:
            raise ValueError(
                f"reference[{layer}] must be 1-D, got shape {tuple(ref.shape)}"
            )
        target_norm = ref.to(torch.float32).norm().item()
        if target_norm == 0.0:
            out[layer] = torch.zeros_like(ref, dtype=torch.float32)
            continue

        gen = torch.Generator().manual_seed(seed + layer * 1_000_003)
        v = torch.randn(ref.shape, generator=gen, dtype=torch.float32)
        v = v * (target_norm / v.norm())
        out[layer] = v

    return out
