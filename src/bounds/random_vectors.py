"""Random steering-vector generators — two controls for the real vectors.

1. ``generate_random_steering_vector`` (per-layer matched): each layer's
   random vector has the same L2 norm as the reference vector at that
   layer. This is the natural "per-layer same magnitude" control — if
   the real per-layer vectors are unit-norm, so are the random ones.
   Because random directions in high-d space are near-orthogonal, the
   aggregate ``‖Σ r_i‖ ≈ √n`` while the real aggregate is ``~ n`` when
   the per-layer directions are trained to be correlated. So this
   control matches per-layer magnitude but DOES NOT match aggregate
   steering magnitude.

2. ``generate_random_steering_vector_aggregate_matched`` (aggregate
   matched): picks a per-layer norm so that ``‖Σ r_i‖`` matches
   ``‖Σ ref_i‖`` over a given set of target layers. Each per-layer
   random vector is rescaled uniformly by ``target_aggregate / √n``
   (approximately — we use the actual observed random aggregate for
   exactness). This control matches AGGREGATE steering magnitude but
   has larger per-layer norms (~√n×) than the reference.

Which control is "right" depends on the question:
- Q: "at matched per-layer magnitude, does direction matter?" → (1)
- Q: "at matched aggregate steering, does direction matter?" → (2)

Both are kept and the bounds pipeline picks one via config.
"""

from __future__ import annotations

import torch


def generate_random_steering_vector(
    reference: dict[int, torch.Tensor],
    seed: int,
) -> dict[int, torch.Tensor]:
    """Generate a random steering vector matched per-layer to the reference.

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
        Same keys and shapes as ``reference``; each output tensor has the
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


def generate_random_steering_vector_aggregate_matched(
    reference: dict[int, torch.Tensor],
    target_layers: list[int],
    seed: int,
) -> dict[int, torch.Tensor]:
    """Generate a random steering vector matched to the reference's aggregate.

    Computes ``target_aggregate = ‖Σ reference[i] for i in target_layers‖``,
    then draws iid Gaussian unit vectors for each layer in the reference
    dict (not just target_layers), and rescales every per-layer vector by
    the same factor so ``‖Σ out[i] for i in target_layers‖ == target_aggregate``.

    Per-layer random norms end up at approximately ``target_aggregate / √n``
    where ``n = len(target_layers)``, significantly larger than the
    reference's per-layer norm (which is 1 for normalized control vectors).
    This is the tradeoff: we match the aggregate scale at the cost of
    larger per-layer magnitudes.

    Parameters
    ----------
    reference : dict[int, torch.Tensor]
        Reference vectors, one per layer. All layers in ``target_layers``
        must appear as keys.
    target_layers : list[int]
        Which layers to use when computing the aggregate target norm.
        Must all be keys of ``reference``.
    seed : int
        Base seed for deterministic sampling.

    Returns
    -------
    dict[int, torch.Tensor]
        Same keys and shapes as ``reference``; on ``target_layers`` the
        sum has L2 norm equal to ``‖Σ reference[i] for i in target_layers‖``.
        Dtype is float32, device is CPU.
    """
    if not target_layers:
        raise ValueError("target_layers must be non-empty")
    for i in target_layers:
        if i not in reference:
            raise KeyError(
                f"target_layers[{i}] not found in reference (keys={sorted(reference.keys())})"
            )

    # Compute the target aggregate norm from the reference.
    ref_sum = torch.zeros_like(
        reference[target_layers[0]], dtype=torch.float32
    )
    for i in target_layers:
        ref_sum = ref_sum + reference[i].to(torch.float32)
    target_aggregate = float(ref_sum.norm().item())

    if target_aggregate == 0.0:
        return {
            layer: torch.zeros_like(ref, dtype=torch.float32)
            for layer, ref in reference.items()
        }

    # Draw iid Gaussian unit vectors for every layer in the reference.
    # We use unit norms because any constant scale would factor out
    # uniformly; the final scale is set by the aggregate-match step.
    raw: dict[int, torch.Tensor] = {}
    for layer, ref in reference.items():
        if ref.ndim != 1:
            raise ValueError(
                f"reference[{layer}] must be 1-D, got shape {tuple(ref.shape)}"
            )
        gen = torch.Generator().manual_seed(seed + layer * 1_000_003)
        v = torch.randn(ref.shape, generator=gen, dtype=torch.float32)
        raw[layer] = v / v.norm().clamp_min(1e-12)

    # Compute the random aggregate (over target_layers only) and rescale.
    random_sum = torch.zeros_like(raw[target_layers[0]], dtype=torch.float32)
    for i in target_layers:
        random_sum = random_sum + raw[i]
    random_aggregate = float(random_sum.norm().item())
    if random_aggregate == 0.0:
        # Degenerate case: all random unit vectors cancelled out. Retry
        # would typically suffice but we'll raise loudly per project policy.
        raise RuntimeError(
            "Random vector aggregate is zero — extraordinarily unlikely for "
            "Gaussian draws, likely a bug."
        )

    scale_factor = target_aggregate / random_aggregate
    return {layer: v * scale_factor for layer, v in raw.items()}
