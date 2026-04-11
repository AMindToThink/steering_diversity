"""Bounds-verification experiments — parallel pipeline to scripts/01-05.

Tests the mathematical claims in docs/paper_sections/ about what steering does
to the distribution of residual-stream activations at an RMSNorm site.
"""

from __future__ import annotations

from src.bounds.config import (
    BoundsDatasetConfig,
    BoundsExperimentConfig,
    BoundsSteeringConfig,
    SteeringVerificationConfig,
)

__all__ = [
    "BoundsDatasetConfig",
    "BoundsExperimentConfig",
    "BoundsSteeringConfig",
    "SteeringVerificationConfig",
]
