"""Config dataclasses for the bounds-verification pipeline.

Reuses `ModelConfig` from `src/config.py` by composition, adds bounds-specific
steering/dataset/verification blocks. `from_yaml` mirrors the pattern at
`src/config.py:73-86`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

from src.config import ModelConfig


@dataclasses.dataclass
class BoundsSteeringConfig:
    """Steering-vector source + scale sweep.

    Exactly one of `vector_path` and `random_reference_path` must be set:
    - `vector_path`: load a real steering vector from a `.gguf` file.
    - `random_reference_path` + `random_seed`: load a reference vector only
      to inherit its target layers and per-layer norms, then draw a norm-
      matched random vector deterministic in `random_seed`.
    """

    target_layers: list[int]
    scale_sweep: list[float]
    vector_path: str | None = None
    random_reference_path: str | None = None
    random_seed: int | None = None


@dataclasses.dataclass
class BoundsDatasetConfig:
    name: str = "HuggingFaceFW/fineweb-edu"
    num_prompts: int = 1000
    max_seq_len: int = 256
    batch_size: int = 8


def _default_auto_escalate() -> list[float]:
    return [2.0, 4.0, 8.0, 16.0]


def _default_sample_prompts() -> list[str]:
    return [
        "Once upon a time, in a quiet village,",
        "The algorithm computes the result by",
        "I am feeling",
        "Write a short story about a dragon who",
    ]


@dataclasses.dataclass
class SteeringVerificationConfig:
    """How `scripts/bounds/01_verify_steering.py` behaves.

    The verification script samples from the steered model alongside the
    unsteered baseline and runs cheap automated sanity checks (KL divergence
    on next-token distributions, cosine similarity of per-token residual
    deltas with the steering direction). `auto_escalate_scales` is the
    geometric sweep tried when the initial `verification_scale` is ambiguous.
    """

    enabled: bool = True
    sample_prompts: list[str] = dataclasses.field(default_factory=_default_sample_prompts)
    verification_scale: float = 4.0
    kl_threshold: float = 0.05
    # Primary "did anything happen?" gate: mean(‖delta‖ / ‖pre_unsteered‖)
    # over last real tokens of the verification prompts. Default 0.02
    # is well above float-noise but below typical scale-4 residual shifts
    # on Qwen / Llama (observed: 0.03-0.1).
    magnitude_ratio_threshold: float = 0.02
    # Informational only — the cosine of the residual delta with the
    # raw sum(scale*s_layer) direction can be near-zero even when the
    # intervention is landing perfectly, because of propagation through
    # intermediate sublayers. NOT gated on.
    cosine_threshold: float = 0.1
    auto_escalate_scales: list[float] = dataclasses.field(default_factory=_default_auto_escalate)
    max_new_tokens: int = 32
    # Sampling vs greedy for decoded samples. Greedy (False) can hide
    # steering effects when the argmax next token doesn't flip; sampling
    # reveals the distributional shift better.
    do_sample: bool = True
    sample_temperature: float = 1.0


def _default_capture_specs() -> list[dict]:
    return [{"site": "final", "tier": "full"}]


@dataclasses.dataclass
class BoundsExperimentConfig:
    run_name: str
    seed: int
    model: ModelConfig
    steering: BoundsSteeringConfig
    dataset: BoundsDatasetConfig = dataclasses.field(default_factory=BoundsDatasetConfig)
    capture_specs: list[dict] = dataclasses.field(default_factory=_default_capture_specs)
    steering_verification: SteeringVerificationConfig = dataclasses.field(
        default_factory=SteeringVerificationConfig
    )
    reservoir_size: int = 1024
    dtype: str = "bfloat16"

    @property
    def output_dir(self) -> Path:
        return Path("outputs/bounds") / self.run_name

    @classmethod
    def from_yaml(cls, path: str | Path) -> BoundsExperimentConfig:
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        steering_raw = raw["steering"]
        has_vector = bool(steering_raw.get("vector_path"))
        has_random = bool(steering_raw.get("random_reference_path"))
        if has_vector == has_random:
            raise ValueError(
                "BoundsSteeringConfig: exactly one of `vector_path` or "
                "`random_reference_path` must be set, got "
                f"vector_path={steering_raw.get('vector_path')!r}, "
                f"random_reference_path={steering_raw.get('random_reference_path')!r}"
            )

        return cls(
            run_name=raw["run_name"],
            seed=raw.get("seed", 42),
            model=ModelConfig(**raw["model"]),
            steering=BoundsSteeringConfig(**steering_raw),
            dataset=BoundsDatasetConfig(**raw.get("dataset", {})),
            capture_specs=raw.get("capture_specs", _default_capture_specs()),
            steering_verification=SteeringVerificationConfig(
                **raw.get("steering_verification", {})
            ),
            reservoir_size=raw.get("reservoir_size", 1024),
            dtype=raw.get("dtype", "bfloat16"),
        )
