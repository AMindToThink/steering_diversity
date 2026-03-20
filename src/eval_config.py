"""Configuration for pass@k evaluation pipelines.

Eval configs reference existing steering YAML files to inherit model and
steering parameters, then layer on eval-specific settings (endpoint, sampling,
datasets).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

from src.config import ExperimentConfig, ModelConfig, SteeringConfig


@dataclasses.dataclass
class EndpointConfig:
    """Where the EasySteer vLLM server is running and where the proxy listens."""

    base_url: str = "http://localhost:8017/v1"
    proxy_port: int = 8018
    steering_mode: str = "proxy"  # "server" or "proxy"


@dataclasses.dataclass
class PassAtKConfig:
    """Sampling and evaluation parameters for pass@k."""

    n_samples: int = 200
    temperatures: list[float] = dataclasses.field(default_factory=lambda: [0.8])
    max_tokens: int = 512
    k_values: list[int] = dataclasses.field(
        default_factory=lambda: [1, 2, 5, 10, 25, 50, 100, 200]
    )


@dataclasses.dataclass
class CodeEvalConfig:
    """Configuration for code-domain pass@k evaluation.

    Inherits model and steering parameters from an existing steering config
    YAML, then adds eval-specific fields.
    """

    run_name: str
    seed: int
    steering: SteeringConfig
    model: ModelConfig
    endpoint: EndpointConfig
    pass_at_k: PassAtKConfig
    datasets: list[str]
    scales: list[float]
    vector_path: str  # Path to the pre-computed steering vector (.gguf)

    # Path to the steering config this was loaded from (for provenance)
    steering_config_path: str = ""

    @property
    def output_dir(self) -> Path:
        return Path("outputs") / self.run_name

    @classmethod
    def from_yaml(cls, path: str | Path) -> CodeEvalConfig:
        """Load eval config from YAML, inheriting from the referenced steering config."""
        path = Path(path)
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        # Resolve steering config path relative to this config's directory
        steering_config_rel = raw["steering_config"]
        steering_config_path = path.parent / steering_config_rel
        base_cfg = ExperimentConfig.from_yaml(steering_config_path)

        # Scales: override from eval config or inherit from steering config
        scales = raw.get("scales", base_cfg.steering.scales)

        # Endpoint
        endpoint_raw = raw.get("endpoint", {})
        endpoint = EndpointConfig(**endpoint_raw)

        # Pass@k config
        pak_raw = raw.get("pass_at_k", {})
        pass_at_k_cfg = PassAtKConfig(**pak_raw)

        # Datasets
        datasets = raw.get("datasets", ["humaneval"])

        # Vector path — required, resolved relative to config directory
        vector_path = raw["vector_path"]
        vector_path_resolved = path.parent / vector_path
        if not vector_path_resolved.exists():
            raise FileNotFoundError(
                f"Steering vector not found: {vector_path_resolved}"
            )

        return cls(
            run_name=raw["run_name"],
            seed=raw.get("seed", 42),
            steering=base_cfg.steering,
            model=base_cfg.model,
            endpoint=endpoint,
            pass_at_k=pass_at_k_cfg,
            datasets=datasets,
            scales=scales,
            vector_path=str(vector_path_resolved),
            steering_config_path=str(steering_config_path),
        )
