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
    """Where the vLLM server is running and where the proxy listens."""

    base_url: str = "http://localhost:8017/v1"
    proxy_port: int = 8018
    steering_mode: str = "proxy"  # "server", "proxy", or "none"


@dataclasses.dataclass
class PassAtKConfig:
    """Sampling and evaluation parameters for pass@k."""

    n_samples: int = 200
    temperatures: list[float] = dataclasses.field(default_factory=lambda: [0.8])
    max_tokens: int = 512
    k_values: list[int] = dataclasses.field(
        default_factory=lambda: [1, 2, 5, 10, 25, 50, 100, 200]
    )
    batch_size: int = 10  # BigCodeBench writes after each batch (crash resilience)
    id_range: str | None = None  # e.g. "0,500" to eval subset of problems


@dataclasses.dataclass
class CodeEvalConfig:
    """Configuration for code-domain pass@k evaluation.

    Two modes:
    1. With ``steering_config``: inherits model and steering parameters from an
       existing steering YAML (EasySteer workflow).
    2. Without ``steering_config``: reads ``model_name`` directly from the eval
       YAML. Used for standalone models (e.g. weight-modified Qwen3-32B) where
       steering is baked into the weights and no runtime steering is needed.
    """

    run_name: str
    seed: int
    steering: SteeringConfig | None
    model: ModelConfig
    endpoint: EndpointConfig
    pass_at_k: PassAtKConfig
    datasets: list[str]
    scales: list[float]
    vector_path: str | None  # Path to the pre-computed steering vector (.gguf)

    # Path to the steering config this was loaded from (for provenance)
    steering_config_path: str = ""

    @property
    def output_dir(self) -> Path:
        return Path("outputs") / self.run_name

    @classmethod
    def from_yaml(cls, path: str | Path) -> CodeEvalConfig:
        """Load eval config from YAML.

        If ``steering_config`` is present, inherits model/steering from that
        file (original behavior). Otherwise, reads ``model_name`` directly
        and sets ``steering`` to None.
        """
        path = Path(path)
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        # Endpoint
        endpoint_raw = raw.get("endpoint", {})
        endpoint = EndpointConfig(**endpoint_raw)

        # Pass@k config
        pak_raw = raw.get("pass_at_k", {})
        pass_at_k_cfg = PassAtKConfig(**pak_raw)

        # Datasets
        datasets = raw.get("datasets", ["humaneval"])

        steering_config_rel = raw.get("steering_config")

        if steering_config_rel is not None:
            # --- Original path: inherit from steering config ---
            steering_config_path = path.parent / steering_config_rel
            base_cfg = ExperimentConfig.from_yaml(steering_config_path)

            scales = raw.get("scales", base_cfg.steering.scales)
            steering = base_cfg.steering
            model = base_cfg.model

            # Vector path — required when using a steering config
            vector_path_raw = raw["vector_path"]
            vector_path_resolved = path.parent / vector_path_raw
            if not vector_path_resolved.exists():
                raise FileNotFoundError(
                    f"Steering vector not found: {vector_path_resolved}"
                )
            vector_path: str | None = str(vector_path_resolved)
            steering_config_path_str = str(steering_config_path)
        else:
            # --- Standalone path: model_name directly in eval config ---
            model_name = raw.get("model_name")
            if model_name is None:
                raise ValueError(
                    "Either 'steering_config' or 'model_name' must be specified "
                    f"in {path}"
                )
            scales = raw.get("scales", [0.0])
            steering = None
            model = ModelConfig(name=model_name, model_type="standalone")

            # Vector path is optional in standalone mode
            vector_path_raw = raw.get("vector_path")
            if vector_path_raw is not None:
                vector_path_resolved = path.parent / vector_path_raw
                if not vector_path_resolved.exists():
                    raise FileNotFoundError(
                        f"Steering vector not found: {vector_path_resolved}"
                    )
                vector_path = str(vector_path_resolved)
            else:
                vector_path = None
            steering_config_path_str = ""

        return cls(
            run_name=raw["run_name"],
            seed=raw.get("seed", 42),
            steering=steering,
            model=model,
            endpoint=endpoint,
            pass_at_k=pass_at_k_cfg,
            datasets=datasets,
            scales=scales,
            vector_path=vector_path,
            steering_config_path=steering_config_path_str,
        )
