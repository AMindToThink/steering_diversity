"""Experiment configuration loaded from YAML."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclasses.dataclass
class ModelConfig:
    name: str
    model_type: str  # e.g. "qwen2.5" — used by EasySteer


@dataclasses.dataclass
class SteeringConfig:
    concept: str  # e.g. "deception"
    contrastive_pairs_path: str
    scales: list[float]
    target_layers: list[int]
    token_pos: int | str = -1
    normalize: bool = True


@dataclasses.dataclass
class GenerationConfig:
    num_prompts: int
    responses_per_prompt: int
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.95
    prompt_dataset: str = "euclaise/writingprompts"
    prompt_split: str = "test"
    system_prompt: str | None = None


@dataclasses.dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 64


@dataclasses.dataclass
class ClusteringConfig:
    min_cluster_size: int = 5
    min_samples: int | None = None
    metric: str = "euclidean"


@dataclasses.dataclass
class ExperimentConfig:
    run_name: str
    seed: int
    model: ModelConfig
    steering: SteeringConfig
    generation: GenerationConfig
    embedding: EmbeddingConfig
    clustering: ClusteringConfig

    @property
    def output_dir(self) -> Path:
        return Path("outputs") / self.run_name

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load config from a YAML file."""
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        return cls(
            run_name=raw["run_name"],
            seed=raw.get("seed", 42),
            model=ModelConfig(**raw["model"]),
            steering=SteeringConfig(**raw["steering"]),
            generation=GenerationConfig(**raw["generation"]),
            embedding=EmbeddingConfig(**raw.get("embedding", {})),
            clustering=ClusteringConfig(**raw.get("clustering", {})),
        )
