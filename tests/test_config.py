"""Tests for configuration loading."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from src.config import ExperimentConfig


@pytest.fixture
def sample_yaml(tmp_path: Path) -> Path:
    cfg = {
        "run_name": "test_run",
        "seed": 123,
        "model": {"name": "test-model", "model_type": "qwen2.5"},
        "steering": {
            "concept": "deception",
            "contrastive_pairs_path": "data/contrastive_pairs/deception.json",
            "scales": [0.0, 1.0, 2.0],
            "target_layers": [10, 11, 12],
        },
        "generation": {
            "num_prompts": 5,
            "responses_per_prompt": 3,
        },
    }
    path = tmp_path / "test_config.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return path


def test_load_config(sample_yaml: Path) -> None:
    cfg = ExperimentConfig.from_yaml(sample_yaml)

    assert cfg.run_name == "test_run"
    assert cfg.seed == 123
    assert cfg.model.name == "test-model"
    assert cfg.model.model_type == "qwen2.5"
    assert cfg.steering.scales == [0.0, 1.0, 2.0]
    assert cfg.steering.target_layers == [10, 11, 12]
    assert cfg.steering.concept == "deception"
    assert cfg.steering.normalize is True  # default
    assert cfg.generation.num_prompts == 5
    assert cfg.generation.responses_per_prompt == 3
    assert cfg.generation.max_tokens == 256  # default
    assert cfg.embedding.model_name == "all-MiniLM-L6-v2"  # default
    assert cfg.clustering.min_cluster_size == 5  # default


def test_output_dir(sample_yaml: Path) -> None:
    cfg = ExperimentConfig.from_yaml(sample_yaml)
    assert cfg.output_dir == Path("outputs") / "test_run"


def test_config_with_all_fields(tmp_path: Path) -> None:
    cfg = {
        "run_name": "full",
        "seed": 0,
        "model": {"name": "m", "model_type": "llama"},
        "steering": {
            "concept": "c",
            "contrastive_pairs_path": "p.json",
            "scales": [0.0],
            "target_layers": [1],
            "token_pos": "mean",
            "normalize": False,
        },
        "generation": {
            "num_prompts": 1,
            "responses_per_prompt": 1,
            "max_tokens": 64,
            "temperature": 0.7,
            "top_p": 0.9,
            "prompt_dataset": "custom/ds",
            "prompt_split": "train",
        },
        "embedding": {"model_name": "custom-emb", "batch_size": 16},
        "clustering": {"min_cluster_size": 10, "min_samples": 3, "metric": "cosine"},
    }
    path = tmp_path / "full.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)

    loaded = ExperimentConfig.from_yaml(path)
    assert loaded.steering.token_pos == "mean"
    assert loaded.steering.normalize is False
    assert loaded.generation.temperature == 0.7
    assert loaded.embedding.batch_size == 16
    assert loaded.clustering.min_samples == 3
    assert loaded.clustering.metric == "cosine"
