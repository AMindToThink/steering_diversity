"""Tests for src/bounds/config.py — BoundsExperimentConfig YAML round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.bounds.config import (
    BoundsDatasetConfig,
    BoundsExperimentConfig,
    BoundsSteeringConfig,
    SteeringVerificationConfig,
)


def _full_cfg() -> dict:
    return {
        "run_name": "test_run",
        "seed": 7,
        "model": {"name": "Qwen/Qwen2.5-1.5B-Instruct", "model_type": "qwen2.5"},
        "steering": {
            "vector_path": "EasySteer/vectors/happy_diffmean.gguf",
            "target_layers": list(range(10, 26)),
            "scale_sweep": [0.0, 0.5, 1.0, 2.0, 4.0, 8.0],
        },
        "dataset": {
            "name": "HuggingFaceFW/fineweb-edu",
            "num_prompts": 1000,
            "max_seq_len": 256,
            "batch_size": 8,
        },
        "capture_specs": [{"site": "final", "tier": "full"}],
        "steering_verification": {
            "enabled": True,
            "sample_prompts": [
                "Once upon a time",
                "The cat sat on",
                "In science, we observe",
                "The best recipe",
            ],
            "verification_scale": 4.0,
            "kl_threshold": 0.05,
            "cosine_threshold": 0.1,
            "auto_escalate_scales": [2.0, 4.0, 8.0, 16.0],
            "max_new_tokens": 32,
        },
        "reservoir_size": 1024,
        "dtype": "bfloat16",
    }


def test_from_yaml_full_roundtrip(tmp_path: Path) -> None:
    cfg = _full_cfg()
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))

    loaded = BoundsExperimentConfig.from_yaml(p)

    assert loaded.run_name == "test_run"
    assert loaded.seed == 7
    assert loaded.model.name == "Qwen/Qwen2.5-1.5B-Instruct"
    assert loaded.model.model_type == "qwen2.5"

    assert isinstance(loaded.steering, BoundsSteeringConfig)
    assert loaded.steering.vector_path == "EasySteer/vectors/happy_diffmean.gguf"
    assert loaded.steering.target_layers == list(range(10, 26))
    assert loaded.steering.scale_sweep == [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    assert loaded.steering.random_reference_path is None
    assert loaded.steering.random_seed is None

    assert isinstance(loaded.dataset, BoundsDatasetConfig)
    assert loaded.dataset.name == "HuggingFaceFW/fineweb-edu"
    assert loaded.dataset.num_prompts == 1000
    assert loaded.dataset.max_seq_len == 256
    assert loaded.dataset.batch_size == 8

    assert loaded.capture_specs == [{"site": "final", "tier": "full"}]

    assert isinstance(loaded.steering_verification, SteeringVerificationConfig)
    assert loaded.steering_verification.enabled is True
    assert len(loaded.steering_verification.sample_prompts) == 4
    assert loaded.steering_verification.verification_scale == 4.0
    assert loaded.steering_verification.kl_threshold == 0.05
    assert loaded.steering_verification.cosine_threshold == 0.1
    assert loaded.steering_verification.auto_escalate_scales == [2.0, 4.0, 8.0, 16.0]
    assert loaded.steering_verification.max_new_tokens == 32

    assert loaded.reservoir_size == 1024
    assert loaded.dtype == "bfloat16"
    assert loaded.output_dir == Path("outputs/bounds/test_run")


def test_from_yaml_minimal_defaults(tmp_path: Path) -> None:
    cfg = {
        "run_name": "min",
        "seed": 0,
        "model": {"name": "sshleifer/tiny-gpt2", "model_type": "gpt2"},
        "steering": {
            "vector_path": "some/vector.gguf",
            "target_layers": [1, 2],
            "scale_sweep": [0.0, 1.0],
        },
    }
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(cfg))

    loaded = BoundsExperimentConfig.from_yaml(p)

    assert loaded.dataset.name == "HuggingFaceFW/fineweb-edu"
    assert loaded.dataset.num_prompts == 1000
    assert loaded.capture_specs == [{"site": "final", "tier": "full"}]
    assert loaded.steering_verification.enabled is True
    assert loaded.steering_verification.auto_escalate_scales == [2.0, 4.0, 8.0, 16.0]
    assert loaded.reservoir_size == 1024
    assert loaded.dtype == "bfloat16"


def test_random_vector_config(tmp_path: Path) -> None:
    cfg = _full_cfg()
    cfg["steering"] = {
        "random_reference_path": "EasySteer/vectors/happy_diffmean.gguf",
        "random_seed": 0,
        "target_layers": list(range(10, 26)),
        "scale_sweep": [0.0, 1.0, 2.0],
    }
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump(cfg))

    loaded = BoundsExperimentConfig.from_yaml(p)
    assert loaded.steering.vector_path is None
    assert loaded.steering.random_reference_path == "EasySteer/vectors/happy_diffmean.gguf"
    assert loaded.steering.random_seed == 0


def test_from_yaml_rejects_both_real_and_random_vector(tmp_path: Path) -> None:
    cfg = _full_cfg()
    cfg["steering"]["random_reference_path"] = "foo.gguf"
    cfg["steering"]["random_seed"] = 0
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(cfg))

    with pytest.raises(ValueError, match="exactly one of"):
        BoundsExperimentConfig.from_yaml(p)


def test_from_yaml_rejects_neither_vector(tmp_path: Path) -> None:
    cfg = {
        "run_name": "bad",
        "seed": 0,
        "model": {"name": "gpt2", "model_type": "gpt2"},
        "steering": {"target_layers": [1], "scale_sweep": [0.0, 1.0]},
    }
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(cfg))

    with pytest.raises(ValueError, match="exactly one of"):
        BoundsExperimentConfig.from_yaml(p)
