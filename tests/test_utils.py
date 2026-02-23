"""Tests for utility functions including provenance tracking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.config import ExperimentConfig
from src.utils import save_provenance


@pytest.fixture
def cfg(tmp_path: Path) -> ExperimentConfig:
    raw = {
        "run_name": "test_run",
        "seed": 42,
        "model": {"name": "test-model", "model_type": "qwen2.5"},
        "steering": {
            "concept": "deception",
            "contrastive_pairs_path": "data/contrastive_pairs/deception.json",
            "scales": [0.0, 1.0],
            "target_layers": [10, 11],
        },
        "generation": {
            "num_prompts": 5,
            "responses_per_prompt": 3,
        },
    }
    path = tmp_path / "cfg.yaml"
    with open(path, "w") as f:
        yaml.dump(raw, f)
    return ExperimentConfig.from_yaml(path)


def test_save_provenance_creates_sidecar(tmp_path: Path, cfg: ExperimentConfig) -> None:
    output_file = tmp_path / "responses.jsonl"
    output_file.write_text("")

    save_provenance(
        step="02_generate_responses",
        config_path="configs/test.yaml",
        cfg=cfg,
        inputs={"steering_vector": "outputs/test/vec.gguf"},
        outputs=[str(output_file)],
    )

    sidecar = Path(str(output_file) + ".provenance.json")
    assert sidecar.exists()

    data = json.loads(sidecar.read_text())
    assert data["step"] == "02_generate_responses"
    assert data["config_path"] == "configs/test.yaml"
    assert data["inputs"] == {"steering_vector": "outputs/test/vec.gguf"}
    assert data["outputs"] == [str(output_file)]
    assert "config_snapshot" in data
    assert data["config_snapshot"]["run_name"] == "test_run"
    assert isinstance(data["git_commit"], str)
    assert isinstance(data["timestamp"], str)


def test_save_provenance_git_commit_populated(tmp_path: Path, cfg: ExperimentConfig) -> None:
    """git_commit should be non-empty since we're running inside a git repo."""
    output_file = tmp_path / "out.npz"
    output_file.write_text("")

    save_provenance(
        step="03_embed_responses",
        config_path="configs/test.yaml",
        cfg=cfg,
        inputs={"responses": "outputs/test/responses.jsonl"},
        outputs=[str(output_file)],
    )

    sidecar = Path(str(output_file) + ".provenance.json")
    data = json.loads(sidecar.read_text())
    assert len(data["git_commit"]) == 40  # full SHA


def test_save_provenance_multiple_outputs(tmp_path: Path, cfg: ExperimentConfig) -> None:
    outputs = [str(tmp_path / f"plot_{i}.png") for i in range(3)]
    for p in outputs:
        Path(p).write_text("")

    save_provenance(
        step="05_visualize",
        config_path="configs/test.yaml",
        cfg=cfg,
        inputs={"embeddings": "emb.npz", "metrics": "metrics.json"},
        outputs=outputs,
    )

    for p in outputs:
        sidecar = Path(p + ".provenance.json")
        assert sidecar.exists()
        data = json.loads(sidecar.read_text())
        assert data["step"] == "05_visualize"
        assert data["outputs"] == outputs


def test_save_provenance_has_all_expected_keys(tmp_path: Path, cfg: ExperimentConfig) -> None:
    output_file = tmp_path / "metrics.json"
    output_file.write_text("")

    save_provenance(
        step="04_compute_metrics",
        config_path="configs/test.yaml",
        cfg=cfg,
        inputs={"embeddings": "emb.npz"},
        outputs=[str(output_file)],
    )

    sidecar = Path(str(output_file) + ".provenance.json")
    data = json.loads(sidecar.read_text())

    expected_keys = {
        "step",
        "config_path",
        "config_snapshot",
        "inputs",
        "outputs",
        "git_commit",
        "timestamp",
    }
    assert set(data.keys()) == expected_keys
