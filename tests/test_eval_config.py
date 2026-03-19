"""Tests for eval_config module."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.eval_config import CodeEvalConfig, EndpointConfig, PassAtKConfig


@pytest.fixture()
def config_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with a steering config, eval config, and dummy vector."""
    # Write a minimal steering config
    steering_yaml = textwrap.dedent("""\
        run_name: "test_steering"
        seed: 42
        model:
          name: "Qwen/Qwen2.5-1.5B-Instruct"
          model_type: "qwen2.5"
        steering:
          concept: "deception"
          scales: [0.0, 1.0, 2.0]
          target_layers: [10, 11, 12]
        generation:
          num_prompts: 5
          responses_per_prompt: 3
    """)
    (tmp_path / "steering.yaml").write_text(steering_yaml)

    # Create a dummy vector file
    (tmp_path / "dummy_vector.gguf").write_bytes(b"fake")

    # Write an eval config that references it
    eval_yaml = textwrap.dedent("""\
        run_name: "test_eval"
        seed: 123
        steering_config: "steering.yaml"
        vector_path: "dummy_vector.gguf"
        scales: [0.0, 0.5, 1.0]
        endpoint:
          base_url: "http://localhost:9999/v1"
          proxy_port: 9998
        pass_at_k:
          n_samples: 50
          temperatures: [0.8, 1.0]
          max_tokens: 256
          k_values: [1, 5, 10, 50]
        datasets: ["humaneval", "mbpp"]
    """)
    (tmp_path / "eval.yaml").write_text(eval_yaml)

    return tmp_path


class TestCodeEvalConfig:
    def test_loads_from_yaml(self, config_dir: Path) -> None:
        cfg = CodeEvalConfig.from_yaml(config_dir / "eval.yaml")
        assert cfg.run_name == "test_eval"
        assert cfg.seed == 123

    def test_inherits_model_from_steering(self, config_dir: Path) -> None:
        cfg = CodeEvalConfig.from_yaml(config_dir / "eval.yaml")
        assert cfg.model.name == "Qwen/Qwen2.5-1.5B-Instruct"
        assert cfg.model.model_type == "qwen2.5"

    def test_inherits_steering_params(self, config_dir: Path) -> None:
        cfg = CodeEvalConfig.from_yaml(config_dir / "eval.yaml")
        assert cfg.steering.concept == "deception"
        assert cfg.steering.target_layers == [10, 11, 12]

    def test_overrides_scales(self, config_dir: Path) -> None:
        cfg = CodeEvalConfig.from_yaml(config_dir / "eval.yaml")
        assert cfg.scales == [0.0, 0.5, 1.0]

    def test_inherits_scales_when_not_overridden(self, config_dir: Path) -> None:
        eval_yaml = textwrap.dedent("""\
            run_name: "test_no_scales"
            steering_config: "steering.yaml"
            vector_path: "dummy_vector.gguf"
            datasets: ["humaneval"]
        """)
        (config_dir / "eval_no_scales.yaml").write_text(eval_yaml)
        cfg = CodeEvalConfig.from_yaml(config_dir / "eval_no_scales.yaml")
        assert cfg.scales == [0.0, 1.0, 2.0]

    def test_endpoint_config(self, config_dir: Path) -> None:
        cfg = CodeEvalConfig.from_yaml(config_dir / "eval.yaml")
        assert cfg.endpoint.base_url == "http://localhost:9999/v1"
        assert cfg.endpoint.proxy_port == 9998

    def test_pass_at_k_config(self, config_dir: Path) -> None:
        cfg = CodeEvalConfig.from_yaml(config_dir / "eval.yaml")
        assert cfg.pass_at_k.n_samples == 50
        assert cfg.pass_at_k.temperatures == [0.8, 1.0]
        assert cfg.pass_at_k.max_tokens == 256
        assert cfg.pass_at_k.k_values == [1, 5, 10, 50]

    def test_datasets(self, config_dir: Path) -> None:
        cfg = CodeEvalConfig.from_yaml(config_dir / "eval.yaml")
        assert cfg.datasets == ["humaneval", "mbpp"]

    def test_output_dir(self, config_dir: Path) -> None:
        cfg = CodeEvalConfig.from_yaml(config_dir / "eval.yaml")
        assert cfg.output_dir == Path("outputs/test_eval")

    def test_steering_config_path_recorded(self, config_dir: Path) -> None:
        cfg = CodeEvalConfig.from_yaml(config_dir / "eval.yaml")
        assert "steering.yaml" in cfg.steering_config_path

    def test_vector_path_resolved(self, config_dir: Path) -> None:
        cfg = CodeEvalConfig.from_yaml(config_dir / "eval.yaml")
        assert cfg.vector_path.endswith("dummy_vector.gguf")
        assert Path(cfg.vector_path).exists()

    def test_defaults(self, config_dir: Path) -> None:
        eval_yaml = textwrap.dedent("""\
            run_name: "minimal"
            steering_config: "steering.yaml"
            vector_path: "dummy_vector.gguf"
        """)
        (config_dir / "minimal.yaml").write_text(eval_yaml)
        cfg = CodeEvalConfig.from_yaml(config_dir / "minimal.yaml")
        assert cfg.seed == 42
        assert cfg.datasets == ["humaneval"]
        assert cfg.endpoint.base_url == "http://localhost:8017/v1"
        assert cfg.pass_at_k.n_samples == 200

    def test_missing_steering_config_raises(self, tmp_path: Path) -> None:
        (tmp_path / "dummy_vector.gguf").write_bytes(b"fake")
        eval_yaml = textwrap.dedent("""\
            run_name: "broken"
            steering_config: "nonexistent.yaml"
            vector_path: "dummy_vector.gguf"
        """)
        (tmp_path / "eval.yaml").write_text(eval_yaml)
        with pytest.raises(FileNotFoundError):
            CodeEvalConfig.from_yaml(tmp_path / "eval.yaml")

    def test_missing_vector_raises(self, config_dir: Path) -> None:
        eval_yaml = textwrap.dedent("""\
            run_name: "broken"
            steering_config: "steering.yaml"
            vector_path: "nonexistent.gguf"
        """)
        (config_dir / "bad_vector.yaml").write_text(eval_yaml)
        with pytest.raises(FileNotFoundError, match="Steering vector not found"):
            CodeEvalConfig.from_yaml(config_dir / "bad_vector.yaml")


class TestEndpointConfig:
    def test_defaults(self) -> None:
        cfg = EndpointConfig()
        assert cfg.base_url == "http://localhost:8017/v1"
        assert cfg.proxy_port == 8018


class TestPassAtKConfig:
    def test_defaults(self) -> None:
        cfg = PassAtKConfig()
        assert cfg.n_samples == 200
        assert cfg.temperatures == [0.8]
        assert cfg.max_tokens == 512
        assert cfg.k_values == [1, 2, 5, 10, 25, 50, 100, 200]
