"""Tests for src/bounds/gguf_loader.py — reading EasySteer .gguf files."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.bounds.gguf_loader import load_steering_vector_gguf

# Layout of each real .gguf file in this repo. All three files happen to contain
# EVERY layer (selection happens at apply time via target_layers), so the
# expected layer set here is the full range the file covers, not the paper's
# target-layer list.
REAL_VECTORS = [
    (Path("EasySteer/vectors/happy_diffmean.gguf"), 1536, set(range(0, 28))),
    (
        Path("EasySteer/replications/steerable_chatbot/style-probe.gguf"),
        1536,
        set(range(0, 28)),
    ),
    (
        Path("EasySteer/replications/creative_writing/create.gguf"),
        4096,
        set(range(0, 32)),
    ),
]


@pytest.mark.parametrize("path,hidden_size,expected_layers", REAL_VECTORS)
def test_loads_real_gguf(path: Path, hidden_size: int, expected_layers: set[int]) -> None:
    if not path.exists():
        pytest.skip(f"{path} not present (EasySteer submodule?)")

    vecs = load_steering_vector_gguf(path)

    assert set(vecs.keys()) == expected_layers, f"{path}: layer mismatch"
    for layer, v in vecs.items():
        assert isinstance(layer, int)
        assert isinstance(v, torch.Tensor)
        assert v.ndim == 1
        assert v.shape[0] == hidden_size
        assert v.dtype == torch.float32
        assert v.device.type == "cpu"
        # Not a silent-zero bug: real steering vectors have nonzero norm.
        assert v.norm().item() > 0.0


def test_all_three_have_distinct_directions() -> None:
    """The three real vectors should point in different directions.

    A loader that returned the same bytes for every file would pass the
    shape/dtype tests above. This catches that failure mode.
    """
    loaded = []
    for path, _, _ in REAL_VECTORS:
        if not path.exists():
            pytest.skip(f"{path} not present")
        loaded.append(load_steering_vector_gguf(path))

    # Pick any layer index present in the first two (both Qwen, layers 0-27).
    layer = 10
    happy = loaded[0][layer]
    style = loaded[1][layer]
    cos = torch.dot(happy, style) / (happy.norm() * style.norm() + 1e-12)
    # They're trained on different tasks so cos should be far from 1.
    assert cos.item() < 0.99, f"happy and style are suspiciously parallel: cos={cos.item()}"


def test_loads_float_dtype_correctly(tmp_path: Path) -> None:
    """Round-trip a synthetic GGUF to verify we read the bytes correctly."""
    import gguf

    d = 16
    n_layers = 4
    writer = gguf.GGUFWriter(str(tmp_path / "synth.gguf"), "controlvector")
    expected: dict[int, torch.Tensor] = {}
    for i in range(n_layers):
        # Distinct vectors so a loader that only reads zeros or constants fails.
        v = torch.arange(d, dtype=torch.float32) + float(i) * 100.0
        expected[i] = v
        writer.add_tensor(f"direction.{i}", v.numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    loaded = load_steering_vector_gguf(tmp_path / "synth.gguf")

    assert set(loaded.keys()) == set(expected.keys())
    for i in expected:
        assert torch.equal(loaded[i], expected[i])


def test_raises_on_unknown_tensor_name(tmp_path: Path) -> None:
    """Loud failure on a malformed GGUF — no silent skip per CLAUDE.md."""
    import gguf

    writer = gguf.GGUFWriter(str(tmp_path / "bad.gguf"), "controlvector")
    writer.add_tensor("this_is_not_a_direction", torch.zeros(8).numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    with pytest.raises(ValueError, match="Unexpected tensor name"):
        load_steering_vector_gguf(tmp_path / "bad.gguf")
