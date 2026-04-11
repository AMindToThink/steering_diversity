"""CPU integration tests for src/bounds/nnsight_runner.py.

Uses ``sshleifer/tiny-gpt2`` (2 layers, hidden size 2) so every test runs
under a second on CPU. The goal is to pin the shape + behavior contract:

- ``run_bounds_forward_pass`` returns pre/post tensors of the right shape
- scale=0 matches "no steering"
- a nonzero steering vector produces measurable downstream changes
- ``sample_from_steered_model`` returns one string per prompt
- zero-steering samples match unsteered generations byte-for-byte
- the two code paths share the same intervention helper so perturbations
  taken through the forward-pass and generate paths are consistent
"""

from __future__ import annotations

import torch
from nnsight import LanguageModel

from src.bounds.nnsight_runner import (
    run_bounds_forward_pass,
    sample_from_steered_model,
)


MODEL_ID = "sshleifer/tiny-gpt2"


def _make_model() -> LanguageModel:
    return LanguageModel(MODEL_ID, device_map="cpu")


def _zero_steering(lm: LanguageModel) -> dict[int, torch.Tensor]:
    d = lm.config.n_embd
    n = lm.config.n_layer
    return {i: torch.zeros(d, dtype=torch.float32) for i in range(n)}


def _nonzero_steering(lm: LanguageModel) -> dict[int, torch.Tensor]:
    d = lm.config.n_embd
    n = lm.config.n_layer
    return {i: torch.arange(d, dtype=torch.float32) + float(i) for i in range(n)}


def test_forward_pass_returns_expected_shapes() -> None:
    lm = _make_model()
    d = lm.config.n_embd
    prompts = ["hello world", "activation steering is fun"]

    out = run_bounds_forward_pass(
        lm,
        prompts,
        steering=None,
        scale=0.0,
        target_layers=[],
        capture_specs=[{"site": "final", "tier": "full"}],
        max_seq_len=16,
    )

    assert set(out.keys()) == {"attention_mask", "final"}
    assert out["attention_mask"].shape == (2, out["attention_mask"].shape[1])
    assert out["attention_mask"].shape[1] <= 16
    final = out["final"]
    B, T = out["attention_mask"].shape
    assert final["pre"].shape == (B, T, d)
    assert final["post"].shape == (B, T, d)


def test_zero_steering_matches_no_steering() -> None:
    lm = _make_model()
    prompts = ["the quick brown fox", "a b c"]

    a = run_bounds_forward_pass(
        lm,
        prompts,
        steering=None,
        scale=0.0,
        target_layers=[],
        capture_specs=[{"site": "final"}],
        max_seq_len=16,
    )
    b = run_bounds_forward_pass(
        lm,
        prompts,
        steering=_zero_steering(lm),
        scale=1.0,
        target_layers=[0, 1],
        capture_specs=[{"site": "final"}],
        max_seq_len=16,
    )
    assert torch.allclose(a["final"]["pre"], b["final"]["pre"], atol=1e-5)
    assert torch.allclose(a["final"]["post"], b["final"]["post"], atol=1e-5)


def test_nonzero_steering_changes_residual_stream() -> None:
    lm = _make_model()
    prompts = ["the quick brown fox", "a b c"]

    a = run_bounds_forward_pass(
        lm,
        prompts,
        steering=None,
        scale=0.0,
        target_layers=[],
        capture_specs=[{"site": "final"}],
        max_seq_len=16,
    )
    b = run_bounds_forward_pass(
        lm,
        prompts,
        steering=_nonzero_steering(lm),
        scale=1.0,
        target_layers=[0, 1],
        capture_specs=[{"site": "final"}],
        max_seq_len=16,
    )
    # Pre-RMSNorm residual should differ — intervention landed.
    diff_pre = (a["final"]["pre"] - b["final"]["pre"]).abs().max().item()
    assert diff_pre > 0.1, f"pre-RMSNorm unchanged (diff={diff_pre}), intervention missed"


def test_unknown_site_raises() -> None:
    lm = _make_model()
    try:
        run_bounds_forward_pass(
            lm,
            ["hello"],
            steering=None,
            scale=0.0,
            target_layers=[],
            capture_specs=[{"site": "layer_0"}],
            max_seq_len=8,
        )
    except NotImplementedError:
        return
    raise AssertionError("Expected NotImplementedError for layer_0 site")


def test_missing_steering_layer_raises() -> None:
    lm = _make_model()
    # steering dict only has layer 0, but target_layers asks for layer 1.
    steering = {0: torch.ones(lm.config.n_embd)}
    try:
        run_bounds_forward_pass(
            lm,
            ["hello"],
            steering=steering,
            scale=1.0,
            target_layers=[0, 1],
            capture_specs=[{"site": "final"}],
            max_seq_len=8,
        )
    except KeyError:
        return
    raise AssertionError("Expected KeyError for missing steering layer")


def test_sample_from_steered_model_unsteered_baseline() -> None:
    lm = _make_model()
    prompts = ["The quick brown", "Once upon a"]
    out = sample_from_steered_model(
        lm,
        prompts,
        steering=None,
        scale=0.0,
        target_layers=[],
        max_new_tokens=4,
    )
    assert len(out) == 2
    assert all(isinstance(s, str) for s in out)
    assert all(len(s) > 0 for s in out)


def test_sample_from_steered_model_zero_steering_matches_baseline() -> None:
    lm = _make_model()
    prompts = ["The quick brown"]
    baseline = sample_from_steered_model(
        lm, prompts, steering=None, scale=0.0, target_layers=[], max_new_tokens=4
    )
    zero = sample_from_steered_model(
        lm,
        prompts,
        steering=_zero_steering(lm),
        scale=1.0,
        target_layers=[0, 1],
        max_new_tokens=4,
    )
    assert baseline == zero, f"baseline {baseline!r} != zero-steering {zero!r}"


def test_sample_from_steered_model_nonzero_steering_changes_output() -> None:
    lm = _make_model()
    prompts = ["The quick brown fox"]

    baseline = sample_from_steered_model(
        lm, prompts, steering=None, scale=0.0, target_layers=[], max_new_tokens=4
    )

    # Make the steering big enough to push tiny-gpt2 far off course.
    d = lm.config.n_embd
    n = lm.config.n_layer
    big_steering = {
        i: torch.tensor([100.0] + [0.0] * (d - 1), dtype=torch.float32)
        for i in range(n)
    }
    perturbed = sample_from_steered_model(
        lm,
        prompts,
        steering=big_steering,
        scale=1.0,
        target_layers=list(range(n)),
        max_new_tokens=4,
    )
    assert baseline != perturbed, (
        f"big steering didn't change tiny-gpt2 output: {baseline!r} == {perturbed!r}"
    )
