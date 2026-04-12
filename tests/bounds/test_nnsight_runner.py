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


def test_steering_affects_every_batch_element() -> None:
    """Regression test for the .output[0] batch-indexing bug.

    On Qwen2/Llama3, ``layer.output`` is a bare tensor ``[B, T, d]`` (not
    a tuple), so ``layer.output[0]`` selects batch element 0 rather than
    unpacking a tuple. A buggy intervention using ``.output[0]`` would
    only modify prompt 0 in a multi-prompt batch, silently producing
    contaminated stats. This test asserts that EVERY batch element shows
    an intervention — not just prompt 0.

    This test runs on tiny-gpt2 (tuple-output architecture), so it only
    catches regressions where someone hardcodes ``.output[0]`` without
    the architecture dispatch. For Qwen/Llama specifically, the
    dispatcher in ``_is_tuple_output_architecture`` is the safety net,
    and GPU verification runs exercise the non-tuple path.
    """
    lm = _make_model()
    # Use 4 very different prompts so any failure to steer one of them
    # is visible.
    prompts = [
        "the quick brown fox",
        "alpha beta gamma",
        "once upon a time",
        "goodbye cruel world",
    ]

    unsteered = run_bounds_forward_pass(
        lm, prompts, steering=None, scale=0.0, target_layers=[],
        capture_specs=[{"site": "final"}], max_seq_len=16,
    )
    steered = run_bounds_forward_pass(
        lm, prompts, steering=_nonzero_steering(lm), scale=1.0,
        target_layers=[0, 1], capture_specs=[{"site": "final"}], max_seq_len=16,
    )

    pre_u = unsteered["final"]["pre"]  # [B, T, d]
    pre_s = steered["final"]["pre"]

    assert pre_u.shape[0] == 4 and pre_s.shape[0] == 4, (
        f"expected batch 4, got unsteered={pre_u.shape}, steered={pre_s.shape}"
    )

    # Every batch element's pre-RMSNorm residual should differ from the
    # unsteered version. The per-element max abs delta must be above a
    # noise threshold. A buggy impl would leave batch elements 1, 2, 3
    # unchanged while only modifying batch 0.
    per_batch_max_diff = (pre_u - pre_s).abs().flatten(1).max(dim=1).values
    assert per_batch_max_diff.shape == (4,)
    for i, delta in enumerate(per_batch_max_diff.tolist()):
        assert delta > 1e-4, (
            f"batch element {i} shows delta={delta:.2e} — steering did not "
            f"reach it. Per-batch deltas: {per_batch_max_diff.tolist()}"
        )


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
