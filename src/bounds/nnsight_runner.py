"""nnsight 0.6 forward-pass runner for the bounds-verification pipeline.

Exposes two functions that share one intervention helper, so the steering
applied during verification (``sample_from_steered_model``) is byte-identical
to the steering applied during bounds recording (``run_bounds_forward_pass``).
This is a hard requirement of the "trust no silent steering" policy in the
plan — the verification script MUST exercise the exact code path the
recording script uses.

Model architecture support:

- GPT-2 style (``sshleifer/tiny-gpt2``): decoder layers at ``lm.transformer.h``,
  final norm at ``lm.transformer.ln_f``.
- Llama / Qwen style: decoder layers at ``lm.model.layers``, final norm at
  ``lm.model.norm``.

``capture_specs`` is a list of dicts describing which site(s) to record at
and at what tier. v1 only implements the ``"final"`` site; per-layer capture
is wired into the API from day one so later experiments need only a config
change.
"""

from __future__ import annotations

from typing import Any

import torch
from nnsight import LanguageModel


# ---------------------------------------------------------------------------
# Architecture resolution
# ---------------------------------------------------------------------------


def _get_layer_list(lm: LanguageModel) -> Any:
    """Return the module-tree handle for the decoder layer list.

    Works for both GPT-2 (``lm.transformer.h``) and Llama/Qwen
    (``lm.model.layers``). Raises loudly on unknown architectures — do not
    silently return an empty list or guess, per the loud-errors policy.
    """
    if hasattr(lm, "transformer") and hasattr(lm.transformer, "h"):
        return lm.transformer.h
    if hasattr(lm, "model") and hasattr(lm.model, "layers"):
        return lm.model.layers
    raise ValueError(
        f"Cannot locate decoder layer list on {type(lm).__name__}. "
        "Expected lm.transformer.h (GPT-2) or lm.model.layers (Llama/Qwen)."
    )


def _get_final_norm(lm: LanguageModel) -> Any:
    """Return the module-tree handle for the final RMSNorm / LayerNorm."""
    if hasattr(lm, "transformer") and hasattr(lm.transformer, "ln_f"):
        return lm.transformer.ln_f
    if hasattr(lm, "model") and hasattr(lm.model, "norm"):
        return lm.model.norm
    raise ValueError(
        f"Cannot locate final norm on {type(lm).__name__}. "
        "Expected lm.transformer.ln_f (GPT-2) or lm.model.norm (Llama/Qwen)."
    )


def _model_dtype_device(lm: LanguageModel) -> tuple[torch.dtype, torch.device]:
    """Inspect the first parameter to determine the model's runtime dtype + device."""
    for p in lm.parameters():
        return p.dtype, p.device
    raise RuntimeError("Model has no parameters")


def _is_tuple_output_architecture(lm: LanguageModel) -> bool:
    """Whether decoder-layer ``.output`` is a tuple ``(hidden_states, ...)``
    or a bare tensor ``[B, T, d]``.

    GPT-2's ``Block.forward`` returns ``(hidden_states, present, attn_w)``
    so ``.output[0]`` gives the hidden state. Qwen2 and Llama3 (in HF
    transformers ≥ 4.40) return the hidden-state tensor directly, so
    ``.output[0]`` selects **batch element 0**, silently steering only 1/B
    of the batch. This function tells callers which convention to use.

    If this function is wrong for a new architecture, shape assertions
    downstream will fire immediately (see ``_add_steering_at_layers``).
    """
    if hasattr(lm, "transformer") and hasattr(lm.transformer, "h"):
        return True  # GPT-2 style
    return False  # Qwen2 / Llama3 style


# ---------------------------------------------------------------------------
# Tokenization (external — avoids colliding with nnsight's internal padding=True)
# ---------------------------------------------------------------------------


def _tokenize_for_trace(
    lm: LanguageModel, prompts: list[str], max_seq_len: int
) -> dict[str, torch.Tensor]:
    """Tokenize a batch of prompts for use inside ``lm.trace({...})``.

    nnsight's ``trace()`` passes ``padding=True`` to the tokenizer internally,
    so passing it again via kwargs raises ``KeyError: 'padding'``. We tokenize
    externally and pass a dict, sidestepping the collision.
    """
    tok = lm.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    }


# ---------------------------------------------------------------------------
# Shared intervention helper — verification and recording BOTH call this.
# ---------------------------------------------------------------------------


def _validate_steering_covers_layers(
    steering: dict[int, torch.Tensor] | None,
    scale: float,
    target_layers: list[int],
) -> None:
    """Precondition check, raised BEFORE entering any nnsight trace context.

    nnsight wraps exceptions raised from inside a ``trace()`` / ``generate()``
    block into an ``NNsightException``, which loses the concrete ``KeyError``
    the caller might be catching. Raising here — before the trace —
    preserves clean error propagation.
    """
    if steering is None or scale == 0.0:
        return
    missing = [i for i in target_layers if i not in steering]
    if missing:
        raise KeyError(
            f"target_layers {missing} have no entries in the steering dict "
            f"(available keys: {sorted(steering.keys())})"
        )


def _add_steering_at_layers(
    layer_list: Any,
    steering: dict[int, torch.Tensor] | None,
    scale: float,
    target_layers: list[int],
    dtype: torch.dtype,
    device: torch.device,
    tuple_output: bool,
) -> None:
    """Mutate the residual stream in-place at ``target_layers``.

    MUST be called from inside a ``lm.trace()`` or ``lm.generate()``
    context. The ``tuple_output`` flag determines whether the decoder
    layer's ``.output`` is a tuple (GPT-2) or a bare tensor (Qwen2/Llama3).

    This function is the single source of truth for "where and how steering
    gets applied to the residual stream."
    """
    if steering is None or scale == 0.0:
        return
    for layer_idx in target_layers:
        s = (scale * steering[layer_idx]).to(dtype=dtype, device=device)
        if tuple_output:
            # GPT-2: .output is (hidden_states, ...), [0] unpacks the tuple.
            layer_list[layer_idx].output[0][:] = layer_list[layer_idx].output[0] + s
        else:
            # Qwen2 / Llama3: .output IS the hidden_states tensor [B, T, d].
            # Using .output[0] here would silently select batch element 0!
            layer_list[layer_idx].output[:] = layer_list[layer_idx].output + s


# ---------------------------------------------------------------------------
# Public API: forward pass for bounds recording
# ---------------------------------------------------------------------------


def run_bounds_forward_pass(
    lm: LanguageModel,
    prompts: list[str],
    steering: dict[int, torch.Tensor] | None,
    scale: float,
    target_layers: list[int],
    capture_specs: list[dict],
    max_seq_len: int,
) -> dict[str, Any]:
    """Run one non-autoregressive forward pass through ``lm`` on ``prompts``,
    apply steering at ``target_layers`` with strength ``scale``, and capture
    activations at each requested site.

    Returns
    -------
    dict
        ``{"attention_mask": Tensor[B, T],
           "<site_name>": {"pre": Tensor[B, T, d], "post": Tensor[B, T, d]}, ...}``
        where ``<site_name>`` is drawn from ``capture_specs[*]["site"]``. v1
        only supports ``"final"``; other sites raise ``NotImplementedError``
        so callers get a clear error instead of silent missing data.
    """
    _validate_steering_covers_layers(steering, scale, target_layers)
    enc = _tokenize_for_trace(lm, prompts, max_seq_len)
    layer_list = _get_layer_list(lm)
    final_norm = _get_final_norm(lm)
    dtype, device = _model_dtype_device(lm)
    tuple_output = _is_tuple_output_architecture(lm)

    # Proxies captured inside the trace will materialize after __exit__.
    proxies: dict[str, dict[str, Any]] = {}

    with lm.trace(enc):
        _add_steering_at_layers(
            layer_list, steering, scale, target_layers, dtype, device,
            tuple_output=tuple_output,
        )
        for spec in capture_specs:
            site = spec.get("site", "final")
            if site == "final":
                proxies["final"] = {
                    "pre": final_norm.input.save(),
                    "post": final_norm.output.save(),
                }
            else:
                raise NotImplementedError(
                    f"capture site {site!r} not implemented in v1; only "
                    "'final' is supported right now. Per-layer capture will "
                    "be added in a follow-up."
                )

    # Materialize the proxies into concrete CPU tensors.
    B, T = enc["attention_mask"].shape
    result: dict[str, Any] = {"attention_mask": enc["attention_mask"].cpu()}
    for site, pair in proxies.items():
        pre = pair["pre"].detach().cpu()
        post = pair["post"].detach().cpu()
        assert pre.ndim == 3, (
            f"Expected pre-RMSNorm shape [B, T, d], got {tuple(pre.shape)}. "
            "If ndim == 2 this likely means .output[0] selected batch element 0 "
            "instead of unpacking a tuple — check _is_tuple_output_architecture."
        )
        assert pre.shape[0] == B, (
            f"pre batch dim {pre.shape[0]} != expected {B}"
        )
        assert post.ndim == 3 and post.shape[0] == B, (
            f"post-RMSNorm shape mismatch: {tuple(post.shape)}, expected B={B}"
        )
        result[site] = {"pre": pre, "post": post}
    return result


# ---------------------------------------------------------------------------
# Public API: generate-with-steering for verification
# ---------------------------------------------------------------------------


def run_verification_forward_pass(
    lm: LanguageModel,
    prompts: list[str],
    steering: dict[int, torch.Tensor] | None,
    scale: float,
    target_layers: list[int],
    max_seq_len: int,
) -> dict[str, Any]:
    """Forward pass for ``scripts/bounds/01_verify_steering.py``.

    Captures three things the verification script needs that the main
    bounds recording path doesn't need:

    1. Pre-RMSNorm residual (same as ``run_bounds_forward_pass``), for the
       residual-delta cosine check against the steering direction.
    2. Next-token logits from ``lm_head.output``, for the KL-divergence
       sanity check between steered and unsteered distributions.
    3. Attention mask, for selecting the last real token per row.

    Kept separate from ``run_bounds_forward_pass`` so the bounds recording
    path stays minimal — verification runs are few and small, recording
    runs are many and large.
    """
    _validate_steering_covers_layers(steering, scale, target_layers)
    enc = _tokenize_for_trace(lm, prompts, max_seq_len)
    layer_list = _get_layer_list(lm)
    final_norm = _get_final_norm(lm)
    dtype, device = _model_dtype_device(lm)
    tuple_output = _is_tuple_output_architecture(lm)

    with lm.trace(enc):
        _add_steering_at_layers(
            layer_list, steering, scale, target_layers, dtype, device,
            tuple_output=tuple_output,
        )
        pre = final_norm.input.save()
        logits = lm.lm_head.output.save()

    return {
        "attention_mask": enc["attention_mask"].cpu(),
        "pre": pre.detach().cpu(),
        "logits": logits.detach().cpu(),
    }


def sample_from_steered_model(
    lm: LanguageModel,
    prompts: list[str],
    steering: dict[int, torch.Tensor] | None,
    scale: float,
    target_layers: list[int],
    max_new_tokens: int = 64,
    do_sample: bool = False,
) -> list[str]:
    """Generate text from ``lm`` with steering applied at every decode step.

    Uses ``lm.generate()`` combined with ``layer.all():`` so the intervention
    fires on the prompt forward AND on every subsequent autoregressive step —
    not just the first token. Shares ``_add_steering_at_layers`` with
    ``run_bounds_forward_pass`` so the two code paths are guaranteed to apply
    identical interventions.

    Returns
    -------
    list[str]
        Decoded generations, one per input prompt. If ``steering`` is ``None``
        or ``scale == 0.0``, the output is an unperturbed baseline generation.
    """
    _validate_steering_covers_layers(steering, scale, target_layers)

    tok = lm.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    layer_list = _get_layer_list(lm)
    dtype, device = _model_dtype_device(lm)
    tuple_output = _is_tuple_output_architecture(lm)

    with lm.generate(
        prompts, max_new_tokens=max_new_tokens, do_sample=do_sample
    ) as tracer:
        if steering is not None and scale != 0.0:
            with tracer.all():
                for layer_idx in target_layers:
                    s = (scale * steering[layer_idx]).to(dtype=dtype, device=device)
                    if tuple_output:
                        layer_list[layer_idx].output[0][:] = (
                            layer_list[layer_idx].output[0] + s
                        )
                    else:
                        layer_list[layer_idx].output[:] = (
                            layer_list[layer_idx].output + s
                        )
        out_ids = lm.generator.output.save()

    # Decode each row. out_ids shape: [B, prompt_len + max_new_tokens]
    decoded = [
        tok.decode(row, skip_special_tokens=True) for row in out_ids
    ]
    return decoded
