"""Probe: what is nnsight 0.6's API shape on a decoder layer?

## Question we were answering

When building ``src/bounds/nnsight_runner.py``, we needed to know exactly
how nnsight 0.6 exposes the residual stream at each decoder layer. The
specific questions:

1. What does ``lm.transformer.h[i].output`` (GPT-2) or
   ``lm.model.layers[i].output`` (Qwen2/Llama3) actually return — a tensor
   or a tuple?
2. Can we capture ``layer_norm.input`` as well as ``layer_norm.output``?
3. Does an in-place modification like ``layer.output[0][:] = ...`` actually
   propagate through the forward pass?
4. How do we batch multiple prompts in a single ``lm.trace()`` call?

## What we found

1. **On tiny-gpt2 (GPT-2 style), ``layer.output`` is a tuple
   ``(hidden_states, ...)`` and ``.output[0]`` unpacks it to the
   ``[B, T, d]`` hidden state.** Confirmed by tracing one prompt and
   observing ``block[-1].output[0].shape = (1, 4, 2)``.
2. **``layer_norm.input`` and ``layer_norm.output`` are both saveable.**
   Shapes match what we'd expect for ``[B, T, d]`` residual stream.
3. **In-place modification via ``.output[0][:] = .output[0] + delta`` DOES
   propagate** — we observe a nonzero difference in the final layernorm
   output when the last block's residual is perturbed by +1000.
4. **Batched input via ``lm.trace(prompts, padding=True, ...)`` fails** with
   ``KeyError: 'padding'`` because nnsight already injects
   ``padding=True`` to the tokenizer internally. **Fix: tokenize externally
   and pass a dict** ``{"input_ids": ..., "attention_mask": ...}``.

## The bug this investigation DID NOT catch (but was close to)

On Qwen2/Llama3 under ``transformers >= 4.40``, ``layer.output`` is a
**bare tensor** ``[B, T, d]``, NOT a tuple. So ``layer.output[0]`` selects
batch element 0, not "the hidden state out of a tuple." That single bug
silently contaminated the first 5 full GPU runs of experiment 1 — every
prompt except batch index 0 was **unsteered**. It was caught much later,
while trying to add per-layer capture to the diagnostic, by yet another
shape-mismatch probe.

The lesson: this script should have been re-run against a real Qwen model
(not just tiny-gpt2) before any production code relied on
``.output[0]``. The GPT-2 success gave false confidence.

## Why it still matters

This file serves as the regression oracle for the nnsight API shape on
GPT-2. If a future ``transformers`` upgrade changes the DecoderLayer
return convention (tuple ↔ bare tensor), re-running this script is the
fastest way to catch it. The production code that depends on this finding
is:

- ``src/bounds/nnsight_runner.py::_is_tuple_output_architecture`` — the
  dispatcher introduced to handle the GPT-2 vs Qwen2/Llama3 divergence.
- ``src/bounds/nnsight_runner.py::_add_steering_at_layers`` — branches on
  the dispatcher to use ``.output[0]`` or ``.output``.
- ``tests/bounds/test_nnsight_runner.py::test_steering_affects_every_batch_element`` —
  regression test that would catch a re-introduction of the batch-indexing
  bug.

Runs on CPU in <5 seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from nnsight import LanguageModel

# 4 levels up: investigations/ → bounds/ → scripts/ → PROJECT_ROOT
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> None:
    lm = LanguageModel("sshleifer/tiny-gpt2", device_map="cpu")

    print("type(lm):", type(lm).__name__)
    print("transformer.h len:", len(lm.transformer.h))
    print("config hidden:", lm.config.n_embd, "n_layer:", lm.config.n_layer)
    print()

    # Probe 1: trace with save on block output + final layernorm output.
    with lm.trace("hello world there friend"):
        h_last = lm.transformer.h[-1].output[0].save()
        ln_out = lm.transformer.ln_f.output.save()
    print("block[-1].output[0].shape:", tuple(h_last.shape))
    print("ln_f.output.shape:", tuple(ln_out.shape))

    # Probe 2: try to capture final layernorm input — exception (if any) will
    # be raised when the trace executes, i.e. on __exit__.
    try:
        with lm.trace("hello world there friend"):
            ln_in_proxy = lm.transformer.ln_f.input.save()
        try:
            shape_or_repr = tuple(ln_in_proxy.shape)
        except Exception:
            shape_or_repr = f"not-a-tensor, repr={repr(ln_in_proxy)[:200]}"
        print("ln_f.input capture:", shape_or_repr)
    except Exception as e:
        print(f"ln_f.input.save() failed: {type(e).__name__}: {str(e)[:200]}")

    # Probe 3: does in-place modification at a block output propagate?
    with lm.trace("hello world there friend"):
        baseline = lm.transformer.ln_f.output.save()

    with lm.trace("hello world there friend"):
        delta = torch.full((lm.config.n_embd,), 1000.0)
        lm.transformer.h[-1].output[0][:] = lm.transformer.h[-1].output[0] + delta
        perturbed = lm.transformer.ln_f.output.save()

    diff = (perturbed - baseline).abs().max().item()
    print(f"max |perturbed − baseline| after +1000 at last block: {diff}")

    # Probe 4: tokenize externally and pass dict input (nnsight already calls
    # tokenizer(padding=True) internally so passing padding=True in trace()
    # collides; external tokenization avoids that).
    tok = lm.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = ["short", "a somewhat longer sentence for testing padding"]
    enc = tok(prompts, padding=True, truncation=True, max_length=16, return_tensors="pt")
    print(f"attention_mask from tokenizer: {enc['attention_mask'].tolist()}")

    with lm.trace(
        {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    ):
        h = lm.transformer.h[-1].output[0].save()
        ln_in = lm.transformer.ln_f.input.save()
        ln_out = lm.transformer.ln_f.output.save()
    print(f"batched via dict input:")
    print(f"  last block out: {tuple(h.shape)}")
    print(f"  ln_f input:     {tuple(ln_in.shape)}")
    print(f"  ln_f output:    {tuple(ln_out.shape)}")


if __name__ == "__main__":
    main()
