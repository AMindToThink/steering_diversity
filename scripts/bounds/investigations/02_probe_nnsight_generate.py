"""Probe: how to apply a steering intervention at every decode step
during ``lm.generate()``.

## Question we were answering

``run_bounds_forward_pass`` only does a single non-autoregressive forward
pass per batch — the intervention fires once, captures happen, trace exits.
But ``sample_from_steered_model`` needs to ``lm.generate()`` new tokens one
at a time, and EACH of those internal forward passes must apply the same
steering (otherwise the intervention only affects the prompt-processing
forward and decays as new tokens are generated).

So: how do we make an nnsight intervention fire on every decode step, not
just the first?

## What we found

The answer is the ``tracer.all()`` context manager (``lm.generate(...) as
tracer: with tracer.all(): ...``). Everything placed inside ``tracer.all()``
is re-executed on each forward pass during generation. Inside that block,
the same ``layer.output[0][:] = layer.output[0] + delta`` assignment pattern
works as it does for a single-pass trace.

Confirmed by running tiny-gpt2 at ``max_new_tokens=6``:
- Baseline (no intervention): ``"The quick brown fox factors factors factors factors factors factors"``
- Perturbed (+1000 at last block): ``"The quick brown fox stairs stairs stairs stairs stairs stairs"``
- ``differ: True``

## Deprecation note

nnsight 0.6 emits a ``DeprecationWarning`` saying ``model.all()`` has been
renamed to ``tracer.all()`` — we use the new form. An older form
``with lm.transformer.h[-1].all(): ...`` still works on single layers but
is deprecated for the multi-layer case.

## Why it still matters

This file documents the correct ``tracer.all()`` pattern that the
production function ``src/bounds/nnsight_runner.py::sample_from_steered_model``
uses. If a future nnsight version changes the per-step intervention API,
re-running this probe on tiny-gpt2 is the fastest way to confirm the new
convention. The production code currently at
``scripts/bounds/01_verify_steering.py`` relies on ``sample_from_steered_model``
to produce side-by-side verification samples; a break in ``tracer.all()``
would silently un-steer the verification run — which means the sentinel
pipeline's "trust no silent steering" policy depends on this pattern being
correct.

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
    tok = lm.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    prompt = "The quick brown fox"

    # Baseline generation (no intervention).
    with lm.generate(prompt, max_new_tokens=6) as tracer:
        out_baseline = lm.generator.output.save()
    print("baseline out shape:", tuple(out_baseline.shape))
    print("baseline decoded:", tok.decode(out_baseline[0], skip_special_tokens=True))

    # Generation with a huge additive intervention at last block.
    d = lm.config.n_embd
    delta = torch.zeros(d)
    delta[0] = 1000.0

    with lm.generate(prompt, max_new_tokens=6) as tracer:
        # Apply at every forward pass by using `.all()` on the layer.
        with lm.transformer.h[-1].all():
            lm.transformer.h[-1].output[0][:] = lm.transformer.h[-1].output[0] + delta
        out_perturbed = lm.generator.output.save()
    print("perturbed decoded:", tok.decode(out_perturbed[0], skip_special_tokens=True))
    print("differ:", not torch.equal(out_baseline, out_perturbed))


if __name__ == "__main__":
    main()
