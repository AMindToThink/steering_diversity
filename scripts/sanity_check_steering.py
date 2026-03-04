#!/usr/bin/env python3
"""Sanity check: verify steering vector produces visibly different outputs.

Mirrors EasySteer's docker_test.py pattern — same prompt, temperature=0,
baseline vs steered side-by-side.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from vllm import LLM, SamplingParams  # type: ignore[import-untyped]
from vllm.steer_vectors.request import SteerVectorRequest  # type: ignore[import-untyped]

VECTOR_PATH = "EasySteer/vectors/happy_diffmean.gguf"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TARGET_LAYERS = list(range(10, 26))

PROMPTS = [
    "Alice's dog has passed away. Please comfort her.",
    "Describe the weather outside today.",
    "Tell me about your day so far.",
]

SCALES = [0, 2.0, 8.0, 32.0]


def main() -> None:
    llm = LLM(
        model=MODEL,
        enable_steer_vector=True,
        enforce_eager=True,
        enable_chunked_prefill=False,
    )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=200)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    for raw_prompt in PROMPTS:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

        print(f"\n{'='*70}")
        print(f"PROMPT: {raw_prompt}")
        print(f"{'='*70}")

        for scale in SCALES:
            request = SteerVectorRequest(
                steer_vector_name=f"happy_scale_{scale}",
                steer_vector_int_id=int(scale * 1000) + 1,
                steer_vector_local_path=VECTOR_PATH,
                scale=scale,
                target_layers=TARGET_LAYERS,
                prefill_trigger_tokens=[-1],
                generate_trigger_tokens=[-1],
            )
            output = llm.generate(text, steer_vector_request=request, sampling_params=sampling_params)
            response = output[0].outputs[0].text

            label = "BASELINE" if scale == 0 else f"SCALE={scale}"
            print(f"\n  [{label}]")
            print(f"  {response[:500]}")


if __name__ == "__main__":
    main()
