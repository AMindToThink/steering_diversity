#!/usr/bin/env python3
"""Sanity check: verify creativity steering vector produces visibly different outputs.

Uses the create vector from EasySteer's creative_writing replication.
Higher scale -> more creative/stylistic writing.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from vllm import LLM, SamplingParams  # type: ignore[import-untyped]
from vllm.steer_vectors.request import SteerVectorRequest  # type: ignore[import-untyped]

VECTOR_PATH = "EasySteer/replications/creative_writing/create.gguf"
MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
TARGET_LAYERS = list(range(16, 30))

PROMPTS = [
    "Write a story about a town.",
    "Describe the ocean at sunset.",
    "Tell me about a journey through the mountains.",
]

SCALES = [0.0, 1.5, 4.0]


def main() -> None:
    llm = LLM(
        model=MODEL,
        enable_steer_vector=True,
        enforce_eager=True,
        enable_chunked_prefill=False,
    )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

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
                steer_vector_name=f"creativity_scale_{scale}",
                steer_vector_int_id=int(scale * 1000) + 1,
                steer_vector_local_path=VECTOR_PATH,
                scale=scale,
                target_layers=TARGET_LAYERS,
                prefill_trigger_tokens=[-1],
                generate_trigger_tokens=[-1],
                algorithm="direct",
            )
            output = llm.generate(text, steer_vector_request=request, sampling_params=sampling_params)
            response = output[0].outputs[0].text

            label = "BASELINE" if scale == 0 else f"SCALE={scale}"
            print(f"\n  [{label}]")
            print(f"  {response[:500]}")


if __name__ == "__main__":
    main()
