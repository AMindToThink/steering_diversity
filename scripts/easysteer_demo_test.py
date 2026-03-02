#!/usr/bin/env python3
"""Exact replica of EasySteer's docker_test.py using our local vector."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from vllm import LLM, SamplingParams  # type: ignore[import-untyped]
from vllm.steer_vectors.request import SteerVectorRequest  # type: ignore[import-untyped]

llm = LLM(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    enable_steer_vector=True,
    enforce_eager=True,
    tensor_parallel_size=1,
    enable_chunked_prefill=False,
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=128)
text = "<|im_start|>user\nAlice's dog has passed away. Please comfort her.<|im_end|>\n<|im_start|>assistant\n"
target_layers = list(range(10, 26))

baseline_request = SteerVectorRequest(
    "baseline", 1,
    steer_vector_local_path="EasySteer/vectors/happy_diffmean.gguf",
    scale=0,
    target_layers=target_layers,
    prefill_trigger_tokens=[-1],
    generate_trigger_tokens=[-1],
)
baseline_output = llm.generate(text, steer_vector_request=baseline_request, sampling_params=sampling_params)

happy_request = SteerVectorRequest(
    "happy", 2,
    steer_vector_local_path="EasySteer/vectors/happy_diffmean.gguf",
    scale=2.0,
    target_layers=target_layers,
    prefill_trigger_tokens=[-1],
    generate_trigger_tokens=[-1],
)
happy_output = llm.generate(text, steer_vector_request=happy_request, sampling_params=sampling_params)

print("=== BASELINE (scale=0) ===")
print(baseline_output[0].outputs[0].text)
print()
print("=== HAPPY (scale=2.0) ===")
print(happy_output[0].outputs[0].text)
