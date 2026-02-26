"""Prompt loading and EasySteer-based steered generation (GPU only)."""

from __future__ import annotations

from typing import Any

from datasets import load_dataset
from tqdm.auto import tqdm

from .config import ExperimentConfig
from .utils import load_contrastive_pairs, save_jsonl


def load_prompts(cfg: ExperimentConfig) -> list[str]:
    """Sample writing prompts from the configured HuggingFace dataset."""
    gen_cfg = cfg.generation
    ds = load_dataset(gen_cfg.prompt_dataset, split=gen_cfg.prompt_split)
    # Deterministically select prompts by slicing
    prompts: list[str] = [ds[i]["prompt"] for i in range(gen_cfg.num_prompts)]
    return prompts


def format_chat_prompt(text: str, model_name: str, system_prompt: str | None = None) -> str:
    """Wrap a user message in the model's chat template.

    For steering-vector extraction we use the tokenizer's chat template
    so the hidden-state positions align with real inference.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": text})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def compute_steering_vector(cfg: ExperimentConfig) -> Any:
    """Compute a DiffMean steering vector using EasySteer.

    Returns a ``StatisticalControlVector`` that can be exported to GGUF.
    Requires GPU (vLLM).
    """
    from vllm import LLM  # type: ignore[import-untyped]

    from easysteer.hidden_states import get_all_hidden_states_generate
    from easysteer.steer import DiffMeanExtractor

    pairs = load_contrastive_pairs(cfg.steering.contrastive_pairs_path)

    positive_prompts = [
        format_chat_prompt(p["positive"], cfg.model.name) for p in pairs
    ]
    negative_prompts = [
        format_chat_prompt(p["negative"], cfg.model.name) for p in pairs
    ]
    all_prompts = positive_prompts + negative_prompts

    llm = LLM(model=cfg.model.name, task="embed", enforce_eager=True)

    all_hidden_states, _ = get_all_hidden_states_generate(llm, all_prompts, max_tokens=1)

    vector = DiffMeanExtractor.extract(
        all_hidden_states=all_hidden_states,
        positive_indices=list(range(len(positive_prompts))),
        negative_indices=list(range(len(positive_prompts), len(all_prompts))),
        model_type=cfg.model.model_type,
        normalize=cfg.steering.normalize,
        token_pos=cfg.steering.token_pos,
    )
    return vector


def generate_steered_responses(cfg: ExperimentConfig, vector_path: str) -> list[dict[str, Any]]:
    """Generate responses at each steering scale.

    Returns a list of records ready to be saved as JSONL.
    Requires GPU (vLLM with steer-vector support).
    """
    from vllm import LLM, SamplingParams  # type: ignore[import-untyped]
    from vllm.steer_vectors.request import SteerVectorRequest  # type: ignore[import-untyped]

    prompts = load_prompts(cfg)
    gen_cfg = cfg.generation
    steer_cfg = cfg.steering

    sampling_params = SamplingParams(
        max_tokens=gen_cfg.max_tokens,
        temperature=gen_cfg.temperature,
        top_p=gen_cfg.top_p,
    )

    llm = LLM(
        model=cfg.model.name,
        enable_steer_vector=True,
        enforce_eager=True,
    )

    records: list[dict[str, Any]] = []

    for scale in steer_cfg.scales:
        sv_request = SteerVectorRequest(
            steer_vector_name=steer_cfg.concept,
            steer_vector_int_id=1,
            steer_vector_local_path=vector_path,
            scale=scale,
            target_layers=steer_cfg.target_layers,
            prefill_trigger_tokens=[-1],
            generate_trigger_tokens=[-1],
        )

        for prompt_idx, prompt in enumerate(tqdm(prompts, desc=f"scale={scale}")):
            chat_prompt = format_chat_prompt(prompt, cfg.model.name, system_prompt=gen_cfg.system_prompt)
            for response_idx in range(gen_cfg.responses_per_prompt):
                outputs = llm.generate(
                    chat_prompt,
                    steer_vector_request=sv_request,
                    sampling_params=sampling_params,
                )
                text = outputs[0].outputs[0].text
                records.append(
                    {
                        "prompt": prompt,
                        "prompt_idx": prompt_idx,
                        "response_idx": response_idx,
                        "scale": scale,
                        "response": text,
                    }
                )

    return records
