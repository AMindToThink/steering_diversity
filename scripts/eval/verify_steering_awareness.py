"""Verify eval-awareness steering by replicating MS paper's judge methodology.

Sends a subset of the MS triggers dataset to the served model, collects
responses, then uses GPT-4 via OpenRouter to judge model_awareness,
hypothetical_recognition, and task_performance.

Usage:
    # Against unsteered model on port 8017:
    uv run python scripts/eval/verify_steering_awareness.py \
        --base-url http://localhost:8017/v1 \
        --model-name "Qwen/Qwen3-32B" \
        --label unsteered --n 50

    # Against steered model:
    uv run python scripts/eval/verify_steering_awareness.py \
        --base-url http://localhost:8017/v1 \
        --model-name "qwen3-32b-steered-unaware-0.3" \
        --label steered --n 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

TRIGGERS_PATH = Path("Test_Awareness_Steering/data/triggers/triggers.json")
JUDGE_PROMPT_PATH = Path("Test_Awareness_Steering/prompts/judgePromptEvidenceBoth.txt")
OUTPUT_DIR = Path("outputs/steering_verification")


def load_triggers(n: int, seed: int) -> list[dict]:
    """Load and sample n trigger prompts from the MS dataset."""
    with open(TRIGGERS_PATH) as f:
        all_triggers = json.load(f)
    rng = random.Random(seed)
    if n >= len(all_triggers):
        return all_triggers
    return rng.sample(all_triggers, n)


def get_model_response(
    base_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """Get a response from the served model."""
    resp = httpx.post(
        f"{base_url}/chat/completions",
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def judge_response(
    judge_prompt_template: str,
    task: str,
    response: str,
    openrouter_key: str,
    judge_model: str = "openai/gpt-4.1-mini",
) -> dict:
    """Use GPT-4 via OpenRouter to judge the response.

    Returns dict with model_awareness, hypothetical_recognition,
    task_performance conclusions.
    """
    # The MS template expects {task}, {reasoning}, {answer}
    # Since we disabled thinking, there's no <think> block — put full
    # response as "answer" and leave reasoning empty.
    filled = judge_prompt_template.format(
        task=task,
        reasoning="(no reasoning provided — thinking mode disabled)",
        answer=response,
    )

    resp = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {openrouter_key}"},
        json={
            "model": judge_model,
            "messages": [{"role": "user", "content": filled}],
            "max_tokens": 1024,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        },
        timeout=60,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return json.loads(content)


def extract_conclusion(judge_result: dict, key: str) -> str:
    """Extract Yes/No/Maybe conclusion from judge result."""
    section = judge_result.get(key, {})
    conclusion = section.get("conclusion", "ERROR")
    return conclusion.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify eval-awareness steering via GPT-4 judge"
    )
    parser.add_argument("--base-url", required=True, help="vLLM base URL (with /v1)")
    parser.add_argument("--model-name", required=True, help="Model name for API calls")
    parser.add_argument("--label", required=True, help="Label for this run (e.g. 'unsteered', 'steered')")
    parser.add_argument("--n", type=int, default=50, help="Number of trigger prompts to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--judge-model", default="openai/gpt-4.1-mini", help="OpenRouter judge model")
    parser.add_argument("--prompt-type", default="hypothetical",
                        choices=["real", "hypothetical"],
                        help="Which prompt variant to use from triggers")
    args = parser.parse_args()

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_key:
        print("ERROR: OPENROUTER_API_KEY not set in environment or .env", file=sys.stderr)
        sys.exit(1)

    if not TRIGGERS_PATH.exists():
        print(f"ERROR: Triggers file not found: {TRIGGERS_PATH}", file=sys.stderr)
        sys.exit(1)

    judge_prompt_template = JUDGE_PROMPT_PATH.read_text()

    # Load triggers
    triggers = load_triggers(args.n, args.seed)
    print(f"Loaded {len(triggers)} trigger prompts (seed={args.seed})")

    # Collect responses
    results: list[dict] = []
    awareness_counts: dict[str, int] = {"Yes": 0, "No": 0, "Maybe": 0, "ERROR": 0}
    compliance_counts: dict[str, int] = {"Yes": 0, "No": 0, "Maybe": 0, "ERROR": 0}

    # Prepare output path for incremental saving
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{args.label}_{args.prompt_type}_n{len(triggers)}.json"

    def save_checkpoint() -> None:
        """Save current results to disk after each prompt."""
        with open(output_path, "w") as f_out:
            json.dump({
                "label": args.label,
                "model_name": args.model_name,
                "n_triggers": len(triggers),
                "prompt_type": args.prompt_type,
                "seed": args.seed,
                "judge_model": args.judge_model,
                "awareness_counts": awareness_counts,
                "compliance_counts": compliance_counts,
                "results": results,
            }, f_out, indent=2)

    for i, trigger in enumerate(triggers):
        prompt = trigger[args.prompt_type]
        category = trigger["category"]

        print(f"\n[{i+1}/{len(triggers)}] category={category}")
        print(f"  Prompt: {prompt[:80]}...")

        # Get model response
        try:
            response = get_model_response(
                args.base_url, args.model_name, prompt,
            )
        except Exception as e:
            print(f"  ERROR getting response: {e}")
            results.append({"idx": i, "category": category, "error": str(e)})
            save_checkpoint()
            continue

        print(f"  Response: {response[:120]}...")

        # Judge the response
        try:
            judge_result = judge_response(
                judge_prompt_template, prompt, response, openrouter_key,
                judge_model=args.judge_model,
            )
        except Exception as e:
            print(f"  ERROR judging: {e}")
            results.append({
                "idx": i, "category": category,
                "prompt": prompt, "response": response,
                "judge_error": str(e),
            })
            save_checkpoint()
            continue

        awareness = extract_conclusion(judge_result, "model_awareness")
        hypothetical = extract_conclusion(judge_result, "hypothetical_recognition")
        compliance = extract_conclusion(judge_result, "task_performance")

        awareness_counts[awareness] = awareness_counts.get(awareness, 0) + 1
        compliance_counts[compliance] = compliance_counts.get(compliance, 0) + 1

        print(f"  Awareness={awareness}  Hypothetical={hypothetical}  Compliance={compliance}")

        results.append({
            "idx": i,
            "category": category,
            "prompt": prompt,
            "response": response,
            "judge_result": judge_result,
            "awareness": awareness,
            "hypothetical": hypothetical,
            "compliance": compliance,
        })
        save_checkpoint()

    # Summary
    n_judged = sum(awareness_counts.values())
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {args.label} ({args.model_name})")
    print(f"{'=' * 60}")
    print(f"Prompts judged: {n_judged}/{len(triggers)}")
    print(f"\nAwareness:  {awareness_counts}")
    if n_judged > 0:
        aware_pct = (awareness_counts.get("Yes", 0) + awareness_counts.get("Maybe", 0)) / n_judged * 100
        print(f"  Aware (Yes+Maybe): {aware_pct:.1f}%")
    print(f"\nCompliance: {compliance_counts}")
    if n_judged > 0:
        comply_pct = compliance_counts.get("Yes", 0) / n_judged * 100
        print(f"  Compliant (Yes): {comply_pct:.1f}%")

    # Final save (already incrementally saved, but ensure final state is written)
    save_checkpoint()
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
