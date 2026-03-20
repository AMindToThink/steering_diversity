# /// script
# dependencies = ["httpx"]
# ///
"""Benchmark steered generation throughput: eager vs CUDA graphs.

Sends N steered chat completion requests sequentially and reports
total wall time and tokens/sec.

Usage:
    uv run scripts/bench_steering_speed.py --url http://localhost:8017 \
        --vector /path/to/vector.gguf --n 20 --max-tokens 200
"""
from __future__ import annotations

import argparse
import time

import httpx


def run_bench(
    url: str,
    vector_path: str,
    scale: float,
    target_layers: list[int],
    n: int,
    max_tokens: int,
) -> dict:
    prompts = [
        "Describe a rainy Monday morning.",
        "Write a short story about a lost cat.",
        "Explain how a bicycle works.",
        "What makes a good cup of coffee?",
        "Describe the view from a mountaintop.",
    ]

    client = httpx.Client(timeout=300.0)
    total_tokens = 0
    t0 = time.perf_counter()

    for i in range(n):
        prompt = prompts[i % len(prompts)]
        resp = client.post(
            f"{url}/v1/chat/completions",
            json={
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.8,
                "steer_vector_request": {
                    "steer_vector_local_path": vector_path,
                    "scale": scale,
                    "target_layers": target_layers,
                    "algorithm": "direct",
                    "normalize": True,
                    "prefill_trigger_tokens": [-1],
                    "generate_trigger_tokens": [-1],
                },
            },
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Request {i} failed: {resp.status_code} {resp.text[:200]}")
        data = resp.json()
        total_tokens += data["usage"]["completion_tokens"]

    elapsed = time.perf_counter() - t0
    client.close()

    return {
        "n_requests": n,
        "total_tokens": total_tokens,
        "elapsed_s": round(elapsed, 2),
        "tokens_per_sec": round(total_tokens / elapsed, 1),
        "avg_latency_ms": round(elapsed / n * 1000, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark steered generation")
    parser.add_argument("--url", default="http://localhost:8017")
    parser.add_argument("--vector", required=True)
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument("--target-layers", type=int, nargs="+",
                        default=list(range(10, 26)))
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    # Warmup request
    print("Warming up...")
    run_bench(args.url, args.vector, args.scale, args.target_layers, n=2, max_tokens=50)

    print(f"Benchmarking: {args.n} requests, max_tokens={args.max_tokens}")
    result = run_bench(
        args.url, args.vector, args.scale, args.target_layers,
        args.n, args.max_tokens,
    )
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
