# PR: Server-level steering with CUDA graphs — 2.3x speedup

**From:** `AMindToThink/EasySteer-vllm-v1` branch `feat/steer-allow-cuda-graphs`
**To:** `ZJU-REAL/EasySteer-vllm-v1` branch `main`
**Status:** Ready for review

## Summary

Steered generation is **2.3x slower** than unsteered because `enforce_eager=True` disables CUDA graphs. The root cause: CUDA graph capture happens at startup before any steering vector is loaded, so graphs record the identity path (no steering). During replay, steering is silently skipped during decode.

This PR adds **server-level steering**: the steering vector is configured at startup and loaded before CUDA graph capture. Graphs record `hidden_states + vector`, so steering is applied correctly during decode replay.

Key results on Quadro RTX 8000, Qwen2.5-1.5B-Instruct:
- **2.34x throughput improvement** (40.9 → 95.6 tok/s)
- **0.000000 logprob difference** between eager and CUDA graph modes (verified across 1200 tokens)
- Per-request `steer_vector_request` is rejected when server-level steering is active (prevents silent config conflicts)
- Runtime scale changes via `POST /v1/steering` with bit-identical logprob restoration

## User experience

```bash
# Start server with fixed steering config + CUDA graphs
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --steer-vector-path vectors/happy.gguf \
  --steer-scale 4.0 \
  --steer-target-layers 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 \
  --steer-normalize \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill

# Requests need no steer_vector_request — steering is applied server-side
curl http://localhost:8000/v1/chat/completions \
  -d '{"model": "Qwen/Qwen2.5-1.5B-Instruct", "messages": [{"role": "user", "content": "Hello"}]}'

# Change scale at runtime
curl -X POST http://localhost:8000/v1/steering -d '{"scale": 2.0}'

# Check current config
curl http://localhost:8000/v1/steering
```

## Correctness enforcement

Steering is incompatible with chunked prefill and prefix caching. This PR adds **hard errors** (not warnings) if either is enabled with steering:

- **Prefix caching** keys on `steer_vector_name` but not scale — silently reuses KV states across different scales, disabling steering. This caused a real data-invalidation bug (commit `9b999cb`).
- **Chunked prefill** is not supported by EasySteer's steering wrappers.

The server now raises `ValueError` at startup if either is detected.

## Files changed (10 modified, 4 new)

**Config & CLI:**
- `vllm/config/steer_vector.py` — Server-level fields (`server_vector_path`, `server_scale`, `server_target_layers`, `server_algorithm`, `server_normalize`) + `has_server_config` property
- `vllm/engine/arg_utils.py` — CLI flags (`--steer-vector-path`, `--steer-scale`, `--steer-target-layers`, `--steer-algorithm`, `--steer-normalize`). `--steer-vector-path` implies `--enable-steer-vector` and `--steer-allow-cuda-graphs`
- `vllm/config/vllm.py` — Hard errors for chunked prefill / prefix caching with steering

**Startup loading:**
- `vllm/v1/worker/steer_vector_model_runner_mixin.py` — `_maybe_load_server_steer_vector()` loads the vector during model init, before CUDA graph capture

**API:**
- `vllm/entrypoints/openai/serving_engine.py` — `_maybe_get_steer_vector()` raises `ValueError` when per-request steering conflicts with server config
- `vllm/entrypoints/openai/api_server.py` — `GET/POST /v1/steering` admin endpoints with `asyncio.Lock` for safe concurrent access
- `vllm/v1/engine/async_llm.py` — `add_steer_vector()` async method

**Buffer strategy:**
- `vllm/steer_vectors/algorithms/template.py` — In-place `copy_()` in `set_steer_vector()` and `set_active_tensor()` preserves tensor addresses for CUDA graph replay. Runtime scale changes compute `vector * new_scale` and copy into the same buffer.

**Docs & tests:**
- `docs/design/cuda_graphs.md` — Server-level steering docs, verification script reference, corrected divergence claims
- `tests/basic_correctness/test_steer_vector_cuda_graphs.py` — Updated with server-level and per-request test paths, tolerance tightened to 1e-4
- `tests/basic_correctness/test_server_level_steering.py` — **New**: 12 unit tests (config, CLI args, buffer reuse)
- `scripts/verify_steering_correctness.py` — **New**: Pre-experiment verification script (4 checks)
- `scripts/bench_eager_vs_cudagraphs.py` — **New**: End-to-end benchmark

## Reproducing correctness

### Verification script (recommended before any experiment)

```bash
cd EasySteer/vllm-steer
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/verify_steering_correctness.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --vector /absolute/path/to/happy_diffmean.gguf \
    --target-layers 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
```

Runs 4 checks:

| Check | What it verifies |
|-------|-----------------|
| Chunked prefill ON vs OFF (plain) | Disabling chunked prefill is safe for your model |
| Plain vs scale=0 steering | Steering infrastructure is transparent when inactive |
| Steered eager vs CUDA graphs | CUDA graphs don't skip the steering intervention |
| scale=4 vs scale=0 | Steering actually changes output (sanity check) |

Our results (Qwen2.5-1.5B, RTX 8000, FlashInfer, float16):
```
  [PASS] Chunked prefill ON vs OFF (plain, no steering): max_diff=0.000000 (tol=0.0001)
  [PASS] Plain vs scale=0 steering (both chunked OFF): max_diff=0.000000 (tol=0.0001)
  [PASS] Steered eager vs CUDA graphs (scale=4.0): max_diff=0.000000 (tol=0.0001)
  [PASS] Steering has effect (scale=4.0 vs scale=0): outputs differ
*** ALL 4 CHECKS PASSED — safe to run experiments ***
```

### POST /v1/steering integration test

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tests/basic_correctness/demo_post_steering.py
```

Tests: initial config → generate at scale=4 → POST scale=0 (output changes) → POST scale=4 (logprobs restored to 0.000000 diff) → per-request rejection → structural change rejection. All 7 checks pass.

### Unit tests (no GPU required)

```bash
.venv/bin/python -m pytest tests/basic_correctness/test_server_level_steering.py -v
```

12 tests covering SteerVectorConfig fields, EngineArgs parsing, and AlgorithmTemplate buffer reuse.

## Reproducing performance

```bash
cd EasySteer/vllm-steer
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/bench_eager_vs_cudagraphs.py \
    --vector /absolute/path/to/happy_diffmean.gguf \
    --n 20 --max-tokens 200
```

Starts both servers sequentially, sends 20 sequential requests to each, compares throughput.

Our results (Quadro RTX 8000, Qwen2.5-1.5B-Instruct):

|                    | Eager (per-request) | CUDA graphs (server-level) | Speedup |
|--------------------|--------------------:|---------------------------:|--------:|
| Wall time          | 96.5s               | 41.3s                      | **2.34x** |
| Throughput (tok/s) | 40.9                | 95.6                       | **2.34x** |
| Avg latency (ms)   | 4826                | 2063                       | **2.34x** |

## What we are confident about

- **Bit-identical logprobs** between eager and CUDA graph modes with server-level steering (0.000000 diff across 1200+ tokens at temperature=0)
- **scale=0 transparency**: steered vLLM at scale=0 is bit-identical to plain vLLM (200 tokens, 5 prompts)
- **POST /v1/steering** correctly updates scale at runtime with exact logprob restoration
- **Buffer `copy_()` strategy** preserves tensor addresses through CUDA graph replay
- **Hard error enforcement** prevents silent correctness bugs from prefix caching and chunked prefill

## What we are less confident about

- **Other configurations**: Only verified on Qwen2.5-1.5B-Instruct / Quadro RTX 8000 / FlashInfer / float16. The verification script (`scripts/verify_steering_correctness.py`) should be run on any new model/GPU/backend combination before experiments.
- **Concurrent POST + generation**: The `asyncio.Lock` serializes POST calls, but doesn't block in-flight generation. We believe `copy_()` is atomic from the GPU perspective, but haven't stress-tested this.
- **Server vector ID**: Hardcoded `steer_vector_int_id=1` for the server vector. No collision risk in normal use (per-request steering is rejected), but not guaranteed by design.

## How it works

1. `--steer-vector-path` implies `--enable-steer-vector` and `--steer-allow-cuda-graphs`
2. `VllmConfig.__init__` enforces: `enforce_eager=False`, `CompilationMode.NONE`, `CUDAGraphMode.FULL_DECODE_ONLY`. Raises errors if chunked prefill or prefix caching are enabled.
3. During model loading, `_wrap_model_with_steer_vectors()` calls `_maybe_load_server_steer_vector()`, which builds a `SteerVectorRequest` from the server config and loads the vector via `add_adapter()`. This happens **before** CUDA graph capture.
4. `AlgorithmTemplate.set_steer_vector()` uses `clone()` for the first load and `copy_()` for subsequent loads (same shape), preserving the buffer address for CUDA graph replay.
5. CUDA graph capture records `hidden_states + vector` at the steered layers. On replay, the steering addition executes as part of the captured graph.
6. `POST /v1/steering` reloads the vector with the new scale. `copy_()` into the existing buffer updates the graph's data without re-capture.
