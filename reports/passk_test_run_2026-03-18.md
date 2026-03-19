# Pass@k Evaluation Infrastructure — Test Run Report

**Date:** 2026-03-18
**Author:** Matthew (with Claude)
**Model:** Qwen2.5-1.5B-Instruct (unsteered baseline)
**GPU:** Quadro RTX 8000 (48GB), single GPU
**Dataset:** HumanEval+ (164 problems)

## What we built

A pass@k evaluation pipeline that measures whether activation steering collapses LLM output diversity through a capability lens. The architecture:

```
EvalPlus  →  Steering Proxy (:8018)  →  EasySteer vLLM (:8017)
              (injects extra_body)        (applies steering)
```

New code (all tests passing, 39 new tests + 47 existing = 86 total):

| File | Purpose |
|------|---------|
| `src/pass_at_k.py` | Unbiased pass@k estimator for arbitrary k |
| `src/eval_config.py` | Config that inherits from existing steering YAMLs |
| `src/steering_proxy.py` | FastAPI proxy injecting steering into OpenAI requests |
| `scripts/eval/eval_code.py` | Orchestrator: loop over scales → proxy → EvalPlus |
| `scripts/eval/plot_pass_at_k.py` | Visualization: pass@k curves + crossover analysis |

## Test run results

**Configuration:** n=10 samples per problem, temperature=0.8, no steering (scale=0.0), full HumanEval+ (164 problems).

### Baseline pass@k (Qwen2.5-1.5B-Instruct, unsteered)

| Metric | HumanEval (base tests) | HumanEval+ (base + extra tests) |
|--------|------------------------|----------------------------------|
| pass@1 | 0.432 | 0.386 |
| pass@10 | 0.756 | 0.713 |

These numbers are reasonable for a 1.5B model. The ~5 percentage point drop from base→plus shows HumanEval+ catches some false positives, as expected.

### Timing breakdown

| Step | Time | Rate |
|------|------|------|
| vLLM startup (cached) | ~1s | — |
| Codegen (164 × 10 samples) | 641s (10.7 min) | ~0.39s/sample |
| Evaluate (1640 solutions) | 66s (1.1 min) | ~0.04s/sample |
| **Total per condition** | **~12 min** | — |

Codegen is 91% of wall time. Evaluation is cheap.

### Lessons learned

1. **EvalPlus requires all problems present.** The `--id-range` flag generates only a subset but `evalplus.evaluate` asserts all 164 problems exist in the samples file. No way to eval a subset — must always generate full dataset.

2. **EvalPlus CLI uses positional args** for model and dataset (not `--model`/`--dataset`), and `--id-range` takes a JSON list string like `"[0,5]"`. Our `eval_code.py` orchestrator script needs to match this interface — the current implementation uses `--model`/`--dataset` flags which won't work. This needs fixing.

3. **Codegen throughput is bottlenecked by sequential problem processing.** EvalPlus sends one problem at a time in batches of n samples. At 0.39s/sample this scales linearly with n.

4. **vLLM at 50% GPU memory utilization is sufficient** for Qwen2.5-1.5B-Instruct. The model is small enough that we could potentially run a second process on the same GPU, though this won't help since we're bottlenecked on sequential generation.

## Cost estimates for full experiments

### Qwen2.5-1.5B-Instruct (dev model)

Based on 0.39s/sample, 164 problems per dataset:

| Configuration | Codegen time | Eval time | Total |
|---------------|-------------|-----------|-------|
| n=10, 1 scale, HumanEval | 10.7 min | 1.1 min | ~12 min |
| n=10, 5 scales, HumanEval | 53 min | 5.5 min | ~1 hr |
| n=50, 5 scales, HumanEval | 4.4 hr | ~28 min | ~5 hr |
| n=100, 5 scales, HumanEval | 8.9 hr | ~55 min | ~10 hr |
| n=200, 5 scales, HumanEval | 17.8 hr | ~1.8 hr | ~20 hr |
| n=200, 5 scales, HumanEval + MBPP | 35.6 hr | ~3.6 hr | ~40 hr |

### Qwen3-30B-A3B (final model)

Unknown throughput — MoE architecture may be faster or slower per token than dense 1.5B depending on vLLM's MoE efficiency. Needs benchmarking. Rough guess: 2-5× slower per sample, so multiply codegen times accordingly.

## Next steps

### Immediate (before running steered experiments)

1. **Fix `eval_code.py` CLI interface** to match EvalPlus's actual positional-arg syntax. The current script uses `--model`/`--dataset` flags that don't exist.

2. **Decide on n for the dev model.** n=200 is expensive (~20hr for 5 scales on HumanEval alone). Consider:
   - n=50 for initial exploration (~5hr, gives pass@k up to k=50)
   - n=100 for the paper (~10hr, smooth curves up to k=100)
   - n=200 only if we need pass@k at k=200

3. **Compute a steering vector** using the existing pipeline (step 01) before we can test the proxy + steering end-to-end.

### Short-term

4. **Validate the proxy end-to-end** with EasySteer's vLLM fork and a real steering vector. Key question: does EasySteer's vLLM accept `steer_vector_request` via the proxy's injection correctly?

5. **Run the dev smoke test** (`configs/eval_code_dev.yaml`) through the full orchestrator to validate the automation.

6. **Benchmark Qwen3-30B-A3B** throughput to get accurate time estimates for the final model.

### Medium-term

7. **Run full experiment** on the dev model (Qwen2.5-1.5B-Instruct) across multiple scales to verify the crossover phenomenon exists.

8. **Add MBPP** once HumanEval results look good.

9. **Run on final model** (Qwen3-30B-A3B) for paper-quality results.
