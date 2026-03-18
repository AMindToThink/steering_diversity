# Evaluating Output Diversity via pass@k

## Motivation

We want to measure how activation steering affects the diversity of a model's output distribution. The eventual application is eval-awareness steering (steering a model to not recognize it's being evaluated, then checking whether this degrades general capabilities). But the current task is just to build the pass@k evaluation infrastructure: given a model served via vLLM, measure its pass@k across code, math, and fuzzy benchmarks. The steering integration comes later.

## The Idea

Use pass@k as a diversity metric across three evaluation domains. If steering compresses the model's output distribution, this shows up as a changed pass@k curve shape — even if pass@1 is preserved. The pass@k curve reflects the extent to which the model has the ability to output a diverse set of strategies for solving different tasks.

## Three Domains, Unified Metric

### Code (binary correctness)
- **Benchmarks:** HumanEval+, MBPP+ (via EvalPlus — 80× / 35× more tests than originals)
- **Scoring:** automatic test execution, binary pass/fail
- **Tooling:** Serve steered model via vLLM, point EvalPlus at it. EvalPlus handles generation + scoring end-to-end with `--backend vllm`. Save all generations for later analysis with other diversity metrics.
- **pass@k:** standard unbiased estimator (Chen et al. 2021)

### Math (binary correctness via answer matching)
- **Benchmarks:** MATH-500, GSM8K, possibly LiveMathBench
- **Scoring:** exact-match answer checking
- **Tooling:** Serve steered model via vLLM, run lm-eval-harness against it with `repeats` + `--log_samples`. Compute pass@k post-hoc from saved samples.
- **pass@k:** same estimator, applied to answer correctness

### Broad/Fuzzy Tasks (judge-scored)
- **Benchmark:** WildBench (1,024 diverse real-world tasks, ICLR 2025)
- **Scoring:** LLM judge (Gemini 2.5 Flash via OpenRouter — very cheap, fast, configurable thinkingBudget) with task-specific checklists (scores 1–10). Threshold at quality bar → binary pass/fail.
- **pass@k:** same estimator, with "pass" defined by judge score ≥ threshold
- **Why WildBench:** diversity enforced by construction (cosine similarity filtering, one conversation per user), covers writing/coding/math/roleplay/planning, easy tasks filtered out, 0.98 Pearson with Chatbot Arena

For the fuzzy domain, the quality threshold that defines "pass" is a free parameter. Report results across multiple thresholds.

## What pass@k Captures

Aggregate pass@k across a benchmark reflects the *distribution* of per-problem pass rates. Jensen's inequality means two models with the same average pass@1 but different per-problem pass-rate distributions will have different pass@k curves. Steering that concentrates mass on fewer problems (or fewer strategies) reshapes this distribution, which shows up as a changed curve shape.

Within a single prompt, pass@k is sensitive to diversity because a model locked into one output strategy has pass@k ≈ pass@1 (that strategy either clears the bar or doesn't), while a model producing varied approaches has pass@k > pass@1 on prompts where some approaches work and others don't.

## Related Work

- **RLVR pass@k crossover** (Yue et al., Li et al. 2025) — RL-trained models outperform base models at pass@1 but underperform at large k. Demonstrates that training interventions reshape the pass@k curve. This is the key precedent: we predict analogous curve reshaping from activation steering.
- **The Elicitation Game** (Hofstätter et al., 2025) — compares prompting, activation steering, and finetuning for eliciting hidden capabilities. Focuses on success rate, not diversity. Our paper fills this gap.
- **Chen et al. 2021 (Codex paper)** — introduces the unbiased pass@k estimator.
- **Wehner et al. 2025 (RepE survey, TMLR)** — reviewed 130+ steering papers; none systematically compare output diversity across elicitation methods.

## Practical Notes

- **Temperature:** must be fixed and justified (RLVR literature uses 0.6–1.0). Show robustness across a range.
- **Judge consistency (fuzzy domain):** Gemini 2.5 Flash with thinkingBudget=0 and temperature 0 for deterministic scoring. Verify consistency (same response → same score across runs) with a small reliability test before the full experiment.
- **Threshold sensitivity (fuzzy domain):** report results across multiple thresholds.
- **Steering:** EasySteer library for applying CAA vectors through vLLM.
- **Generation:** Serve steered model via EasySteer + vLLM, use each domain's standard tooling (EvalPlus, lm-eval-harness, WildBench) pointed at the vLLM endpoint. n=200 samples per problem. Save all raw generations across all domains for later analysis with other diversity metrics.
- **Models:** 8B / 3B-active models (qwen3-30b-a3b primary), so n=200 is slow but doable.
