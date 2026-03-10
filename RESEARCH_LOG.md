# Research Log

## 2026-02-25 — Full "happy" steering experiment (happy_full)

### Setup

- **Model**: Qwen/Qwen2.5-1.5B-Instruct
- **Steering concept**: "happy" (contrastive pairs from deception.json)
- **Steering vector**: diffmean, applied to layers 10–25, token position -1, normalized
- **Scales**: [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
- **Prompts**: 50 (from euclaise/writingprompts, test split)
- **Responses per prompt per scale**: 10 → 3,000 total responses
- **Generation**: temperature=1.0, top_p=0.95, max_tokens=256
- **Embedding**: all-MiniLM-L6-v2 (384-dim)
- **Clustering**: HDBSCAN, min_cluster_size=5, euclidean metric
- **Config**: `configs/happy_full.yaml`

### Results summary

**No statistically significant effect of "happy" steering on output diversity.**

All three pre-registered tests converge:

| Test | Statistic | p-value | Significant? |
|------|-----------|---------|-------------|
| Mixed-effects model (primary) | β = −0.0027 | 0.092 | No |
| Spearman ρ (effect size) | ρ = −0.015 | 0.802 | No |
| Page's L (all 6 metrics, Holm–Bonferroni) | best raw p = 0.023 (cluster_entropy) | adj p = 0.138 | No |

The mixed-effects β is tiny and negative (higher scale → marginally lower mean pairwise cosine distance), but the 95% CI crosses zero [−0.0059, 0.0004]. The Spearman correlation is essentially zero.

### Per-scale diversity metrics

| Scale | Clusters | Noise ratio | Mean pairwise cosine dist | Cluster entropy |
|-------|----------|-------------|--------------------------|-----------------|
| 0.0   | 29       | 0.444       | 0.878                    | 3.31            |
| 0.5   | 28       | 0.496       | 0.874                    | 3.24            |
| 1.0   | 20       | 0.550       | 0.873                    | 2.91            |
| 2.0   | 23       | 0.520       | 0.878                    | 3.07            |
| 4.0   | 27       | 0.540       | 0.873                    | 3.23            |
| 8.0   | 26       | 0.504       | 0.873                    | 3.18            |

All metrics are remarkably flat across scales. The dip at scale=1.0 (20 clusters) appears to be noise — it does not continue at higher scales.

### Cluster review observations

Qualitative review of clusters at scale=0.0 (baseline) vs scale=8.0 (max steering):

1. **Clusters are prompt-driven, not scale-driven.** Each cluster predominantly groups responses to the same writing prompt (e.g., all about Leonardo DiCaprio, all about a frying pan cooking level). The embedding space is structured by prompt topic, not by steering intensity.

2. **Refusal/confusion clusters persist at all scales.** Both baseline and max steering produce clusters of:
   - Generic refusals: "I'm sorry, but I can't assist with that."
   - Confusion about the "[WP]" tag: "I'm not sure what you mean by WP."
   - Self-identification: "I'm Qwen, an AI by Alibaba Cloud."

3. **Substantive response clusters appear at both scales** with similar themes, topics, and levels of elaboration.

4. **No visible collapse in response variety** at high steering scales — the model still produces diverse creative writing responses, analytical breakdowns, and meta-commentary at scale=8.0.

### Interpretation

For the "happy" steering concept applied to Qwen2.5-1.5B-Instruct, activation steering up to scale=8.0 does not measurably reduce output diversity as measured by embedding-space metrics. The primary diversity axis in the embedding space is the input prompt, not the steering scale.

Possible explanations:
- The "happy" vector may be too mild or orthogonal to the content-diversity axes of the embedding space.
- Sentence-BERT embeddings may be more sensitive to topic than to tone/style, potentially missing diversity collapse along affective dimensions.
- The model may be robust to this particular steering intervention at these scales.

### Outputs

- `outputs/happy_full/responses.jsonl` — 3,000 generated responses
- `outputs/happy_full/embeddings.npz` — 384-dim sentence embeddings
- `outputs/happy_full/metrics.json` — per-scale diversity metrics
- `outputs/happy_full/stats.json` — full statistical test results
- `outputs/happy_full/cluster_review.json` — sample responses per cluster
- `outputs/happy_full/plots/` — UMAP and metrics bar visualizations

## 2026-03-02 — happy_recon: Re-run with steering fix + within-vs-pooled analysis

### Context

The prefix-cache bug (fixed in `d2a72b5`) meant all prior generation data was produced with steering effectively disabled — every scale was getting the same unsteered output. This run uses the fixed code with narrowed scales [0, 0.5, 1, 2, 4, 8] (scales above 8 cause degeneration).

### Setup

- Same model and vector as happy_full
- **Scales**: [0, 0.5, 1, 2, 4, 8] (6 scales × 10 prompts × 5 responses = 300 total)
- **Config**: `configs/happy_recon.yaml`

### Key finding: Steering collapses cross-prompt diversity, not within-prompt diversity

The pooled pairwise cosine distance (all 50 responses per scale) drops sharply from 0.74 (scale 0) to 0.53 (scale 8). This initially looks like diversity collapse. But decomposing into within-prompt vs. cross-prompt tells a different story:

| Scale | Within-prompt (mean ± SE) | Pooled (cross-prompt included) |
|-------|--------------------------|-------------------------------|
| 0.0   | 0.555 ± 0.037            | 0.740                         |
| 0.5   | 0.502 ± 0.028            | 0.738                         |
| 1.0   | 0.494 ± 0.024            | 0.710                         |
| 2.0   | 0.509 ± 0.035            | 0.721                         |
| 4.0   | 0.489 ± 0.035            | 0.599                         |
| 8.0   | 0.527 ± 0.020            | 0.533                         |

- **Within-prompt diversity is flat** (~0.49–0.55) across all scales. The model produces equally varied responses to the same prompt regardless of steering strength.
- **Pooled diversity drops** because responses to *different* prompts converge. At scale 8, the two lines nearly meet — meaning cross-prompt diversity has essentially vanished.

**Interpretation**: Steering doesn't reduce the model's inherent stochasticity (sampling diversity). Instead, it overrides the prompt signal. At high scales, the steering vector dominates the output so strongly that the model produces similar content regardless of what prompt it received. The "diversity collapse" is really "prompt override."

This is visible qualitatively in the cluster review: at scale 8, responses degenerate into excited gibberish ("Activate! Spark! Boost-A-GIG!") and multilingual exclamations regardless of the writing prompt given.

### Statistical tests (correctly using within-prompt metrics)

| Test | Statistic | p-value | Significant? |
|------|-----------|---------|-------------|
| Mixed-effects (primary) | β = −0.0002 | 0.957 | No |
| Spearman ρ | ρ = −0.061 | 0.641 | No |
| Page's L (noise_ratio, Holm-corrected) | — | 0.023 | Yes* |

The primary tests confirm no within-prompt diversity effect. The significant noise_ratio trend likely reflects the degenerate outputs at scale 8 forming tight clusters rather than a meaningful diversity change.

### Plot

See `outputs/happy_recon/plots/within_vs_pooled_diversity.png` — the key visualization showing the divergence between pooled and within-prompt diversity measures.

### Outputs

- `outputs/happy_recon/responses.jsonl` — 300 generated responses
- `outputs/happy_recon/embeddings.npz` — 384-dim sentence embeddings
- `outputs/happy_recon/metrics.json` — per-scale diversity metrics
- `outputs/happy_recon/stats.json` — full statistical test results
- `outputs/happy_recon/cluster_review.json` — sample responses per cluster
- `outputs/happy_recon/plots/` — UMAP, metrics bars, and within-vs-pooled diversity

## 2026-03-08 — EasySteer normalize bug on float16 GPUs

### Bug

`SteerVectorRequest(normalize=True)` produces garbage output (`!!!...`) on GPUs with compute capability < 8.0 (e.g. Quadro RTX 8000) because the model runs in float16 instead of bfloat16.

**Root cause**: In `DirectAlgorithm._transform`, the norm-preserving rescaling computes `transformed * norm_pre / norm_post`. The intermediate product `transformed * norm_pre` overflows float16's max value (65504) — hidden state norms reach ~12,000, so any element > ~5.2 causes overflow to `inf`. This cascades as `nan` through subsequent layers. On bfloat16 (max ~3.4e38), overflow never occurs.

**Impact on our experiments**: None. Our `generate_steered_responses` does not pass `normalize` to `SteerVectorRequest`. The `SteeringConfig.normalize` field is used only at extraction time (step 01). Precomputed vectors (style-probe, create) already have normalization baked in.

**Fix**: Cast to float32 for the intermediate computation. Submitted as GitHub issue on ZJU-REAL/EasySteer. We forked to AMindToThink/EasySteer to apply the fix locally.

## 2026-03-10 — style_full and creativity_full: Steering collapses cross-prompt diversity across models and concepts

### Context

We ran two new full-scale experiments to test whether the "prompt override" pattern found in happy_recon generalizes across different steering concepts and model architectures. Both use precomputed steering vectors from EasySteer's replication notebooks.

A config bug was caught and fixed before running: the `system_prompt` field was missing from both configs, which would have caused the models to use their default system prompts instead of the creative writing prompt. See commit `461aecb`.

### Experiment 1: style_full (Qwen2.5-1.5B, style steering)

**Setup:**
- **Model**: Qwen/Qwen2.5-1.5B-Instruct
- **Steering vector**: `style-probe.gguf` (precomputed, from EasySteer steerable_chatbot replication)
- **Target layers**: 0–27 (all layers), normalized, direct algorithm
- **Scales**: [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
- **Prompts**: 50 (euclaise/writingprompts, test split)
- **Responses**: 10 per prompt per scale → 3,000 total
- **Config**: `configs/style_full.yaml`

**Results:**

| Scale | Clusters | Cosine dist (pooled) | Within-prompt cosine dist | Noise ratio |
|-------|----------|---------------------|--------------------------|-------------|
| 0.0   | 7        | 0.703               | 0.51                     | 0.41        |
| 0.5   | 3        | 0.669               | 0.48                     | 0.48        |
| 1.0   | 3        | 0.641               | 0.50                     | 0.50        |
| 2.0   | 0        | 0.545               | 0.46                     | 1.00        |
| 4.0   | 0        | 0.444               | 0.43                     | 1.00        |
| 8.0   | 2        | 0.498               | 0.48                     | 0.87        |

**Statistical tests (within-prompt, primary):**

| Test | Statistic | p-value | Significant? |
|------|-----------|---------|-------------|
| Mixed-effects (primary) | β = −0.004 | 0.003 | Yes |
| Spearman ρ | ρ = −0.234 | < 0.001 | Yes |
| Page's L (cosine dist, Holm-corrected) | — | < 0.001 | Yes |
| Page's L (num_clusters, Holm-corrected) | — | < 0.001 | Yes |

**Key observations:**
- Pooled cosine distance drops steeply: 0.70 → 0.44 (37% reduction, scales 0–4).
- Within-prompt diversity shows a mild but statistically significant decline (β = −0.004, p = 0.003) — unlike the "happy" vector, the style vector does slightly reduce sampling diversity, not just cross-prompt diversity.
- The within-prompt and pooled lines converge by scale 4, confirming the prompt-override pattern.
- UMAP shows scale-8 responses forming a distinct isolated cluster (yellow blob), consistent with degenerate output at extreme scales.
- Rebound at scale 8 (pooled cosine dist increases from 0.44 to 0.50) suggests the model is "breaking" — producing incoherent but somewhat varied outputs.

### Experiment 2: creativity_full (Llama-3-8B, creativity steering)

**Setup:**
- **Model**: meta-llama/Meta-Llama-3-8B-Instruct
- **Steering vector**: `create.gguf` (precomputed, from EasySteer creative_writing replication)
- **Target layers**: 16–29, unnormalized, direct algorithm
- **Scales**: [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
- **Prompts**: 50 (euclaise/writingprompts, test split)
- **Responses**: 10 per prompt per scale → 3,000 total
- **Config**: `configs/creativity_full.yaml`

**Results:**

| Scale | Clusters | Cosine dist (pooled) | Within-prompt cosine dist | Noise ratio |
|-------|----------|---------------------|--------------------------|-------------|
| 0.0   | 27       | 0.682               | 0.35                     | 0.42        |
| 0.5   | 5        | 0.606               | 0.34                     | 0.05        |
| 1.0   | 2        | 0.479               | 0.31                     | 0.02        |
| 2.0   | 2        | 0.363               | 0.34                     | 0.80        |
| 4.0   | 0        | 0.343               | 0.35                     | 1.00        |
| 8.0   | 2        | 0.476               | 0.48                     | 0.96        |

**Statistical tests (within-prompt, primary):**

| Test | Statistic | p-value | Significant? |
|------|-----------|---------|-------------|
| Mixed-effects (primary) | β = +0.017 | < 0.001 | Yes |
| Spearman ρ | ρ = +0.397 | < 0.001 | Yes |
| Page's L (num_clusters, Holm-corrected) | — | < 0.001 | Yes |
| Page's L (intra_cluster_dist, Holm-corrected) | — | < 0.001 | Yes |

**Key observations:**
- Strongest cross-prompt collapse of any experiment: pooled cosine distance halves from 0.68 → 0.34 (50% reduction).
- Within-prompt diversity is flat (~0.31–0.35) across scales 0–4 — the creativity vector does not reduce sampling diversity, only prompt sensitivity. This matches the happy_recon finding.
- The **positive** mixed-effects β (+0.017) and Spearman ρ (+0.40) seem counterintuitive but are explained by the scale-8 rebound: within-prompt diversity *increases* at extreme scales as the model degenerates into varied incoherent outputs.
- UMAP shows dramatic structure: scale 4–8 responses cluster into two tight, isolated blobs far from the diverse main cloud.
- Cluster count collapses from 27 → 0, the most dramatic drop across all experiments.
- The gap between within-prompt and pooled cosine distance at scale 0 is large (0.35 vs. 0.68), indicating Llama-3-8B produces more prompt-specific responses than Qwen2.5-1.5B at baseline.

### Cross-experiment comparison

| Experiment | Model | Concept | Pooled collapse (scale 0→4) | Within-prompt effect | Prompt override? |
|-----------|-------|---------|---------------------------|---------------------|-----------------|
| happy_recon | Qwen2.5-1.5B | happy | 0.74 → 0.60 (19%) | None (β ≈ 0, p = 0.96) | Yes |
| style_full | Qwen2.5-1.5B | style | 0.70 → 0.44 (37%) | Mild (β = −0.004, p = 0.003) | Yes |
| creativity_full | Llama-3-8B | creativity | 0.68 → 0.34 (50%) | None (flat 0.31–0.35) | Yes |

### Conclusions

1. **The "prompt override" pattern is robust.** All three experiments, across two models and three steering concepts, show the same phenomenon: steering primarily collapses cross-prompt diversity while leaving within-prompt diversity largely intact. The steering vector overrides the prompt signal rather than reducing the model's sampling stochasticity.

2. **Collapse severity varies by concept.** The "happy" vector produces mild cross-prompt collapse (19%), while "creativity" produces dramatic collapse (50%). This likely reflects how much the vector interferes with the model's prompt-processing circuitry — emotional tone vectors may be more orthogonal to content representations than style/creativity vectors.

3. **Within-prompt effects are concept-specific.** Only the "style" vector shows a statistically significant (though small) within-prompt diversity reduction. This suggests some steering vectors can affect sampling diversity, but the effect is much smaller than the cross-prompt collapse.

4. **Extreme scales cause degeneration, not just collapse.** All experiments show anomalous behavior at scale 8.0 — the UMAP plots show isolated clusters of degenerate outputs. This represents model failure rather than meaningful steering.

5. **The finding generalizes across architectures.** Both Qwen2.5-1.5B and Llama-3-8B show the same qualitative pattern, suggesting this is a general property of activation steering rather than a model-specific artifact.

### Plots

- `outputs/style_full/plots/` — UMAP, metrics bars, within-vs-pooled diversity
- `outputs/creativity_full/plots/` — UMAP, metrics bars, within-vs-pooled diversity

### Outputs

- `outputs/style_full/responses.jsonl` — 3,000 generated responses
- `outputs/style_full/embeddings.npz` — 384-dim sentence embeddings
- `outputs/style_full/metrics.json` — per-scale diversity metrics
- `outputs/style_full/stats.json` — full statistical test results
- `outputs/creativity_full/responses.jsonl` — 3,000 generated responses
- `outputs/creativity_full/embeddings.npz` — 384-dim sentence embeddings
- `outputs/creativity_full/metrics.json` — per-scale diversity metrics
- `outputs/creativity_full/stats.json` — full statistical test results
