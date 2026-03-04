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
