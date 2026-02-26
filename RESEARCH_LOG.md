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
