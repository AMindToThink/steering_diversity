# Pre-registered Statistical Analysis Plan

Date: 2026-02-25
Committed before any experimental data is generated.

## Hypothesis
Activation steering at increasing intensities causes monotonic decrease
in output diversity of Qwen2.5-1.5B-Instruct.

## Experimental Design
- Model: Qwen/Qwen2.5-1.5B-Instruct
- Steering vector: EasySteer happy_diffmean.gguf (layers 10–25)
- Scales: 0.0, 0.5, 1.0, 2.0, 4.0, 8.0
- Prompts: 50 (from euclaise/writingprompts, test split, first 50)
- Responses per prompt per scale: 10
- Total responses: 3,000
- Seed: 42

## Primary Outcome
Per-prompt mean pairwise cosine distance (computed from Sentence-BERT
all-MiniLM-L6-v2 embeddings of the 10 responses for each prompt at
each scale). This metric is clustering-independent — it has no
sensitivity to HDBSCAN hyperparameters.

## Primary Test: Page's L Test
- Tests for monotonic trend across ordered repeated-measures conditions
- One-sided (alternative: diversity decreases as scale increases)
- α = 0.05
- Unit of observation: per-prompt diversity at each of 6 scales (50 × 6)

## Secondary Tests
Page's L on per-prompt values of:
- num_clusters
- noise_ratio
- cluster_entropy
- mean_intra_cluster_distance
- mean_inter_cluster_distance

All 6 tests (including primary) corrected with Holm-Bonferroni.

## Sensitivity Analysis
Linear mixed-effects model:
  mean_pairwise_cosine_distance ~ scale + (1|prompt_idx)
Reports: slope β, 95% CI, p-value.
Agreement between Page's L and mixed model = robust finding.
Disagreement = report both and investigate.

## Effect Size
Spearman's ρ between scale and per-prompt diversity, with 95% bootstrap CI
(10,000 resamples).

## Reporting Commitment
All results reported regardless of significance. No tests added post-hoc
without explicit labeling as exploratory.
