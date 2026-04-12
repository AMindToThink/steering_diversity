# claim1_covariance_structure — LHS vs RHS

**Run:** `bounds_qwen_random`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Points on or near the diagonal mean the Taylor approximation from Proposition 1 matches empirical `tr(Σ_z)` (where `Σ_z` is the covariance of unit-normalized pre-RMSNorm residuals). Points well below the diagonal mean the true spherical variance is *smaller* than the Taylor bound — bound holds but is loose.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 0.481 | 0.8934 | 0.5384 |
| 1 | 0.4809 | 0.8927 | 0.5386 |
| 2 | 0.4806 | 0.8915 | 0.5391 |
| 4 | 0.4798 | 0.889 | 0.5397 |
| 8 | 0.4774 | 0.8841 | 0.5399 |

