# claim1_covariance_structure — LHS vs RHS

**Run:** `bounds_qwen_happy_extended`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Points on or near the diagonal mean the Taylor approximation from Proposition 1 matches empirical `tr(Σ_z)` (where `Σ_z` is the covariance of unit-normalized pre-RMSNorm residuals). Points well below the diagonal mean the true spherical variance is *smaller* than the Taylor bound — bound holds but is loose.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 1 | 0.5056 | 0.9636 | 0.5247 |
| 8 | 0.5222 | 1.037 | 0.5035 |
| 32 | 0.627 | 1.264 | 0.4959 |
| 128 | 0.6114 | 0.6853 | 0.8922 |

