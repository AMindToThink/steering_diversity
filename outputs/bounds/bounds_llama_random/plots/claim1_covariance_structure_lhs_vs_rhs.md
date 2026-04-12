# claim1_covariance_structure — LHS vs RHS

**Run:** `bounds_llama_random`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Points on or near the diagonal mean the Taylor approximation from Proposition 1 matches empirical `tr(Σ_z)` (where `Σ_z` is the covariance of unit-normalized pre-RMSNorm residuals). Points well below the diagonal mean the true spherical variance is *smaller* than the Taylor bound — bound holds but is loose.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 0.855 | 5.564 | 0.1537 |
| 1 | 0.8536 | 5.498 | 0.1553 |
| 2 | 0.8497 | 5.331 | 0.1594 |
| 4 | 0.8378 | 4.829 | 0.1735 |
| 8 | 0.819 | 4.024 | 0.2035 |

