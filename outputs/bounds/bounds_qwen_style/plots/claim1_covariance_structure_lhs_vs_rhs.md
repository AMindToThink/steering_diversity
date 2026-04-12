# claim1_covariance_structure — LHS vs RHS

**Run:** `bounds_qwen_style`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Points on or near the diagonal mean the Taylor approximation from Proposition 1 matches empirical `tr(Σ_z)` (where `Σ_z` is the covariance of unit-normalized pre-RMSNorm residuals). Points well below the diagonal mean the true spherical variance is *smaller* than the Taylor bound — bound holds but is loose.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 0.4802 | 0.8904 | 0.5393 |
| 1 | 0.4791 | 0.8869 | 0.5402 |
| 2 | 0.4767 | 0.8796 | 0.5419 |
| 4 | 0.473 | 0.869 | 0.5443 |
| 8 | 0.4789 | 0.8859 | 0.5406 |

