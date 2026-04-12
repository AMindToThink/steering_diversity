# claim1_covariance_structure — LHS vs RHS

**Run:** `bounds_qwen_happy`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Points on or near the diagonal mean the Taylor approximation from Proposition 1 matches empirical `tr(Σ_z)` (where `Σ_z` is the covariance of unit-normalized pre-RMSNorm residuals). Points well below the diagonal mean the true spherical variance is *smaller* than the Taylor bound — bound holds but is loose.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 0.4814 | 0.8955 | 0.5376 |
| 1 | 0.4818 | 0.8974 | 0.5369 |
| 2 | 0.4826 | 0.902 | 0.535 |
| 4 | 0.4849 | 0.915 | 0.53 |
| 8 | 0.5014 | 0.9684 | 0.5178 |

