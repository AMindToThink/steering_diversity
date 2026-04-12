# claim1_covariance_structure — LHS vs RHS

**Run:** `bounds_llama_creativity`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Points on or near the diagonal mean the Taylor approximation from Proposition 1 matches empirical `tr(Σ_z)` (where `Σ_z` is the covariance of unit-normalized pre-RMSNorm residuals). Points well below the diagonal mean the true spherical variance is *smaller* than the Taylor bound — bound holds but is loose.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 0.8528 | 5.482 | 0.1556 |
| 1 | 0.8508 | 5.38 | 0.1581 |
| 2 | 0.8533 | 5.321 | 0.1604 |
| 4 | 0.8551 | 4.422 | 0.1934 |
| 8 | 0.8586 | 1.964 | 0.4372 |

