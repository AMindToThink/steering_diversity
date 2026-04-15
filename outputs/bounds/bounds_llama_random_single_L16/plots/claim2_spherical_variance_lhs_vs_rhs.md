# claim2_spherical_variance — LHS vs RHS

**Run:** `bounds_llama_random_single_L16`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Global bound: `V = 1 − ‖R̄‖ ≤ 2 E[‖x‖] / ‖s_eff‖`. Points below the diagonal mean the bound holds. The gap between the points and the diagonal measures how tight the bound is; a large gap means the natural residual stream is far from the asymptotic 1/‖s‖ regime the bound describes.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 0.6205 | 174.3 | 0.00356 |
| 1 | 0.6197 | 86.62 | 0.007154 |
| 2 | 0.6165 | 42.53 | 0.0145 |
| 4 | 0.6033 | 20.06 | 0.03008 |
| 8 | 0.5454 | 8.358 | 0.06525 |

