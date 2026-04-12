# claim2_spherical_variance — LHS vs RHS

**Run:** `bounds_llama_random`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Global bound: `V = 1 − ‖R̄‖ ≤ 2 E[‖x‖] / ‖s_eff‖`. Points below the diagonal mean the bound holds. The gap between the points and the diagonal measures how tight the bound is; a large gap means the natural residual stream is far from the asymptotic 1/‖s‖ regime the bound describes.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 0.6122 | 35.94 | 0.01704 |
| 1 | 0.5966 | 17.88 | 0.03337 |
| 2 | 0.5458 | 8.773 | 0.06222 |
| 4 | 0.391 | 4.145 | 0.09433 |
| 8 | 0.1202 | 2.137 | 0.05625 |

