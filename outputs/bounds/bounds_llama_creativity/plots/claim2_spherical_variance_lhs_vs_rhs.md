# claim2_spherical_variance — LHS vs RHS

**Run:** `bounds_llama_creativity`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Global bound: `V = 1 − ‖R̄‖ ≤ 2 E[‖x‖] / ‖s_eff‖`. Points below the diagonal mean the bound holds. The gap between the points and the diagonal measures how tight the bound is; a large gap means the natural residual stream is far from the asymptotic 1/‖s‖ regime the bound describes.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 0.6163 | 36.96 | 0.01667 |
| 1 | 0.6137 | 18.58 | 0.03304 |
| 2 | 0.617 | 9.849 | 0.06265 |
| 4 | 0.6194 | 5.918 | 0.1047 |
| 8 | 0.6239 | 3.219 | 0.1938 |

