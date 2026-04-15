# claim2_spherical_variance — LHS vs RHS

**Run:** `bounds_llama_random_single_L29`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Global bound: `V = 1 − ‖R̄‖ ≤ 2 E[‖x‖] / ‖s_eff‖`. Points below the diagonal mean the bound holds. The gap between the points and the diagonal measures how tight the bound is; a large gap means the natural residual stream is far from the asymptotic 1/‖s‖ regime the bound describes.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 0.62 | 144.3 | 0.004297 |
| 1 | 0.6188 | 72.05 | 0.008589 |
| 2 | 0.6152 | 35.93 | 0.01712 |
| 4 | 0.6033 | 17.88 | 0.03374 |
| 8 | 0.5639 | 8.875 | 0.06354 |

