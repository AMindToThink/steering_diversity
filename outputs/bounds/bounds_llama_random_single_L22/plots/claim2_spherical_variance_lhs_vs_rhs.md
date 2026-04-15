# claim2_spherical_variance — LHS vs RHS

**Run:** `bounds_llama_random_single_L22`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Global bound: `V = 1 − ‖R̄‖ ≤ 2 E[‖x‖] / ‖s_eff‖`. Points below the diagonal mean the bound holds. The gap between the points and the diagonal measures how tight the bound is; a large gap means the natural residual stream is far from the asymptotic 1/‖s‖ regime the bound describes.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 0.6198 | 124 | 0.004999 |
| 1 | 0.6183 | 61.84 | 0.009998 |
| 2 | 0.6137 | 30.76 | 0.01995 |
| 4 | 0.5976 | 15.21 | 0.03928 |
| 8 | 0.5416 | 7.442 | 0.07278 |

