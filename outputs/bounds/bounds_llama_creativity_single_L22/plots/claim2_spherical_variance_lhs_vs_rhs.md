# claim2_spherical_variance — LHS vs RHS

**Run:** `bounds_llama_creativity_single_L22`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Global bound: `V = 1 − ‖R̄‖ ≤ 2 E[‖x‖] / ‖s_eff‖`. Points below the diagonal mean the bound holds. The gap between the points and the diagonal measures how tight the bound is; a large gap means the natural residual stream is far from the asymptotic 1/‖s‖ regime the bound describes.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 0.6183 | 109.7 | 0.005634 |
| 1 | 0.6153 | 55.15 | 0.01116 |
| 2 | 0.6075 | 27.79 | 0.02186 |
| 4 | 0.5858 | 14.02 | 0.04177 |
| 8 | 0.5239 | 7.044 | 0.07438 |

