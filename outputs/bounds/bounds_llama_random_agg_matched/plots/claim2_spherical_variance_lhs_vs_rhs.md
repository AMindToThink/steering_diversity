# claim2_spherical_variance — LHS vs RHS

**Run:** `bounds_llama_random_agg_matched`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Global bound: `V = 1 − ‖R̄‖ ≤ 2 E[‖x‖] / ‖s_eff‖`. Points below the diagonal mean the bound holds. The gap between the points and the diagonal measures how tight the bound is; a large gap means the natural residual stream is far from the asymptotic 1/‖s‖ regime the bound describes.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 0.5697 | 11.16 | 0.05105 |
| 1 | 0.4604 | 5.35 | 0.08606 |
| 2 | 0.208 | 2.574 | 0.08081 |
| 4 | 0.02239 | 1.355 | 0.01652 |
| 8 | 0.005036 | 0.8228 | 0.00612 |

