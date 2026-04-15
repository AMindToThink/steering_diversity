# claim2_spherical_variance — LHS vs RHS

**Run:** `bounds_qwen_random_single_L17`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Global bound: `V = 1 − ‖R̄‖ ≤ 2 E[‖x‖] / ‖s_eff‖`. Points below the diagonal mean the bound holds. The gap between the points and the diagonal measures how tight the bound is; a large gap means the natural residual stream is far from the asymptotic 1/‖s‖ regime the bound describes.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 0.2799 | 1203 | 0.0002326 |
| 1 | 0.28 | 601.4 | 0.0004657 |
| 2 | 0.2804 | 300.5 | 0.0009332 |
| 4 | 0.2811 | 149.9 | 0.001875 |
| 8 | 0.282 | 74.36 | 0.003793 |

