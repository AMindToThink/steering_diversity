# claim2_spherical_variance — LHS vs RHS

**Run:** `bounds_qwen_style`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Global bound: `V = 1 − ‖R̄‖ ≤ 2 E[‖x‖] / ‖s_eff‖`. Points below the diagonal mean the bound holds. The gap between the points and the diagonal measures how tight the bound is; a large gap means the natural residual stream is far from the asymptotic 1/‖s‖ regime the bound describes.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 0.2743 | 54.54 | 0.00503 |
| 1 | 0.2668 | 27.02 | 0.009873 |
| 2 | 0.2453 | 13.18 | 0.01861 |
| 4 | 0.1926 | 6.609 | 0.02914 |
| 8 | 0.1213 | 3.825 | 0.0317 |

