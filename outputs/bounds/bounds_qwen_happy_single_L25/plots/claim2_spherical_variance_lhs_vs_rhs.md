# claim2_spherical_variance — LHS vs RHS

**Run:** `bounds_qwen_happy_single_L25`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Global bound: `V = 1 − ‖R̄‖ ≤ 2 E[‖x‖] / ‖s_eff‖`. Points below the diagonal mean the bound holds. The gap between the points and the diagonal measures how tight the bound is; a large gap means the natural residual stream is far from the asymptotic 1/‖s‖ regime the bound describes.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 0.2798 | 1030 | 0.0002716 |
| 1 | 0.28 | 515.2 | 0.0005434 |
| 2 | 0.2803 | 257.6 | 0.001088 |
| 4 | 0.2808 | 128.9 | 0.002179 |
| 8 | 0.2818 | 64.5 | 0.004369 |

