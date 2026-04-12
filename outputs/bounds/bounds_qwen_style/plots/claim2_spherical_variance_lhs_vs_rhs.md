# claim2_spherical_variance — LHS vs RHS

**Run:** `bounds_qwen_style`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Global bound: `V = 1 − ‖R̄‖ ≤ 2 E[‖x‖] / ‖s_eff‖`. Points below the diagonal mean the bound holds. The gap between the points and the diagonal measures how tight the bound is; a large gap means the natural residual stream is far from the asymptotic 1/‖s‖ regime the bound describes.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 0.279 | 427.5 | 0.0006527 |
| 1 | 0.2783 | 211.8 | 0.001314 |
| 2 | 0.2766 | 103.3 | 0.002677 |
| 4 | 0.274 | 52.01 | 0.005269 |
| 8 | 0.2781 | 30.16 | 0.009222 |

