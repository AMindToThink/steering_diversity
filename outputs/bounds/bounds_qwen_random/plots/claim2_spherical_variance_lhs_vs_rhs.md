# claim2_spherical_variance — LHS vs RHS

**Run:** `bounds_qwen_random`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Global bound: `V = 1 − ‖R̄‖ ≤ 2 E[‖x‖] / ‖s_eff‖`. Points below the diagonal mean the bound holds. The gap between the points and the diagonal measures how tight the bound is; a large gap means the natural residual stream is far from the asymptotic 1/‖s‖ regime the bound describes.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 0.279 | 261.6 | 0.001067 |
| 1 | 0.2782 | 130.7 | 0.002129 |
| 2 | 0.2762 | 65.21 | 0.004235 |
| 4 | 0.2702 | 32.34 | 0.008356 |
| 8 | 0.2509 | 15.68 | 0.016 |

