# claim2_spherical_variance — LHS vs RHS

**Run:** `bounds_qwen_happy_extended`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Global bound: `V = 1 − ‖R̄‖ ≤ 2 E[‖x‖] / ‖s_eff‖`. Points below the diagonal mean the bound holds. The gap between the points and the diagonal measures how tight the bound is; a large gap means the natural residual stream is far from the asymptotic 1/‖s‖ regime the bound describes.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 1 | 0.2969 | 279.8 | 0.001061 |
| 8 | 0.3088 | 38.05 | 0.008115 |
| 32 | 0.3892 | 9.938 | 0.03917 |
| 128 | 0.3766 | 3.234 | 0.1165 |

