# claim2_spherical_variance — LHS vs RHS

**Run:** `bounds_qwen_happy_single_L17`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Global bound: `V = 1 − ‖R̄‖ ≤ 2 E[‖x‖] / ‖s_eff‖`. Points below the diagonal mean the bound holds. The gap between the points and the diagonal measures how tight the bound is; a large gap means the natural residual stream is far from the asymptotic 1/‖s‖ regime the bound describes.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 0.28 | 860.1 | 0.0003255 |
| 1 | 0.2803 | 430.5 | 0.0006511 |
| 2 | 0.2809 | 215.7 | 0.001303 |
| 4 | 0.2821 | 108.2 | 0.002607 |
| 8 | 0.2841 | 54.29 | 0.005234 |

