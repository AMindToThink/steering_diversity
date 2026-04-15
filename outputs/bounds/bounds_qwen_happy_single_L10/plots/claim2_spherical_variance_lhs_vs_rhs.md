# claim2_spherical_variance — LHS vs RHS

**Run:** `bounds_qwen_happy_single_L10`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Global bound: `V = 1 − ‖R̄‖ ≤ 2 E[‖x‖] / ‖s_eff‖`. Points below the diagonal mean the bound holds. The gap between the points and the diagonal measures how tight the bound is; a large gap means the natural residual stream is far from the asymptotic 1/‖s‖ regime the bound describes.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 0.2795 | 1255 | 0.0002228 |
| 1 | 0.2794 | 628.3 | 0.0004447 |
| 2 | 0.2791 | 315 | 0.0008858 |
| 4 | 0.2782 | 158.1 | 0.00176 |
| 8 | 0.276 | 79.02 | 0.003493 |

