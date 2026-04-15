# claim2_spherical_variance — LHS vs RHS

**Run:** `bounds_llama_creativity_single_L16`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Global bound: `V = 1 − ‖R̄‖ ≤ 2 E[‖x‖] / ‖s_eff‖`. Points below the diagonal mean the bound holds. The gap between the points and the diagonal measures how tight the bound is; a large gap means the natural residual stream is far from the asymptotic 1/‖s‖ regime the bound describes.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 0.6205 | 125.8 | 0.004932 |
| 1 | 0.6197 | 62.02 | 0.009991 |
| 2 | 0.6156 | 29.97 | 0.02054 |
| 4 | 0.597 | 13.75 | 0.04341 |
| 8 | 0.5061 | 5.873 | 0.08618 |

