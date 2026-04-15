# claim2_spherical_variance — LHS vs RHS

**Run:** `bounds_llama_creativity_single_L29`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Global bound: `V = 1 − ‖R̄‖ ≤ 2 E[‖x‖] / ‖s_eff‖`. Points below the diagonal mean the bound holds. The gap between the points and the diagonal measures how tight the bound is; a large gap means the natural residual stream is far from the asymptotic 1/‖s‖ regime the bound describes.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 0.6202 | 130.9 | 0.004738 |
| 1 | 0.6193 | 65.91 | 0.009395 |
| 2 | 0.6161 | 33.39 | 0.01845 |
| 4 | 0.605 | 17.06 | 0.03547 |
| 8 | 0.5683 | 8.757 | 0.0649 |

