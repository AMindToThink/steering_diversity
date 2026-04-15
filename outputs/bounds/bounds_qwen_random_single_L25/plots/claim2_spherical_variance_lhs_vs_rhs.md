# claim2_spherical_variance — LHS vs RHS

**Run:** `bounds_qwen_random_single_L25`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Global bound: `V = 1 − ‖R̄‖ ≤ 2 E[‖x‖] / ‖s_eff‖`. Points below the diagonal mean the bound holds. The gap between the points and the diagonal measures how tight the bound is; a large gap means the natural residual stream is far from the asymptotic 1/‖s‖ regime the bound describes.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 0.2797 | 1046 | 0.0002673 |
| 1 | 0.2797 | 523.2 | 0.0005347 |
| 2 | 0.2798 | 261.6 | 0.00107 |
| 4 | 0.2799 | 130.8 | 0.002141 |
| 8 | 0.28 | 65.35 | 0.004284 |

