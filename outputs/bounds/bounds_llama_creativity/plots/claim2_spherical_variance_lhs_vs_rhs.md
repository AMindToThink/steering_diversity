# claim2_spherical_variance — LHS vs RHS

**Run:** `bounds_llama_creativity`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Global bound: `V = 1 − ‖R̄‖ ≤ 2 E[‖x‖] / ‖s_eff‖`. Points below the diagonal mean the bound holds. The gap between the points and the diagonal measures how tight the bound is; a large gap means the natural residual stream is far from the asymptotic 1/‖s‖ regime the bound describes.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 0.5625 | 9.416 | 0.05974 |
| 1 | 0.4558 | 4.733 | 0.09631 |
| 2 | 0.2435 | 2.504 | 0.09724 |
| 4 | 0.03776 | 1.506 | 0.02507 |
| 8 | 0.005373 | 0.8201 | 0.006552 |

