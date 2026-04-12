# claim2_spherical_variance — LHS vs RHS

**Run:** `bounds_qwen_random`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Global bound: `V = 1 − ‖R̄‖ ≤ 2 E[‖x‖] / ‖s_eff‖`. Points below the diagonal mean the bound holds. The gap between the points and the diagonal measures how tight the bound is; a large gap means the natural residual stream is far from the asymptotic 1/‖s‖ regime the bound describes.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 0.2796 | 2056 | 0.000136 |
| 1 | 0.2795 | 1028 | 0.000272 |
| 2 | 0.2793 | 512.6 | 0.0005448 |
| 4 | 0.2787 | 254.2 | 0.001097 |
| 8 | 0.2771 | 123 | 0.002253 |

