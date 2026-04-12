# claim2_spherical_variance — LHS vs RHS

**Run:** `bounds_qwen_happy`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Global bound: `V = 1 − ‖R̄‖ ≤ 2 E[‖x‖] / ‖s_eff‖`. Points below the diagonal mean the bound holds. The gap between the points and the diagonal measures how tight the bound is; a large gap means the natural residual stream is far from the asymptotic 1/‖s‖ regime the bound describes.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 0.2808 | 71.33 | 0.003937 |
| 1 | 0.2811 | 36.37 | 0.007729 |
| 2 | 0.2797 | 18.81 | 0.01487 |
| 4 | 0.2723 | 9.844 | 0.02766 |
| 8 | 0.2685 | 4.844 | 0.05543 |

