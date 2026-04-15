# claim2_spherical_variance — LHS vs RHS

**Run:** `bounds_qwen_random_single_L10`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Global bound: `V = 1 − ‖R̄‖ ≤ 2 E[‖x‖] / ‖s_eff‖`. Points below the diagonal mean the bound holds. The gap between the points and the diagonal measures how tight the bound is; a large gap means the natural residual stream is far from the asymptotic 1/‖s‖ regime the bound describes.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 0.2795 | 1787 | 0.0001564 |
| 1 | 0.2794 | 892.9 | 0.0003129 |
| 2 | 0.2791 | 445.6 | 0.0006263 |
| 4 | 0.2784 | 221.6 | 0.001256 |
| 8 | 0.2761 | 108.8 | 0.002539 |

