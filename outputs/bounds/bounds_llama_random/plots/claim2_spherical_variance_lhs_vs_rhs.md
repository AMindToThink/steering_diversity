# claim2_spherical_variance — LHS vs RHS

**Run:** `bounds_llama_random`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Global bound: `V = 1 − ‖R̄‖ ≤ 2 E[‖x‖] / ‖s_eff‖`. Points below the diagonal mean the bound holds. The gap between the points and the diagonal measures how tight the bound is; a large gap means the natural residual stream is far from the asymptotic 1/‖s‖ regime the bound describes.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 0.6193 | 141.1 | 0.004389 |
| 1 | 0.6174 | 70.15 | 0.008802 |
| 2 | 0.6124 | 34.37 | 0.01781 |
| 4 | 0.5972 | 16.24 | 0.03678 |
| 8 | 0.5746 | 8.38 | 0.06857 |

