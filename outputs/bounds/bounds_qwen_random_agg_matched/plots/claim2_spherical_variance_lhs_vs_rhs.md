# claim2_spherical_variance — LHS vs RHS

**Run:** `bounds_qwen_random_agg_matched`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Global bound: `V = 1 − ‖R̄‖ ≤ 2 E[‖x‖] / ‖s_eff‖`. Points below the diagonal mean the bound holds. The gap between the points and the diagonal measures how tight the bound is; a large gap means the natural residual stream is far from the asymptotic 1/‖s‖ regime the bound describes.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 0.2775 | 93.97 | 0.002953 |
| 1 | 0.2741 | 46.78 | 0.00586 |
| 2 | 0.2639 | 23.04 | 0.01145 |
| 4 | 0.2298 | 10.88 | 0.02112 |
| 8 | 0.1493 | 4.992 | 0.02991 |

