# claim3_pole_concentration — LHS vs RHS

**Run:** `bounds_qwen_random`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Exact global bound (Proposition 2): `max ‖φ_s(x) − ŝ‖ ≤ 2 max ‖x‖ / ‖s_eff‖`. Points below the diagonal are required — this is a hard bound, not an approximation. A point above the diagonal is a real bug.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 1.46 | 8340 | 0.0001751 |
| 1 | 1.429 | 4167 | 0.0003429 |
| 2 | 1.442 | 2079 | 0.0006936 |
| 4 | 1.469 | 1031 | 0.001425 |
| 8 | 1.46 | 498.7 | 0.002927 |

