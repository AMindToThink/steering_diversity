# claim3_pole_concentration — LHS vs RHS

**Run:** `bounds_qwen_style`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Exact global bound (Proposition 2): `max ‖φ_s(x) − ŝ‖ ≤ 2 max ‖x‖ / ‖s_eff‖`. Points below the diagonal are required — this is a hard bound, not an approximation. A point above the diagonal is a real bug.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 1.46 | 1734 | 0.0008422 |
| 1 | 1.429 | 858.8 | 0.001664 |
| 2 | 1.442 | 419 | 0.003442 |
| 4 | 1.469 | 210.9 | 0.006964 |
| 8 | 1.458 | 122.3 | 0.01192 |

