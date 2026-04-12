# claim3_pole_concentration — LHS vs RHS

**Run:** `bounds_qwen_happy_extended`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Exact global bound (Proposition 2): `max ‖φ_s(x) − ŝ‖ ≤ 2 max ‖x‖ / ‖s_eff‖`. Points below the diagonal are required — this is a hard bound, not an approximation. A point above the diagonal is a real bug.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 1 | 1.484 | 1013 | 0.001465 |
| 8 | 1.519 | 137.8 | 0.01103 |
| 32 | 1.479 | 35.98 | 0.04111 |
| 128 | 1.446 | 11.71 | 0.1235 |

