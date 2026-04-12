# claim3_pole_concentration — LHS vs RHS

**Run:** `bounds_llama_creativity`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Exact global bound (Proposition 2): `max ‖φ_s(x) − ŝ‖ ≤ 2 max ‖x‖ / ‖s_eff‖`. Points below the diagonal are required — this is a hard bound, not an approximation. A point above the diagonal is a real bug.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 1.475 | 277 | 0.005325 |
| 1 | 1.564 | 139.2 | 0.01123 |
| 2 | 1.561 | 73.81 | 0.02115 |
| 4 | 1.533 | 44.35 | 0.03457 |
| 8 | 1.493 | 24.12 | 0.06188 |

