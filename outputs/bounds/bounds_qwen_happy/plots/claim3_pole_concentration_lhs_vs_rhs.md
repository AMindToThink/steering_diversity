# claim3_pole_concentration — LHS vs RHS

**Run:** `bounds_qwen_happy`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Exact global bound (Proposition 2): `max ‖φ_s(x) − ŝ‖ ≤ 2 max ‖x‖ / ‖s_eff‖`. Points below the diagonal are required — this is a hard bound, not an approximation. A point above the diagonal is a real bug.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 1.46 | 2301 | 0.0006345 |
| 1 | 1.428 | 1175 | 0.001216 |
| 2 | 1.441 | 607.8 | 0.002371 |
| 4 | 1.502 | 317.2 | 0.004735 |
| 8 | 1.455 | 154.9 | 0.009387 |

