# claim3_pole_concentration — LHS vs RHS

**Run:** `bounds_llama_random`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Exact global bound (Proposition 2): `max ‖φ_s(x) − ŝ‖ ≤ 2 max ‖x‖ / ‖s_eff‖`. Points below the diagonal are required — this is a hard bound, not an approximation. A point above the diagonal is a real bug.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 1.516 | 1057 | 0.001434 |
| 1 | 1.556 | 525.7 | 0.002959 |
| 2 | 1.556 | 257.6 | 0.006038 |
| 4 | 1.55 | 121.7 | 0.01274 |
| 8 | 1.527 | 62.8 | 0.02431 |

