# claim5_diameter — LHS vs RHS

**Run:** `bounds_qwen_random`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Exact global bound (Corollary 3): `diam(φ_s(D)) ≤ 4 R / ‖s_eff‖`. Points below the diagonal are required. Computed from the reservoir.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 1.522 | 1.668e+04 | 9.124e-05 |
| 1 | 1.487 | 8335 | 0.0001784 |
| 2 | 1.504 | 4158 | 0.0003618 |
| 4 | 1.516 | 2062 | 0.0007352 |
| 8 | 1.55 | 997.5 | 0.001553 |

