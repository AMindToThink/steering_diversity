# claim5_diameter — LHS vs RHS

**Run:** `bounds_qwen_happy_extended`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Exact global bound (Corollary 3): `diam(φ_s(D)) ≤ 4 R / ‖s_eff‖`. Points below the diagonal are required. Computed from the reservoir.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 1 | 1.529 | 2026 | 0.0007548 |
| 8 | 1.558 | 275.5 | 0.005654 |
| 32 | 1.568 | 71.95 | 0.02179 |
| 128 | 1.526 | 23.41 | 0.06518 |

