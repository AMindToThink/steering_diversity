# claim5_diameter — LHS vs RHS

**Run:** `bounds_llama_creativity`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Exact global bound (Corollary 3): `diam(φ_s(D)) ≤ 4 R / ‖s_eff‖`. Points below the diagonal are required. Computed from the reservoir.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 1.651 | 554 | 0.00298 |
| 1 | 1.636 | 278.4 | 0.005876 |
| 2 | 1.664 | 147.6 | 0.01127 |
| 4 | 1.659 | 88.7 | 0.0187 |
| 8 | 1.643 | 48.25 | 0.03404 |

