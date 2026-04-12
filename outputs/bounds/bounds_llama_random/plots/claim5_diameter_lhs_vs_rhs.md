# claim5_diameter — LHS vs RHS

**Run:** `bounds_llama_random`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Exact global bound (Corollary 3): `diam(φ_s(D)) ≤ 4 R / ‖s_eff‖`. Points below the diagonal are required. Computed from the reservoir.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 1.668 | 2115 | 0.0007888 |
| 1 | 1.671 | 1051 | 0.001589 |
| 2 | 1.664 | 515.3 | 0.003229 |
| 4 | 1.659 | 243.4 | 0.006816 |
| 8 | 1.643 | 125.6 | 0.01308 |

