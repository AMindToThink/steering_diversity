# claim5_diameter — LHS vs RHS

**Run:** `bounds_qwen_style`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Exact global bound (Corollary 3): `diam(φ_s(D)) ≤ 4 R / ‖s_eff‖`. Points below the diagonal are required. Computed from the reservoir.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 1.522 | 3468 | 0.0004389 |
| 1 | 1.487 | 1718 | 0.0008656 |
| 2 | 1.504 | 838.1 | 0.001795 |
| 4 | 1.516 | 421.9 | 0.003593 |
| 8 | 1.55 | 244.6 | 0.006334 |

