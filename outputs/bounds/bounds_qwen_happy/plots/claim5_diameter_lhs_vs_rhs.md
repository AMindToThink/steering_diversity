# claim5_diameter — LHS vs RHS

**Run:** `bounds_qwen_happy`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Exact global bound (Corollary 3): `diam(φ_s(D)) ≤ 4 R / ‖s_eff‖`. Points below the diagonal are required. Computed from the reservoir.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 1.522 | 4603 | 0.0003306 |
| 1 | 1.487 | 2350 | 0.0006328 |
| 2 | 1.504 | 1216 | 0.001238 |
| 4 | 1.541 | 634.4 | 0.002429 |
| 8 | 1.55 | 309.9 | 0.005 |

