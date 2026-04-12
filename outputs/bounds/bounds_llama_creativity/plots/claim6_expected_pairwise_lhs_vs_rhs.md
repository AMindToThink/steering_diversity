# claim6_expected_pairwise — LHS vs RHS

**Run:** `bounds_llama_creativity`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Corollary 2: `E[‖φ(x₁) − φ(x₂)‖] ≤ 4 E[‖x‖] / ‖s_eff‖`. The LHS is estimated from the closed-form identity `E[‖u−v‖²] = 2(1 − ‖R̄‖²)` for iid unit vectors.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 1.306 | 73.92 | 0.01767 |
| 1 | 1.304 | 37.15 | 0.03511 |
| 2 | 1.306 | 19.7 | 0.06632 |
| 4 | 1.308 | 11.84 | 0.1105 |
| 8 | 1.31 | 6.438 | 0.2035 |

