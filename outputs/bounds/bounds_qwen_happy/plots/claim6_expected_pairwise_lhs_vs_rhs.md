# claim6_expected_pairwise — LHS vs RHS

**Run:** `bounds_qwen_happy`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Corollary 2: `E[‖φ(x₁) − φ(x₂)‖] ≤ 4 E[‖x‖] / ‖s_eff‖`. The LHS is estimated from the closed-form identity `E[‖u−v‖²] = 2(1 − ‖R̄‖²)` for iid unit vectors.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 0.9813 | 1135 | 0.0008646 |
| 1 | 0.9816 | 579.4 | 0.001694 |
| 2 | 0.9824 | 299.7 | 0.003278 |
| 4 | 0.9848 | 156.4 | 0.006296 |
| 8 | 1.001 | 76.41 | 0.01311 |

