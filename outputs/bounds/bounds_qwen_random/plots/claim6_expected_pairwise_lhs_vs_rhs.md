# claim6_expected_pairwise — LHS vs RHS

**Run:** `bounds_qwen_random`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Corollary 2: `E[‖φ(x₁) − φ(x₂)‖] ≤ 4 E[‖x‖] / ‖s_eff‖`. The LHS is estimated from the closed-form identity `E[‖u−v‖²] = 2(1 − ‖R̄‖²)` for iid unit vectors.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 0.9808 | 4113 | 0.0002385 |
| 1 | 0.9807 | 2055 | 0.0004772 |
| 2 | 0.9804 | 1025 | 0.0009562 |
| 4 | 0.9796 | 508.3 | 0.001927 |
| 8 | 0.9771 | 245.9 | 0.003973 |

