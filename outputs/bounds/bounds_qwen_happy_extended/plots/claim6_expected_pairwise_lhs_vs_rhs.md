# claim6_expected_pairwise — LHS vs RHS

**Run:** `bounds_qwen_happy_extended`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Corollary 2: `E[‖φ(x₁) − φ(x₂)‖] ≤ 4 E[‖x‖] / ‖s_eff‖`. The LHS is estimated from the closed-form identity `E[‖u−v‖²] = 2(1 − ‖R̄‖²)` for iid unit vectors.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 1 | 1.006 | 559.6 | 0.001797 |
| 8 | 1.022 | 76.1 | 0.01343 |
| 32 | 1.12 | 19.88 | 0.05634 |
| 128 | 1.106 | 6.467 | 0.171 |

