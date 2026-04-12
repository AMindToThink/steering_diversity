# claim6_expected_pairwise — LHS vs RHS

**Run:** `bounds_llama_random`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Corollary 2: `E[‖φ(x₁) − φ(x₂)‖] ≤ 4 E[‖x‖] / ‖s_eff‖`. The LHS is estimated from the closed-form identity `E[‖u−v‖²] = 2(1 − ‖R̄‖²)` for iid unit vectors.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 1.308 | 282.2 | 0.004634 |
| 1 | 1.307 | 140.3 | 0.009313 |
| 2 | 1.304 | 68.75 | 0.01896 |
| 4 | 1.294 | 32.48 | 0.03986 |
| 8 | 1.28 | 16.76 | 0.07637 |

