# claim6_expected_pairwise — LHS vs RHS

**Run:** `bounds_qwen_style`

Empirical LHS plotted against theoretical RHS across the scale sweep,
log-log. The dashed line is `LHS = RHS` (where the bound is tight).
Points are colored by steering scale (viridis).

**Interpretation.** Corollary 2: `E[‖φ(x₁) − φ(x₂)‖] ≤ 4 E[‖x‖] / ‖s_eff‖`. The LHS is estimated from the closed-form identity `E[‖u−v‖²] = 2(1 − ‖R̄‖²)` for iid unit vectors.

**Observed values in this run:**

| scale | empirical LHS | theoretical RHS | ratio LHS/RHS |
|------:|--------------:|----------------:|--------------:|
| 0.5 | 0.98 | 855 | 0.001146 |
| 1 | 0.9789 | 423.5 | 0.002311 |
| 2 | 0.9764 | 206.6 | 0.004725 |
| 4 | 0.9726 | 104 | 0.009351 |
| 8 | 0.9787 | 60.32 | 0.01623 |

