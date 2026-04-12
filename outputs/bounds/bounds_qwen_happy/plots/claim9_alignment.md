# Claim 9 — lost-dimension alignment with m̂

**Run:** `bounds_qwen_happy`

Absolute cosine between the smallest-eigenvalue eigenvector of the
post-normalization covariance `Σ_z` (estimated from the steered
reservoir, unit-normalized per sample) and the direction
`m̂ = μ_steered / ‖μ_steered‖`. The paper (synthesis.tex) predicts
this should approach 1 as ‖s‖ grows — the direction the normalization
projects out should be the steering pole.

**Interpretation.** A random-control run should stay near 0 at every
scale (random steering shouldn't align with any preferred direction).
A real steering vector should grow toward 1 as ‖s_eff‖ enters the
asymptotic regime. The 0.5 pass threshold is arbitrary — treat this
chart as a *trend* check, not a hard gate.

**Per-scale values:**

| scale | |cos(λ_min eigvec, m̂)| | passes (> 0.5)? |
|------:|----------------------:|:----------------:|
| 0.5 | 0.0008189 | ❌ |
| 1 | 0.003988 | ❌ |
| 2 | 0.002713 | ❌ |
| 4 | 0.01087 | ❌ |
| 8 | 1.434e-06 | ❌ |

