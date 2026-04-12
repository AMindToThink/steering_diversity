# Pre-RMSNorm Σ_x eigenvalue spectrum

**Run:** `bounds_qwen_happy_extended`

Eigenvalues of the pre-RMSNorm residual covariance matrix `Σ_x`,
sorted descending, log y-axis. One curve per scale in the sweep,
colored by scale (viridis: dark = low scale, yellow = high scale).

**Interpretation.** Claim 1 (Proposition 1) predicts that under
steering the projection `P_⊥m̂` annihilates variance along `m̂`,
dropping the effective rank by one. Visually this shows up as the
smallest eigenvalue sagging lower at larger scales — if the theory
is in force here. In practice, large steering scales can also
*inject* variance (steering noise), causing the top eigenvalues to
grow instead of shrink.

**Top and estimated-rank eigenvalues:**

| scale | λ₁ (max) | λ at est. rank | est. rank (eig > 1e-6 λ₁) |
|------:|---------:|---------------:|--------------------------:|
| 0 | 5947 | 0.9206 | 1536 |
| 1 | 5925 | 0.9222 | 1536 |
| 8 | 5925 | 0.9048 | 1536 |
| 32 | 2.623e+04 | 0.8608 | 1536 |
| 128 | 2.308e+05 | 1.002 | 1536 |

