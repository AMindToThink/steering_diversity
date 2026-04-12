# Pre-RMSNorm Σ_x eigenvalue spectrum

**Run:** `bounds_qwen_random`

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
| 0 | 5019 | 1.04 | 1536 |
| 0.5 | 5021 | 1.041 | 1536 |
| 1 | 5024 | 1.041 | 1536 |
| 2 | 5030 | 1.041 | 1536 |
| 4 | 5039 | 1.04 | 1536 |
| 8 | 5053 | 1.033 | 1536 |

