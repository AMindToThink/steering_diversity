# Pre-RMSNorm Σ_x eigenvalue spectrum

**Run:** `bounds_llama_random`

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
| 0 | 144 | 0.02116 | 4096 |
| 0.5 | 144.2 | 0.02118 | 4096 |
| 1 | 144.2 | 0.02103 | 4096 |
| 2 | 143.7 | 0.02041 | 4096 |
| 4 | 140.2 | 0.01918 | 4096 |
| 8 | 413 | 0.01678 | 4096 |

