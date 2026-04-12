# Scaling law plot (Claim 8)

**Run:** `bounds_llama_random`

Two log-log plots of empirical quantities against `‖s_eff‖`
(the effective steering at the final RMSNorm, computed as
`μ(pre_steered) − μ(pre_unsteered)`). Dashed reference lines show
the slopes predicted by the paper in the asymptotic regime.

- **Left** — spherical variance `1 − ‖R̄‖²` (which equals `tr(Σ_z)`
  for unit-normalized samples). Predicted slope: **−1**. Empirical
  slope: **-0.410**.
- **Right** — pre-RMSNorm `tr(Σ_x)`. Predicted slope: **−2**.
  Empirical slope: **-0.314**.

**Interpretation.** The paper predicts that as ‖s_eff‖ grows, the
steering dominates the residual stream and the post-normalization
distribution concentrates near `ŝ`, so spherical variance shrinks
as `1/‖s‖`. Empirically, whether this holds depends on whether the
achievable ‖s_eff‖ is large compared to the natural residual stream
magnitude `E[‖x‖]`.

**Verdict for this run:** ❌ SLOPES DO NOT MATCH THEORY

**Observed values:**

| scale | ‖s_eff‖ | 1 − ‖R̄‖² | tr(Σ_x) |
|------:|--------:|----------:|---------:|
| 0.5 | 2.74 | 0.8496 | 2164 |
| 1 | 5.51 | 0.8372 | 2151 |
| 2 | 11.2 | 0.7937 | 2080 |
| 4 | 23.8 | 0.6291 | 1786 |
| 8 | 46.1 | 0.226 | 772.3 |

