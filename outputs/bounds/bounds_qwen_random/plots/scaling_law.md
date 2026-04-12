# Scaling law plot (Claim 8)

**Run:** `bounds_qwen_random`

Two log-log plots of empirical quantities against `‖s_eff‖`
(the effective steering at the final RMSNorm, computed as
`μ(pre_steered) − μ(pre_unsteered)`). Dashed reference lines show
the slopes predicted by the paper in the asymptotic regime.

- **Left** — spherical variance `1 − ‖R̄‖²` (which equals `tr(Σ_z)`
  for unit-normalized samples). Predicted slope: **−1**. Empirical
  slope: **-0.029**.
- **Right** — pre-RMSNorm `tr(Σ_x)`. Predicted slope: **−2**.
  Empirical slope: **-0.020**.

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
| 0.5 | 2.3 | 0.4802 | 4.619e+04 |
| 1 | 4.6 | 0.4791 | 4.623e+04 |
| 2 | 9.23 | 0.4761 | 4.621e+04 |
| 4 | 18.6 | 0.4674 | 4.578e+04 |
| 8 | 38.4 | 0.4389 | 4.33e+04 |

