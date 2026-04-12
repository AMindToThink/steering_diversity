# Scaling law plot (Claim 8)

**Run:** `bounds_qwen_happy`

Two log-log plots of empirical quantities against `‖s_eff‖`
(the effective steering at the final RMSNorm, computed as
`μ(pre_steered) − μ(pre_unsteered)`). Dashed reference lines show
the slopes predicted by the paper in the asymptotic regime.

- **Left** — spherical variance `1 − ‖R̄‖²` (which equals `tr(Σ_z)`
  for unit-normalized samples). Predicted slope: **−1**. Empirical
  slope: **-0.015**.
- **Right** — pre-RMSNorm `tr(Σ_x)`. Predicted slope: **−2**.
  Empirical slope: **-0.150**.

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
| 0.5 | 8.44 | 0.4828 | 4.586e+04 |
| 1 | 16.5 | 0.4832 | 4.538e+04 |
| 2 | 32 | 0.4812 | 4.387e+04 |
| 4 | 61.1 | 0.4705 | 3.95e+04 |
| 8 | 124 | 0.4649 | 2.982e+04 |

