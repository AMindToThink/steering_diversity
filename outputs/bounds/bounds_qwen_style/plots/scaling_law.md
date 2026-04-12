# Scaling law plot (Claim 8)

**Run:** `bounds_qwen_style`

Two log-log plots of empirical quantities against `‖s_eff‖`
(the effective steering at the final RMSNorm, computed as
`μ(pre_steered) − μ(pre_unsteered)`). Dashed reference lines show
the slopes predicted by the paper in the asymptotic regime.

- **Left** — spherical variance `1 − ‖R̄‖²` (which equals `tr(Σ_z)`
  for unit-normalized samples). Predicted slope: **−1**. Empirical
  slope: **-0.254**.
- **Right** — pre-RMSNorm `tr(Σ_x)`. Predicted slope: **−2**.
  Empirical slope: **-0.229**.

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
| 0.5 | 11 | 0.4734 | 4.62e+04 |
| 1 | 22.3 | 0.4624 | 4.587e+04 |
| 2 | 45.6 | 0.4304 | 4.408e+04 |
| 4 | 91 | 0.3481 | 3.747e+04 |
| 8 | 157 | 0.2278 | 2.312e+04 |

