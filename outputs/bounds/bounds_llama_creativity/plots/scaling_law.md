# Scaling law plot (Claim 8)

**Run:** `bounds_llama_creativity`

Two log-log plots of empirical quantities against `‖s_eff‖`
(the effective steering at the final RMSNorm, computed as
`μ(pre_steered) − μ(pre_unsteered)`). Dashed reference lines show
the slopes predicted by the paper in the asymptotic regime.

- **Left** — spherical variance `1 − ‖R̄‖²` (which equals `tr(Σ_z)`
  for unit-normalized samples). Predicted slope: **−1**. Empirical
  slope: **-1.773**.
- **Right** — pre-RMSNorm `tr(Σ_x)`. Predicted slope: **−2**.
  Empirical slope: **-1.128**.

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
| 0.5 | 10.5 | 0.8086 | 2061 |
| 1 | 20.8 | 0.7039 | 1850 |
| 2 | 39.3 | 0.4278 | 1274 |
| 4 | 65.4 | 0.07409 | 331 |
| 8 | 120 | 0.01072 | 152.3 |

