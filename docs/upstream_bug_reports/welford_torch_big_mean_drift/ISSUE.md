# `OnlineCovariance.add_all` loses > 100% relative accuracy in float32 when the running mean is far from zero (catastrophic cancellation)

## Summary

In float32, `OnlineCovariance.add_all` produces a covariance matrix that disagrees with `numpy.cov` by more than 100% relative error — and in some batch-size regimes produces a covariance with a **negative trace**, which is mathematically impossible — whenever the samples have a large consistent bias relative to their per-coordinate standard deviation (e.g. `x ~ N(mu=1000, sigma^2=0.01^2 * I)`). The single-observation `.add()` path and `float64` are unaffected. Warming the state by calling `.add(x[0])` once before any `.add_all()` call fully recovers precision, which localizes the root cause to the cold-start subtraction inside `add_all`.

This is a real concern for anyone using `OnlineCovariance` to accumulate statistics over LLM residual-stream activations or other quantities with a large consistent bias. In our case (streaming stats over Qwen2.5 / Llama-3 residual streams) the resulting covariance was unusable, and we ended up hand-rolling Chan-Golub-LeVeque's batched merge instead.

**Standalone reproducer:** [`reproducer.py`](https://github.com/AMindToThink/steering_diversity/blob/058dea8/docs/upstream_bug_reports/welford_torch_big_mean_drift/reproducer.py) — depends only on `welford_torch`, `torch`, and `numpy`; copy-paste into a fresh venv and run.

## Environment

- `welford-torch == 0.2.5` (latest on PyPI)
- `torch == 2.9.1+cu128`
- `numpy == 2.2.6`
- Python 3.10.12
- Linux

## Reproducer

Standalone script, depends only on `welford_torch`, `torch`, `numpy`. Full file is attached as `reproducer.py`; running it on a fresh venv produces:

```
Distribution: x ~ N(mu=1000.0, sigma^2 I), d=64, N=5000
Ground truth trace (numpy, ddof=1): 0.00640744
------------------------------------------------------------------------------
Scenario                                                        trace    rel_err
------------------------------------------------------------------------------
.add_all() batches of 37, cold start                        0.0137938       1.15
.add_all() batches of 37, float64                          0.00640637   0.000167
.add_all() after 1 warm-up .add()                          0.00640704   6.14e-05
all samples via .add() one at a time                       0.00642125    0.00216
.add_all() one batch of N=5000                              -0.446952       70.8
```

The key rows:

- **`.add_all()` batches of 37, cold start** — trace `0.0138` vs. ground truth `0.0064`, **115% relative error**.
- **`.add_all()` one batch of `N=5000`** — trace `−0.447`. A covariance matrix cannot have a negative trace; this value is impossible.
- **`float64`** — `rel_err = 2e-4`. The formula is correct; the issue is precision.
- **Warmed with one `.add()`** — `rel_err = 6e-5`. The entire drift comes from the *first* `.add_all()` call against a cold-start state with running mean = 0.

Minimal form:

```python
import numpy as np
import torch
from welford_torch import OnlineCovariance

torch.manual_seed(0)
d, N, mu, sigma = 64, 5000, 1000.0, 0.01
x = torch.randn(N, d, dtype=torch.float32) * sigma + mu

oc = OnlineCovariance(dtype=torch.float32, device="cpu")
for start in range(0, N, 37):
    oc.add_all(x[start : start + 37])

trace = float((oc.cov * N / (N - 1)).trace().item())
ref   = float(np.var(x.numpy(), axis=0, ddof=1).sum())
print(f"welford-torch trace = {trace:.6g}, numpy trace = {ref:.6g}, rel_err = {abs(trace - ref)/ref:.3g}")
# welford-torch trace = 0.0137938, numpy trace = 0.00640744, rel_err = 1.15
```

## Root cause

Inside `covariance_torch.py::OnlineCovariance.add_all`:

```python
old_mean = self.__mean.clone()          # starts at 0 on cold start
delta_xs   = xs - old_mean              # magnitude ≈ |mu|  (e.g. ~1000)
self.__mean.add_(delta_xs.sum(dim=0) / self.__count)
new_mean = self.__mean                  # now ≈ |mu|
delta_xs_2 = xs - new_mean              # magnitude ≈ |sigma|  (e.g. ~0.01)

batch_cov_update = einops.einsum(
    delta_xs, delta_xs_2,
    "n_tokens ... pos_i, n_tokens ... pos_j -> ... pos_i pos_j",
) / self.__count
```

On the first call, `old_mean` is zero and `new_mean` equals the batch mean. For samples with `|mu| ≫ |sigma|`:

- `delta_xs[i] = x_i - 0 ≈ mu`  (magnitude ~1000 per coord)
- `delta_xs_2[i] = x_i - new_mean ≈ sigma * noise`  (magnitude ~0.01 per coord)

Expanding the einsum coordinate-wise:

```
batch_cov_update[i, j]
  = sum_k (x_k[i] - 0) * (x_k[j] - new_mean[j]) / n
  = sum_k x_k[i] * x_k[j] / n  −  new_mean[j] * sum_k x_k[i] / n
```

Both terms are order `|mu|^2 ≈ 10^6` in magnitude, and they must cancel to yield the true per-coord variance of order `|sigma|^2 ≈ 10^-4`. In `float32`, each term carries at most ~7 significant decimal digits, so the subtraction has absolute error `~10^6 * 2^-23 ≈ 0.1`, giving a result that is swamped by precision loss — the true signal (`~10^-4`) is four orders of magnitude smaller than the numerical noise.

This is textbook catastrophic cancellation and is exactly why Welford's online algorithm exists. The single-observation `.add()` path avoids it by computing `delta_1 = x - m_old` and `delta_2 = x - m_new` where the running mean tracks the true mean after each sample, so both deltas stay O(σ). `add_all` breaks that invariant on the first call because the running mean leaps from 0 to ~1000 in one step.

The fact that **larger batches are worse** (one giant batch yields a negative trace!) follows from the same mechanism: a bigger batch means more terms are summed into the two large intermediate tensors before they get subtracted, so more precision is lost before cancellation. This is the opposite of the usual "bigger batches are more accurate" intuition.

## Why the three mitigations work

- **`float64`**: 15–17 significant decimal digits. `10^6 * 2^-52 ≈ 2e-10`, way below the true signal of `10^-4`. No cancellation pressure.
- **Warm-up with one `.add()`**: after `.add(x[0])`, the running mean is `≈ x[0] ≈ 1000`. The next `.add_all()` then computes `delta_xs = xs - old_mean ≈ sigma * noise`, which is O(0.01) instead of O(1000). The product `delta_xs * delta_xs_2` is now O(0.01 * 0.01) = O(1e-4), matching the signal magnitude exactly — no cancellation.
- **All singletons via `.add()`**: same reason, the running mean tracks the true mean after every single sample, so deltas stay small throughout.

## Suggested fix: Chan-Golub-LeVeque batched merge

The standard numerically stable batched Welford update is [Chan, Golub, LeVeque 1979](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm). Sketch:

```python
def add_all(self, xs):
    n_b = xs.shape[0]
    mean_b = xs.mean(dim=0)
    # Center WITHIN the batch first — this is the crucial step.
    centered = xs - mean_b
    m2_b = centered.T @ centered          # [d, d], all entries O(sigma^2)

    if self.__count == 0:
        # Cold start: just absorb the batch's own statistics.
        self.__mean = mean_b.clone()
        self.__m2 = m2_b.clone()
        self.__count = n_b
        return

    n_a = self.__count
    delta = mean_b - self.__mean
    n = n_a + n_b
    self.__mean = self.__mean + delta * (n_b / n)
    self.__m2 = self.__m2 + m2_b + torch.outer(delta, delta) * (n_a * n_b / n)
    self.__count = n

def finalize_sample_cov(self):
    return self.__m2 / max(self.__count - 1, 1)
```

`m2_b` is the sum of outer products of *within-batch-centered* samples, so every entry is O(σ²) regardless of how large the absolute mean is. `delta = mean_b - self.__mean` is also O(σ) once the first batch is absorbed. All intermediate quantities stay on the scale of the real signal, so there is no cancellation.

This is mathematically identical to the current `add_all` formula — both compute the exact same covariance in the limit of infinite precision — but Chan's version never forms the large intermediate `(x_k − 0)(x_k − new_mean)` product on the cold-start call.

It would also fix the "larger batches are worse" behaviour: under Chan's merge, bigger batches monotonically improve accuracy (more in-batch samples → better within-batch variance estimate), which matches intuition.

(I've verified Chan's merge produces `rel_err < 1e-2` across every scenario in the reproducer including the pathological big-mean float32 case.)

## Workaround for users

Until a fix lands, any of the following restores precision in float32:

1. Prime the running state before any `.add_all()` call by calling `.add(x[0])` once.
2. Use `dtype=torch.float64`.
3. For batched workloads, use `.add()` in a loop over singletons (slower but correct).

## Related

My project's cross-check test that caught this:
[`tests/bounds/test_activation_streams_crosscheck.py::test_welford_torch_drifts_in_big_mean_regime`](https://github.com/AMindToThink/steering_diversity/blob/058dea8/tests/bounds/test_activation_streams_crosscheck.py) — we compare `welford_torch.OnlineCovariance` against a hand-rolled Chan-Golub-LeVeque implementation ([`src/bounds/activation_streams.py::FullMoments`](https://github.com/AMindToThink/steering_diversity/blob/058dea8/src/bounds/activation_streams.py)) and against `numpy.cov` as ground truth. The test currently asserts that `welford_torch` drifts > 20% in the big-mean regime, so if this issue is fixed the test will fail and prompt us to re-enable delegation to your library.

## Note

Thanks for building this library — Carsten Schelp's derivation is a nice reference and the Welford single-observation path is clean.
