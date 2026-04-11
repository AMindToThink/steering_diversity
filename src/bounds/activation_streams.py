"""Streaming statistics for the bounds-verification pipeline.

All accumulators use **Chan-Golub-LeVeque batched Welford updates** (the
batched generalization of Welford 1962) — we never accumulate
``Σ xx^T − N μ μ^T``, because that form suffers catastrophic cancellation
when ``E‖x‖²`` and ``‖μ‖²`` are close in magnitude (very likely for residual
streams with large consistent biases).

Chan's merge formula for two chunks ``A`` and ``B``::

    delta = mean_B - mean_A
    n     = n_A + n_B
    mean  = mean_A + delta * (n_B / n)
    M2    = M2_A + M2_B + outer(delta, delta) * (n_A * n_B / n)

where ``M2 = Σ (x_i - mean)(x_i - mean)^T`` is the centered second moment.
Same per-batch cost as the naive path, numerically stable in float32.

Two tiers:

- **Cheap** (``CheapMoments``): ``O(d)`` state. Stores only the diagonal
  ``M2`` (sufficient for ``tr(Σ)``) and cheap scalars. Hand-rolled because
  no external library covers the trace-only path — critical for keeping
  per-layer capture at ~0.8 GB instead of ~54 GB per run.
- **Full** (``FullMoments``): ``O(d²)`` state. Also hand-rolled with the same
  Chan-Golub-LeVeque merge. We evaluated delegating to
  ``welford_torch.OnlineCovariance`` (a widely-used PyTorch streaming
  library) but a numpy cross-check — see
  ``tests/bounds/test_activation_streams_crosscheck.py`` — revealed that
  ``welford_torch`` drifts by >100% relative error in the big-mean /
  small-batch regime because its ``add_all`` centers against the *running*
  mean, squaring large intermediates. Chan's merge centers within each batch
  first, so the squared differences are always small regardless of absolute
  scale.  The library is still imported in the cross-check test so the
  regression is guarded going forward.

Correctness of both tiers is cross-checked against authoritative numpy
references (``np.cov`` / ``np.var``) in
``tests/bounds/test_activation_streams_crosscheck.py``. See the numerical
stability regression test ``test_big_mean_full_pipeline_matches_numpy`` for
the exact regime where the naive formulas fail.
"""

from __future__ import annotations

import torch


def _batch_moments(x: torch.Tensor, full: bool) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Compute ``(n_B, mean_B, M2_B)`` for one batch.

    For ``full=True`` returns the d×d centered outer-product sum; otherwise
    returns the per-coordinate ``[d]`` sum of squares.
    """
    n_b = x.shape[0]
    mean_b = x.mean(dim=0)
    centered = x - mean_b
    if full:
        m2_b = centered.T @ centered
    else:
        m2_b = (centered * centered).sum(dim=0)
    return n_b, mean_b, m2_b


def _chan_merge(
    n_a: int,
    mean_a: torch.Tensor,
    m2_a: torch.Tensor,
    n_b: int,
    mean_b: torch.Tensor,
    m2_b: torch.Tensor,
    full: bool,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Chan-Golub-LeVeque merge of two running moment states."""
    if n_a == 0:
        return n_b, mean_b.clone(), m2_b.clone()
    n = n_a + n_b
    delta = mean_b - mean_a
    mean = mean_a + delta * (n_b / n)
    if full:
        m2 = m2_a + m2_b + torch.outer(delta, delta) * (n_a * n_b / n)
    else:
        m2 = m2_a + m2_b + (delta * delta) * (n_a * n_b / n)
    return n, mean, m2


class _MomentsBase:
    """Shared state for cheap and full tiers.

    Subclasses differ only in whether ``m2`` is d-shaped (cheap, trace only)
    or d×d (full, full covariance).
    """

    _FULL: bool = False  # overridden

    def __init__(self, d: int, device: str = "cuda", dtype: torch.dtype = torch.float32):
        self.d = d
        self.device = device
        self.dtype = dtype
        self.count = 0
        self.mean = torch.zeros(d, device=device, dtype=dtype)
        if self._FULL:
            self.m2 = torch.zeros(d, d, device=device, dtype=dtype)
        else:
            self.m2 = torch.zeros(d, device=device, dtype=dtype)
        # Cheap scalar extras for claim bounds that depend on ‖x‖.
        self.sum_norm = torch.zeros((), device=device, dtype=dtype)
        self.sum_sq_norm = torch.zeros((), device=device, dtype=dtype)
        self.max_norm = torch.zeros((), device=device, dtype=dtype)

    def update(self, x: torch.Tensor) -> None:
        """Fold a mini-batch ``x: [B, d]`` into the running state."""
        if x.ndim != 2 or x.shape[1] != self.d:
            raise ValueError(
                f"update() expected shape [B, {self.d}], got {tuple(x.shape)}"
            )
        if x.shape[0] == 0:
            return
        x = x.to(self.device, self.dtype)

        n_b, mean_b, m2_b = _batch_moments(x, full=self._FULL)
        self.count, self.mean, self.m2 = _chan_merge(
            self.count, self.mean, self.m2, n_b, mean_b, m2_b, full=self._FULL
        )

        # Cheap scalar extras (iteration-order-independent, no stability issues).
        norms = x.norm(dim=1)
        self.sum_norm += norms.sum()
        self.sum_sq_norm += (norms * norms).sum()
        if norms.numel() > 0:
            self.max_norm = torch.maximum(self.max_norm, norms.max())

    def finalize(self) -> dict:
        n = self.count
        if n == 0:
            raise RuntimeError("finalize() called with empty state (no updates seen)")
        out: dict = {
            "count": n,
            "mean": self.mean.detach().cpu(),
            "E_norm": (self.sum_norm / n).item(),
            "E_sq_norm": (self.sum_sq_norm / n).item(),
            "R": self.max_norm.item(),
        }
        if self._FULL:
            cov = self.m2 / max(n - 1, 1)
            out["cov"] = cov.detach().cpu()
            out["trace_cov"] = float(cov.trace().item())
        else:
            out["trace_cov"] = float((self.m2 / max(n - 1, 1)).sum().item())
        return out


class CheapMoments(_MomentsBase):
    """Cheap-tier streaming stats. ``O(d)`` state, gives ``tr(Σ)`` but not ``Σ``."""

    _FULL = False


class FullMoments(_MomentsBase):
    """Full-tier streaming stats. ``O(d²)`` state, gives the full covariance."""

    _FULL = True


class SphericalMoments:
    """Running statistics on the unit sphere ``S^(d−1)``.

    Accumulates ``R̄ = E[φ(x)]`` (running mean of L2-normalized samples),
    and optionally tracks ``max ‖φ(x) − ŝ‖`` for a fixed pole ``s``.

    Claim 6's ``E[‖φ(x_i) − φ(x_j)‖²] = 2(1 − ‖R̄‖²)`` holds for iid unit
    vectors, so we don't need pairs to estimate it.
    """

    def __init__(
        self,
        d: int,
        device: str = "cuda",
        pole: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.d = d
        self.device = device
        self.dtype = dtype
        self.count = 0
        self.sum_unit = torch.zeros(d, device=device, dtype=dtype)
        self.max_chord_to_pole = torch.zeros((), device=device, dtype=dtype)
        if pole is None:
            self.pole = None
        else:
            pole = pole.to(device, dtype)
            pole_norm = pole.norm()
            if pole_norm == 0:
                self.pole = None
            else:
                self.pole = (pole / pole_norm).contiguous()

    def update(self, x: torch.Tensor) -> None:
        if x.ndim != 2 or x.shape[1] != self.d:
            raise ValueError(
                f"update() expected shape [B, {self.d}], got {tuple(x.shape)}"
            )
        if x.shape[0] == 0:
            return
        x = x.to(self.device, self.dtype)
        norms = x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        u = x / norms
        self.count += u.shape[0]
        self.sum_unit += u.sum(dim=0)
        if self.pole is not None:
            chord = (u - self.pole).norm(dim=1)
            self.max_chord_to_pole = torch.maximum(self.max_chord_to_pole, chord.max())

    def finalize(self) -> dict:
        if self.count == 0:
            raise RuntimeError("SphericalMoments.finalize() with empty state")
        r_bar = self.sum_unit / self.count
        r_bar_norm = r_bar.norm().item()
        return {
            "count": self.count,
            "R_bar": r_bar.detach().cpu(),
            "R_bar_norm": r_bar_norm,
            "spherical_variance": 1.0 - r_bar_norm,
            "expected_pair_sq_chord": 2.0 * (1.0 - r_bar_norm * r_bar_norm),
            "max_chord_to_pole": self.max_chord_to_pole.item(),
        }


class Reservoir:
    """Vitter's Algorithm R reservoir sampling.

    Maintains a uniform random sample of size ``K`` from a stream of samples
    of arbitrary length. The reservoir is stored on ``device`` (default CPU
    since it's only consulted once at the end for pairwise claim stats).
    """

    def __init__(self, K: int, d: int, seed: int = 0, device: str = "cpu"):
        self.K = K
        self.d = d
        self.device = device
        self.seen = 0
        self.samples = torch.zeros(K, d, device=device, dtype=torch.float32)
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

    def update(self, x: torch.Tensor) -> None:
        if x.ndim != 2 or x.shape[1] != self.d:
            raise ValueError(
                f"update() expected shape [B, {self.d}], got {tuple(x.shape)}"
            )
        if x.shape[0] == 0:
            return
        x_cpu = x.to(self.device, torch.float32)
        for i in range(x_cpu.shape[0]):
            if self.seen < self.K:
                self.samples[self.seen] = x_cpu[i]
            else:
                j = int(torch.randint(0, self.seen + 1, (1,), generator=self.gen).item())
                if j < self.K:
                    self.samples[j] = x_cpu[i]
            self.seen += 1

    def finalize(self) -> torch.Tensor:
        """Return the current reservoir contents (size ``min(K, seen)``)."""
        if self.seen < self.K:
            return self.samples[: self.seen].clone()
        return self.samples.clone()
