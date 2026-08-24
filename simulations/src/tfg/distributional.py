"""[A4] Distributional-predictor helpers for the generalised TFG engine.

Everything here is opt-in and lives OUTSIDE the engine: the engine only sees a
``log_f(x, n_t, eta_keys)`` callable.  This module provides

* :func:`repository_schedule` -- the checkpointed ``DiffusionModel``'s cosine
  schedule REBUILT from its formula (``Diffusion.py:25-29``) in a requested
  dtype, as a :class:`~tfg.schedule.DiffusionSchedule`.  In float32 it is
  bit-identical to ``model.baralphas`` / ``model.betas``; in float64 it is the
  same formula at higher precision (a rebuild, never a cast of the float32
  attributes).  The engine's ``T`` is ``diffusion_steps - 1`` because the
  repository loop ``range(T-1, 0, -1)`` stops at ``t = 1`` and the model's
  ``baralphas[0] == 1`` plays the role of ``alphabar_0``.
* :class:`CMSampler` -- a noise-injectable, tape-keyed wrapper around
  ``ConsistencyModeliCT.sample``.  Noise sources:
    ``tape``    one tape draw ``("eta", t, i)`` of shape ``(len(ts), d_y)`` per
                conditional sample (C5 keying, order-independent);
    ``legacy``  a ``torch.Generator`` seeded exactly as ``experiments/_guided.py``
                seeds the GLOBAL generator (``key_seed("cond", restart, t, j)``),
                replaying its draw order (spatial delta first when M > 1, then
                the sampler's ``len(ts)`` draws) -- proven bit-identical to the
                ``torch.manual_seed`` path in ``tests/test_engine_matches_guided.py``.
  Options: ``antithetic`` (second half of the batch reuses the negated noise of
  the first half -- EXACT in distribution, variance-reducing for the MMD
  estimate), ``cache`` (exact reuse of identical ``(x, key)`` draws within a
  step; used by the adaptive-n agreement statistic so the half-batch gradients
  are free).  Every actual generator draw is counted in ``cm_samples``.
* :class:`DistributionalLoss` -- the repository MMD (``LossFunctions.MMDLoss``,
  RBF with 5 bandwidths, ``mul_factor=2``, biased V-statistic) with explicit,
  opt-in bandwidth policies and output transforms:
    bandwidth: ``fixed`` (value given; what the synthetic benchmark uses),
               ``pooled`` (repository default: mean off-diagonal squared
               distance of the POOLED ``(X;Y)`` -- a function of the samples,
               so the gradient flows through it and tiny batches can collapse
               it), ``target`` (same rule on the target only, fixed and
               detached), ``pooled_floor`` (pooled, but never below
               ``floor_frac`` x the target rule, detached).
    transform: ``mmd2`` (raw V-statistic), ``sqrt_abs_eps`` (the SD code's
               ``sqrt(|MMD^2| + eps)``, gradient up to ``1/(2 sqrt eps)``),
               ``sqrt_floor`` (``sqrt(MMD^2 + c) - sqrt(c)`` with ``c`` the
               V-statistic's own finite-sample floor ``k(0) (1/n + 1/m)``
               times ``floor_frac`` -- bounded gradient, same asymptote).
"""

import math

import torch

from tfg.schedule import DiffusionSchedule

LEGACY_DELTA_TAG = "delta"


# ---------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------

class RepositorySchedule(DiffusionSchedule):
    """``DiffusionSchedule`` whose constants are the repository model's cosine
    schedule, rebuilt from ``Diffusion.DiffusionModel.__init__`` verbatim."""

    def __init__(self, diffusion_steps=100, dtype=torch.float32, device="cpu", s=0.008):
        # deliberately NOT calling super().__init__: different formula/clipping
        self.T = int(diffusion_steps) - 1
        self.dtype = dtype
        self.device = torch.device(device)
        # Diffusion.py:25-29, with `dtype` in place of the hard-coded float32.
        timesteps = torch.tensor(range(0, int(diffusion_steps)), dtype=dtype)
        schedule = torch.cos((timesteps / int(diffusion_steps) + s) / (1 + s) * torch.pi / 2) ** 2
        baralphas = (schedule / schedule[0]).to(self.device)
        betas = (1 - baralphas / torch.cat([baralphas[0:1], baralphas[:-1]])).to(self.device)
        self.alphabar = baralphas            # length diffusion_steps = T + 1
        self.min_alphabar = None
        # model.betas[t] = 1 - ab[t]/ab[t-1] for t >= 1; betas[0] = 0 is dropped
        # so that self.betas[t-1] is the per-step beta of step t (as in the
        # cosine DiffusionSchedule).
        self.betas = betas[1:]
        self.model_betas = betas             # the model's own indexing, kept for checks

    def matches_model(self, model):
        """True iff the model's float32 attributes equal ours bit-for-bit."""
        ab = model.baralphas.detach().cpu()
        be = model.betas.detach().cpu()
        return (self.dtype == torch.float32
                and torch.equal(ab, self.alphabar.cpu())
                and torch.equal(be, self.model_betas.cpu()))


def repository_schedule(model_or_steps, dtype=torch.float32, device="cpu"):
    steps = (model_or_steps if isinstance(model_or_steps, int)
             else int(model_or_steps.diffusion_steps))
    return RepositorySchedule(steps, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# sampler
# ---------------------------------------------------------------------------

def sample_with_noise(model, cond, ts, Z):
    """``ConsistencyModeliCT.sample`` with the noise supplied.

    ``Z`` has shape ``(len(ts), n, d_y)``; ``Z[0]`` is the initial draw and
    ``Z[k]`` the draw consumed at ``ts[k]``.  Arithmetic is the sampler's own,
    in the same order, so identical noise gives identical output.
    """
    x = Z[0] * ts[0]
    for k, t in enumerate(ts[1:], start=1):
        x = x + math.sqrt(t ** 2 - model.eps ** 2) * Z[k]
        x = model(x, t, cond=cond)
    return x


def legacy_seed(*parts):
    """``experiments/_common.key_seed``, duplicated so this module stays
    importable without the experiments package."""
    import hashlib
    return int.from_bytes(
        hashlib.blake2b(repr(parts).encode(), digest_size=8).digest(),
        "big") % (2 ** 31 - 1)


class LegacyStream:
    """Replays ``_guided.py``'s global-RNG consumption for one ``(restart, t, j)``
    with a private ``torch.Generator`` (CPU mt19937 -- identical values to
    ``torch.manual_seed`` + ``torch.randn``).  Draw order: ``delta`` of shape
    ``delta_shape`` iff ``with_delta`` (drawn at construction), then
    ``len(ts)`` draws of ``(n, d_y)`` (drawn on the first ``Z`` request).
    """

    def __init__(self, restart, t, j, with_delta, delta_shape, dtype):
        self.g = torch.Generator(device="cpu").manual_seed(
            legacy_seed("cond", int(restart), int(t), int(j)))
        self.dtype = dtype
        self.delta = (torch.randn(delta_shape, generator=self.g, dtype=dtype)
                      if with_delta else None)
        self._Z = None

    def Z(self, n, d_y, n_ts):
        if self._Z is None:
            self._Z = torch.stack([torch.randn(n, d_y, generator=self.g, dtype=self.dtype)
                                   for _ in range(n_ts)])
        if self._Z.shape[1] != n:
            raise ValueError("legacy stream re-requested with a different n")
        return self._Z


class LegacyTape:
    """A tape facade that serves the engine's ``("delta", t, j)`` requests from
    the legacy stream and everything else from a wrapped ``NoiseTape``.

    Only ``randn`` is used by the engine; delegation covers the rest.  The
    ``streams`` dict is shared with the :class:`CMSampler` in ``legacy`` mode
    so the delta and the conditional noise come from ONE generator in the
    original order (this is the order dependence the real tape removes; it is
    reproduced here only to prove equivalence with the historical runs).
    """

    def __init__(self, base_tape, restart, with_delta, delta_shape, dtype):
        self.base = base_tape
        self.restart = int(restart)
        self.with_delta = bool(with_delta)
        self.delta_shape = tuple(delta_shape)
        self.dtype = dtype
        self.streams = {}
        self.seed = base_tape.seed

    def stream(self, t, j):
        key = (int(t), int(j))
        if key not in self.streams:
            self.streams[key] = LegacyStream(self.restart, t, j, self.with_delta,
                                             self.delta_shape, self.dtype)
        return self.streams[key]

    def randn(self, key, shape, device=None, dtype=None):
        if isinstance(key, tuple) and key and key[0] == LEGACY_DELTA_TAG and self.with_delta:
            _, t, j = key
            s = self.stream(t, j)
            return s.delta.to(device=device or "cpu", dtype=dtype or self.dtype)
        return self.base.randn(key, shape, device=device, dtype=dtype)

    def __getattr__(self, name):
        return getattr(self.base, name)


class CMSampler:
    """Noise-injectable conditional sampler; see module docstring."""

    def __init__(self, model, ts, tape, source="tape", antithetic=False,
                 cache=False, dtype=None, device="cpu"):
        if source not in ("tape", "legacy"):
            raise ValueError(f"unknown noise source {source!r}")
        self.model = model
        self.ts = list(ts)
        self.tape = tape
        self.source = source
        self.antithetic = bool(antithetic)
        self.cache_on = bool(cache)
        self.device = device
        self.dtype = dtype or next(model.parameters()).dtype
        self.cm_samples = 0          # ACTUAL conditional generator draws
        self.cm_calls = 0
        self._cache = {}
        self._cache_x = None

    def reset_counts(self):
        self.cm_samples = 0
        self.cm_calls = 0
        self._cache.clear()
        self._cache_x = None

    def _noise(self, keys, n, d_y):
        n_ts = len(self.ts)
        if self.source == "legacy":
            t, j = keys[0][1], (keys[0][2] if len(keys[0]) == 4 else 0)
            return self.tape.stream(t, j).Z(n, d_y, n_ts)
        if self.antithetic and n >= 2:
            h = n // 2
            base = torch.stack([self.tape.randn(k, (n_ts, d_y), device=self.device,
                                                dtype=self.dtype)
                                for k in keys[:h]], dim=1)           # (n_ts, h, d_y)
            parts = [base, -base]
            if n % 2:
                parts.append(self.tape.randn(keys[-1], (n_ts, d_y), device=self.device,
                                             dtype=self.dtype).unsqueeze(1))
            return torch.cat(parts, dim=1)
        return torch.stack([self.tape.randn(k, (n_ts, d_y), device=self.device,
                                            dtype=self.dtype) for k in keys], dim=1)

    def __call__(self, x, keys):
        """Return ``n = len(keys)`` conditional samples given condition ``x``
        (shape ``(1, d_x)``), differentiable in ``x``.

        With ``cache=True`` every produced row is remembered per ``(key,
        x-bytes)`` and a later call with the SAME ``x`` and a subset of the keys
        (e.g. the half batches of the agreement statistic) is served from the
        cache -- exact reuse, no new generator draws.  The cache keeps rows of
        the most recent ``x`` values only (bounded).
        """
        n = len(keys)
        d_y = self.model.nfeatures
        self.cm_calls += 1
        if not self.cache_on:
            Z = self._noise(keys, n, d_y)
            cond = x.reshape(1, -1).repeat(n, 1)
            y = sample_with_noise(self.model, cond, self.ts, Z)
            self.cm_samples += n
            return y
        xb = x.detach().cpu().numpy().tobytes()
        if xb != self._cache_x:
            self._cache_x = xb
            self._cache.clear()
        missing = [k for k in keys if k not in self._cache]
        if missing:
            Z = self._noise(missing, len(missing), d_y)
            cond = x.reshape(1, -1).repeat(len(missing), 1)
            y_new = sample_with_noise(self.model, cond, self.ts, Z)
            self.cm_samples += len(missing)
            for i, k in enumerate(missing):
                self._cache[k] = y_new[i:i + 1]
        return torch.cat([self._cache[k] for k in keys], dim=0)


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------

def pooled_bandwidth(Z):
    d2 = torch.cdist(Z, Z, p=2) ** 2
    m = Z.shape[0]
    return d2.sum() / (m ** 2 - m)


class DistributionalLoss:
    """``loss(Y, S_G) -> 0-dim tensor`` (smaller is better); ``log_f = -loss``."""

    def __init__(self, S_G, bandwidth="fixed", bandwidth_value=None, transform="mmd2",
                 n_kernels=5, mul_factor=2.0, floor_frac=0.1, eps=1e-8, device="cpu",
                 backend="reference"):
        """``backend``: ``reference`` = ``LossFunctions.MMDLoss`` (the repository
        code); ``fast`` = ``tfg.fast_mmd.MMDFixedTarget`` (exact cached-target
        evaluation, same value and gradient to 1e-12 in float64; supports the
        ``fixed``, ``target`` and ``pooled`` bandwidth policies)."""
        if bandwidth not in ("fixed", "pooled", "target", "pooled_floor"):
            raise ValueError(f"unknown bandwidth policy {bandwidth!r}")
        if transform not in ("mmd2", "sqrt_abs_eps", "sqrt_floor"):
            raise ValueError(f"unknown transform {transform!r}")
        from LossFunctions import MMDLoss, RBF
        self.S_G = S_G.detach()
        self.bandwidth = bandwidth
        self.transform = transform
        self.n_kernels = n_kernels
        self.floor_frac = float(floor_frac)
        self.eps = float(eps)
        self.target_bw = float(pooled_bandwidth(self.S_G.double()))
        if bandwidth == "fixed":
            if bandwidth_value is None:
                raise ValueError("bandwidth='fixed' needs bandwidth_value")
            bw = float(bandwidth_value)
        elif bandwidth == "target":
            bw = self.target_bw
        else:
            bw = None                                  # pooled: the repository rule
        self.kernel = RBF(n_kernels=n_kernels, mul_factor=mul_factor, bandwidth=bw, device=device)
        self.mmd = MMDLoss(kernel=self.kernel, device=device)
        self.last_bandwidth = bw
        if backend not in ("reference", "fast"):
            raise ValueError(f"unknown backend {backend!r}")
        self.backend = backend
        self._fast = None
        if backend == "fast":
            if bandwidth == "pooled_floor":
                raise ValueError("backend='fast' does not implement bandwidth='pooled_floor'")
            from tfg.fast_mmd import MMDFixedTarget
            self._fast = MMDFixedTarget(self.S_G, bandwidth=bw, n_kernels=n_kernels,
                                        mul_factor=mul_factor)

    def _mmd2(self, Y):
        if self._fast is not None:
            out = self._fast(Y)
            if self.bandwidth == "pooled":
                with torch.no_grad():
                    Z = torch.vstack([Y, self.S_G.to(Y.dtype)])
                    self.last_bandwidth = float(pooled_bandwidth(Z))
            return out
        if self.bandwidth == "pooled_floor":
            Z = torch.vstack([Y, self.S_G.to(Y.dtype)])
            bw = pooled_bandwidth(Z).detach()
            bw = torch.clamp(bw, min=self.floor_frac * self.target_bw)
            self.kernel.bandwidth = bw
            self.last_bandwidth = float(bw)
            out = self.mmd(Y, self.S_G)
            self.kernel.bandwidth = None
            return out
        out = self.mmd(Y, self.S_G)
        if self.bandwidth == "pooled":
            with torch.no_grad():
                Z = torch.vstack([Y, self.S_G.to(Y.dtype)])
                self.last_bandwidth = float(pooled_bandwidth(Z))
        return out

    def __call__(self, Y):
        m2 = self._mmd2(Y)
        if self.transform == "mmd2":
            return m2
        if self.transform == "sqrt_abs_eps":
            return torch.sqrt(m2.abs() + self.eps)
        n, m = Y.shape[0], self.S_G.shape[0]
        c = self.floor_frac * self.n_kernels * (1.0 / n + 1.0 / m)
        return torch.sqrt(torch.clamp(m2, min=0.0) + c) - math.sqrt(c)
