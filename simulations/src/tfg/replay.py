"""[Agent M] Sample-replay MMD for the distributional guidance path (opt-in).

At outer step ``t`` the guidance MMD normally uses only the ``n_t`` conditional
samples generated at this step.  This module mixes in DETACHED conditional
samples from past steps with geometrically decaying representation
(``ReplayConfig``): generalisation of the upstream ``reuse_frac`` mechanism
(branch ``upstream/claude/hybrid-sampling-optimization-55fv3b``,
``Optimization.optimize_LGD``; see
``experiments/model-optimization/replay/ORI_IMPLEMENTATION.md``).  Derivations
(gradient with only fresh rows differentiable, bias as a trajectory-smoothed
objective, variance/ESS) in ``experiments/model-optimization/replay/THEORY.md``.

Everything lives OUTSIDE the engine, wrapped around the ``log_f`` callable
(:func:`wrap_log_f`), exactly like the rest of ``tfg/distributional.py``:

* :func:`replay_counts` -- deterministic largest-remainder split of a total
  MMD batch ``B`` into per-group counts ``[m_0..m_depth]`` with
  ``m_k ~ decay^k`` (``m_0`` is the fresh count; ``decay=0`` gives
  ``[B, 0, ..]`` -- replay exactly off).
* :class:`ReplayBuffer` -- stores the DETACHED fresh rows of the last
  ``depth`` steps, one buffer per spatial-perturbation index ``j`` (as the
  upstream code keeps ``prev_target_samples[j]``); a re-push for the same
  ``(t, j)`` (recurrence) replaces the entry.
* :func:`wrap_log_f` -- the loss assembler.  ``mode="subsample"``: draw
  ``r_k`` rows from the step-``t+k`` cache (uniform without replacement,
  NoiseTape-keyed ``("replay", t, j, k)`` so runs are tape-deterministic and
  order-independent) and evaluate the UNCHANGED ``DistributionalLoss`` (either
  backend) on the stacked fresh+replay batch.  ``mode="weighted"``: use every
  cached row with per-group weights ``W_k ~ decay^k`` in the weighted
  V-statistic :func:`weighted_mmd2` (fixed bandwidth, ``transform='mmd2'``;
  uses the fast backend's cached target blocks when available).

Only the fresh rows carry gradient; replayed rows are constants.  Conditional
calls are counted by the engine / ``CMSampler`` as usual, so the calls saving
of the ``batch_total`` mode is visible in ``cm_samples``.
"""

import math

import torch


def replay_counts(B, decay, depth):
    """Split ``B`` rows into groups ``k = 0..depth`` with ``m_k ~ decay^k``.

    Largest-remainder rounding (ties resolved toward smaller ``k``), so the
    counts always sum to ``B`` and are a deterministic function of the
    arguments.  ``decay = 0`` returns ``[B, 0, ..., 0]``.
    ``replay_counts(n, p/(1-p), 1)`` reproduces the upstream
    ``n_reuse = round(p * n)`` split for every ``p < 0.5`` fraction used there.
    """
    B, depth = int(B), int(depth)
    if B < 1:
        raise ValueError("B must be >= 1")
    if decay == 0.0:
        return [B] + [0] * depth
    w = [decay ** k for k in range(depth + 1)]
    s = sum(w)
    raw = [B * wk / s for wk in w]
    counts = [int(math.floor(r)) for r in raw]
    deficit = B - sum(counts)
    order = sorted(range(depth + 1), key=lambda k: (-(raw[k] - counts[k]), k))
    for k in order[:deficit]:
        counts[k] += 1
    if counts[0] < 1:                       # always keep at least one fresh row
        donor = max(range(1, depth + 1), key=lambda k: counts[k])
        counts[donor] -= 1
        counts[0] += 1
    return counts


def fill_counts(B, f, decay, depth):
    """[M-9] Top-up split: fresh count stays ``f``; ``max(B - f, 0)`` recycled
    rows are allocated over ``k = 1..depth`` proportional to ``decay^k``
    (largest-remainder, ties toward smaller ``k``).  Returns
    ``[f, m_1..m_depth]``.  ``decay = 0`` or ``B <= f`` gives no recycling.
    Counts are the PLAN; at run time each ``m_k`` is clamped by the cache
    size (``f`` rows per buffered step), so at ``f = 1`` the realised batch
    is capped at ``f + depth`` regardless of ``B``.
    """
    B, f, depth = int(B), int(f), int(depth)
    if f < 1:
        raise ValueError("f must be >= 1")
    top = max(B - f, 0)
    if top == 0 or decay == 0.0:
        return [f] + [0] * depth
    w = [decay ** k for k in range(1, depth + 1)]
    sw = sum(w)
    raw = [top * wk / sw for wk in w]
    counts = [int(math.floor(r)) for r in raw]
    deficit = top - sum(counts)
    order = sorted(range(depth), key=lambda k: (-(raw[k] - counts[k]), k))
    for k in order[:deficit]:
        counts[k] += 1
    return [f] + counts


def fifo_counts(B, f, depth=None):
    """[M-10] Uniform FIFO plan: the ``B - f`` recycled rows are the most
    recent rows, cohort by cohort (``f`` per past step), oldest partially
    included when ``B - f`` is not a multiple of ``f``.  Returns
    ``[f, m_1, ...]`` with ``sum = min(B, f * (1 + len-1)) = B`` for
    ``B >= f``; no recycling when ``B <= f``."""
    B, f = int(B), int(f)
    if f < 1:
        raise ValueError("f must be >= 1")
    top = max(B - f, 0)
    counts = [f]
    while top > 0:
        take = min(f, top)
        counts.append(take)
        top -= take
    if depth is not None:
        counts += [0] * (int(depth) + 1 - len(counts))
    return counts


def cohort_counts(B, f, depth=None):
    """[M-10] Capped-cohort thinning plan: cohort ``k`` (the ``k``-th most
    recent past step) keeps ``min(f, ceil(B / 2**k))`` rows, until the
    ``B - f`` recycled budget is spent (last cohort may be partial) -- an
    implicit smooth decay with a longer memory tail than a fixed-depth
    geometric."""
    B, f = int(B), int(f)
    if f < 1:
        raise ValueError("f must be >= 1")
    top = max(B - f, 0)
    counts = [f]
    k = 1
    while top > 0:
        cap = min(f, -(-B // (2 ** k)))          # ceil(B / 2^k)
        take = min(cap, top)
        if take == 0:
            break                                # cap hit 0 (cannot happen: ceil >= 1)
        counts.append(take)
        top -= take
        k += 1
    if depth is not None:
        counts += [0] * (int(depth) + 1 - len(counts))
    return counts


class ReplayBuffer:
    """Detached conditional samples of the last ``depth`` outer steps.

    The outer loop runs ``t = T..1`` downward, so the sample set generated
    ``k`` steps ago carries step index ``t + k``.  One independent buffer per
    spatial-perturbation index ``j``.
    """

    def __init__(self, depth):
        self.depth = int(depth)
        self._store = {}                      # j -> {t: rows}

    def push(self, t, j, rows):
        """Store the FRESH rows of step ``t`` (detached copy); replaces any
        existing entry for ``(t, j)`` and evicts entries older than ``depth``
        steps relative to ``t``."""
        d = self._store.setdefault(int(j), {})
        d[int(t)] = rows.detach().clone()
        for s in [s for s in d if s > int(t) + self.depth or s < int(t)]:
            del d[s]

    def entries(self, t, j):
        """``[(k, rows)]`` for the steps ``t + k``, ``k = 1..depth``, oldest
        last; only steps actually buffered appear."""
        d = self._store.get(int(j), {})
        return [(k, d[int(t) + k]) for k in range(1, self.depth + 1)
                if int(t) + k in d]


def subsample_rows(cache, r, tape, key):
    """``r`` rows of ``cache``, uniform without replacement, tape-determined.

    The tape draw ``randn(key, (len(cache),))`` induces a uniformly random
    permutation via argsort; the first ``r`` positions are the subsample.
    Deterministic given the tape seed and ``key``; independent across keys.
    """
    c = cache.shape[0]
    if r >= c:
        return cache
    v = tape.randn(key, (c,), dtype=torch.float64)
    idx = torch.argsort(v)[:r]
    return cache[idx]


def weighted_mmd2(loss, X, row_weights):
    """Weighted V-statistic ``sum_ij w_i w_j k(x_i,x_j) - 2 sum_i w_i mean_a
    k(x_i,y_a) + mean_ab k(y_a,y_b)`` against ``loss``'s target.

    ``loss`` is a ``tfg.distributional.DistributionalLoss`` with
    ``bandwidth='fixed'`` and ``transform='mmd2'`` (the screening
    configuration); ``row_weights`` must sum to 1.  Uses the fast backend's
    cached target blocks when ``loss`` was built with ``backend='fast'``,
    otherwise the reference ``RBF`` on the stacked ``(X;Y)`` -- both are the
    same value (see ``tests/test_fast_mmd_integration.py``).  With uniform
    weights this equals the unweighted ``loss(X)`` exactly.
    """
    if loss.bandwidth != "fixed":
        raise NotImplementedError("weighted replay needs bandwidth='fixed'")
    if loss.transform != "mmd2":
        raise NotImplementedError("weighted replay needs transform='mmd2'")
    w = row_weights.to(X.dtype)
    if abs(float(w.sum()) - 1.0) > 1e-4:
        raise ValueError("row_weights must sum to 1")
    fast = loss._fast
    if fast is not None:
        from tfg.fast_mmd import _sq_dists
        Xc = X.to(fast.dtype)
        wc = w.to(fast.dtype)
        X_sq = (Xc * Xc).sum(-1, keepdim=True)
        scales = fast._scales(fast.bandwidth)
        K_xx = fast._K(_sq_dists(Xc, Xc, X_sq, X_sq), scales)
        K_xy = fast._K(_sq_dists(Xc, fast.Y, X_sq, fast.Y_sq), scales)
        xx = wc @ K_xx @ wc
        xy = wc @ K_xy.mean(dim=1)
        return xx - 2.0 * xy + fast.YY_fixed
    S = loss.S_G.to(X.dtype)
    K = loss.kernel(torch.vstack([X, S]))
    n = X.shape[0]
    xx = w @ K[:n, :n] @ w
    xy = w @ K[:n, n:].mean(dim=1)
    yy = K[n:, n:].mean()
    return xx - 2.0 * xy + yy


def _parse_key(eta_keys):
    """Step ``t`` and perturbation ``j`` from the engine's eta keys
    (``("eta", t, i)`` or ``("eta", t, j, i)``)."""
    k0 = eta_keys[0]
    t = int(k0[1])
    j = int(k0[2]) if len(k0) == 4 else 0
    return t, j


def wrap_log_f(sampler, loss, tape, rcfg):
    """Replay-augmented ``log_f(x, n_t=None, eta_keys=None)``.

    ``sampler``/``loss`` are the ``CMSampler`` / ``DistributionalLoss`` of the
    plain distributional path; the fresh rows are ``sampler(x, eta_keys)``
    exactly as without replay (same tape keys, same count -- the engine's
    ``n_t`` IS the fresh count).  With ``rcfg.enabled=False`` or
    ``rcfg.decay=0`` the returned callable computes exactly
    ``-loss(sampler(x, eta_keys))``.
    """
    rcfg.validate()
    buf = ReplayBuffer(rcfg.depth)

    def log_f(x, n_t=None, eta_keys=None):
        if eta_keys is None:
            raise ValueError("replay requires the keyed log_f protocol "
                             "(n_schedule.enabled with eta keys)")
        fresh = sampler(x, eta_keys)
        if not rcfg.enabled or rcfg.decay == 0.0:
            return -loss(fresh)
        t, j = _parse_key(eta_keys)
        f = fresh.shape[0]
        entries = buf.entries(t, j)
        if rcfg.mode == "subsample":
            if rcfg.batch_total > 0 and rcfg.fill:
                policy = getattr(rcfg, "policy", "geometric")
                if policy == "fifo":
                    counts = fifo_counts(rcfg.batch_total, f, rcfg.depth)
                elif policy == "cohort":
                    counts = cohort_counts(rcfg.batch_total, f, rcfg.depth)
                else:
                    counts = fill_counts(rcfg.batch_total, f, rcfg.decay, rcfg.depth)
                if len(counts) > rcfg.depth + 1:
                    raise ValueError(
                        f"replay policy {policy!r} needs depth >= {len(counts) - 1}, "
                        f"got {rcfg.depth}")
            elif rcfg.batch_total > 0:
                counts = replay_counts(rcfg.batch_total, rcfg.decay, rcfg.depth)
                if counts[0] != f:
                    raise ValueError(
                        f"batch_total={rcfg.batch_total} implies fresh count "
                        f"{counts[0]}, but the engine supplied n_t={f}; set "
                        "n_schedule.n_max = replay_counts(B, decay, depth)[0]")
            else:
                counts = [f] + [int(round(f * rcfg.decay ** k))
                                for k in range(1, rcfg.depth + 1)]
            rows = [fresh]
            for k, cache in entries:
                r = min(counts[k], cache.shape[0])
                if r > 0:
                    sel = subsample_rows(cache, r, tape, ("replay", t, j, k))
                    rows.append(sel.to(fresh.dtype))
            out = -loss(torch.cat(rows, dim=0))
        elif rcfg.mode == "weighted":
            groups = [(0, fresh)] + list(entries)
            gw = torch.tensor([rcfg.decay ** k for k, _ in groups],
                              dtype=torch.float64)
            gw = gw / gw.sum()
            row_w = torch.cat([torch.full((rows.shape[0],),
                                          float(gw[i]) / rows.shape[0],
                                          dtype=torch.float64)
                               for i, (_, rows) in enumerate(groups)])
            out = -weighted_mmd2(loss, torch.cat([r for _, r in groups], dim=0),
                                 row_w)
        else:                                  # pragma: no cover - validate()d
            raise ValueError(rcfg.mode)
        buf.push(t, j, fresh)
        return out

    log_f.buffer = buf
    return log_f
