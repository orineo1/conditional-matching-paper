"""[Agent B] Importance-selected backpropagation for the guidance MMD (opt-in).

At each predictor evaluation the guidance gradient decomposes as

    dL/dx = sum_{i=1}^{n} J_i^T g_i,   J_i = dy_i/dx,   g_i = dL/dy_i,

where the ``g_i`` are kernel-only (cheap) and each ``J_i^T g_i`` needs a
backward pass through the conditional sampler (expensive: graph memory +
VJP).  This module (theory: ``experiments/model-optimization/backsel/THEORY.md``)

1. generates ALL ``n`` conditional samples under ``torch.no_grad()`` (full
   forward, zero autograd graphs),
2. computes the full-batch loss value and every output-space gradient ``g_i``
   (one autograd call on a leaf copy of ``Y`` -- no conditional model
   involved),
3. selects ``k << n`` samples by a tape-keyed rule (``("backsel", t, j)``),
4. regenerates EXACTLY those samples WITH graphs by replaying their
   per-sample eta keys (``CMSampler`` keys ``("eta", t[, j], i)`` are
   per-sample-index: the replayed NOISE is bit-identical; the regenerated
   rows equal the no-grad rows to round-off -- 1e-14 in float64, ~1e-6
   relative in float32, because batched BLAS rounding depends on the batch
   dimension; ``tests/test_backsel.py`` asserts both),
5. returns a scalar whose VALUE is the full-batch ``-loss`` and whose
   gradient w.r.t. ``x`` is ``-sum_{i in S} w_i J_i^T g_i`` via the surrogate
   ``sum_{i in S} (w_i g_i)^T y_i`` with ``g_i`` detached.

Unbiasedness of the two stochastic rules (``h_i := J_i^T g_i``, ``G = sum h_i``):

* ``uniform``: k of n without replacement, inclusion probability
  ``pi_i = k/n``, Horvitz-Thompson weight ``w_i = n/k``:
  ``E[G_hat] = sum_i pi_i (n/k) h_i = G``.
* ``importance``: k iid draws from
  ``p_i = (1-eps) ||g_i||/sum_j ||g_j|| + eps/n`` (``eps = floor``),
  ``G_hat = (1/k) sum_m h_{d_m}/p_{d_m}``:
  ``E[G_hat] = (1/k) k sum_i p_i h_i/p_i = G``; de-duplicating repeated
  draws into ``w_i = c_i/(k p_i)`` (``c_i`` the multiplicity) leaves the
  estimator, and hence its expectation, unchanged.  The floor bounds every
  weight by ``n/(k*eps)`` and the variance by ``1/eps`` times uniform's.

``kcenter`` is the APPROXIMATE third rule: greedy k-center on the ``y_i``
(tape-keyed start row), differentiate only the ``k`` centers and push the
cluster-AGGREGATED output gradient ``g_eff(c) = sum_{i in C_c} g_i`` through
each center's Jacobian -- all output-gradient mass kept, bias = the
within-cluster Jacobian substitution error, zero selection variance.

All rules reduce to the identity (every sample, unit weights) when
``k >= n``, restoring the exact full gradient (float64 test at 1e-12).
Everything lives OUTSIDE the engine, wrapped around ``log_f``
(:func:`wrap_log_f`), exactly like ``tfg/replay.py``.
"""

import torch


def output_gradients(loss, Y):
    """``(g, value)``: the output-space gradient ``dL/dY`` at detached ``Y``
    (shape of ``Y``) and the detached full-batch loss value.

    One autograd call on a leaf copy of ``Y`` through ``loss`` -- for the RBF
    V-statistic this is the closed-form kernel gradient (THEORY.md sec 1) at
    the cost of one extra kernel evaluation; no conditional model involved.
    """
    Yl = Y.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        val = loss(Yl)
        (g,) = torch.autograd.grad(val, Yl)
    return g.detach(), val.detach()


def _identity(g):
    return list(range(g.shape[0])), g.clone()


def select_uniform(g, k, tape, key):
    """k of n uniform without replacement (tape argsort); returns
    ``(indices, g_eff)`` with ``g_eff_i = (n/k) g[idx_i]`` (Horvitz-Thompson).
    """
    n = g.shape[0]
    if k >= n:
        return _identity(g)
    v = tape.randn(key + ("uni",), (n,), dtype=torch.float64)
    idx = sorted(torch.argsort(v)[:k].tolist())
    return idx, g[idx] * (n / k)


def select_importance(g, k, tape, key, floor=0.25):
    """k iid draws from ``p_i ~ (1-floor) ||g_i||/sum + floor/n``, de-duplicated;
    returns ``(indices, g_eff)`` with ``g_eff_i = (c_i / (k p_i)) g[idx_i]``.
    Unbiased for any ``floor`` in (0, 1]; see module docstring.
    """
    n = g.shape[0]
    if k >= n:
        return _identity(g)
    norms = g.detach().double().norm(dim=-1)
    s = float(norms.sum())
    if s > 0:
        p = (1.0 - floor) * norms / s + floor / n
    else:
        p = torch.full((n,), 1.0 / n, dtype=torch.float64)
    p = p / p.sum()                                   # exact normalisation
    # k iid categorical draws from the tape: normals -> uniforms -> inverse CDF
    z = tape.randn(key + ("is",), (k,), dtype=torch.float64)
    u = torch.special.ndtr(z).clamp(1e-12, 1.0 - 1e-12)
    cdf = torch.cumsum(p, 0)
    cdf[-1] = 1.0
    draws = torch.searchsorted(cdf, u)
    counts = torch.bincount(draws, minlength=n)
    idx = sorted(torch.nonzero(counts, as_tuple=False).reshape(-1).tolist())
    w = torch.tensor([float(counts[i]) / (k * float(p[i])) for i in idx],
                     dtype=torch.float64)
    return idx, g[idx] * w.to(g.dtype).unsqueeze(-1)


def select_kcenter(Y, g, k, tape, key):
    """Greedy k-center on the rows of ``Y`` (tape-keyed start), one
    representative per cluster; returns ``(center_indices, g_eff)`` with
    ``g_eff_c = sum_{i in cluster c} g_i`` (all mass kept, weights folded in).
    Deterministic given the tape.  ``k >= n``: singleton clusters (identity).
    """
    n = Y.shape[0]
    if k >= n:
        return _identity(g)
    Yd = Y.detach().double()
    start = int(torch.argmin(tape.randn(key + ("kc",), (n,), dtype=torch.float64)))
    centers = [start]
    d2 = ((Yd - Yd[start]) ** 2).sum(-1)
    for _ in range(k - 1):
        nxt = int(torch.argmax(d2))
        centers.append(nxt)
        d2 = torch.minimum(d2, ((Yd - Yd[nxt]) ** 2).sum(-1))
    centers = sorted(centers)
    C = Yd[centers]                                    # (k, d)
    assign = torch.cdist(Yd, C).argmin(dim=1)          # (n,)
    g_eff = torch.zeros((len(centers),) + g.shape[1:], dtype=g.dtype)
    g_eff.index_add_(0, assign, g)
    return centers, g_eff


def select_stratified(Y, g, k, tape, key):
    """[B-R7] Stratified: k-center clusters, ONE tape-random member per
    cluster as representative (not the center); ``g_eff_c`` = the cluster's
    summed g.  ``k >= n``: identity."""
    n = Y.shape[0]
    if k >= n:
        return _identity(g)
    centers, _ = select_kcenter(Y, g, k, tape, key)
    Yd = Y.detach().double()
    assign = torch.cdist(Yd, Yd[centers]).argmin(dim=1)
    u = tape.randn(key + ("strat",), (n,), dtype=torch.float64)
    reps = [max(torch.nonzero(assign == c).reshape(-1).tolist(), key=lambda i: float(u[i]))
            for c in range(len(centers))]
    order = sorted(range(len(reps)), key=lambda c: reps[c])
    g_eff = torch.zeros((len(reps),) + g.shape[1:], dtype=g.dtype)
    g_eff.index_add_(0, assign, g)
    return [reps[c] for c in order], g_eff[order]


def kcenter_centers(Y, k, start):
    """Greedy farthest-point k-center on the rows of ``Y`` (float64, Euclidean)
    from ``start``; returns ``(sorted centers, nearest-center assignment)``."""
    Yd = Y.detach().double()
    centers = [int(start)]
    d2 = ((Yd - Yd[start]) ** 2).sum(-1)
    for _ in range(k - 1):
        nxt = int(torch.argmax(d2))
        centers.append(nxt)
        d2 = torch.minimum(d2, ((Yd - Yd[nxt]) ** 2).sum(-1))
    centers = sorted(centers)
    assign = torch.cdist(Yd, Yd[centers]).argmin(dim=1)
    return centers, assign


def balanced_assignment(Y, centers, capacity):
    """Capacity-constrained assignment of every row to one of ``centers``:
    rows are processed closest-first (distance to their nearest center) and
    each takes its nearest center with free capacity.  Returns ``assign`` (n,)
    with values in ``range(len(centers))``; every stratum holds <= capacity."""
    Yd = Y.detach().double()
    D = torch.cdist(Yd, Yd[list(centers)])                # (n, k)
    n, k = D.shape
    assign = torch.full((n,), -1, dtype=torch.long)
    load = torch.zeros(k, dtype=torch.long)
    for i in torch.argsort(D.min(dim=1).values).tolist():
        for c in torch.argsort(D[i]).tolist():
            if load[c] < capacity:
                assign[i] = c
                load[c] += 1
                break
    return assign


def select_stratified_balanced(Y, g, k, tape, key):
    """[Agent S] The SD ``strat`` rule: k-center strata with capacity
    ``ceil(n/k)`` (``balanced_assignment``), ONE tape-random member per stratum
    as representative, ``g_eff_r = |C_c| g_r``.  Unbiased (stratified
    Horvitz-Thompson: ``E[|C| J_r^T g_r] = sum_{i in C} J_i^T g_i``), weights
    ``<= ceil(n/k)``, and no more than ``ceil(n/k)`` gradients ever sit behind
    one Jacobian -- the failure mode of ``kcenter`` on SD
    (``experiments/model-optimization/sd/BACKSEL_DIAG.md``).  ``k >= n``: identity.
    Returns ``(reps, g_eff, sizes)``."""
    n = Y.shape[0]
    if k >= n:
        idx, ge = _identity(g)
        return idx, ge, [1] * n
    start = int(torch.argmin(tape.randn(key + ("kc",), (n,), dtype=torch.float64)))
    centers, _ = kcenter_centers(Y, k, start)
    assign = balanced_assignment(Y, centers, -(-n // k))
    u = tape.randn(key + ("strat",), (n,), dtype=torch.float64)
    reps, sizes = [], []
    for c in range(len(centers)):
        members = torch.nonzero(assign == c).reshape(-1).tolist()
        if not members:
            continue
        reps.append(max(members, key=lambda i: float(u[i])))
        sizes.append(len(members))
    order = sorted(range(len(reps)), key=lambda c: reps[c])
    idx = [reps[c] for c in order]
    w = torch.tensor([float(sizes[c]) for c in order], dtype=g.dtype)
    return idx, g[idx] * w.reshape(-1, *([1] * (g.dim() - 1))), [sizes[c] for c in order]


def soft_tau(Y, idx, mode="local", scale=1.0, bandwidth=None):
    """[Agent S] The temperature of :func:`soft_aggregate`, ONE rule for both
    pipelines.

    ``local``      ``scale`` x median over the NON-selected rows of the squared
                   distance to their nearest representative (scale-free; the SD
                   default -- the global MMD bandwidth measures the target
                   spread and is ~100x too large, backsel/REPORT.md sec 7).
    ``bandwidth``  ``scale`` x ``bandwidth`` when given (the synthetic path passes
                   the loss's bandwidth), else ``scale`` x the median heuristic
                   on ``Y`` (median off-diagonal squared distance / 2).
    """
    Yd = Y.detach().double()
    n = Yd.shape[0]
    sel = torch.tensor(list(idx), dtype=torch.long)
    if mode == "local":
        mask = torch.ones(n, dtype=torch.bool)
        mask[sel] = False
        D = torch.cdist(Yd[mask], Yd[sel]) ** 2
        tau = float(torch.median(D.min(dim=1).values)) if D.numel() else 1.0
    elif mode == "bandwidth":
        if bandwidth is not None:
            tau = float(bandwidth)
        else:
            d2 = torch.cdist(Yd, Yd) ** 2
            off = d2[~torch.eye(n, dtype=torch.bool)]
            tau = float(torch.median(off)) / 2.0
    else:
        raise ValueError(f"unknown soft tau_mode {mode!r}")
    return max(tau * float(scale), 1e-12)


def soft_aggregate(Y, g, idx, tau, return_mass=False):
    """[B-R7] Soft assignment: every NON-selected row ``j`` spreads ``g_j``
    over the selected rows ``i in idx`` with
    ``a_ji = softmax_i(-||y_j - y_i||^2 / tau)``; selected rows keep their own
    ``g_i`` in full.  Returns ``g_eff`` (k, d) with ``sum_i g_eff_i = sum_j g_j``
    exactly.  ``tau -> 0`` is the hard nearest-representative assignment
    (= ``select_kcenter``'s aggregation when ``idx`` are the centers);
    ``tau -> inf`` splits every non-selected ``g_j`` equally over ``idx``."""
    Yd = Y.detach().double()
    gd = g.detach().double()
    n = Yd.shape[0]
    sel = torch.tensor(idx, dtype=torch.long)
    d2 = torch.cdist(Yd, Yd[sel]) ** 2                       # (n, k)
    A = torch.softmax(-d2 / float(tau), dim=1)
    mask = torch.ones(n, dtype=torch.bool)
    mask[sel] = False
    g_eff = gd[sel] + A[mask].T @ gd[mask]
    if return_mass:                      # units of gradient mass per representative
        return g_eff.to(g.dtype), (1.0 + A[mask].sum(dim=0))
    return g_eff.to(g.dtype)


def _parse_key(eta_keys):
    """Step ``t`` and perturbation ``j`` from the engine's eta keys
    (``("eta", t, i)`` or ``("eta", t, j, i)``)."""
    k0 = eta_keys[0]
    t = int(k0[1])
    j = int(k0[2]) if len(k0) == 4 else 0
    return t, j


def _plain_inner(sampler, loss):
    def log_f(x, n_t=None, eta_keys=None):
        return -loss(sampler(x, eta_keys))
    return log_f


def wrap_log_f(sampler, loss, tape, bcfg, inner=None):
    """Backsel-wrapped ``log_f(x, n_t=None, eta_keys=None)``.

    ``sampler``/``loss`` are the ``CMSampler`` / ``DistributionalLoss`` of the
    plain distributional path.  With ``bcfg.enabled=False`` the returned
    callable computes exactly ``-loss(sampler(x, eta_keys))``.  The wrapper
    exposes ``log_f.stats`` with the two cost currencies:
    ``forward_samples`` (no-grad + regenerated forwards) and ``diff_samples``
    (samples actually carrying autograd graphs).

    ``inner`` (optional composition hook): a factory
    ``inner(sampler_like, loss_like) -> log_f`` that assembles the MMD batch
    from the fresh rows ``sampler_like(x, eta_keys)`` and evaluates
    ``loss_like(X_stacked)`` -- e.g.
    ``lambda s, l: tfg.replay.wrap_log_f(s, l, tape, rcfg)`` (subsample /
    fill / cohort modes; ``weighted`` is not supported).  Backsel then runs
    the assembled batch ONCE under no_grad (the proxies record the fresh rows
    and the output gradients of the FIRST ``n`` rows of the stacked batch),
    selects among the fresh rows only (the recycled rows are constants, so
    ``dL/dx = sum_{i in fresh} J_i^T g_i`` exactly) and regenerates the
    selected fresh rows with graphs.  With ``bcfg.enabled=False`` the plain
    ``inner(sampler, loss)`` callable is returned.
    """
    bcfg.validate()
    if getattr(sampler, "cache_on", False):
        raise ValueError("backsel requires CMSampler(cache=False): cached rows "
                         "were produced under no_grad and carry no graph")
    stats = {"forward_samples": 0, "diff_samples": 0}
    inner = _plain_inner if inner is None else inner
    plain = inner(sampler, loss)
    if not bcfg.enabled:
        plain.stats = stats
        return plain
    rec = {}

    class _SamplerProxy:                 # fresh rows, no graphs, recorded
        cache_on = False

        def __call__(self, x, keys):
            with torch.no_grad():
                Y = sampler(x, keys)
            rec["Y"], rec["n"] = Y, len(keys)
            return Y

    class _LossProxy:                    # full-batch value + output gradients
        def __call__(self, X):
            g, val = output_gradients(loss, X)
            rec["g"], rec["val"] = g[:rec["n"]], val
            return val

    assembled = inner(_SamplerProxy(), _LossProxy())

    def log_f(x, n_t=None, eta_keys=None):
        if eta_keys is None:
            raise ValueError("backsel requires the keyed log_f protocol "
                             "(n_schedule.enabled with eta keys)")
        n = len(eta_keys)
        # (1)+(2) full forward without graphs; full-batch value and the
        # output-space gradients of the fresh rows (kernel-only)
        rec.clear()
        assembled(x, n_t, eta_keys)
        Y, g, val = rec["Y"], rec["g"], rec["val"]
        # (3) tape-keyed selection
        t, j = _parse_key(eta_keys)
        key = ("backsel", t, j)
        if bcfg.rule == "uniform":
            idx, g_eff = select_uniform(g, bcfg.k, tape, key)
        elif bcfg.rule == "importance":
            idx, g_eff = select_importance(g, bcfg.k, tape, key, floor=bcfg.floor)
        elif bcfg.rule == "stratified":
            idx, g_eff = select_stratified(Y, g, bcfg.k, tape, key)
        elif bcfg.rule == "stratified_balanced":       # [Agent S]
            idx, g_eff, _ = select_stratified_balanced(Y, g, bcfg.k, tape, key)
        else:                                          # "kcenter", validate()d
            idx, g_eff = select_kcenter(Y, g, bcfg.k, tape, key)
        if bcfg.weighting == "soft" and len(idx) < n:  # [B-R7]
            bw = getattr(loss, "last_bandwidth", None)
            if bcfg.tau_mode == "bandwidth" and bw is None:
                raise ValueError("backsel weighting='soft' needs a fixed-bandwidth loss")
            tau = soft_tau(Y, idx, mode=bcfg.tau_mode, scale=bcfg.tau_mult,
                           bandwidth=(float(bw) if bw is not None else None))
            g_eff = soft_aggregate(Y, g, idx, tau)
        # (4) regenerate exactly the selected samples WITH graphs
        y_sel = sampler(x, [eta_keys[i] for i in idx])
        stats["forward_samples"] += n + len(idx)
        stats["diff_samples"] += len(idx)
        # (5) full-batch VALUE, subset gradient
        surr = (y_sel * g_eff.to(y_sel.dtype)).sum()
        return -(val.to(surr.dtype) + surr - surr.detach())

    log_f.stats = stats
    return log_f
