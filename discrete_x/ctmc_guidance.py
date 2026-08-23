"""Rate-ratio guidance on a continuous-time masked-diffusion chain
(TFG-Flow / Nisonoff-style), as an alternative to twisted SMC.

Single chain, no particles, no resampling, no importance weights. The
scalar predictor f(x1) = exp(-beta * L(x1)) enters by reweighting the
CTMC jump destinations: for a masked coordinate d unmasking at time s,

    R(x, d->v | y)  =  rate(s) * E[f(x1) 1{x1^d=v} | x_s] / E[f(x1) | x_s],

(TFG-Flow eq. 5 specialized to masking, gamma=1). Two estimators for the
posterior expectations:

  mc        K full decodes x^(k) ~ prod_d p_{1|s}(. | x_s), weights
            w_k = f(x^(k)); guided destination law at coordinate d is the
            f-weighted empirical distribution of the decodes' values at d
            (TFG-Flow eq. 6). Needs only the scalar loss oracle — no
            gradients, no cross-modal surrogate. K oracle calls per step.

  additive  the relaxing assumption: f is log-linear across coordinates,
            log f(x1) = sum_d theta_d(x1^d). Then the expectation
            factorizes and the guided law is softmax(log p_{1|s} +
            gamma * theta_d) in closed form (eq. 7): exact, zero variance,
            ZERO oracle calls during sampling. theta is fitted once per
            run as the least-squares additive projection of -beta*L on a
            design of decoded prompts.

Time convention: s in [0,1], all-masked at s=0, clean data at s=1, linear
schedule kappa(s)=s. A masked coordinate unmasks on [s, s+ds] w.p.
ds/(1-s); guidance only redistributes the destination VALUE (under
masking, the total unmask rate is the same for every x1, so timing is
unguided — a structural property, not a choice).

Generic over the denoiser: `marginals_fn(x) -> list of (V_d,) arrays`
(p_{1|s}(x^d=v | x); point mass on observed tokens), and over the oracle:
`loss_fn(x) -> float` for a fully decoded int canvas.
"""

import numpy as np

MASK = -1


def logsumexp(a, axis=None):
    if axis is None:
        m = np.max(a)
        return float(m + np.log(np.sum(np.exp(a - m))))
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def sample_factorized(marg, masked, x, K, rng):
    """K full decodes from the factorized posterior (unmasked coords kept)."""
    X = np.tile(x, (K, 1))
    for d in masked:
        X[:, d] = rng.choice(len(marg[d]), size=K, p=marg[d])
    return X


def mc_guided_law(marg_d, decodes_d, logf, gamma):
    """Guided destination law at one coordinate from K weighted decodes.
    q(v) ∝ phat(v) * exp(gamma * (logfbar(v) - logfbar)), which at gamma=1
    reduces to sum_k f_k 1{x_k^d = v} (TFG-Flow eq. 5/6)."""
    V = len(marg_d)
    logq = np.full(V, -np.inf)
    # only destinations seen among the K decodes can carry mass (coverage
    # limit of the MC estimator, kept deliberately — see writeup)
    for v in np.unique(decodes_d):
        sel = decodes_d == v
        n_v = int(sel.sum())
        logfbar_v = logsumexp(logf[sel]) - np.log(n_v)
        logq[v] = np.log(n_v / len(decodes_d)) + gamma * logfbar_v
    if not np.isfinite(logq).any():
        return marg_d  # all mass vanished numerically: fall back to prior
    q = np.exp(logq - logq.max())
    return q / q.sum()


def additive_guided_law(marg_d, theta_d, gamma, scale=1.0):
    """Closed-form guided law under the additive assumption (eq. 7):
    q(v) ∝ p_{1|s}(v) * exp(gamma * scale * theta_d(v))."""
    logq = np.log(marg_d + 1e-300) + gamma * scale * theta_d
    q = np.exp(logq - logq.max())
    return q / q.sum()


def fit_additive_theta(vocab_sizes, X_design, losses, beta, ridge=1e-3):
    """Least-squares additive projection of log f = -beta*L onto per-slot
    one-hot features: theta_d(v) minimizing ||y - sum_d theta_d(x^d)||^2.
    Returns list of (V_d,) arrays (centered per slot; the global constant
    is irrelevant to the guided law)."""
    D = len(vocab_sizes)
    offs = np.concatenate([[0], np.cumsum(vocab_sizes)])
    F = np.zeros((len(X_design), offs[-1]))
    for n, x in enumerate(X_design):
        for d in range(D):
            F[n, offs[d] + int(x[d])] = 1.0
    y = -beta * np.asarray(losses, dtype=float)
    A = F.T @ F + ridge * np.eye(offs[-1])
    th = np.linalg.solve(A, F.T @ y)
    return [th[offs[d]:offs[d + 1]] - th[offs[d]:offs[d + 1]].mean()
            for d in range(D)]


def guided_ctmc_sample(marginals_fn, loss_fn, vocab_sizes, rng,
                       n_steps=20, beta=200.0, gamma=1.0,
                       estimator="mc", K=64, theta=None,
                       x_init=None, s0=0.0, anneal=True, eta=0.0,
                       remask_guided=False, record=None):
    """One guided reverse chain from s0 to 1. estimator in
    {"unguided", "mc", "additive"}. x_init: source canvas for SDEdit-style
    starts (already containing MASKs); None = all-masked from s0=0.
    eta: DFM detailed-balance stochasticity (Campbell et al. 2024): the
    generating rate is augmented with R^DB satisfying detailed balance
    w.r.t. the conditional marginals, which for masking means unmasked
    coordinates REMASK at rate eta while the unmask rate grows to
    (kappa' + eta*kappa)/(1-kappa) — same marginal flow for the prior,
    but the chain can now revise committed tokens.
    remask_guided: SELECTIVE revision (mc estimator only). Keeps the same
    total churn (sum of remask probabilities = eta*ds*|unmasked|) but
    allocates it by softmax of the per-coordinate commit regret
    Delta_d = L_commit(d) - L_maskavg(d), both cached for free from the
    decodes at d's commit step: tokens that underperformed their own
    refill distribution re-mask preferentially (rate-ratio guidance
    applied to the remask jumps, zero extra oracle calls; stats are
    stale by design — commit-time context).
    Returns the decoded canvas (int array)."""
    D = len(vocab_sizes)
    if x_init is None:
        x = np.full(D, MASK, dtype=int)
    else:
        x = np.asarray(x_init, dtype=int).copy()
    L_commit = np.zeros(D)   # mean loss of decodes that chose the token
    L_maskavg = np.zeros(D)  # mean loss over all refills at commit time
    grid = np.linspace(s0, 1.0, n_steps + 1)
    for j in range(n_steps):
        s, s_next = grid[j], grid[j + 1]
        masked = np.where(x == MASK)[0]
        last = j == n_steps - 1
        if len(masked) == 0 and (eta == 0 or last):
            break
        ds = s_next - s
        rate_un = (1.0 + eta * s) / max(1.0 - s, 1e-12)
        p_un = 1.0 if last else min(1.0, ds * rate_un)
        to_unmask = masked[rng.random(len(masked)) < p_un]
        # DFM detailed-balance remask, decided from the start-of-step
        # state (tau-leap simultaneity), never on the final step
        to_remask = np.array([], dtype=int)
        if eta > 0 and not last:
            unmasked = np.where(x != MASK)[0]
            if remask_guided and len(unmasked) > 1:
                delta = L_commit[unmasked] - L_maskavg[unmasked]
                tau = max(float(np.std(delta)), 1e-4)
                w = np.exp((delta - delta.max()) / tau)
                w /= w.sum()
                p_rm = np.minimum(1.0, len(unmasked) * eta * ds * w)
            else:
                p_rm = np.full(len(unmasked), min(1.0, eta * ds))
            to_remask = unmasked[rng.random(len(unmasked)) < p_rm]
        if len(to_unmask) == 0 and len(to_remask) == 0:
            continue
        marg = marginals_fn(x)
        b_s = beta * s_next if anneal else beta
        if len(to_unmask) == 0:
            pass  # remask-only step: no destination choice, no oracle
        elif estimator == "mc":
            Xd = sample_factorized(marg, masked, x, K, rng)
            losses_k = np.array([loss_fn(Xd[k]) for k in range(K)])
            logf = -b_s * losses_k
            for d in to_unmask:
                q = mc_guided_law(marg[d], Xd[:, d], logf, gamma)
                x[d] = rng.choice(vocab_sizes[d], p=q)
                sel = Xd[:, d] == x[d]
                L_maskavg[d] = losses_k.mean()
                L_commit[d] = (losses_k[sel].mean() if sel.any()
                               else L_maskavg[d])
        elif estimator == "additive":
            scale = s_next if anneal else 1.0
            for d in to_unmask:
                q = additive_guided_law(marg[d], theta[d], gamma, scale)
                x[d] = rng.choice(vocab_sizes[d], p=q)
        else:  # unguided
            for d in to_unmask:
                x[d] = rng.choice(vocab_sizes[d], p=marg[d])
        if len(to_remask):
            x[to_remask] = MASK
        if record is not None:
            record.append({"s": float(s_next), "n_masked": int((x == MASK).sum())})
    assert (x != MASK).all()
    return x
