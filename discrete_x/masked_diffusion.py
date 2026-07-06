"""Absorbing (masked) discrete diffusion over the sanity-task prompt space,
with an exact denoiser derived from the mixture-of-products prior.

Forward process (linear schedule, alpha_t = 1 - t/T): each position is masked
independently; at t=T everything is masked. Reverse ancestral step t -> t-1:
each masked position unmasks with probability
    (alpha_{t-1} - alpha_t) / (1 - alpha_t) = 1/t,
and an unmasked position is filled from the denoiser posterior p(x_0^i | x_t).

The denoiser is FACTORIZED, like a learned MDLM head: newly unmasked positions
are sampled independently from their per-position marginals given x_t. The
marginals themselves are exact (Bayes over the mixture component given the
unmasked tokens), so the only approximation is the factorization — the same
one every real masked-diffusion LM makes.
"""

import numpy as np

from sanity_task import MASK


def component_posterior(task, x):
    """p(k | unmasked tokens of x), shape (K,)."""
    logp = np.log(task.pi).copy()
    for i in range(task.L):
        if x[i] != MASK:
            logp += np.log(task.theta[i][:, x[i]])
    logp -= logp.max()
    p = np.exp(logp)
    return p / p.sum()

def posterior_marginals(task, x):
    """Per-position denoiser marginals p(x_0^i = v | x_t), list of (V_i,).
    For unmasked positions the marginal is a point mass on the observed token."""
    w = component_posterior(task, x)
    marg = []
    for i in range(task.L):
        if x[i] == MASK:
            marg.append(w @ task.theta[i])
        else:
            m = np.zeros(task.vocab_sizes[i])
            m[x[i]] = 1.0
            marg.append(m)
    return marg

def reverse_step(task, x, t, rng):
    """One ancestral reverse step t -> t-1 (unmask each masked position
    w.p. 1/t; t=1 unmasks everything)."""
    x = x.copy()
    masked = np.where(x == MASK)[0]
    if len(masked) == 0:
        return x
    p_unmask = 1.0 if t <= 1 else 1.0 / t
    to_unmask = masked[rng.random(len(masked)) < p_unmask]
    if len(to_unmask) == 0:
        return x
    marg = posterior_marginals(task, x)
    for i in to_unmask:
        x[i] = rng.choice(task.vocab_sizes[i], p=marg[i])
    return x
