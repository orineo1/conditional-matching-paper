"""Twisted SMC / Feynman-Kac guidance for masked discrete diffusion.

The loss L(x_0) is consumed strictly as a scalar — no gradients w.r.t. x.
Particles are propagated by the untwisted reverse kernel p_theta(x_{t-1}|x_t)
and reweighted by h_{t-1}(x_{t-1}) / h_t(x_t), with

    h_t(x_t) = exp(-beta * L(x0_hat(x_t)))            [estimator="mode"]
    h_t(x_t) = (1/n_dec) sum_j exp(-beta * L(x0_j))   [estimator="sampled"],
               x0_j ~ p_theta(x_0 | x_t)

The sampled variant averages exp(-beta L), NOT L: it is a plug-in Monte Carlo
estimate of the exact twist E[e^{-beta L(x_0)} | x_t]. Systematic resampling
fires when ESS < ess_frac * N.

Both estimators share this skeleton, the same RNG seeds, and the same task.loss
implementation; separate RNG streams for propagation / estimator decodes /
resampling guarantee the ONLY difference between runs is the x0_hat estimator
(the extra decode draws in "sampled" cannot desync the propagation stream).
"""

import json
import time

import numpy as np

from masked_diffusion import posterior_marginals, reverse_step
from sanity_task import MASK


def logsumexp(a):
    m = np.max(a)
    return m + np.log(np.sum(np.exp(a - m)))


def estimate_twist(task, x, estimator, n_dec, beta, rng):
    """Return (log h_t(x), L_hat, decoded_text) for one particle.

    L_hat is the per-particle scalar logged as "L across particles":
    the single decode's loss for "mode", the mean over decodes for "sampled".
    """
    masked = np.where(x == MASK)[0]
    if len(masked) == 0:  # fully decoded: both estimators reduce to L(x) exactly
        l = task.loss(x)
        return -beta * l, l, task.decode(x)
    marg = posterior_marginals(task, x)
    if estimator == "mode":
        x0 = x.copy()
        for i in masked:
            x0[i] = int(np.argmax(marg[i]))
        l = task.loss(x0)
        return -beta * l, l, task.decode(x0)
    elif estimator == "sampled":
        losses = np.empty(n_dec)
        first_decode = None
        for j in range(n_dec):
            x0 = x.copy()
            for i in masked:
                x0[i] = rng.choice(task.vocab_sizes[i], p=marg[i])
            losses[j] = task.loss(x0)
            if j == 0:
                first_decode = task.decode(x0)
        logh = logsumexp(-beta * losses) - np.log(n_dec)
        return logh, float(losses.mean()), first_decode
    raise ValueError(f"unknown estimator: {estimator}")


def systematic_resample(W, rng):
    N = len(W)
    positions = (rng.random() + np.arange(N)) / N
    return np.minimum(np.searchsorted(np.cumsum(W), positions), N - 1)


def run_smc(
    task,
    estimator,
    n_particles=64,
    T=10,
    beta=10.0,
    n_dec=4,
    seed=0,
    ess_frac=0.5,
    log_path=None,
    beta_anneal=False,
    n_dec_early=None,
):
    """beta_anneal: use a time-dependent twist beta_s = beta * (T - s) / T for
    state time s (weak guidance early where the x0_hat estimate is noisiest,
    full beta at s=0, so the final FK target P(x0) e^{-beta L(x0)} is
    unchanged). Time-dependent twists are standard FK; the telescoping weight
    h_{s}(x_s)/h_{s+1}(x_{s+1}) remains valid.

    n_dec_early: for the "sampled" estimator, use this many decodes (instead of
    n_dec) while s > T/2, where most positions are masked and the Monte Carlo
    twist estimate has the highest variance. Defaults preserve the original
    constant-beta / constant-n_dec behavior exactly."""
    rng_prop = np.random.default_rng(10_000 + seed)
    rng_est = np.random.default_rng(20_000 + seed)
    rng_res = np.random.default_rng(30_000 + seed)

    def beta_at(s):
        return beta * (T - s) / T if beta_anneal else beta

    def n_dec_at(s):
        if n_dec_early is not None and s > T / 2:
            return n_dec_early
        return n_dec

    N = n_particles
    particles = np.full((N, task.L), MASK, dtype=int)
    logw = np.zeros(N)

    # h_T on the (identical, fully masked) initial particles — a common
    # constant, computed once so the first incremental weight telescopes.
    logh_prev = np.full(
        N,
        estimate_twist(
            task, particles[0], estimator, n_dec_at(T), beta_at(T), rng_est
        )[0],
    )

    records = []
    n_resamples = 0
    t_start = time.perf_counter()

    for t in range(T, 0, -1):
        logh_new = np.empty(N)
        l_hat = np.empty(N)
        p0_text = None
        b_s, nd_s = beta_at(t - 1), n_dec_at(t - 1)  # twist params of state x_{t-1}
        for n in range(N):
            particles[n] = reverse_step(task, particles[n], t, rng_prop)
            lh, l, dec = estimate_twist(
                task, particles[n], estimator, nd_s, b_s, rng_est
            )
            logh_new[n] = lh
            l_hat[n] = l
            if n == 0:
                p0_text = dec

        logw += logh_new - logh_prev
        logh_prev = logh_new

        lw = logw - logsumexp(logw)
        W = np.exp(lw)
        ess = 1.0 / np.sum(W**2)

        resampled = False
        if ess < ess_frac * N:
            idx = systematic_resample(W, rng_res)
            particles = particles[idx]
            logh_prev = logh_prev[idx]
            logw = np.zeros(N)
            n_resamples += 1
            resampled = True

        records.append(
            {
                "t": t - 1,  # state is x_{t-1} after the step
                "ess": float(ess),
                "w_var": float(np.var(W)),
                "L_min": float(l_hat.min()),
                "L_max": float(l_hat.max()),
                "resampled": resampled,
                "p0_x0_decode": p0_text,
            }
        )

    wall_clock = time.perf_counter() - t_start

    # Final particles are fully decoded; L(x_0) is exact for each.
    final_losses = np.array([task.loss(particles[n]) for n in range(N)])
    best = int(np.argmin(final_losses))
    x_best = particles[best].copy()

    result = {
        "estimator": estimator,
        "seed": seed,
        "beta": beta,
        "beta_anneal": beta_anneal,
        "n_dec": n_dec if estimator == "sampled" else 1,
        "n_dec_early": n_dec_early if estimator == "sampled" else None,
        "x_best": x_best.tolist(),
        "x_best_decode": task.decode(x_best),
        "L_best": float(final_losses[best]),
        "mmd2_eval": float(task.eval_mmd(x_best)),
        "emb_dist_to_true": task.emb_dist_to_true(x_best),
        "slot_acc_vs_true": float(np.mean(x_best == task.x_true)),
        "exact_recovery": bool(np.all(x_best == task.x_true)),
        "final_L_mean": float(final_losses.mean()),
        "n_resamples": n_resamples,
        "wall_clock_s": wall_clock,
        "ess_traj": [r["ess"] for r in records],
    }

    if log_path is not None:
        with open(log_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    return result, records
