"""Frozen reference implementation of TFG Algorithm 1.

IMPORTANT PROVENANCE NOTE
-------------------------
This is a **new** implementation, written for this project directly from
Algorithm 1 of

    Ye, Lin, Han, Xu, Liu, Liang, Ma, Zou, Ermon.
    "TFG: Unified Training-Free Guidance for Diffusion Models." NeurIPS 2024.
    arXiv:2409.15761v2, Algorithm 1, page 5.

It is **not** a pre-existing implementation from this repository.  Reconnaissance
established that no TFG implementation existed here -- only the LGD subspace
(``Optimization.py:88 optimize_LGD``, which is ``N_recur=1, N_iter=0, mu=0``,
with ``rho`` hard-coded to 1).  Consequently the equivalence tests in
``tests/test_equivalence.py`` compare *two new implementations* of the same
algorithm, which is a weaker guarantee than checking a new engine against
established code.  This limitation must be stated in any writeup.

This module is deliberately dumb: no configuration objects, no extension hooks,
no caching, no adaptivity.  It transcribes the pseudocode and nothing else.  It
is imported only by tests.  Do not add features here.

Algorithm 1, verbatim:

     1  Input: eps_theta, f, rho, mu, gamma_bar, T, N_recur, N_iter
     2  x_T ~ N(0, I)
     3  for t = T, ..., 1 do
     4      f_tilde(x) = E_{delta~N(0,I)} f(x + gamma_bar * sqrt(1 - alphabar_t) * delta)
     5      for r = 1, ..., N_recur do
     6          x_{0|t} = (x_t - sqrt(1 - alphabar_t) * eps_theta(x_t, t)) / sqrt(alphabar_t)
     7          Delta_t = rho_t * grad_{x_t} log f_tilde(x_{0|t})
     8          Delta_0 = Delta_0 + mu_t * grad_{x_0|t} log f_tilde(x_{0|t} + Delta_0)
     9          x_{t-1} = Sample(x_t, x_{0|t}, t) + Delta_t / sqrt(alpha_t)
                          + sqrt(alphabar_{t-1}) * Delta_0
    10          x_t ~ N(sqrt(alpha_t) x_{t-1}, (1 - alpha_t) I)
    11      end for
    12  end for
    13  Output: x_0

Pinned conventions where the pseudocode is ambiguous.  Both this module and
``tfg/engine.py`` follow them identically; they are the two places equivalence
could silently break.

  C1. Line 4 draws the smoothing noise ONCE PER OUTER STEP ``t``, before the
      recurrence loop.  The same ``delta`` values are reused across all ``r``
      and across all ``N_iter`` inner iterations.  Tape key: ``("delta", t, j)``
      -- deliberately no ``r`` and no inner index.
  C2. Line 10 re-noises only when ``r < N_recur``.  Re-noising on the final
      recurrence would discard the ``x_{t-1}`` that the step just produced.
  C3. Line 8 differentiates with respect to ``x_{0|t}`` treated as a leaf
      (detached), so the mean-guidance branch does not backpropagate through
      ``eps_theta``.  Line 7 does backpropagate through ``eps_theta``.
  C4a. Line 9 divides Delta_t by ``sqrt(alpha_t)`` -- the PER-STEP alpha, NOT
      ``sqrt(alphabar_t)``.  Verified against the published text, where line 6
      reads ``sqrt(\bar alpha_t)`` (with bar) and line 9 reads
      ``sqrt(\alpha_t)`` (no bar).  The derivation agrees: perturbing x_t by
      Delta_t with eps held fixed moves x_{0|t} by Delta_t/sqrt(alphabar_t),
      so x_{t-1} moves by sqrt(alphabar_{t-1}) * Delta_t / sqrt(alphabar_t)
      = Delta_t * sqrt(alphabar_{t-1}/alphabar_t) = Delta_t / sqrt(alpha_t).
      An earlier revision of this file used sqrt(alphabar_t) and was wrong by
      a factor of 1/sqrt(alphabar_{t-1}), up to 64x at T=100.
  C4. ``Sample`` on line 9 is DDIM with ``eta = 0``:
      ``x_{t-1} = sqrt(alphabar_{t-1}) * x_{0|t} + sqrt(1 - alphabar_{t-1}) * eps_theta``.
  C5. ``log f_tilde`` is evaluated in log space as
      ``logsumexp_j log f(...) - log(n_mc)``, never as ``log(mean(exp(...)))``,
      which underflows for the composite MMD predictor.
"""

import math

import torch


def _noop_trace(name, t, r, k, tensor):
    pass


def log_f_tilde(log_f, x, t, gamma_bar, n_mc, schedule, tape):
    """Monte-Carlo estimate of ``log E_delta f(x + gamma_bar*sqrt(1-ab_t)*delta)``.

    Implements convention C1 (key without ``r``) and C5 (log-space average).
    ``gamma_bar == 0`` still draws and still averages, so that the number of
    predictor evaluations does not depend on ``gamma_bar``.  With the tape this
    costs nothing in correctness terms and keeps the two engines comparable.
    """
    scale = gamma_bar * torch.sqrt(1.0 - schedule.alphabar[t])
    terms = []
    for j in range(n_mc):
        delta = tape.randn(("delta", int(t), int(j)), x.shape,
                           device=x.device, dtype=x.dtype)
        terms.append(log_f(x + scale * delta))
    stacked = torch.stack(terms)
    return torch.logsumexp(stacked, dim=0) - math.log(n_mc)


def run_reference_tfg(eps_theta, log_f, schedule, tape, shape,
                      N_recur=1, N_iter=0, rho=None, mu=None,
                      gamma_bar=0.0, n_mc=1, trace=None):
    """Run TFG Algorithm 1.

    Parameters
    ----------
    eps_theta:
        ``eps_theta(x, t) -> tensor`` with the same shape as ``x``.
    log_f:
        ``log_f(x) -> 0-dim tensor``, the log target predictor. Must be
        differentiable in ``x``.
    schedule:
        :class:`tfg.schedule.DiffusionSchedule`.
    tape:
        :class:`tfg.noise_tape.NoiseTape`.
    shape:
        Shape of ``x_T``, e.g. ``(1, d)``.
    rho, mu:
        Length ``T+1`` tensors indexed by ``t``.
    trace:
        Optional ``trace(name, t, r, k, tensor)`` callback. Must not consume
        randomness or alter values; see ``tfg/trace.py``.

    Returns
    -------
    The final ``x_0``, detached.
    """
    trace = _noop_trace if trace is None else trace
    T = schedule.T
    dtype, device = schedule.dtype, schedule.device

    if rho is None:
        rho = torch.zeros(T + 1, dtype=dtype, device=device)
    if mu is None:
        mu = torch.zeros(T + 1, dtype=dtype, device=device)

    # Line 2
    x_t = tape.randn(("x_T",), shape, device=device, dtype=dtype)
    trace("x_T", T, None, None, x_t)

    # Line 3
    for t in range(T, 0, -1):
        # Independence note: we read the raw ``alphabar`` array and form every
        # derived quantity here, rather than calling schedule.sqrt_ab() /
        # .alpha() as tfg/engine.py does.  The two modules must share the
        # schedule *constants* -- they are the same physical schedule -- but
        # they must not share the algebra that turns those constants into an
        # update step, or the equivalence test becomes circular.
        ab_t = schedule.alphabar[t]
        ab_prev = schedule.alphabar[t - 1]
        sqrt_ab_t = torch.sqrt(ab_t)
        sqrt_1mab_t = torch.sqrt(1.0 - ab_t)
        sqrt_ab_prev = torch.sqrt(ab_prev)
        sqrt_1mab_prev = torch.sqrt(1.0 - ab_prev)
        alpha_t = ab_t / ab_prev

        # Line 5
        for r in range(1, N_recur + 1):
            x_t = x_t.detach().requires_grad_(True)
            trace("x_t_in", t, r, None, x_t)

            # Line 6
            eps = eps_theta(x_t, t)
            trace("eps_theta", t, r, None, eps)
            x0 = (x_t - sqrt_1mab_t * eps) / sqrt_ab_t
            trace("x0_pred", t, r, None, x0)

            # Line 7 -- gradient w.r.t. x_t, through eps_theta (C3)
            lf = log_f_tilde(log_f, x0, t, gamma_bar, n_mc, schedule, tape)
            trace("log_f_tilde_rho", t, r, None, lf)
            grad_rho, = torch.autograd.grad(lf, x_t, retain_graph=False)
            trace("grad_rho_raw", t, r, None, grad_rho)
            Delta_t = rho[t] * grad_rho
            trace("Delta_t", t, r, None, Delta_t)

            # Line 8 -- gradient w.r.t. x_{0|t} as a leaf (C3)
            x0_leaf = x0.detach()
            Delta_0 = torch.zeros_like(x0_leaf)
            for k in range(N_iter):
                probe = (x0_leaf + Delta_0).detach().requires_grad_(True)
                lf0 = log_f_tilde(log_f, probe, t, gamma_bar, n_mc, schedule, tape)
                grad_mu, = torch.autograd.grad(lf0, probe)
                trace("grad_mu_raw", t, r, k, grad_mu)
                Delta_0 = Delta_0 + mu[t] * grad_mu
                trace("Delta_0_iter", t, r, k, Delta_0)
            trace("Delta_0", t, r, None, Delta_0)

            # Line 9 -- DDIM eta=0 (C4), then guidance
            with torch.no_grad():
                x_ddim = sqrt_ab_prev * x0_leaf + sqrt_1mab_prev * eps.detach()
                trace("x_ddim", t, r, None, x_ddim)
                x_prev = (x_ddim
                          + Delta_t.detach() / torch.sqrt(alpha_t)
                          + sqrt_ab_prev * Delta_0)
                trace("x_prev", t, r, None, x_prev)

                # Line 10 -- re-noise only if another recurrence follows (C2)
                if r < N_recur:
                    noise = tape.randn(("renoise", int(t), int(r)), x_prev.shape,
                                       device=device, dtype=dtype)
                    trace("renoise_eps", t, r, None, noise)
                    x_t = torch.sqrt(alpha_t) * x_prev + torch.sqrt(1.0 - alpha_t) * noise
                    trace("x_t_out", t, r, None, x_t)
                else:
                    x_t = x_prev

    trace("x_0", 0, None, None, x_t)
    return x_t.detach()
