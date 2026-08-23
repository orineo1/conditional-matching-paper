"""Generalised TFG engine.

With every extension disabled (:meth:`TFGConfig.all_extensions_disabled`) this
executes TFG Algorithm 1 and must agree with ``tfg.reference`` to within
float64 round-off.  The two modules were written separately and share only the
:class:`~tfg.noise_tape.NoiseTape`, the :class:`~tfg.schedule.DiffusionSchedule`,
and the caller-supplied ``eps_theta`` / ``log_f``.

Conventions C1-C5 documented in ``tfg/reference.py`` are followed here
identically.  They are the load-bearing agreements; if equivalence ever breaks,
check them first.

Extension status:
  * Component 1 (``n_schedule``)          -- implemented, off by default.
  * Component 2 (``temporal_cache``)      -- rejected by config validation.
  * Component 3 (``adaptive_recurrence``) -- rejected by config validation.
"""

import math

import torch

from tfg import n_schedule as n_sched
from tfg.adam_guidance import AdamGuidance
from tfg.config import TFGConfig
from tfg.schedule import structured_vector


def _noop_trace(name, t, r, k, tensor):
    pass


class CallCounter:
    """Counts the two compute currencies the experiments must match on."""

    def __init__(self):
        self.denoiser_calls = 0
        self.conditional_calls = 0
        self.predictor_evals = 0
        self.n_t_history = []
        self.recurrence_history = []

    def as_dict(self):
        return {
            "denoiser_calls": self.denoiser_calls,
            "conditional_calls": self.conditional_calls,
            "predictor_evals": self.predictor_evals,
            "n_t_mean": (sum(self.n_t_history) / len(self.n_t_history)
                         if self.n_t_history else None),
            "n_t_history": list(self.n_t_history),
            "recurrence_mean": (sum(self.recurrence_history) / len(self.recurrence_history)
                                if self.recurrence_history else None),
            "recurrence_history": list(self.recurrence_history),
        }


class GeneralizedTFG:
    """TFG Algorithm 1 with switchable extensions.

    Parameters
    ----------
    eps_theta:
        ``eps_theta(x, t) -> tensor``.
    log_f:
        ``log_f(x, n_t=None, eta_keys=None) -> 0-dim tensor``. When every
        extension is disabled the engine calls it as ``log_f(x)``, exactly as
        the reference does, so a plain single-argument predictor works
        unchanged. ``n_t``/``eta_keys`` are supplied only when the
        ``n_schedule`` extension is on.
    schedule, tape, config:
        See the respective modules.
    """

    def __init__(self, eps_theta, log_f, schedule, tape, config=None):
        self.eps_theta = eps_theta
        self.log_f = log_f
        self.schedule = schedule
        self.tape = tape
        self.config = (config or TFGConfig()).validate()
        self.counter = CallCounter()

        T = schedule.T
        if self.config.T != T:
            raise ValueError(
                f"config.T={self.config.T} disagrees with schedule.T={T}"
            )
        # Temporal operator on the rho-branch gradient. inv_sqrt_alpha=False
        # because the engine already divides Delta_t by sqrt(alpha_t) on line 9;
        # enabling both would apply the factor twice.
        t = self.config.temporal
        self._adam = (AdamGuidance(beta1=t.beta1, beta2=t.beta2, delta=t.delta,
                                   rho=t.adam_rho, inv_sqrt_alpha=False)
                      if t.mode == "adam" else None)
        self._prev_used = None

        self.rho = structured_vector(self.config.rho_scalar,
                                     self.config.rho_structure, schedule)
        self.mu = structured_vector(self.config.mu_scalar,
                                    self.config.mu_structure, schedule)

    # -- predictor ---------------------------------------------------------

    def _call_log_f(self, x, n_t, eta_keys):
        self.counter.predictor_evals += 1
        if self.config.n_schedule.enabled:
            self.counter.conditional_calls += n_t
            return self.log_f(x, n_t=n_t, eta_keys=eta_keys)
        self.counter.conditional_calls += 1
        return self.log_f(x)

    def _log_f_tilde(self, x, t, n_t, eta_keys):
        """C1 + C5: delta keyed on (t, j) only; averaged in log space."""
        cfg = self.config
        scale = cfg.gamma_bar * self.schedule.sqrt_one_minus_ab(t)
        terms = []
        for j in range(cfg.n_mc):
            delta = self.tape.randn(("delta", int(t), int(j)), x.shape,
                                    device=x.device, dtype=x.dtype)
            terms.append(self._call_log_f(x + scale * delta, n_t, eta_keys))
        return torch.logsumexp(torch.stack(terms), dim=0) - math.log(cfg.n_mc)

    def _temporal(self, grad):
        """Apply the configured temporal operator to the rho-branch gradient."""
        mode = self.config.temporal.mode
        if mode == "none":
            return grad
        if mode == "adam":
            return self._adam.step(grad)
        # "lambda": retained option, not evaluated in the current experiments.
        lam = self.config.temporal.lam_max
        out = grad if self._prev_used is None else (
            (1 - lam) * grad + lam * self._prev_used)
        self._prev_used = out.detach()
        return out

    # -- main loop ---------------------------------------------------------

    def run(self, shape, trace=None):
        trace = _noop_trace if trace is None else trace
        cfg = self.config
        sch = self.schedule
        T = sch.T
        dtype, device = sch.dtype, sch.device

        x_t = self.tape.randn(("x_T",), shape, device=device, dtype=dtype)
        trace("x_T", T, None, None, x_t)

        for t in range(T, 0, -1):
            sqrt_ab_t = sch.sqrt_ab(t)
            sqrt_1mab_t = sch.sqrt_one_minus_ab(t)
            sqrt_ab_prev = sch.sqrt_ab(t - 1)
            sqrt_1mab_prev = sch.sqrt_one_minus_ab(t - 1)
            alpha_t = sch.alpha(t)

            # Component 1: conditional sample count for this outer step. The
            # eta keys carry no recurrence and no loss-evaluation index, so the
            # same conditional draws are reused across all recurrences and all
            # loss evaluations within this outer step, and the next outer step
            # draws fresh ones.
            if cfg.n_schedule.enabled:
                n_t = n_sched.n_at(t, sch, cfg.n_schedule.n_max,
                                   cfg.n_schedule.kappa, cfg.n_schedule.type)
                eta_keys = n_sched.conditional_seed_keys(t, n_t)
            else:
                n_t, eta_keys = None, None
            self.counter.n_t_history.append(n_t if n_t is not None else 1)

            n_recur_used = 0
            for r in range(1, cfg.N_recur + 1):
                n_recur_used = r
                x_t = x_t.detach().requires_grad_(True)
                trace("x_t_in", t, r, None, x_t)

                eps = self.eps_theta(x_t, t)
                self.counter.denoiser_calls += 1
                trace("eps_theta", t, r, None, eps)

                x0 = (x_t - sqrt_1mab_t * eps) / sqrt_ab_t
                trace("x0_pred", t, r, None, x0)

                # rho branch: gradient w.r.t. x_t, through the denoiser.
                lf = self._log_f_tilde(x0, t, n_t, eta_keys)
                trace("log_f_tilde_rho", t, r, None, lf)
                grad_rho, = torch.autograd.grad(lf, x_t, retain_graph=False)
                trace("grad_rho_raw", t, r, None, grad_rho)

                # Temporal treatment, applied to the rho-branch gradient before
                # rho_t scaling. With mode="adam" this reproduces upstream's
                #   guidance = AdaptiveMomentEstimate(g); guidance *= strength
                #   x_prev  += guidance / alpha_t ** 0.5
                # since line 9 below supplies the 1/sqrt(alpha_t).
                grad_used = self._temporal(grad_rho)
                # Traced ONLY when a temporal operator is active, so the
                # default path emits exactly the key set that
                # tfg/reference.py emits and the Algorithm 1 equivalence
                # comparison stays structurally identical.
                if cfg.temporal.mode != "none":
                    trace("grad_rho_used", t, r, None, grad_used)
                Delta_t = self.rho[t] * grad_used
                trace("Delta_t", t, r, None, Delta_t)

                # mu branch: gradient w.r.t. x_{0|t} treated as a leaf (C3).
                x0_leaf = x0.detach()
                Delta_0 = torch.zeros_like(x0_leaf)
                for k in range(cfg.N_iter):
                    probe = (x0_leaf + Delta_0).detach().requires_grad_(True)
                    lf0 = self._log_f_tilde(probe, t, n_t, eta_keys)
                    grad_mu, = torch.autograd.grad(lf0, probe)
                    trace("grad_mu_raw", t, r, k, grad_mu)
                    Delta_0 = Delta_0 + self.mu[t] * grad_mu
                    trace("Delta_0_iter", t, r, k, Delta_0)
                trace("Delta_0", t, r, None, Delta_0)

                with torch.no_grad():
                    x_ddim = sqrt_ab_prev * x0_leaf + sqrt_1mab_prev * eps.detach()
                    trace("x_ddim", t, r, None, x_ddim)
                    # Line 9 divides by sqrt(alpha_t), the per-step alpha,
                    # NOT sqrt(alphabar_t). See convention C4a in
                    # tfg/reference.py for the derivation.
                    x_prev = (x_ddim
                              + Delta_t.detach() / torch.sqrt(alpha_t)
                              + sqrt_ab_prev * Delta_0)
                    trace("x_prev", t, r, None, x_prev)

                    # C2: re-noise only when a further recurrence follows.
                    if r < cfg.N_recur:
                        noise = self.tape.randn(("renoise", int(t), int(r)),
                                                x_prev.shape,
                                                device=device, dtype=dtype)
                        trace("renoise_eps", t, r, None, noise)
                        x_t = (torch.sqrt(alpha_t) * x_prev
                               + torch.sqrt(1.0 - alpha_t) * noise)
                        trace("x_t_out", t, r, None, x_t)
                    else:
                        x_t = x_prev

            self.counter.recurrence_history.append(n_recur_used)

        trace("x_0", 0, None, None, x_t)
        return x_t.detach()
