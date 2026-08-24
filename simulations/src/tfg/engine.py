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
  * Component 1 (``n_schedule``)          -- implemented, off by default;
    [A4] ``type="adaptive"`` (gradient-agreement / improvement policies),
    ``eta_per_perturbation``, ``eta_keying="frozen"``.
  * Component 2 (``temporal_cache``)      -- [A4] ``implementation="stale"``
    implemented (approximate, stale-gradient reuse); ``"gated"`` still refused.
  * Component 3 (``adaptive_recurrence``) -- [A4] ``implementation="v1"``
    implemented (early stopping); ``"gated"`` still refused.
  * [A4] legacy-compatibility switches ``init``, ``guidance_scaling``,
    ``smoothing`` and the ``temporal.grad_norm`` pre-processing.  With these the
    engine reproduces ``experiments/_guided.py::run`` bit-for-bit
    (``tests/test_engine_matches_guided.py``).
"""

import math

import torch

from tfg import n_schedule as n_sched
from tfg.adam_guidance import AdamGuidance
from tfg.adaptive import gradient_agreement, recurrence_should_stop
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
        # [A4] per-step diagnostics (empty unless the mechanism is active)
        self.agreement_history = []
        self.grad_norm_history = []
        self.stale_steps = 0

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
            "agreement_history": list(self.agreement_history),
            "grad_norm_history": list(self.grad_norm_history),
            "stale_steps": self.stale_steps,
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
        # [A4] stale-gradient cache (component 2, implementation="stale")
        self._stale_grad = None
        self._stale_age = 0
        # [A4 round 2] history of RAW per-step gradient norms for the
        # scale-free clipping rules (clip_rel / clip_quantile)
        self._raw_norms = []
        self._ema_norm = None

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

    def _smoothing_scale(self, t):
        cfg = self.config
        if cfg.smoothing == "lgd_beta":
            # Repository LGD convention: r_t = beta_t / sqrt(1 + beta_t^2),
            # independent of gamma_bar. beta_t is the schedule's per-step beta.
            beta_t = self.schedule.betas[t - 1]
            return beta_t / torch.sqrt(1 + beta_t ** 2)
        return cfg.gamma_bar * self.schedule.sqrt_one_minus_ab(t)

    def _eta_keys_for(self, t, n_t, j):
        ns = self.config.n_schedule
        if n_t is None:
            return None
        return n_sched.conditional_seed_keys(
            t, n_t, j=(j if ns.eta_per_perturbation else None),
            frozen=(ns.eta_keying == "frozen"))

    def _log_f_tilde(self, x, t, n_t, eta_keys, key_slice=None):
        """C1 + C5: delta keyed on (t, j) only; averaged in log space.

        [A4] ``key_slice`` restricts the conditional keys (half batches for the
        agreement statistic). With ``eta_per_perturbation`` the keys are built
        per ``j`` here and ``eta_keys`` (the shared ones) is ignored.
        """
        cfg = self.config
        scale = self._smoothing_scale(t)
        terms = []
        for j in range(cfg.n_mc):
            delta = self.tape.randn(("delta", int(t), int(j)), x.shape,
                                    device=x.device, dtype=x.dtype)
            keys = (self._eta_keys_for(t, n_t, j)
                    if cfg.n_schedule.eta_per_perturbation else eta_keys)
            n_call = n_t
            if key_slice is not None and keys is not None:
                keys = keys[key_slice]
                n_call = len(keys)
            terms.append(self._call_log_f(x + scale * delta, n_call, keys))
        return torch.logsumexp(torch.stack(terms), dim=0) - math.log(cfg.n_mc)

    def _clip_to(self, grad, nrm, threshold):
        factor = torch.clamp(threshold / (nrm + self.config.temporal.grad_eps), max=1.0)
        return grad * factor

    def _preprocess(self, grad):
        """[A4] gradient-norm clipping / normalisation, before the temporal rule.

        The scale-free rules (``clip_rel``, ``clip_quantile``) use the history
        of RAW norms of the previous steps only (never the current one), so
        the first step is never clipped and the rule is causal."""
        tc = self.config.temporal
        if tc.grad_norm == "none":
            return grad
        nrm = grad.norm()
        if tc.grad_norm == "clip":
            return self._clip_to(grad, nrm, tc.grad_clip)
        if tc.grad_norm == "unit":
            return grad / (nrm + tc.grad_eps)
        raw = float(nrm)
        hist = self._raw_norms
        out = grad
        if tc.grad_norm == "clip_rel":
            if tc.clip_ref == "median":
                ref = (sorted(hist)[len(hist) // 2] if hist else None)
            else:
                ref = self._ema_norm
            if ref is not None:
                out = self._clip_to(grad, nrm, tc.grad_clip * ref)
        elif tc.grad_norm == "clip_quantile":
            if hist:
                srt = sorted(hist)
                idx = min(len(srt) - 1, int(tc.grad_clip * len(srt)))
                out = self._clip_to(grad, nrm, srt[idx])
        # update the history with the RAW norm (after deciding this step)
        hist.append(raw)
        self._ema_norm = (raw if self._ema_norm is None
                          else tc.clip_ema * self._ema_norm + (1 - tc.clip_ema) * raw)
        return out

    def _step_clip(self, Delta_t, t, x_ddim, x_t):
        """[A4 round 2] trust region on the applied step (config.temporal.step_clip)."""
        tc = self.config.temporal
        if tc.step_clip == "none":
            return Delta_t
        if tc.step_clip == "noise":
            ref = tc.step_tau * self.schedule.sqrt_one_minus_ab(t)
        else:
            ref = tc.step_tau * (x_ddim - x_t.detach()).norm()
        nrm = Delta_t.norm()
        factor = torch.clamp(ref / (nrm + tc.grad_eps), max=1.0)
        return Delta_t * factor

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
        ns = cfg.n_schedule
        ar = cfg.adaptive_recurrence
        tc = cfg.temporal_cache
        adaptive_n = ns.enabled and ns.type == "adaptive"
        agreement_on = adaptive_n and ns.policy == "agreement"
        n_recur_max = ar.max_recurrences if ar.enabled else cfg.N_recur

        if cfg.init == "zeros":
            x_t = torch.zeros(shape, device=device, dtype=dtype)
        else:
            x_t = self.tape.randn(("x_T",), shape, device=device, dtype=dtype)
        trace("x_T", T, None, None, x_t)

        # [A4] adaptive-n state carried across outer steps
        adapt_state = None
        if adaptive_n:
            adapt_state = {"n_prev": None, "n_min": ns.n_min, "n_start": ns.n_start,
                           "grow": ns.grow, "policy": ns.policy,
                           "agreement": None, "agreement_threshold": ns.agreement_threshold,
                           "improved": None, "improvement_threshold": ns.improvement_threshold,
                           "budget_remaining": ns.budget_total, "steps_left": T}
        prev_lf_value = None

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
            if ns.enabled:
                if adaptive_n:
                    adapt_state["steps_left"] = t
                    n_t = n_sched.n_at(t, sch, ns.n_max, ns.kappa, ns.type,
                                       state=adapt_state)
                else:
                    n_t = n_sched.n_at(t, sch, ns.n_max, ns.kappa, ns.type)
                eta_keys = self._eta_keys_for(t, n_t, None)
            else:
                n_t, eta_keys = None, None
            self.counter.n_t_history.append(n_t if n_t is not None else 1)

            # [A4] component 2 ("stale"): skip the predictor entirely on
            # non-refresh steps and re-apply the cached rho-branch gradient.
            use_stale = (tc.enabled and self._stale_grad is not None
                         and self._stale_age < tc.refresh_every)

            n_recur_used = 0
            prev_rec = None
            for r in range(1, n_recur_max + 1):
                n_recur_used = r
                x_t = x_t.detach().requires_grad_(True)
                trace("x_t_in", t, r, None, x_t)

                eps = self.eps_theta(x_t, t)
                self.counter.denoiser_calls += 1
                trace("eps_theta", t, r, None, eps)

                x0 = (x_t - sqrt_1mab_t * eps) / sqrt_ab_t
                trace("x0_pred", t, r, None, x0)

                if use_stale:
                    grad_rho = self._stale_grad
                    self._stale_age += 1
                    self.counter.stale_steps += 1
                    lf_value = prev_lf_value
                else:
                    # rho branch: gradient w.r.t. x_t, through the denoiser.
                    lf = self._log_f_tilde(x0, t, n_t, eta_keys)
                    trace("log_f_tilde_rho", t, r, None, lf)
                    if agreement_on and n_t >= 2:
                        h = n_t // 2
                        lf_a = self._log_f_tilde(x0, t, n_t, eta_keys, key_slice=slice(0, h))
                        lf_b = self._log_f_tilde(x0, t, n_t, eta_keys, key_slice=slice(h, n_t))
                        grad_rho, = torch.autograd.grad(lf, x_t, retain_graph=True)
                        g_a, = torch.autograd.grad(lf_a, x_t, retain_graph=True)
                        g_b, = torch.autograd.grad(lf_b, x_t, retain_graph=False)
                        agreement = gradient_agreement(g_a, g_b)
                        self.counter.agreement_history.append(agreement)
                        adapt_state["agreement"] = agreement
                    else:
                        grad_rho, = torch.autograd.grad(lf, x_t, retain_graph=False)
                    lf_value = float(lf.detach())
                    if tc.enabled:
                        fresh = grad_rho.detach()
                        if self._stale_grad is not None and tc.lambda_value > 0:
                            fresh = ((1 - tc.lambda_value) * fresh
                                     + tc.lambda_value * self._stale_grad)
                        self._stale_grad = fresh
                        self._stale_age = 1
                trace("grad_rho_raw", t, r, None, grad_rho)
                if cfg.temporal.grad_norm != "none" or tc.enabled:
                    self.counter.grad_norm_history.append(float(grad_rho.norm()))

                # Temporal treatment, applied to the rho-branch gradient before
                # rho_t scaling. With mode="adam" this reproduces upstream's
                #   guidance = AdaptiveMomentEstimate(g); guidance *= strength
                #   x_prev  += guidance / alpha_t ** 0.5
                # since line 9 below supplies the 1/sqrt(alpha_t).
                grad_used = self._temporal(self._preprocess(grad_rho))
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
                    Delta_applied = self._step_clip(Delta_t.detach(), t, x_ddim, x_t)
                    if cfg.guidance_scaling == "raw":
                        guidance = Delta_applied
                    else:
                        guidance = Delta_applied / torch.sqrt(alpha_t)
                    x_prev = (x_ddim + guidance + sqrt_ab_prev * Delta_0)
                    trace("x_prev", t, r, None, x_prev)

                    # [A4] component 3 ("v1"): stop recurring early.
                    stop = False
                    if ar.enabled and r < n_recur_max:
                        cur_rec = {"log_f": (lf_value if lf_value is not None
                                             else float("nan")),
                                   "x_prev": x_prev, "grad": grad_rho.detach()}
                        stop = recurrence_should_stop(ar.metric, ar.threshold,
                                                      prev_rec, cur_rec)
                        prev_rec = cur_rec

                    # C2: re-noise only when a further recurrence follows.
                    if r < n_recur_max and not stop:
                        noise = self.tape.randn(("renoise", int(t), int(r)),
                                                x_prev.shape,
                                                device=device, dtype=dtype)
                        trace("renoise_eps", t, r, None, noise)
                        x_t = (torch.sqrt(alpha_t) * x_prev
                               + torch.sqrt(1.0 - alpha_t) * noise)
                        trace("x_t_out", t, r, None, x_t)
                    else:
                        x_t = x_prev
                if stop:
                    break

            self.counter.recurrence_history.append(n_recur_used)
            if adaptive_n:
                adapt_state["n_prev"] = n_t
                if ns.budget_total:
                    adapt_state["budget_remaining"] -= n_t
                if lf_value is not None and prev_lf_value is not None:
                    adapt_state["improved"] = lf_value - prev_lf_value
            if not use_stale:
                prev_lf_value = lf_value

        trace("x_0", 0, None, None, x_t)
        return x_t.detach()
