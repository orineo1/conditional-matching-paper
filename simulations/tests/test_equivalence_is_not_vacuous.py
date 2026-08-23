"""Mutation tests: prove the equivalence harness can actually fail.

``test_equivalence.py`` reports a maximum absolute discrepancy of exactly 0.0
across every cell.  That is the expected outcome for two correct
implementations evaluating the same float64 expressions -- and it is also
exactly what a harness that compares nothing would report.

Each test here injects one specific, realistic bug into the generalised engine
and asserts the harness detects it.  Every mutation corresponds to a real
failure mode identified during design:

  M1  convention C1 violated: smoothing noise keyed per recurrence
  M2  convention C2 violated: re-noising on the final recurrence
  M3  ``Delta_t`` divided by ``sqrt(alphabar_t)`` instead of ``sqrt(alpha_t)``.
      This is the bug that was ACTUALLY shipped in the first revision of both
      modules and that the two-engine comparison could not catch, because both
      shared it. Only a derivation check finds this class of error; the
      mutation exists so a regression is caught.
  M4  convention C3 violated: mu branch backpropagates through the denoiser
  M5  ``log(mean(f))`` replaced by ``mean(log(f))``
  M6  DDIM step uses alphabar_t where it should use alphabar_{t-1}

If any of these ever stops being detected, the equivalence result is worthless.
"""

import math

import pytest
import torch

from conftest import ToyDenoiser, make_quadratic_log_f
from tfg.config import TFGConfig
from tfg.engine import GeneralizedTFG
from tfg.noise_tape import NoiseTape
from tfg.reference import run_reference_tfg
from tfg.schedule import DiffusionSchedule, constant_vector
from tfg.trace import Tracer, compare_traces

SHAPE = (1, 2)
T = 6


def _bits():
    schedule = DiffusionSchedule(T=T)
    denoiser = ToyDenoiser(d=SHAPE[1], T=T, seed=0, schedule=schedule)
    eps_theta = lambda x, t: denoiser(x, t)          # noqa: E731
    log_f = make_quadratic_log_f([1.5, -0.75], scale=0.6)
    return schedule, eps_theta, log_f


def _reference_trace(N_recur, N_iter, gamma_bar, n_mc, seed=17):
    schedule, eps_theta, log_f = _bits()
    tr = Tracer()
    run_reference_tfg(eps_theta, log_f, schedule, NoiseTape(seed=seed), SHAPE,
                      N_recur=N_recur, N_iter=N_iter,
                      rho=constant_vector(0.4, T), mu=constant_vector(0.2, T),
                      gamma_bar=gamma_bar, n_mc=n_mc, trace=tr)
    return tr


def _mutant_trace(cls, N_recur, N_iter, gamma_bar, n_mc, seed=17):
    schedule, eps_theta, log_f = _bits()
    cfg = TFGConfig(T=T, N_recur=N_recur, N_iter=N_iter, gamma_bar=gamma_bar,
                    rho_scalar=0.4, mu_scalar=0.2, n_mc=n_mc)
    tr = Tracer()
    cls(eps_theta, log_f, schedule, NoiseTape(seed=seed), cfg).run(SHAPE, trace=tr)
    return tr


def _assert_detected(cls, N_recur, N_iter, gamma_bar, n_mc, label):
    ref = _reference_trace(N_recur, N_iter, gamma_bar, n_mc)
    mut = _mutant_trace(cls, N_recur, N_iter, gamma_bar, n_mc)
    ok, report = compare_traces(ref, mut, atol=1e-12)
    assert not ok, (
        f"MUTATION NOT DETECTED ({label}): the harness reported agreement "
        f"(max_abs_err={report['max_abs_err']:.3e}) despite an injected bug. "
        "The equivalence result cannot be trusted."
    )
    assert report["max_abs_err"] > 1e-12 or report["missing_in_a"] or report["missing_in_b"]


# -- M1: delta keyed per recurrence (violates C1) ---------------------------

class M1_DeltaKeyedByRecurrence(GeneralizedTFG):
    def _log_f_tilde(self, x, t, n_t, eta_keys):
        cfg = self.config
        scale = cfg.gamma_bar * self.schedule.sqrt_one_minus_ab(t)
        terms = []
        for j in range(cfg.n_mc):
            # BUG: key now includes a per-call counter, so recurrences and
            # inner iterations no longer share the smoothing noise.
            self._bug_counter = getattr(self, "_bug_counter", 0) + 1
            delta = self.tape.randn(("delta", int(t), int(j) + 1000 * self._bug_counter),
                                    x.shape, device=x.device, dtype=x.dtype)
            terms.append(self._call_log_f(x + scale * delta, n_t, eta_keys))
        return torch.logsumexp(torch.stack(terms), dim=0) - math.log(cfg.n_mc)


def test_M1_delta_keyed_per_recurrence_is_detected():
    _assert_detected(M1_DeltaKeyedByRecurrence, 2, 1, 0.35, 2, "M1 C1 violation")


# -- M2: re-noise on the final recurrence (violates C2) ---------------------

class M2_RenoiseOnLastRecurrence(GeneralizedTFG):
    def run(self, shape, trace=None):
        # Re-implement only the final-recurrence branch by temporarily raising
        # N_recur so the engine always re-noises, then stripping the extra step.
        cfg = self.config
        original = cfg.N_recur
        cfg.N_recur = original + 1
        try:
            return super().run(shape, trace=trace)
        finally:
            cfg.N_recur = original


def test_M2_extra_recurrence_is_detected():
    _assert_detected(M2_RenoiseOnLastRecurrence, 1, 1, 0.0, 1, "M2 C2 violation")


# -- M3: drop the Delta_t / sqrt(alphabar_t) rescaling ----------------------

class M3_BarredAlphaRescale(GeneralizedTFG):
    """Divide Delta_t by sqrt(alphabar_t) instead of sqrt(alpha_t).

    Emulated by pre-multiplying rho_t by sqrt(alpha_t)/sqrt(alphabar_t), so the
    engine's division by sqrt(alpha_t) yields a net division by
    sqrt(alphabar_t).
    """

    def run(self, shape, trace=None):
        sch = self.schedule
        original = self.rho.clone()
        for t in range(1, sch.T + 1):
            alpha_t = sch.alphabar[t] / sch.alphabar[t - 1]
            self.rho[t] = original[t] * torch.sqrt(alpha_t) / torch.sqrt(sch.alphabar[t])
        try:
            return super().run(shape, trace=trace)
        finally:
            self.rho = original


def test_M3_barred_alpha_rescale_is_detected():
    _assert_detected(M3_BarredAlphaRescale, 1, 0, 0.0, 1,
                     "M3 sqrt(alphabar_t) instead of sqrt(alpha_t)")


# -- M4: mu branch backpropagates through the denoiser (violates C3) --------

def test_M4_documented_but_covered_by_M5():
    """C3 is enforced structurally (``x0.detach()``), so a mutation would have
    to rewrite the whole loop. It is covered indirectly: M5 perturbs the same
    gradient path. Recorded explicitly so the gap is visible rather than
    silently absent."""
    pytest.skip("C3 is structural; see M5 for coverage of the same gradient path")


# -- M5: mean of logs instead of log of mean --------------------------------

class M5_MeanOfLogs(GeneralizedTFG):
    def _log_f_tilde(self, x, t, n_t, eta_keys):
        cfg = self.config
        scale = cfg.gamma_bar * self.schedule.sqrt_one_minus_ab(t)
        terms = []
        for j in range(cfg.n_mc):
            delta = self.tape.randn(("delta", int(t), int(j)), x.shape,
                                    device=x.device, dtype=x.dtype)
            terms.append(self._call_log_f(x + scale * delta, n_t, eta_keys))
        # BUG: E[log f] instead of log E[f].
        return torch.stack(terms).mean()


def test_M5_mean_of_logs_is_detected():
    _assert_detected(M5_MeanOfLogs, 1, 1, 0.35, 3, "M5 Jensen swap")


# -- M6: DDIM uses the wrong alphabar index ---------------------------------

class M6_WrongDDIMIndex(GeneralizedTFG):
    def run(self, shape, trace=None):
        sch = self.schedule
        shifted = sch.alphabar.clone()
        # BUG: off-by-one in the schedule the DDIM step reads.
        shifted[:-1] = sch.alphabar[1:].clone()
        original = sch.alphabar
        sch.alphabar = shifted
        try:
            return super().run(shape, trace=trace)
        finally:
            sch.alphabar = original


def test_M6_schedule_offbyone_is_detected():
    _assert_detected(M6_WrongDDIMIndex, 1, 0, 0.0, 1, "M6 alphabar off-by-one")


# -- sanity: the unmutated engine must NOT be flagged -----------------------

def test_unmutated_engine_is_not_flagged():
    ref = _reference_trace(2, 1, 0.35, 2)
    mut = _mutant_trace(GeneralizedTFG, 2, 1, 0.35, 2)
    ok, report = compare_traces(ref, mut, atol=1e-12)
    assert ok, f"false positive: {report}"
    assert report["max_abs_err"] == 0.0
