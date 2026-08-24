"""[A4] Adaptive policies for the generalised TFG engine.

Two mechanisms, both opt-in:

1. ``adaptive_n``: uncertainty-adaptive conditional sample count.  The engine
   builds a ``state`` dict every outer step and ``n_schedule.n_at(...,
   kind="adaptive", state=state)`` delegates here.  Policies:

   ``agreement``
       At step ``t`` the engine evaluates the predictor on the two independent
       halves of the ``n_t`` conditional draws (keys ``eta[:h]`` and
       ``eta[h:]``) in addition to the full batch and records the normalised
       agreement :func:`gradient_agreement` (cosine-like, graded in 1-D).  Low agreement => the gradient is noise-dominated
       => grow ``n`` for the NEXT step by ``grow``x (up to ``n_max``); high
       agreement => shrink by ``grow``x (down to ``n_min``).  The full-batch
       gradient is what the step uses; the halves are a free by-product when
       the sampler caches identical (x, key) draws within a step
       (``tfg.distributional.CachedSampler``), and cost 2x otherwise.
   ``improvement``
       Grow when ``log f`` failed to improve by more than
       ``improvement_threshold`` over the previous step, shrink otherwise.

   A ``budget_total`` (sum of ``n_t`` over the run) may be given: the policy
   then never plans to exceed it and spends any remainder on the final step,
   so an adaptive run and a constant-``n`` run with ``n * T == budget_total``
   make EXACTLY the same number of conditional calls.

2. ``recurrence_should_stop``: early stopping of the recurrence loop
   (component 3, ``AdaptiveRecurrenceConfig(implementation="v1")``).
"""

import math

import torch


def _cos(a, b, eps=1e-30):
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    return float((a @ b) / (a.norm() * b.norm() + eps))


def gradient_agreement(grad_a, grad_b, eps=1e-30):
    """Normalised agreement of two independent half-batch gradients,

        (||g_a + g_b|| - ||g_a - g_b||) / (||g_a|| + ||g_b||)   in [-1, 1].

    Equal gradients give +1, opposite ones -1, orthogonal ones 0. Unlike the
    cosine it is graded in ONE dimension too (g_a=1, g_b=0.5 -> 0.667), where
    the cosine is just the sign agreement; it equals the cosine when
    ||g_a|| == ||g_b||."""
    a = grad_a.reshape(-1).double()
    b = grad_b.reshape(-1).double()
    return float(((a + b).norm() - (a - b).norm()) / (a.norm() + b.norm() + eps))


def adaptive_n(n_max, state):
    """Next ``n_t`` under the adaptive policy.  ``state`` keys:

    ``n_prev``      n used at the previous outer step (None at the first step)
    ``n_min``, ``n_start``, ``grow``, ``policy``
    ``agreement``   cos(grad_a, grad_b) of the previous step (agreement policy)
    ``agreement_threshold``
    ``improved``    log f(prev) - log f(prev-1) of the previous step (improvement policy)
    ``improvement_threshold``
    ``budget_remaining``, ``steps_left``  (budget accounting; 0 budget = unlimited)
    """
    n_max = int(n_max)
    n_min = int(state.get("n_min", 1))
    grow = int(state.get("grow", 2))
    n_prev = state.get("n_prev")
    if n_prev is None:
        n = int(state.get("n_start") or n_max)
    else:
        policy = state.get("policy", "agreement")
        if policy == "agreement":
            a = state.get("agreement")
            thr = float(state.get("agreement_threshold", 0.5))
            if a is None or not math.isfinite(a):
                n = n_prev
            elif a < thr:
                n = n_prev * grow
            else:
                n = n_prev // grow
        elif policy == "improvement":
            imp = state.get("improved")
            thr = float(state.get("improvement_threshold", 0.0))
            if imp is None or not math.isfinite(imp):
                n = n_prev
            elif imp <= thr:
                n = n_prev * grow
            else:
                n = n_prev // grow
        else:
            raise ValueError(f"unknown adaptive policy {policy!r}")
    n = max(n_min, min(n_max, int(n)))

    budget = int(state.get("budget_remaining") or 0)
    steps_left = int(state.get("steps_left") or 0)
    if budget > 0 and steps_left > 0:
        if steps_left == 1:
            n = budget                       # spend the remainder exactly
        else:
            hi = budget - (steps_left - 1) * n_min      # leave n_min for every later step
            lo = budget - (steps_left - 1) * n_max      # what later steps cannot absorb
            n = max(n_min, min(n, hi))
            n = max(n, lo)                  # spread an unspendable surplus, do not dump it
    return max(1, n)


def recurrence_should_stop(metric, threshold, prev, cur):
    """Early-stop rule for the recurrence loop; ``prev``/``cur`` are dicts with
    ``log_f`` (float), ``x_prev`` (tensor) and ``grad`` (tensor) of the previous
    and current recurrence.  Never stops on the first recurrence."""
    if prev is None:
        return False
    if metric == "clean_proxy":
        den = max(abs(prev["log_f"]), 1.0)
        return abs(cur["log_f"] - prev["log_f"]) / den < threshold
    if metric == "next_state_tweedie":
        den = max(float(prev["x_prev"].norm()), 1.0)
        return float((cur["x_prev"] - prev["x_prev"]).norm()) / den < threshold
    if metric == "grad_stability":
        return _cos(cur["grad"], prev["grad"]) > 1.0 - threshold
    raise ValueError(f"unknown recurrence metric {metric!r}")
