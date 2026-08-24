"""Component 1: adaptive conditional sample count ``n_t``.

    n_t = 1 + floor( (n_max - 1) * p_t^kappa )

with two progress variables, both normalised over the timesteps the sampler
ACTUALLY EXECUTES, namely ``t = T, ..., 1``:

    time :   p_t = (T - t) / (T - 1)
    noise:   p_t = (alphabar_t - alphabar_T) / (alphabar_1 - alphabar_T)

so that ``p_T = 0`` and ``p_1 = 1``, giving ``n_T = 1`` and ``n_1 = n_max``.
``n_t`` is non-decreasing as sampling proceeds from noise toward data.

An earlier revision normalised over ``t = T..0`` (``p_t = 1 - t/T``, denominator
``alphabar_0 - alphabar_T``).  Because the engine loop is ``range(T, 0, -1)``,
``t = 0`` is never queried, so ``n_max`` was never actually reached -- the
largest count used was ``n_at(1) = n_max - 1`` -- and the compute-budget
accounting was correspondingly off.

``constant`` returns ``n_max`` at every step and therefore reproduces the
current fixed-``n_cond`` behaviour exactly.

Uncertainty-adaptive ``n_t`` (``kind="adaptive"``) consumes the optional
``state`` argument of :func:`n_at`; the policy lives in ``tfg/adaptive.py``.
With no state it degenerates to ``constant``.
"""

import torch

VALID_TYPES = ("constant", "time", "noise", "adaptive")


def progress(t, schedule, kind):
    """``p_t`` in [0, 1]; 0 at ``t = T``, 1 at ``t = 1``.

    Normalised over the executed timesteps ``t = T..1``, not ``T..0``.
    """
    T = schedule.T
    if T < 2:
        raise ValueError("progress schedules require T >= 2")
    if kind == "time":
        return (float(T) - float(t)) / (float(T) - 1.0)
    if kind == "noise":
        ab_1 = float(schedule.ab(1))
        ab_T = float(schedule.ab(T))
        denom = ab_1 - ab_T
        if denom <= 0:
            raise ValueError(
                "degenerate schedule: alphabar_1 must exceed alphabar_T for the "
                "noise-progress variable to be defined"
            )
        return (float(schedule.ab(t)) - ab_T) / denom
    raise ValueError(f"unknown progress kind {kind!r}")


def n_at(t, schedule, n_max, kappa=1.0, kind="constant", state=None):
    """Integer conditional sample count at step ``t``, clamped to ``[1, n_max]``.

    ``state`` is consumed ONLY by ``kind="adaptive"`` (see ``tfg/adaptive.py``);
    every other kind ignores it.  ``adaptive`` without a state (or without an
    ``n_prev``/budget) falls back to ``n_max``, so a state-less adaptive
    schedule is the constant one.
    """
    if kind not in VALID_TYPES:
        raise ValueError(f"n_schedule type must be one of {VALID_TYPES}, got {kind!r}")
    n_max = int(n_max)
    if n_max < 1:
        raise ValueError("n_max must be >= 1")

    if kind == "constant":
        return n_max
    if kind == "adaptive":
        if state is None:
            return n_max
        from tfg.adaptive import adaptive_n
        return adaptive_n(n_max, state)

    if kappa <= 0:
        raise ValueError("kappa must be positive")

    p = progress(t, schedule, kind)
    # Clamp before the power: floating point can put p a few ulps outside [0,1]
    # at the endpoints, and a negative base with fractional kappa is nan.
    p = min(1.0, max(0.0, p))
    n = 1 + int((n_max - 1) * (p ** float(kappa)))
    return max(1, min(n_max, n))


def schedule_table(schedule, n_max, kappa=1.0, kind="constant"):
    """``n_t`` for the EXECUTED steps ``t = T..1``, in sampling order."""
    return [n_at(t, schedule, n_max, kappa, kind) for t in range(schedule.T, 0, -1)]


def conditional_seed_keys(t, n_t, tag="eta", j=None, frozen=False):
    """Tape keys for this outer step's conditional-generator draws.

    Keyed by ``(tag, t, i)`` with no recurrence index and no loss-evaluation
    index, which is what makes the same ``eta`` values shared across every loss
    evaluation and every recurrence within outer step ``t``, while the next
    outer step draws fresh ones.

    [A4] ``j`` (perturbation index) adds a per-perturbation component,
    ``(tag, t, j, i)``; ``frozen=True`` drops ``t`` entirely
    (``(tag, "frozen", i)`` / ``(tag, "frozen", j, i)``): common random numbers
    across the whole trajectory (approximate, see config docstring).
    """
    head = (tag, "frozen") if frozen else (tag, int(t))
    if j is not None:
        head = head + (int(j),)
    return [head + (int(i),) for i in range(int(n_t))]


def is_nondecreasing_toward_data(schedule, n_max, kappa, kind):
    """True iff ``n_t`` never decreases as ``t`` goes from ``T`` down to ``0``."""
    table = schedule_table(schedule, n_max, kappa, kind)
    return all(b >= a for a, b in zip(table, table[1:]))
