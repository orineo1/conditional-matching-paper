"""Unit tests for Component 1: the adaptive conditional sample count n_t.

Covers the four properties required of the schedule: boundary values,
monotonicity toward the data end, determinism, and that every schedule type
behaves as specified -- in particular that ``constant`` reproduces the legacy
fixed-``n_cond`` behaviour exactly.
"""

import pytest

from tfg.n_schedule import (
    VALID_TYPES,
    conditional_seed_keys,
    is_nondecreasing_toward_data,
    n_at,
    progress,
    schedule_table,
)
from tfg.schedule import DiffusionSchedule  # noqa: F401


@pytest.fixture
def sch():
    return DiffusionSchedule(T=100)


# -- boundaries -------------------------------------------------------------

@pytest.mark.parametrize("kind", ["time", "noise"])
def test_progress_endpoints(sch, kind):
    """Normalised over the EXECUTED steps: p_T = 0 and p_1 = 1."""
    assert progress(sch.T, sch, kind) == pytest.approx(0.0, abs=1e-12)
    assert progress(1, sch, kind) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("kind", ["time", "noise"])
def test_progress_is_monotone_over_executed_steps(sch, kind):
    ps = [progress(t, sch, kind) for t in range(sch.T, 0, -1)]
    assert all(b >= a for a, b in zip(ps, ps[1:])), "p_t must be non-decreasing"
    assert all(0.0 <= p <= 1.0 for p in ps)


def test_progress_requires_at_least_two_steps():
    with pytest.raises(ValueError, match="T >= 2"):
        progress(1, DiffusionSchedule(T=1), "time")


@pytest.mark.parametrize("kind", ["time", "noise"])
@pytest.mark.parametrize("kappa", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("n_max", [1, 2, 16, 64, 250])
def test_bounds(sch, kind, kappa, n_max):
    for t in range(sch.T, -1, -1):
        n = n_at(t, sch, n_max, kappa, kind)
        assert isinstance(n, int)
        assert 1 <= n <= n_max


@pytest.mark.parametrize("kind", ["time", "noise"])
@pytest.mark.parametrize("kappa", [0.5, 1.0, 2.0])
def test_endpoint_values(sch, kind, kappa):
    """n_T = 1 at the noisiest executed step, n_1 = n_max at the cleanest."""
    n_max = 64
    assert n_at(sch.T, sch, n_max, kappa, kind) == 1
    assert n_at(1, sch, n_max, kappa, kind) == n_max


@pytest.mark.parametrize("kind", ["time", "noise"])
@pytest.mark.parametrize("kappa", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("n_max", [2, 16, 64, 250])
def test_n_max_is_actually_attained(sch, kind, kappa, n_max):
    """The regression this replaces: n_max was never reached, because the
    schedule was normalised over t=T..0 while the engine executes t=T..1."""
    table = schedule_table(sch, n_max, kappa, kind)
    assert table[-1] == n_max, f"last executed step used n={table[-1]}, not n_max={n_max}"
    assert table[0] == 1, f"first executed step used n={table[0]}, not 1"
    assert max(table) == n_max


def test_n_max_one_is_always_one(sch):
    for kind in VALID_TYPES:
        for t in range(sch.T, -1, -1):
            assert n_at(t, sch, 1, 1.0, kind) == 1


# -- monotonicity -----------------------------------------------------------

@pytest.mark.parametrize("kind", ["constant", "time", "noise"])
@pytest.mark.parametrize("kappa", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("n_max", [2, 16, 64, 250])
def test_monotone_nondecreasing_toward_data(sch, kind, kappa, n_max):
    assert is_nondecreasing_toward_data(sch, n_max, kappa, kind)


# -- schedule types ---------------------------------------------------------

def test_constant_reproduces_fixed_n(sch):
    """`constant` must be exactly the legacy fixed-n_cond behaviour."""
    table = schedule_table(sch, n_max=250, kappa=1.0, kind="constant")
    assert set(table) == {250}
    # One entry per EXECUTED step, t = T..1.
    assert len(table) == sch.T


def test_constant_matches_a_hand_written_fixed_n_loop(sch):
    """Constant-schedule equivalence: identical to the legacy fixed-n_cond
    behaviour, step for step, for every t the engine actually visits."""
    legacy = [250 for _ in range(sch.T, 0, -1)]
    assert schedule_table(sch, n_max=250, kappa=1.0, kind="constant") == legacy
    for kappa in (0.5, 1.0, 2.0):
        assert schedule_table(sch, 250, kappa, "constant") == legacy, (
            "constant must ignore kappa entirely"
        )


def test_kappa_orders_the_schedules(sch):
    """Larger kappa delays the ramp-up, so it never spends more samples."""
    n_max = 64
    lo = schedule_table(sch, n_max, kappa=0.5, kind="time")
    mid = schedule_table(sch, n_max, kappa=1.0, kind="time")
    hi = schedule_table(sch, n_max, kappa=2.0, kind="time")
    assert sum(hi) <= sum(mid) <= sum(lo)


def test_time_and_noise_differ(sch):
    """The two progress variables must not be the same schedule."""
    a = schedule_table(sch, 64, 1.0, "time")
    b = schedule_table(sch, 64, 1.0, "noise")
    assert a != b


def test_unknown_type_rejected(sch):
    with pytest.raises(ValueError, match="n_schedule type"):
        n_at(5, sch, 16, 1.0, "uncertainty")


def test_invalid_params_rejected(sch):
    with pytest.raises(ValueError):
        n_at(5, sch, 0, 1.0, "time")
    with pytest.raises(ValueError):
        n_at(5, sch, 16, 0.0, "time")


# -- determinism ------------------------------------------------------------

@pytest.mark.parametrize("kind", VALID_TYPES)
def test_determinism(sch, kind):
    a = schedule_table(sch, 64, 1.3, kind)
    b = schedule_table(sch, 64, 1.3, kind)
    assert a == b


def test_state_argument_is_ignored_for_now(sch):
    """The uncertainty-adaptive hook exists but must not yet change anything."""
    plain = n_at(50, sch, 64, 1.0, "time")
    with_state = n_at(50, sch, 64, 1.0, "time", state={"variance": 123.0})
    assert plain == with_state


# -- conditional seed keys --------------------------------------------------

def test_seed_keys_shape_and_uniqueness():
    keys = conditional_seed_keys(t=17, n_t=4)
    assert keys == [("eta", 17, 0), ("eta", 17, 1), ("eta", 17, 2), ("eta", 17, 3)]
    assert len(set(keys)) == 4


def test_seed_keys_carry_no_recurrence_or_eval_index():
    """Reuse within an outer step, fresh draws at the next one."""
    a = conditional_seed_keys(t=17, n_t=3)
    b = conditional_seed_keys(t=17, n_t=3)
    c = conditional_seed_keys(t=16, n_t=3)
    assert a == b, "keys must be identical across recurrences within a step"
    assert set(a).isdisjoint(set(c)), "the next outer step must draw fresh keys"
