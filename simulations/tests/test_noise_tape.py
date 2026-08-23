"""Unit tests for the semantic NoiseTape."""

import subprocess
import sys

import pytest
import torch

from tfg.noise_tape import NoiseTape, compare_access


def test_same_key_same_value():
    tape = NoiseTape(seed=7)
    a = tape.randn(("delta", 3, 0), (1, 2))
    b = tape.randn(("delta", 3, 0), (1, 2))
    assert torch.equal(a, b)


def test_different_keys_differ():
    tape = NoiseTape(seed=7)
    a = tape.randn(("delta", 3, 0), (1, 2))
    b = tape.randn(("delta", 3, 1), (1, 2))
    c = tape.randn(("delta", 4, 0), (1, 2))
    assert not torch.equal(a, b)
    assert not torch.equal(a, c)


def test_order_independence():
    """The whole point: request order must not affect values."""
    keys = [("x_T",), ("delta", 5, 0), ("renoise", 5, 1), ("delta", 4, 0)]

    fwd = NoiseTape(seed=11)
    got_fwd = {k: fwd.randn(k, (1, 3)) for k in keys}

    rev = NoiseTape(seed=11)
    got_rev = {k: rev.randn(k, (1, 3)) for k in reversed(keys)}

    for k in keys:
        assert torch.equal(got_fwd[k], got_rev[k]), f"value depends on order for {k!r}"


def test_skipping_a_key_does_not_shift_others():
    """An engine that skips a draw must not perturb the others.

    This is the exact failure mode a sequential RNG has: N_iter=0 consumes
    fewer draws than N_iter=4 and every subsequent tensor shifts.
    """
    full = NoiseTape(seed=3)
    a_full = full.randn(("a",), (2,))
    full.randn(("skipped",), (2,))
    c_full = full.randn(("c",), (2,))

    partial = NoiseTape(seed=3)
    a_part = partial.randn(("a",), (2,))
    c_part = partial.randn(("c",), (2,))

    assert torch.equal(a_full, a_part)
    assert torch.equal(c_full, c_part)


def test_seed_changes_values():
    a = NoiseTape(seed=1).randn(("k",), (4,))
    b = NoiseTape(seed=2).randn(("k",), (4,))
    assert not torch.equal(a, b)


def test_shape_mismatch_is_loud():
    tape = NoiseTape(seed=0)
    tape.randn(("k",), (2, 3))
    with pytest.raises(ValueError, match="disagree about this key"):
        tape.randn(("k",), (2, 4))


def test_rejects_unstable_key_atoms():
    tape = NoiseTape(seed=0)
    with pytest.raises(TypeError):
        tape.randn(("k", 0.5), (2,))
    with pytest.raises(TypeError):
        tape.randn(("k", torch.tensor(1)), (2,))


def test_access_log_and_compare():
    a = NoiseTape(seed=0)
    b = NoiseTape(seed=0)
    a.randn(("x",), (1,))
    a.randn(("y",), (1,))
    b.randn(("x",), (1,))
    b.randn(("z",), (1,))
    only_a, only_b = compare_access(a, b)
    assert only_a == [("y",)]
    assert only_b == [("z",)]


def test_does_not_touch_global_rng():
    """The tape must not disturb the ambient torch RNG stream."""
    torch.manual_seed(123)
    baseline = torch.randn(5)

    torch.manual_seed(123)
    tape = NoiseTape(seed=99)
    tape.randn(("a",), (10,))
    tape.randn(("b",), (10,))
    after = torch.randn(5)

    assert torch.equal(baseline, after)


def test_reproducible_across_processes():
    """Guards against Python's per-process string hash salt.

    A tape keyed with :func:`hash` would pass every in-process test above and
    still produce different values in a fresh interpreter.
    """
    code = (
        "import sys; sys.path.insert(0, '.');"
        "import torch;"
        "from tfg.noise_tape import NoiseTape;"
        "t = NoiseTape(seed=42);"
        "print(t.randn(('delta', 7, 2), (3,)).sum().item())"
    )
    outs = []
    for _ in range(2):
        res = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
        assert res.returncode == 0, res.stderr
        outs.append(res.stdout.strip())
    assert outs[0] == outs[1], f"tape not reproducible across processes: {outs}"

    tape = NoiseTape(seed=42)
    here = tape.randn(("delta", 7, 2), (3,)).sum().item()
    assert abs(float(outs[0]) - here) < 1e-15


def test_dtype_and_device_cast_preserves_value():
    tape = NoiseTape(seed=5)
    a = tape.randn(("k",), (4,), dtype=torch.float64)
    b = tape.randn(("k",), (4,), dtype=torch.float32)
    assert torch.allclose(a.float(), b)
