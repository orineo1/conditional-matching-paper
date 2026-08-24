"""The dim(Y) benchmark claims d = 1 reproduces the canonical 2-D benchmark.

Nothing asserted that claim, and it was false: BASE_X[8] read -7.0 against the
canonical file's -8.0. That single component made the d = 1 anchor a different
distribution from the one Experiments 2, 3, 6 and 7 run on, which is exactly the
anchor Experiment 5B uses to decide whether the dim(Y) sweep is comparable
across d. Pin the whole construction against the file.
"""
import sys
from pathlib import Path

import pytest
import torch

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM / "src"))

from tfg import oracle                       # noqa: E402
from tfg.dimy_benchmark import as_params     # noqa: E402

CANONICAL = SIM / "params" / "2D_cond_1D_gmm_params.pt"


@pytest.fixture(scope="module")
def pair():
    if not CANONICAL.exists():
        pytest.skip(f"canonical params missing: {CANONICAL}")
    return as_params(1), oracle.load_params(str(CANONICAL))


def test_component_means_match_canonical(pair):
    d1, ref = pair
    a = torch.stack([m.reshape(-1).double() for m in d1["mu_list"]])
    b = torch.stack([m.reshape(-1).double() for m in ref["mu_list"]])
    assert a.shape == b.shape
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_covariances_and_weights_match_canonical(pair):
    """Tolerance here is float32 storage in the .pt file (~7e-9), not modelling
    slack: the benchmark writes exact float64 constants, the canonical file
    round-trips through float32. 1e-6 still catches any real drift, which would
    be O(0.001) or larger."""
    d1, ref = pair
    for A, B in zip(d1["Sigma_list"], ref["Sigma_list"]):
        torch.testing.assert_close(A.double(), B.double(), rtol=0, atol=1e-6)
    torch.testing.assert_close(d1["alpha"].double(), ref["alpha"].double(),
                               rtol=0, atol=1e-6)


def test_target_and_optimum_match_canonical(pair):
    d1, ref = pair
    torch.testing.assert_close(d1["x_star"].double().reshape(-1),
                               ref["x_star"].double().reshape(-1),
                               rtol=0, atol=0)
    torch.testing.assert_close(d1["target_means"].double().reshape(-1),
                               ref["target_means"].double().reshape(-1),
                               rtol=0, atol=0)
    torch.testing.assert_close(d1["target_weights"].double().reshape(-1),
                               ref["target_weights"].double().reshape(-1),
                               rtol=0, atol=0)


def test_population_objective_agrees_on_a_grid(pair):
    """The end-to-end check: identical L2 at d = 1 and canonical, everywhere."""
    d1, ref = pair
    for x in (-8.0, -6.0, -5.0, -4.0, -2.0, 0.0, 2.0, 5.0, 6.0, 8.0):
        xv = torch.tensor([x], dtype=torch.float64)
        torch.testing.assert_close(
            float(oracle.population_l2_squared(xv, d1)),
            float(oracle.population_l2_squared(xv, ref)),
            rtol=1e-6, atol=1e-9)   # float32 storage, see test above
