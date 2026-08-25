"""The nuisance construction: only coordinate 0 may carry signal about X.

It exists because the primary construction cannot test the dim(Y) question --
its extra coordinates repeat one scalar signal, so estimator noise FALLS with d.
These tests pin the properties that make the nuisance variant different, so a
later edit cannot quietly reintroduce signal into the appended coordinates.
"""
import sys
from pathlib import Path

import torch

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM / "src"))

from tfg.dimy_benchmark import (as_params, as_params_nuisance,  # noqa: E402
                                build, build_nuisance)


def test_m0_reproduces_the_d1_benchmark():
    a, b = build_nuisance(0), build(1)
    torch.testing.assert_close(a["means"], b["means"], rtol=0, atol=0)
    torch.testing.assert_close(a["cov"], b["cov"], rtol=0, atol=0)
    torch.testing.assert_close(a["x_star"], b["x_star"], rtol=0, atol=0)
    pa, pb = as_params_nuisance(0), as_params(1)
    torch.testing.assert_close(pa["target_means"].double(),
                               pb["target_means"].double(), rtol=0, atol=0)


def test_only_coordinate_zero_couples_to_x():
    """X must be uncorrelated with every appended coordinate; otherwise they
    carry signal and the construction is not a nuisance construction."""
    for m in (1, 4, 16):
        cov = build_nuisance(m)["cov"]
        assert float(cov[0, 1]) != 0.0                  # informative coordinate
        assert torch.all(cov[0, 2:] == 0.0)             # nuisance coordinates
        assert torch.all(cov[2:, 0] == 0.0)


def test_nuisance_coordinates_carry_no_component_structure():
    """All components share the same mean in the appended coordinates, so they
    cannot indicate which mixture component (hence which x) generated a sample."""
    for m in (1, 4, 16):
        means = build_nuisance(m)["means"]
        assert torch.all(means[:, 2:] == 0.0)
        assert len(torch.unique(means[:, 1])) > 1       # coord 0 does vary


def test_dimensions_and_optimum():
    for m in (0, 3, 9):
        p = as_params_nuisance(m)
        assert p["dim_y"] == 1 + m
        assert p["target_means"].shape[1] == 1 + m
        assert float(p["x_star"]) == -5.0


def test_covariance_is_positive_definite():
    for m in (0, 1, 8, 32):
        torch.linalg.cholesky(build_nuisance(m)["cov"])
