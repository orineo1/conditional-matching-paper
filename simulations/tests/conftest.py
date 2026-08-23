"""Shared fixtures for the tfg validation suite."""
import sys
import types
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# LossFunctions.py imports POT at module scope but nothing we use touches it.
if "ot" not in sys.modules:
    try:
        import ot  # noqa: F401
    except ImportError:
        sys.modules["ot"] = types.ModuleType("ot")

torch.use_deterministic_algorithms(True)


@pytest.fixture(autouse=True)
def _float64_cpu():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


class ToyDenoiser(torch.nn.Module):
    """Deterministic eps_theta(x, t), parameterised through a BOUNDED x0.

    Not the repository's DiffusionModel: the equivalence tests only need a
    fixed, differentiable, non-trivial function that both engines call
    identically. Parameterising through tanh(x0) keeps x_{0|t} bounded; an
    unconstrained network makes x_{0|t} scale like 1/sqrt(alphabar_t) and the
    traced values reach 1e287, which is legible to no one.
    """

    def __init__(self, d=2, hidden=16, T=100, seed=0, schedule=None):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.T, self.schedule = T, schedule
        self.w1 = torch.nn.Parameter(torch.randn(d + 1, hidden, generator=g, dtype=torch.float64) * 0.5)
        self.b1 = torch.nn.Parameter(torch.randn(hidden, generator=g, dtype=torch.float64) * 0.1)
        self.w2 = torch.nn.Parameter(torch.randn(hidden, d, generator=g, dtype=torch.float64) * 0.5)
        self.b2 = torch.nn.Parameter(torch.randn(d, generator=g, dtype=torch.float64) * 0.1)
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def predicted_x0(self, x, t):
        tt = torch.full((x.shape[0], 1), float(t) / self.T, dtype=x.dtype)
        h = torch.tanh(torch.cat([x, tt], dim=1) @ self.w1 + self.b1)
        return torch.tanh(h @ self.w2 + self.b2)

    def forward(self, x, t):
        m = self.predicted_x0(x, t)
        if self.schedule is None:
            return m
        ab = self.schedule.alphabar[t]
        return (x - torch.sqrt(ab) * m) / torch.sqrt(1.0 - ab)


@pytest.fixture
def denoiser():
    return ToyDenoiser(d=2, T=8, seed=0)


def make_quadratic_log_f(center, scale=1.0):
    """An ordinary point-target TFG predictor: f(x) = exp(-scale*||x - c||^2)."""
    c = torch.as_tensor(center, dtype=torch.float64)

    def log_f(x):
        return -scale * ((x - c) ** 2).sum()

    return log_f
