"""Import shims for running against the legacy modules unmodified.

``LossFunctions.py`` does ``import ot`` (POT) at module scope, but nothing we
use from it -- ``RBF`` and ``MMDLoss`` -- touches POT; both are pure torch.
POT is not installed in this environment and adding a dependency to run an
analysis would be a heavier change than the analysis warrants, so we register
a stub module instead. This keeps the study running against the REAL
repository kernel rather than a re-typed copy, and leaves ``LossFunctions.py``
untouched.

If POT is genuinely installed, this does nothing.
"""

import sys
import types


def _stub(name, attrs=(), package=False):
    m = types.ModuleType(name)
    if package:
        m.__path__ = []          # marks it as a package so submodules import
    for a in attrs:
        setattr(m, a, type(a, (), {}))
    sys.modules[name] = m
    return m


def ensure_ot_stub():
    if "ot" not in sys.modules:
        try:
            import ot  # noqa: F401
        except ImportError:
            sys.modules["ot"] = types.ModuleType("ot")


def ensure_flow_matching_stub():
    """Stub the uninstalled ``flow_matching`` package.

    ``Optimization.py`` imports ``FlowMatching``, which imports
    ``flow_matching`` at module scope. The D-FLOW path is not used by anything
    in this study, but the import must succeed for the LEGACY
    ``Optimization.optimize_LGD`` to be importable at all -- and importing the
    real legacy function is the whole point of the equivalence check. Stubbing
    keeps ``Optimization.py`` untouched.
    """
    try:
        import flow_matching  # noqa: F401
        return
    except ImportError:
        pass
    _stub("flow_matching", package=True)
    _stub("flow_matching.path", ["AffineProbPath"], package=True)
    _stub("flow_matching.path.scheduler", ["CondOTScheduler"])
    _stub("flow_matching.solver", ["Solver", "ODESolver"])
    utils = _stub("flow_matching.utils")
    utils.gradient = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("flow_matching is stubbed; the D-FLOW path is unavailable"))
    fm = sys.modules["flow_matching"]
    fm.path = sys.modules["flow_matching.path"]
    fm.path.scheduler = sys.modules["flow_matching.path.scheduler"]
    fm.solver = sys.modules["flow_matching.solver"]
    fm.utils = utils


def ensure_torchdiffeq_stub():
    """Stub ``torchdiffeq`` (imported by ``FM_Solver_Extension_module``)."""
    try:
        import torchdiffeq  # noqa: F401
        return
    except ImportError:
        pass
    m = _stub("torchdiffeq")
    m.odeint = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("torchdiffeq is stubbed; the D-FLOW path is unavailable"))


def ensure_legacy_importable():
    """Make the legacy ``Optimization`` module importable without installing
    packages that only its unused D-FLOW path needs."""
    ensure_ot_stub()
    ensure_flow_matching_stub()
    ensure_torchdiffeq_stub()
