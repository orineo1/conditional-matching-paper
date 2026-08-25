"""Make the shared ``tfg`` package (simulations/src/tfg) importable from the SD
pipeline without pip: appends ``<repo>/simulations/src`` to ``sys.path`` if
``tfg`` is not already importable.  Appended (not prepended) so the SD ``src``
modules keep precedence over the synthetic ones."""
import importlib.util
import os
import sys

if importlib.util.find_spec("tfg") is None:
    _here = os.path.dirname(os.path.abspath(__file__))
    _sim_src = os.path.abspath(os.path.join(_here, "..", "..", "simulations", "src"))
    if _sim_src not in sys.path:
        sys.path.append(_sim_src)
