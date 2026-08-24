"""Shared setup for Agent 5 (systems) scripts: paths, model loading, timing."""
import resource
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SIM = ROOT / "simulations"
for p in (SIM / "src", SIM / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch  # noqa: E402

from _common import fixed_bandwidth, key_seed, load, target_set  # noqa: E402,F401
from _models import PAPER_TS, SEEDS, conditional_model, unconditional_model  # noqa: E402,F401

TAG = {"2D": "", "5D": "_canonical", "10D": "_canonical"}


def setup(setting="2D"):
    params = load(setting)
    S_G = target_set(params)
    bw = fixed_bandwidth(S_G)
    mc = conditional_model(params, seed=SEEDS[0], tag=TAG[setting])
    mu = unconditional_model(params, seed=SEEDS[0], tag=TAG[setting])
    return params, S_G, bw, mc, mu


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10


def timeit(fn, repeats=5, warmup=1):
    """Return (median, min, max, all) of wall seconds; fn() is called repeats+warmup times."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return st.median(ts), min(ts), max(ts), ts
