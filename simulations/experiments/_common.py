"""Shared setup for the tfg experiments: paths, params, target, bandwidth."""
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

SIM = Path(__file__).resolve().parents[1]
SRC = SIM / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tfg._compat import ensure_ot_stub                      # noqa: E402

ensure_ot_stub()

import dist_utils                                            # noqa: E402
from tfg import oracle                                       # noqa: E402

RESULTS = SIM / "results" / "tfg"
PENALTY = 2.0                    # failure-penalised score cap
SUCCESS_TOL = 0.5                # |x - x*| below this counts as success


def key_seed(*parts):
    return int.from_bytes(
        hashlib.blake2b(repr(parts).encode(), digest_size=8).digest(),
        "big") % (2 ** 31 - 1)


def git_meta():
    def run(*a):
        try:
            return subprocess.run(a, cwd=str(SIM), capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return None
    return {"commit": run("git", "rev-parse", "HEAD"),
            "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": run("git", "status", "--porcelain")}


def load(setting="2D"):
    """Load one of the paper's canonical parameter files."""
    return oracle.load_params(SIM / "params" / f"{setting}_cond_1D_gmm_params.pt")


def target_set(params, size=250, seed=987654):
    """One fixed empirical target set S_G, drawn once and reused everywhere."""
    torch.manual_seed(seed)
    tv = torch.stack([v.float() for v in params["target_variances"]])
    return dist_utils.generate_mog_samples_not_differentiable(
        size, params["target_means"].float(), tv,
        params["target_weights"].float()).detach()


def fixed_bandwidth(S_G):
    """Repository rule (mean off-diagonal squared distance), frozen thereafter."""
    d2 = torch.cdist(S_G, S_G, p=2) ** 2
    m = S_G.shape[0]
    return float(d2.sum() / (m ** 2 - m))


def penalised_score(runs):
    vals = []
    for r in runs:
        v = r.get("L2", float("inf"))
        vals.append(PENALTY if (r.get("diverged") or v != v or v == float("inf"))
                    else min(v, PENALTY))
    return sum(vals) / len(vals)


def paired_stats(a, b, B=20000, P=20000, seed=0):
    """Positive => b better. Bootstrap CI + paired permutation test."""
    import random
    import statistics as st
    diff = [x - y for x, y in zip(a, b)]
    n = len(diff)
    m = st.mean(diff)
    random.seed(seed)
    boots = sorted(st.mean([diff[random.randrange(n)] for _ in range(n)])
                   for _ in range(B))
    random.seed(seed + 1)
    cnt = sum(1 for _ in range(P)
              if abs(st.mean([d if random.random() < .5 else -d for d in diff])) >= abs(m))
    return {"n": n, "mean_diff": m, "median_diff": st.median(diff),
            "ci95": [boots[int(.025 * B)], boots[int(.975 * B)]],
            "wins_for_b": sum(1 for v in diff if v > 0),
            "perm_p": (cnt + 1) / (P + 1)}


def save(name, payload):
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / f"{name}.json"
    payload.setdefault("meta", {}).update({
        **git_meta(), "torch": torch.__version__,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    p.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwritten: {p.relative_to(SIM)}")
    return p
