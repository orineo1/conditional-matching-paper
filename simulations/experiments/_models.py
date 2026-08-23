"""Train-or-load the conditional and unconditional models used by experiments 2-4.

Both follow the paper's Table 3 (2D setting): Blocks 3, Units 128, Epochs
20,000, Batch 1,024. The notebook omits `depth` (falling back to the class
default 6) and uses 7,500 CM epochs; the paper's table is authoritative and is
what we use. Reproduces Appendix Table 6 (CM MMD 0.163 +- 0.214) to within seed
spread -- see `tests/` and the README.

Checkpoints live in `simulations/artifacts/checkpoints/` (git-ignored).
"""
import json
import sys
import time
from functools import partial
from pathlib import Path

import torch

SIM = Path(__file__).resolve().parents[1]
SRC = SIM / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tfg._compat import ensure_ot_stub        # noqa: E402

ensure_ot_stub()

import dist_utils                              # noqa: E402
from ConsistencyModels import ConsistencyModeliCT   # noqa: E402
from Diffusion import DiffusionModel                # noqa: E402

CKPT = SIM / "artifacts" / "checkpoints"
SEEDS = (20240401, 20240402, 20240403)        # main, robustness-1, robustness-2
CM_CFG = dict(nunits=128, depth=3, nepochs=20_000, batch_size=1_024)
UNCOND_CFG = dict(nblocks=3, nunits=128, diffusion_steps=100,
                  nepochs=20_000, batch_size=1_024)


def dims(params):
    """(dim_x, dim_y) for a parameter set. dim_y is 1 in every paper setting."""
    d_joint = int(params["mu_list"][0].reshape(-1).numel())
    d_x = int(params["x_star"].reshape(-1).numel())
    return d_x, d_joint - d_x
# Paper, Appendix A.5: "Sampling follows the multistep iCT procedure with noise
# levels [150, 50, 20, 10, 5, 1]". Note this is the DEFAULT ts of
# ConsistencyModel.sample, NOT of ConsistencyModeliCT.sample (15-entry ladder).
PAPER_TS = [150.0, 50.0, 20.0, 10.0, 5.0, 1.0]


def _generator(params, y_only=False, d_x=1):
    kw = {} if not y_only else {"kernel_func": lambda X: X[:, :d_x]}
    return partial(dist_utils.generate_mog_samples_not_differentiable,
                   means=[m.float() for m in params["mu_list"]],
                   variances=[s.float() for s in params["Sigma_list"]],
                   weights=params["alpha"].float(), **kw)


def conditional_model(params, seed=SEEDS[0], device="cpu", nepochs=None, tag=""):
    CKPT.mkdir(parents=True, exist_ok=True)
    d_x, d_y = dims(params)
    path = CKPT / f"cm_seed{seed}_dx{d_x}dy{d_y}{tag}.pt"
    m = ConsistencyModeliCT(nfeatures=d_y, condition_on=d_x,
                            nunits=CM_CFG["nunits"], depth=CM_CFG["depth"],
                            device=torch.device(device))
    if path.exists():
        m.load_state_dict(torch.load(path, map_location=device,
                                     weights_only=False)["state_dict"])
        m.eval()
        return m
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    m.train_model(X=None, nepochs=nepochs or CM_CFG["nepochs"],
                  batch_size=CM_CFG["batch_size"], device=device,
                  condition=d_x,
                  data_generator=_generator(params), use_improved_training=True)
    m.eval()
    torch.save({"state_dict": m.state_dict(),
                "record": {"seed": seed, "config": CM_CFG,
                           "seconds": time.perf_counter() - t0,
                           "provenance": "RE-TRAINED from the paper's Table 3; "
                                         "not the paper's original weights"}}, path)
    return m


def unconditional_model(params, seed=SEEDS[0], device="cpu", nepochs=None, tag=""):
    CKPT.mkdir(parents=True, exist_ok=True)
    d_x, d_y = dims(params)
    path = CKPT / f"uncond_seed{seed}_dx{d_x}{tag}.pt"
    m = DiffusionModel(nfeatures=d_x,
                       nblocks=UNCOND_CFG["nblocks"], nunits=UNCOND_CFG["nunits"],
                       condition=False,
                       diffusion_steps=UNCOND_CFG["diffusion_steps"],
                       device=torch.device(device))
    if path.exists():
        m.load_state_dict(torch.load(path, map_location=device,
                                     weights_only=False)["state_dict"])
        m.eval()
        return m
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    m.train_model(None, data_generator=_generator(params, y_only=True, d_x=d_x),
                  nepochs=nepochs or UNCOND_CFG["nepochs"],
                  batch_size=UNCOND_CFG["batch_size"], condition_on=d_x)
    m.eval()
    torch.save({"state_dict": m.state_dict(),
                "record": {"seed": seed, "config": UNCOND_CFG,
                           "seconds": time.perf_counter() - t0,
                           "provenance": "RE-TRAINED from the paper's Table 3"}}, path)
    return m
