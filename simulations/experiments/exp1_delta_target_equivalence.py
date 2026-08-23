"""Experiment 1 -- delta-target equivalence.

QUESTION
    Does the generalized engine, configured with the POINT objective and a
    delta target, reproduce ordinary TFG exactly?

WHY IT MATTERS
    Every later claim rests on the engine being a strict generalization. If the
    point-objective path is not exactly ordinary TFG, the MMD path is not a
    generalization of anything.

COMPARISON
    tfg.engine (point objective, delta target)  vs  tfg.reference (frozen
    Algorithm 1 transcription), on identical NoiseTape randomness, comparing
    every traced intermediate per (t, r): x_T, x_{0|t}, raw rho-gradient,
    Delta_t, each mu-gradient, Delta_0, DDIM iterate, x_{t-1}, re-noising
    noise, x_t, and the final x_0.

    Run:  python experiments/exp1_delta_target_equivalence.py
"""
import itertools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import save                                    # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from conftest import ToyDenoiser, make_quadratic_log_f      # noqa: E402

from tfg.config import TFGConfig                            # noqa: E402
from tfg.engine import GeneralizedTFG                       # noqa: E402
from tfg.noise_tape import NoiseTape, compare_access        # noqa: E402
from tfg.reference import run_reference_tfg                 # noqa: E402
from tfg.schedule import DiffusionSchedule, constant_vector  # noqa: E402
from tfg.trace import Tracer, compare_traces                # noqa: E402

SHAPE, T, TOL = (1, 2), 8, 1e-12
MODES = {"rho_only": (0.9, 0.0), "mu_only": (0.0, 0.35), "both": (0.9, 0.35)}


def cell(N_recur, N_iter, gamma_bar, mode, T=T, seed=17):
    sch = DiffusionSchedule(T=T)
    den = ToyDenoiser(d=SHAPE[1], T=T, seed=0, schedule=sch)
    eps = lambda x, t: den(x, t)                            # noqa: E731
    log_f = make_quadratic_log_f([1.5, -0.75], 0.6)         # delta/point target
    rho_s, mu_s = MODES[mode]
    n_mc = 3 if gamma_bar else 1

    tr_ref, tape_ref = Tracer(), NoiseTape(seed=seed)
    x_ref = run_reference_tfg(eps, log_f, sch, tape_ref, SHAPE,
                              N_recur=N_recur, N_iter=N_iter,
                              rho=constant_vector(rho_s, T),
                              mu=constant_vector(mu_s, T),
                              gamma_bar=gamma_bar, n_mc=n_mc, trace=tr_ref)

    cfg = TFGConfig(T=T, N_recur=N_recur, N_iter=N_iter, gamma_bar=gamma_bar,
                    rho_scalar=rho_s, mu_scalar=mu_s, n_mc=n_mc)
    tr_eng, tape_eng = Tracer(), NoiseTape(seed=seed)
    x_eng = GeneralizedTFG(eps, log_f, sch, tape_eng, cfg).run(SHAPE, trace=tr_eng)

    ok, rep = compare_traces(tr_ref, tr_eng, atol=0.0)
    only_a, only_b = compare_access(tape_ref, tape_eng)
    return {"N_recur": N_recur, "N_iter": N_iter, "gamma_bar": gamma_bar,
            "mode": mode, "T": T,
            "max_abs_err": rep["max_abs_err"],
            "keys_compared": rep["n_keys_compared"],
            "tape_only_reference": [str(k) for k in only_a],
            "tape_only_engine": [str(k) for k in only_b],
            "final_x0_abs_err": float((x_ref - x_eng).abs().max()),
            "pass": bool(ok)}


def main():
    torch.set_default_dtype(torch.float64)
    cells = []
    for N_recur, N_iter, gb, mode in itertools.product(
            (1, 2), (0, 1, 4), (0.0, 0.35), ("rho_only", "mu_only", "both")):
        if N_iter == 0 and mode != "rho_only":
            continue                       # mu cannot act; degenerate
        cells.append(cell(N_recur, N_iter, gb, mode))
    cells.append(cell(3, 2, 0.4, "both"))                  # deep recurrence
    cells.append(cell(2, 1, 0.3, "both", T=2))             # t=1 -> t=0 boundary

    worst = max(c["max_abs_err"] for c in cells)
    print(f"{'N_recur':>7}{'N_iter':>7}{'gamma':>7}{'mode':>10}{'T':>4}"
          f"{'keys':>7}{'max_abs_err':>14}")
    for c in cells:
        print(f"{c['N_recur']:>7}{c['N_iter']:>7}{c['gamma_bar']:>7}"
              f"{c['mode']:>10}{c['T']:>4}{c['keys_compared']:>7}"
              f"{c['max_abs_err']:>14.3e}")
    print(f"\ncells: {len(cells)}   worst max_abs_err: {worst:.3e}   "
          f"all pass: {all(c['pass'] for c in cells)}")
    save("exp1_delta_target_equivalence",
         {"cells": cells, "worst_max_abs_err": worst,
          "all_pass": all(c["pass"] for c in cells), "tolerance": TOL})


if __name__ == "__main__":
    main()
