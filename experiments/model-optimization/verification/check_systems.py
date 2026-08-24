"""Agent 6 -- independent re-run of the Agent-5 (systems/runners.py) equivalence
claims and a quiet re-timing with warm-up, plus a micro-timing of the Agent-2
cached-target MMD.  2D synthetic, float32 (as the repo loop), single process.

    cd simulations && /Users/stolk/miniconda3/bin/python ../experiments/model-optimization/verification/check_systems.py

Checks (all against simulations/experiments/_guided.py::run, restarts 0..7):
  1. run_single(flags off)           -> must be bit-identical (generator seeding)
  2. run_single(batched_lgd)         -> must be bit-identical (LGD cell)
  3. run_single(batched_mmd)         -> per-step teacher-forced |dg| and e2e |dx|
  4. run_batched_restarts B=1, B=8   -> per-step teacher-forced |dg| and e2e |dx|
  5. timing: reference vs B=1 vs B=8 batched restarts (median of 5, 1 warm-up)
  6. MMD micro-timing: reference MMDLoss vs fast_mmd MMDFixedTarget (fwd+bwd)
"""
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MO = HERE.parent
sys.path.insert(0, str(MO / "systems"))
sys.path.insert(0, str(MO / "exact_loss"))
from _setup import setup, timeit  # noqa: E402   (adds simulations/src, experiments to sys.path)

import torch  # noqa: E402
from _guided import run as ref_run  # noqa: E402
from LossFunctions import MMDLoss, RBF  # noqa: E402
from runners import run_batched_restarts, run_single  # noqa: E402
from fast_mmd import MMDFixedTarget  # noqa: E402

torch.set_num_threads(4)
RESTARTS = list(range(8))
CELLS = [("no_lgd", "none", 8), ("lgd", "none", 8), ("no_lgd", "adam", 8), ("no_lgd", "none", 32)]


def maxdiff(a, b):
    return float((a.reshape(-1) - b.reshape(-1)).abs().max())


def main():
    print(f"torch {torch.__version__}, threads {torch.get_num_threads()}, load avg {__import__('os').getloadavg()}")
    params, S_G, bw, mc, mu = setup("2D")
    for sp, tp, n in CELLS:
        print(f"\n=== 2D {sp}/{tp} n={n} ===")
        # reference trajectories + grads via run_single(flags off), verified against _guided.run
        refs = {}
        for r in RESTARTS:
            xr, _ = ref_run(mc, mu, S_G, bw, n, sp, tp, r)
            traj, gl = [], []
            xs, _ = run_single(mc, mu, S_G, bw, n, sp, tp, r, trajectory=traj, grad_log=gl)
            refs[r] = (xr, traj, gl)
            assert maxdiff(xr, xs) == 0.0, (r, maxdiff(xr, xs))
        print("1. run_single(flags off) vs _guided.run: bit-identical on 8 restarts -> EXACT")
        if sp == "lgd":
            d = max(maxdiff(refs[r][0], run_single(mc, mu, S_G, bw, n, sp, tp, r, batched_lgd=True)[0])
                    for r in RESTARTS)
            print(f"2. batched_lgd e2e max|dx| over 8 restarts = {d:.3e}")
        # batched mmd: teacher forced
        dg, dx = 0.0, 0.0
        for r in RESTARTS:
            xr, traj, gref = refs[r]
            gl = []
            run_single(mc, mu, S_G, bw, n, sp, tp, r, force_traj=traj, grad_log=gl,
                       batched_lgd=(sp == "lgd"), batched_mmd=True)
            dg = max(dg, max(maxdiff(a[0], b[0]) for a, b in zip(gref, gl)))
            xv, _ = run_single(mc, mu, S_G, bw, n, sp, tp, r, batched_lgd=(sp == "lgd"), batched_mmd=True)
            dx = max(dx, maxdiff(xr, xv))
        gmag = max(float(a[0].abs().max()) for r in RESTARTS for a in refs[r][2])
        print(f"3. batched_mmd (+batched_lgd): teacher-forced per-step max|dg| = {dg:.3e} "
              f"(max|g| = {gmag:.2e}); e2e max|dx| = {dx:.3e}")
        # batched restarts
        for B in (1, 8):
            R = RESTARTS[:B]
            xb, ib = run_batched_restarts(mc, mu, S_G, bw, n, sp, tp, R)
            e2e = max(maxdiff(xb[i], refs[r][0]) for i, r in enumerate(R))
            ft = [torch.stack([refs[r][1][k].reshape(-1) for r in R]) for k in range(len(refs[R[0]][1]))]
            gl = []
            run_batched_restarts(mc, mu, S_G, bw, n, sp, tp, R, force_traj=ft, grad_log=gl)
            trip = max(((maxdiff(gl[k][0][i], refs[r][2][k][0]), float(refs[r][2][k][0].abs().max()), k, r)
                        for k in range(len(gl)) for i, r in enumerate(R)), key=lambda z: z[0])
            dgb, gat, kk, rr = trip
            rel = max(maxdiff(gl[k][0][i], refs[r][2][k][0]) / max(float(refs[r][2][k][0].abs().max()), 1e-12)
                      for k in range(len(gl)) for i, r in enumerate(R))
            n_div = int(ib["diverged"].sum())
            print(f"4. batched_restarts B={B}: teacher-forced per-step max|dg| = {dgb:.3e} "
                  f"(at step idx {kk}, restart {rr}, |g|max there {gat:.2e}); max rel |dg|/|g| = {rel:.2e}; "
                  f"e2e max|dx| = {e2e:.3e}; diverged rows {n_div}; calls {ib['conditional_calls']}")
        # timing
        t_ref, *_ = timeit(lambda: ref_run(mc, mu, S_G, bw, n, sp, tp, 0), repeats=5, warmup=1)
        t_b1, *_ = timeit(lambda: run_batched_restarts(mc, mu, S_G, bw, n, sp, tp, [0], lean_ddim=True), repeats=5, warmup=1)
        t_b8, *_ = timeit(lambda: run_batched_restarts(mc, mu, S_G, bw, n, sp, tp, RESTARTS, lean_ddim=True), repeats=5, warmup=1)
        print(f"5. timing (median of 5, warm-up 1): reference {t_ref:.3f}s/restart; batched B=1 {t_b1:.3f}s "
              f"({t_ref / t_b1:.2f}x); batched B=8 {t_b8:.3f}s = {t_b8 / 8:.4f}s/restart ({t_ref / (t_b8 / 8):.1f}x)")

    # 6. MMD micro-timing (fwd+bwd) float32, n=8/32, m=250, d=2
    print("\n=== MMD micro-timing, float32, m=250, d=d_y, fwd+bwd, median of 200 after 20 warm-up ===")
    ref_mmd = MMDLoss(kernel=RBF(bandwidth=bw))
    for n in (8, 32):
        g = torch.Generator().manual_seed(0)
        X0 = torch.randn(n, S_G.shape[1], generator=g)
        fast = MMDFixedTarget(S_G, bw, dist="mm", kernel_eval="powchain")
        fast_c = MMDFixedTarget(S_G, bw, dist="cdist", kernel_eval="exp")

        def tm(fn):
            for _ in range(20):
                fn()
            ts = []
            for _ in range(200):
                t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
            return st.median(ts) * 1e3

        def f_ref():
            X = X0.clone().requires_grad_(True)
            torch.autograd.grad(ref_mmd(X, S_G), X)

        def f_fast():
            X = X0.clone().requires_grad_(True)
            torch.autograd.grad(fast(X), X)

        def f_fast_c():
            X = X0.clone().requires_grad_(True)
            torch.autograd.grad(fast_c(X), X)
        a, b, c = tm(f_ref), tm(f_fast), tm(f_fast_c)
        print(f"n={n}: reference {a:.3f} ms; fixed_cdist {c:.3f} ms ({a / c:.1f}x); fixed_mm_powchain {b:.3f} ms ({a / b:.1f}x)")


if __name__ == "__main__":
    main()
