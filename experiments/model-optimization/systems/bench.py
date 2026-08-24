"""Agent 5 -- single-factor systems benchmarks against _guided.run (synthetic 2D).

    cd simulations && python ../experiments/model-optimization/systems/bench.py [--quick]

Every cell: warm-up, then REPEATS timed runs (median reported), peak RSS of the
process (monotone; reported per cell as the running max), and an equivalence
record against the reference `_guided.run`:
  * end2end_max_abs  : max |x_final(variant) - x_final(ref)| over the restarts run
  * step_grad_max_abs: max over steps/restarts of |g_variant - g_ref| with the
                       variant TEACHER-FORCED on the reference trajectory
                       (isolates per-step numerical agreement from chaotic
                       trajectory divergence; see BENCH.md)
Writes bench_rows.csv next to this file.
"""
import argparse
import copy
import csv
import platform
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _setup import peak_rss_mb, setup  # noqa: E402

import torch  # noqa: E402
from _guided import run as ref_run  # noqa: E402
from runners import (capture_trajectory, frozen_params, run_batched_restarts,  # noqa: E402
                     run_single)

COMMIT = "6af2081"
CELLS = [("no_lgd", "none", 8), ("no_lgd", "none", 32), ("lgd", "none", 8), ("no_lgd", "adam", 8)]
RESTARTS = [0, 1, 2, 3]


def ref_traj_and_grads(mc, mu, S_G, bw, n, sp, tp, r):
    """Reference trajectory x_t (t=T-1..1) and per-step (g, upd) from
    run_single with all flags off (verified bit-identical to _guided.run)."""
    traj, gl = [], []
    x, _ = run_single(mc, mu, S_G, bw, n, sp, tp, r, trajectory=traj, grad_log=gl)
    return x, traj, gl


def step_equiv_single(mc, mu, S_G, bw, n, sp, tp, r, ref, **kw):
    _, traj, gref = ref
    gl = []
    run_single(mc, mu, S_G, bw, n, sp, tp, r, force_traj=traj, grad_log=gl, **kw)
    return max(float((a[0] - b[0]).abs().max()) for a, b in zip(gref, gl))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--setting", default="2D")
    ap.add_argument("--skip-mps", action="store_true")
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument("--cuda", action="store_true",
                    help="also run batched restarts on CUDA (B=1,8,32,128) if available")
    ap.add_argument("--out", default=None, help="csv path (default: bench_rows.csv next to this file)")
    a = ap.parse_args()
    repeats = 2 if a.quick else a.repeats
    torch.set_num_threads(torch.get_num_threads())
    params, S_G, bw, mc, mu = setup(a.setting)
    rows = []
    hw = f"{platform.machine()}-cpu-{torch.get_num_threads()}thr"

    def record(variant, sp, tp, n, B, wall_med, wall_all, e2e, stepg, calls, status="ok",
               extra=""):
        rows.append({"commit": COMMIT, "candidate": f"sys:{variant}", "task": f"synthetic_{a.setting}",
                     "target": "S_G250_seed987654", "seed": ";".join(map(str, RESTARTS[:B])) if B > 1 else "0",
                     "config": f"spatial={sp};temporal={tp};n={n};B={B};{extra}",
                     "hardware": hw, "dtype": "float64" if "float64" in variant else "float32",
                     "wall_s": f"{wall_med:.4f}", "wall_all": "|".join(f"{w:.4f}" for w in wall_all),
                     "wall_per_restart_s": f"{wall_med / B:.4f}",
                     "restarts_per_s": f"{B / wall_med:.3f}",
                     "peak_mem_mb": f"{peak_rss_mb():.1f}",
                     "score_calls": 99 * B, "cond_calls": 99 * B * (3 if sp == "lgd" else 1),
                     "cond_samples": calls,
                     "end2end_max_abs": f"{e2e:.3e}" if e2e is not None else "",
                     "step_grad_max_abs": f"{stepg:.3e}" if stepg is not None else "",
                     "status": status})
        print(f"{variant:<28} {sp:<6} {tp:<4} n={n:<3} B={B:<2} wall={wall_med:.4f}s "
              f"({wall_med / B:.4f}/restart, {B / wall_med:.2f} r/s) e2e={e2e} stepg={stepg} {status}", flush=True)

    def timed(fn):
        fn()
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        return st.median(ts), ts

    for sp, tp, n in CELLS:
        # ---- reference ------------------------------------------------
        xr, ir = ref_run(mc, mu, S_G, bw, n, sp, tp, 0)
        med, ts = timed(lambda: ref_run(mc, mu, S_G, bw, n, sp, tp, 0))
        record("reference(_guided.run)", sp, tp, n, 1, med, ts, 0.0, 0.0, ir["conditional_calls"])
        ref = ref_traj_and_grads(mc, mu, S_G, bw, n, sp, tp, 0)
        refs_B = {r: ref_traj_and_grads(mc, mu, S_G, bw, n, sp, tp, r) for r in RESTARTS}
        # ---- (0) generator-seeded transcription ------------------------
        x, i = run_single(mc, mu, S_G, bw, n, sp, tp, 0)
        med, ts = timed(lambda: run_single(mc, mu, S_G, bw, n, sp, tp, 0))
        record("generator_seed", sp, tp, n, 1, med, ts, float((x - xr).abs().max()),
               step_equiv_single(mc, mu, S_G, bw, n, sp, tp, 0, ref), i["conditional_calls"])
        # ---- (i) requires_grad_(False) ---------------------------------
        with frozen_params(mc, mu):
            x, i = ref_run(mc, mu, S_G, bw, n, sp, tp, 0)
            med, ts = timed(lambda: ref_run(mc, mu, S_G, bw, n, sp, tp, 0))
            record("frozen_params", sp, tp, n, 1, med, ts, float((x - xr).abs().max()), 0.0 if torch.equal(x, xr) else None,
                   i["conditional_calls"])
        # ---- (ii) batched LGD -----------------------------------------
        if sp == "lgd":
            x, i = run_single(mc, mu, S_G, bw, n, sp, tp, 0, batched_lgd=True)
            med, ts = timed(lambda: run_single(mc, mu, S_G, bw, n, sp, tp, 0, batched_lgd=True))
            record("batched_lgd", sp, tp, n, 1, med, ts, float((x - xr).abs().max()),
                   step_equiv_single(mc, mu, S_G, bw, n, sp, tp, 0, ref, batched_lgd=True), i["conditional_calls"])
        # ---- (iii) batched MMD ---------------------------------------
        kw = dict(batched_lgd=(sp == "lgd"), batched_mmd=True)
        x, i = run_single(mc, mu, S_G, bw, n, sp, tp, 0, **kw)
        med, ts = timed(lambda: run_single(mc, mu, S_G, bw, n, sp, tp, 0, **kw))
        record("batched_mmd(+batched_lgd)", sp, tp, n, 1, med, ts, float((x - xr).abs().max()),
               step_equiv_single(mc, mu, S_G, bw, n, sp, tp, 0, ref, **kw), i["conditional_calls"])
        # ---- (iv) batched restarts -----------------------------------
        Bs = [1, 2, 4] if a.quick else [1, 2, 4, 8, 16, 32]
        for B in Bs:
            R = list(range(B))
            xb, ib = run_batched_restarts(mc, mu, S_G, bw, n, sp, tp, R)
            med, ts = timed(lambda: run_batched_restarts(mc, mu, S_G, bw, n, sp, tp, R))
            # equivalence on the first min(B,4) restarts
            e2e, stepg = 0.0, 0.0
            Rq = R[:len(RESTARTS)]
            for r in Rq:
                if r not in refs_B:
                    refs_B[r] = ref_traj_and_grads(mc, mu, S_G, bw, n, sp, tp, r)
                e2e = max(e2e, float((xb[r] - refs_B[r][0]).abs().max()))
            ft = [torch.cat([refs_B[r][1][k] if r in refs_B else torch.zeros(1, mu.nfeatures)
                             for r in R], 0) for k in range(len(refs_B[0][1]))]
            gl = []
            run_batched_restarts(mc, mu, S_G, bw, n, sp, tp, R, force_traj=ft, grad_log=gl)
            for r in Rq:
                stepg = max(stepg, max(float((g[r] - gr[0].reshape(-1)).abs().max())
                                       for (g, _, _), gr in zip(gl, refs_B[r][2])))
            record("batched_restarts", sp, tp, n, B, med, ts, e2e, stepg, ib["conditional_calls"],
                   extra="batched_lgd=1;batched_mmd=1")
            if B == 8 and not a.quick:
                xb, ib = run_batched_restarts(mc, mu, S_G, bw, n, sp, tp, R, lean_ddim=True)
                med, ts = timed(lambda: run_batched_restarts(mc, mu, S_G, bw, n, sp, tp, R, lean_ddim=True))
                record("batched_restarts+lean_ddim", sp, tp, n, B, med, ts,
                       max(float((xb[r] - refs_B[r][0]).abs().max()) for r in Rq), None,
                       ib["conditional_calls"], extra="batched_lgd=1;batched_mmd=1;lean_ddim=1")
        # ---- (vi) float64 throughout ----------------------------------
        if not a.quick:
            mc64, mu64 = copy.deepcopy(mc).double(), copy.deepcopy(mu).double()
            for name in ("baralphas", "betas", "alphas"):
                setattr(mu64, name, getattr(mu64, name).double())
            S64 = S_G.double()
            torch.set_default_dtype(torch.float64)
            try:
                x, i = ref_run(mc64, mu64, S64, bw, n, sp, tp, 0)
                med, ts = timed(lambda: ref_run(mc64, mu64, S64, bw, n, sp, tp, 0))
            finally:
                torch.set_default_dtype(torch.float32)
            record("float64_throughout", sp, tp, n, 1, med, ts, float((x.double() - xr.double()).abs().max()), None,
                   i["conditional_calls"], status="changes_numerics(float64)")
    # ---- (v) torch.compile on the MLPs ---------------------------------
    if not a.quick and not a.skip_compile:
        sp, tp, n = "no_lgd", "none", 8
        xr, _ = ref_run(mc, mu, S_G, bw, n, sp, tp, 0)
        for mode in ("default",):
            mcc, muc = copy.deepcopy(mc), copy.deepcopy(mu)
            try:
                t0 = time.perf_counter()
                mcc.forward = torch.compile(mcc.forward, mode=mode, dynamic=True)
                muc.forward = torch.compile(muc.forward, mode=mode, dynamic=True)
                x, i = ref_run(mcc, muc, S_G, bw, n, sp, tp, 0)      # compile happens here
                compile_s = time.perf_counter() - t0
                med, ts = timed(lambda: ref_run(mcc, muc, S_G, bw, n, sp, tp, 0))
                record(f"torch.compile[{mode}]", sp, tp, n, 1, med, ts, float((x - xr).abs().max()), None,
                       i["conditional_calls"], extra=f"first_call_incl_compile_s={compile_s:.2f}")
            except Exception as e:  # noqa: BLE001
                record(f"torch.compile[{mode}]", sp, tp, n, 1, float("nan"), [], None, None, 0,
                       status=f"failed:{type(e).__name__}:{str(e)[:80]}")
    # ---- (vii) MPS ------------------------------------------------------
    if a.cuda and torch.cuda.is_available():
        for sp, tp, n in CELLS:
            try:
                mcc, muc = copy.deepcopy(mc).to("cuda"), copy.deepcopy(mu).to("cuda")
                for name in ("baralphas", "betas", "alphas"):
                    setattr(muc, name, getattr(muc, name).to("cuda"))
                mcc.device = torch.device("cuda"); muc.device = torch.device("cuda")
                for B in (1, 8, 32, 128):
                    R = list(range(B))
                    xb, ib = run_batched_restarts(mcc, muc, S_G, bw, n, sp, tp, R, device="cuda", lean_ddim=True)
                    torch.cuda.synchronize()
                    def f():
                        run_batched_restarts(mcc, muc, S_G, bw, n, sp, tp, R, device="cuda", lean_ddim=True)
                        torch.cuda.synchronize()
                    med, ts = timed(f)
                    e2e = max(float((xb[r].cpu() - ref_run(mc, mu, S_G, bw, n, sp, tp, r)[0]).abs().max()) for r in R[:4])
                    record("batched_restarts[cuda]", sp, tp, n, B, med, ts, e2e, None, ib["conditional_calls"],
                           status="ok(cuda,float32)", extra="lean_ddim=1")
            except Exception as e:  # noqa: BLE001
                record("batched_restarts[cuda]", sp, tp, n, 1, float("nan"), [], None, None, 0,
                       status=f"failed:{type(e).__name__}:{str(e)[:80]}")
    if torch.backends.mps.is_available() and not a.quick and not a.skip_mps:
        sp, tp, n = "no_lgd", "none", 8
        try:
            mcm, mum = copy.deepcopy(mc).to("mps"), copy.deepcopy(mu).to("mps")
            for name in ("baralphas", "betas", "alphas"):
                setattr(mum, name, getattr(mum, name).to("mps"))
            mcm.device = torch.device("mps"); mum.device = torch.device("mps")
            for B in (1, 8, 32):
                R = list(range(B))
                xb, ib = run_batched_restarts(mcm, mum, S_G, bw, n, sp, tp, R, device="mps")
                torch.mps.synchronize()
                def f():
                    run_batched_restarts(mcm, mum, S_G, bw, n, sp, tp, R, device="mps"); torch.mps.synchronize()
                med, ts = timed(f)
                e2e = max(float((xb[r].cpu() - ref_run(mc, mu, S_G, bw, n, sp, tp, r)[0]).abs().max()) for r in R[:4])
                record("batched_restarts[mps]", sp, tp, n, B, med, ts, e2e, None, ib["conditional_calls"],
                       status="ok(mps,float32)")
        except Exception as e:  # noqa: BLE001
            record("batched_restarts[mps]", sp, tp, n, 1, float("nan"), [], None, None, 0,
                   status=f"failed:{type(e).__name__}:{str(e)[:80]}")

    out = Path(a.out) if a.out else HERE / "bench_rows.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("written", out)


if __name__ == "__main__":
    main()
