# BASELINE -- CDM / distributional-TFG synthetic loop (Agent 1)

## Provenance

| item | value |
|---|---|
| commit | `6af2081` (branch `tfg-generalization-v2`; working tree had uncommitted edits to `simulations/experiments/README.md`, `_guided.py` and untracked `exp5b`, `exp7`, `experiments/model-optimization/`) |
| loop | `simulations/experiments/_guided.py::run` (what Exp 2-7 call); `simulations/src/tfg/engine.py` is the generalised Algorithm 1 used by tests/Exp 1, not by Exp 2-7 |
| python | `/Users/stolk/miniconda3/bin/python`, 3.13.11 |
| torch | 2.12.0, CPU, `torch.get_num_threads() = 4` |
| platform | macOS-26.0.1-arm64 (Apple M4, 10 cores, 16 GB) |
| dtype | **float32** for both models, the target set and the whole guided loop (`_common.target_set` casts to `.float()`, checkpoints are fp32). Only `evaluate` (exact GMM L2) and `tfg.schedule` are float64. (The campaign README's "float64 as the repo does" is true of the `tfg` engine, not of `_guided.run`.) |
| checkpoints | `simulations/artifacts/checkpoints/cm_seed20240401_dx1dy1.pt`, `uncond_seed20240401_dx1.pt` (2D, tag ""); `cm_seed20240401_dx4dy1_canonical.pt`, `uncond_seed20240401_dx4_canonical.pt` (5D, tag `_canonical`); `cm_seed20240401_dx9dy1_canonical.pt`, `uncond_seed20240401_dx9_canonical.pt` (10D, tag `_canonical`) -- the tags Exp 3 used |
| params | `simulations/params/{2D,5D,10D}_cond_1D_gmm_params.pt` via `_common.load(setting)`; dim(X) = 1 / 4 / 9, dim(Y) = 1 |
| target | `S_G` = 250 samples, `_common.target_set(params)` seed 987654; bandwidth `_common.fixed_bandwidth(S_G)` = 50.197830 (2D), 9.110705 (5D), 9.258415 (10D); RBF 5 bandwidths, `mul_factor = 2` |
| conditional model | `ConsistencyModeliCT` 128 units, depth 3, sampling ladder `PAPER_TS = [150, 50, 20, 10, 5, 1]` (5 network evaluations per sample) |
| unconditional model | `DiffusionModel` 3 blocks, 128 units, `diffusion_steps = 100` -> 99 guided DDIM steps (eta = 0), `x_T = 0` |
| Adam arm | `AdamGuidance(beta1 = 0.9, beta2 = 0.995, delta = 1e-8, rho = 0.4, inv_sqrt_alpha = False)` |
| seeds | restart index r in 0..4; conditional draws seeded `key_seed("cond", r, t, j)` per step/perturbation (global `torch.manual_seed`); model seed 20240401 |
| metric | exact population L2 between the CM conditional at x_hat and the target, `_guided.evaluate` -> `tfg.oracle.population_l2_squared`; `abs_err = |x_hat - x*|` |

## Commands

```bash
cd /Users/stolk/github/conditional-matching-paper/simulations
# all 18 cells (3 settings x 3 arms x n in {8,32}), restarts 0..4, 1 warm-up + 5 timed repeats each:
/Users/stolk/miniconda3/bin/python ../experiments/model-optimization/profiling/run_baseline.py
# one cell, e.g.
/Users/stolk/miniconda3/bin/python ../experiments/model-optimization/profiling/run_baseline.py --cell 2D no_lgd none 8
# hierarchical profile and operator-level profile
/Users/stolk/miniconda3/bin/python ../experiments/model-optimization/profiling/profile_guided.py
/Users/stolk/miniconda3/bin/python ../experiments/model-optimization/profiling/torch_profile.py
```

Each cell is equivalent to (in Python, from `simulations/`, with `experiments/` and
`src/` on `sys.path`):

```python
params = _common.load(setting); S_G = _common.target_set(params); bw = _common.fixed_bandwidth(S_G)
mc = _models.conditional_model(params, seed=20240401, tag=TAG)   # TAG "" (2D) or "_canonical" (5D/10D)
mu = _models.unconditional_model(params, seed=20240401, tag=TAG)
x_hat, info = _guided.run(mc, mu, S_G, bw, n, spatial, temporal, restart)   # schedule="constant", adam_rho=0.4
_guided.evaluate(x_hat, params, info)["L2"]
```

The same cells are what `experiments/exp2_lgd_vs_adam.py --n 8` (2D) and
`experiments/exp3_sample_scaling.py --setting {2D,5D,10D} --tag _canonical` run at
100 restarts; the 2D rows below match `results/tfg/exp2_lgd_vs_adam_n8_canonical.json`
restart-for-restart (e.g. restart 0 no_lgd/none L2 = 0.21824, x_hat = -5.447157).

## Results (median wall of 25 timed runs per cell = 5 restarts x 5 repeats; peak RSS per cell subprocess)

| setting | arm | n | wall s median [min,max] | ms/step | peak RSS MB | cond sampler calls | cond samples | exact L2 (restarts 0..4) | mean L2 | final MMD^2 at x_hat (r0) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2D | no_lgd/none | 8 | 0.188 [0.184,0.194] | 1.90 | 328 | 99 | 792 | 0.218, 0.314, 0.044, 0.181, 1.096 | 0.371 | 0.2074 |
| 2D | no_lgd/none | 32 | 0.215 [0.209,0.233] | 2.17 | 340 | 99 | 3168 | 0.231, 0.255, 0.262, 0.346, 0.138 | 0.247 | 0.0680 |
| 2D | no_lgd/adam | 8 | 0.189 [0.186,0.196] | 1.91 | 327 | 99 | 792 | 0.077, 0.327, 0.786, 0.783, 0.207 | 0.436 | 0.1790 |
| 2D | no_lgd/adam | 32 | 0.216 [0.211,0.238] | 2.18 | 331 | 99 | 3168 | 0.314, 0.252, 0.433, 0.409, 0.155 | 0.313 | 0.0679 |
| 2D | lgd/none | 8 | 0.512 [0.505,0.541] | 5.17 | 352 | 297 | 2376 | 0.077, 0.218, 0.819, 0.053, 0.030 | 0.240 | 0.1539 |
| 2D | lgd/none | 32 | 0.589 [0.582,0.619] | 5.95 | 352 | 297 | 9504 | 0.187, 0.236, 0.277, 0.236, 0.250 | 0.237 | 0.0676 |
| 5D | no_lgd/none | 8 | 0.183 [0.181,0.194] | 1.85 | 331 | 99 | 792 | 0.411, 1.087, 1.078, 0.399, 1.077 | 0.810 | 0.3589 |
| 5D | no_lgd/none | 32 | 0.212 [0.209,0.215] | 2.14 | 327 | 99 | 3168 | 0.412, 0.418, 0.471, 0.417, 0.442 | 0.432 | 0.2914 |
| 5D | no_lgd/adam | 8 | 0.185 [0.184,0.195] | 1.87 | 322 | 99 | 792 | 0.396, 0.568, 0.384, 0.399, 0.412 | 0.432 | 0.3566 |
| 5D | no_lgd/adam | 32 | 0.216 [0.213,0.222] | 2.18 | 330 | 99 | 3168 | 0.405, 0.281, 0.622, 0.420, 0.454 | 0.437 | 0.2431 |
| 5D | lgd/none | 8 | 0.502 [0.495,0.540] | 5.07 | 344 | 297 | 2376 | 0.399, 0.412, 0.444, 0.419, 0.413 | 0.417 | 0.2996 |
| 5D | lgd/none | 32 | 0.606 [0.587,0.718] | 6.12 | 367 | 297 | 9504 | 0.429, 0.425, 0.422, 0.423, 0.427 | 0.425 | 0.2570 |
| 10D | no_lgd/none | 8 | 0.183 [0.180,0.198] | 1.85 | 333 | 99 | 792 | 0.527, 0.581, 0.450, 0.536, 0.730 | 0.565 | 0.3381 |
| 10D | no_lgd/none | 32 | 0.216 [0.210,0.239] | 2.18 | 348 | 99 | 3168 | 0.597, 0.571, 0.403, 0.443, 0.534 | 0.510 | 0.0533 |
| 10D | no_lgd/adam | 8 | 0.186 [0.184,0.191] | 1.88 | 327 | 99 | 792 | 0.558, 0.910, 0.921, 0.505, 0.682 | 0.715 | 0.2443 |
| 10D | no_lgd/adam | 32 | 0.216 [0.210,0.279] | 2.18 | 330 | 99 | 3168 | 0.765, 0.399, 0.729, 0.438, 0.359 | 0.538 | 0.0310 |
| 10D | lgd/none | 8 | 0.502 [0.488,0.584] | 5.07 | 356 | 297 | 2376 | 0.416, 0.323, 0.441, 0.438, 0.399 | 0.404 | 0.2962 |
| 10D | lgd/none | 32 | 0.597 [0.583,0.807] | 6.03 | 359 | 297 | 9504 | 0.451, 0.293, 0.461, 0.412, 0.420 | 0.407 | 0.0509 |

* All 25 repeats of every cell return bit-identical `x_hat` (deterministic given the
  restart index). No divergence in any cell.
* "final MMD^2" = one fresh n-sample CM draw at x_hat vs S_G (the loop logs no loss;
  this is the diagnostic `opt_loss` column in `profiling/baseline_rows.csv`).
* RSS after import + model load, before any run: 293 MB. Per-run increment 30-75 MB.
* Hardware-independent cost: conditional samples per run = 99 x M_t x n (M_t = 1
  no-LGD, 3 LGD); conditional network forwards = 99 x M_t x 5; denoiser calls = 99;
  backward passes = 99; MMD evaluations = 99 x M_t, each on an (n+250)^2 x 5 kernel.

Rows for `results.csv` are in `profiling/baseline_rows.csv` (90 rows, `candidate =
baseline`, `score_calls` = denoiser calls, `cond_calls` = conditional sampler calls,
`cond_samples` = conditional samples, `opt_loss` = final MMD^2, `eval_metric` = exact L2,
`wall_s` = per-restart median of 5 repeats).

Profile, bottleneck ranking, call accounting and the MNIST / SD static analysis:
`profiling/baseline_profile.md`, machine-readable `profiling/profile.json`.
