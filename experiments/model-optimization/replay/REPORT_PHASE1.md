# Agent M -- sample-replay MMD: phase 1 (implementation + smoke)

2026-08-24, branch `tfg-generalization-v2`, python
`/Users/stolk/miniconda3/bin/python`, float32 pipeline. Status: implemented,
tested, pre-registered, smoke-tested; **cluster screening not yet submitted**.

## What exists

* `ORI_IMPLEMENTATION.md` -- Ori's mechanism FOUND in the repo:
  `reuse_frac` on `upstream/claude/hybrid-sampling-optimization-55fv3b`
  (+ `reuse-adamdps-grid`, `sd-reuse-adamdps-schedule`): depth-1 detached
  buffer of the previous step's FRESH conditional samples, `n_reuse =
  round(reuse_frac * nsamples)` reused rows + `n_new` fresh, total MMD batch
  unchanged, so fresh conditional calls are cut by `reuse_frac`. Grids swept
  `reuse_frac in {0.0..0.9}`; **no results are committed anywhere** -- this
  campaign's screening is the first quantitative test.
* `THEORY.md` -- weighted/subsampled replay V-statistics, exact gradient
  (only fresh rows differentiable: `f/B` XY rescale + fresh-replay XX
  repulsion, target direction unchanged), bias = MMD to a geometric mixture
  of the last D+1 conditional laws (lag -> 0 near convergence), ESS argument,
  and the contrast with the failed stale-gradient / CRN candidates (replay
  never freezes a derivative and never freezes noise).
* `simulations/src/tfg/replay.py` -- `replay_counts` (largest-remainder
  geometric split; reproduces Ori's `round(p*n)` at depth 1),
  `ReplayBuffer` (per-perturbation-j, depth-bounded, detached, recurrence
  replaces), `subsample_rows` (NoiseTape argsort, key `("replay", t, j, k)`),
  `weighted_mmd2` (weighted V-stat; uses the fast backend's cached target
  blocks when present; == unweighted at uniform weights to 1e-12),
  `wrap_log_f` (the assembler; `enabled=False` / `decay=0` byte-identical to
  the plain path).
* `simulations/src/tfg/config.py` -- `ReplayConfig` dataclass + one
  `TFGConfig.replay` field (off by default; the engine itself never reads it
  -- wiring is outside, in the runner).
* `estimator/engine_runner.py` -- `replay*` candidate names appended to
  `candidate_spec` + a 4-line opt-in `wrap_log_f` hook in the `log_f` path.
* `simulations/tests/test_replay.py` -- 12 tests: counts/Ori split, buffer
  depth/eviction/detachment, off & `decay=0` identity, tape-seed determinism,
  fresh-only gradient == manual constant-replay batch (both backends,
  1e-12), weighted==unweighted at uniform weights, weighted == subsampled
  expectation up to the EXACT within-group diagonal correction (400-draw MC,
  5 SE), config validation, engine integration (decay=0 bit-identical to
  baseline; replay30 uses 6*T calls vs 8*T; augment/weighted arms run).
  **Full suite: 378 passed, 1 skipped (pre-existing)** -- the original 348
  plus concurrent agents' and these.
* `hypotheses/agentM.yaml` -- pre-registered (M-1..M-7) BEFORE the smoke run.
* `cells.py` + `submit_replay.sh` -- 15 candidates x {2D,5D,10D} x n {4,8,32}
  = 135 cells, no_lgd/none, restarts 0..39 offset 0, glacier CPU array.

## Smoke (2D n=8, restarts 0..9 -- sanity ONLY, noise floor ~0.1 at 10 restarts)

| candidate | score | succ | div | cond calls/run |
|---|---|---|---|---|
| baseline | 0.420 | 20% | 0 | 792 |
| trust_noise1 | 0.146 | 80% | 0 | 792 |
| replay30 (Ori) | 0.398 | 40% | 0 | **594** |
| replay_w30 (weighted twin) | 0.389 | 40% | 0 | **594** |
| replay_geo0.5d3 | 0.255 | 50% | 0 | **396** |
| replay30_aug | 0.238 | 50% | 0 | 792 |
| replay_geo0.5d3_trust | 0.256 | 50% | 0 | **396** |

Sanity checks all pass: no divergence, calls accounting exactly as designed
(replay30: 6 fresh/step = 594; geo0.5d3: 4 fresh/step = 396 -- half the
baseline's), weighted twin ~= subsample twin (M-5's prediction), and the
directions are consistent with M-1/M-3/M-4 (replay ~= baseline at 25% fewer
calls; geometric depth-3 BETTER than baseline at 50% fewer calls; augment
better at equal calls). The `_trust` combination did not add on top of geo
here (0.256 vs trust alone 0.146) -- at 10 restarts this is inside the noise
floor and is exactly what the 40-restart paired screening must settle. **No
promotion claims from these numbers** (pre-registered smoke protocol).

## Next step (NOT executed): cluster screening

```
cd /sci/labs/orzuk/shaulytolk/cdm-perf/simulations
N=$(python ../experiments/model-optimization/replay/cells.py list 2>/dev/null | wc -l)   # 135
sbatch --array=0-$((N-1))%40 ../experiments/model-optimization/replay/submit_replay.sh
# after completion:
python ../experiments/model-optimization/replay/cells.py report   # -> replay_rows.csv, replay_tables.md
```

Prerequisite: the cluster mirror `/sci/labs/orzuk/shaulytolk/cdm-perf/` must
carry the updated `simulations/src/tfg/{config,replay}.py`,
`estimator/engine_runner.py` and `replay/` (rsync before submitting).
Note `replay/runs/` cells are skipped when present, and the local smoke lives
in `replay/smoke/` precisely so it cannot shadow the 40-restart cluster cells.
