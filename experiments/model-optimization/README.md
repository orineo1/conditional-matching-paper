# CDM / Distributional-TFG performance campaign

Campaign root for the brief `FABLE_CDM_PERFORMANCE_ORCHESTRATION.md` (2026-08-23).
The brief's `research/performance/` deliverables live **here** instead
(`experiments/model-optimization/`), by user instruction.

Source of truth for the implemented baseline:

* engine: `simulations/src/tfg/` (`engine.py` = generalised TFG Algorithm 1,
  `reference.py` = frozen transcription, tests in `simulations/tests/`)
* synthetic experiment loop actually used by Exp 2-7:
  `simulations/experiments/_guided.py` (+ `_common.py`, `_models.py`)
* MMD: `simulations/src/LossFunctions.py` (`RBF`, 5 bandwidths, `mul_factor=2`,
  biased V-statistic on the stacked `(X;Y)` kernel)
* MNIST: `MNIST/`, Stable Diffusion: `SD_cond_SD_controlnet/`

Layout

| dir | owner | content |
|---|---|---|
| `profiling/` | Agent 1 | baseline reproduction, hierarchical profile, call accounting |
| `exact_loss/` | Agent 2 | mathematically equivalent MMD accelerations + tests + benchmarks |
| `approx_loss/` | Agent 3 | RFF / Nystrom / linear-time / sliced candidates + diagnostics |
| `estimator/` | Agent 4 | engine-vs-`_guided` equivalence, estimator/update-rule candidates |
| `systems/` | Agent 5 | autograd/systems audit and benchmarks |
| `verification/` | Agent 6 | independent re-runs and verdicts |
| `report_tools/` | Agent 7 | scripts that build `hypotheses.yaml`, `results.csv`, `pareto.*` |
| `hypotheses/` | all | one YAML per agent, merged into `hypotheses.yaml` |

Top-level deliverables: `BASELINE.md`, `HYPOTHESES.md`, `hypotheses.yaml`,
`results.csv`, `VERIFICATION.md`, `FINAL_REPORT.md`.

Conventions

* Python: `/Users/stolk/miniconda3/bin/python` (torch 2.12, CPU). NOTE: the synthetic
  experiment loop runs float32 end to end (only evaluation and `tfg.schedule` are float64). Run from `simulations/` with `sys.path` containing `simulations/src`
  and `simulations/experiments` (see `_common.py`).
* No `pip install`, no git add/commit/push (user-only).
* Timing: warm-up, `time.perf_counter()`, report median of >= 5 repeats, CPU
  peak RSS via `resource.getrusage`; report conditional-model calls as the
  hardware-independent cost.
* `results.csv` columns: commit,candidate,task,target,seed,config,hardware,dtype,
  wall_s,peak_mem_mb,score_calls,cond_calls,cond_samples,opt_loss,eval_metric,status

Cluster mirror used for all heavy runs: `/sci/labs/orzuk/shaulytolk/cdm-perf/` (simulations/ + experiments/model-optimization/), partition glacier (CPU) / catfish (L4). Final deliverables: `FINAL_REPORT.md` (start here), `VERIFICATION.md`, `HYPOTHESES.md`, `BASELINE.md`, `results.csv`, `pareto.md`/`pareto.png`.
