# Ori's previous-step sample reuse: what is actually in the repo

Searched 2026-08-24: `git grep` over every `origin/*` and `upstream/*` branch
(94 remote refs) for `prev_samples | old_samples | keep_frac | replay |
reuse_frac | sample_buffer | prev_step_samples | mix_frac` in `simulations/`
and `SD_cond_SD_controlnet/`, plus `git log --all -S`.

**Found.** The mechanism exists on three upstream branches (author "Claude",
2026-08-20, i.e. a Claude session run from Ori's fork `orineo1/...`), under the
name **`reuse_frac`**:

| branch | file | what |
|---|---|---|
| `upstream/claude/hybrid-sampling-optimization-55fv3b` | `simulations/src/Optimization.py` (`optimize_LGD`), commits `2515aec` -> `334475f` -> `209458b` -> `0fd03e3` | the core mechanism + momentum/Adam options; sanity notebook `simulations/notebooks/Exp_hybrid_sampling_sanity_check.ipynb`; grid scripts `run_hybrid_sweep.py`, `run_reuse_momentum_grid.py` |
| `upstream/claude/reuse-adamdps-grid` (single commit `60c81ef` on top of `main`) | `simulations/src/Optimization.py`, `simulations/scripts/run_reuse_adamdps_grid.py` | minimal port of the same mechanism to main + AdamDPS x reuse grid |
| `upstream/claude/sd-reuse-adamdps-schedule` (`02e87bd`, `6b4c7ce`) | `SD_cond_SD_controlnet/src/generation.py`, `scripts/run_mlgd_f.py` | same mechanism on the SD pipeline: reuses the previous step's **detached CLIP embeddings** (`prev_variation_clip`) inside the MLGD-F loss step |

## Exact mechanism (simulations version, `optimize_LGD`)

Per spatial perturbation index `j` (one buffer per `j`, `num_x_t` buffers):

1. `n_reuse = min(round(reuse_frac * nsamples), buffer[j].shape[0])`,
   `n_new = nsamples - n_reuse`. **Total MMD batch stays `nsamples`; fresh
   conditional-model draws are cut to `n_new`** -- reuse is a calls saver, not
   a batch grower.
2. `target_samples = cat([buffer[j][:n_reuse], fresh])` -- the first
   `n_reuse` rows of the buffer, **no random subsampling**.
3. Buffer update: `buffer[j] = fresh.detach().clone()` -- **only the freshly
   generated rows** are buffered (commit `334475f` fixed an earlier version
   that buffered the reused+new concatenation, which compounded staleness);
   so the buffer is depth 1, "one step old" by construction, and holds
   `n_new < nsamples` rows (hence the clamp in step 1: for
   `reuse_frac > 0.5` the effective reuse is limited by `n_new_prev`).
4. Reused rows are **detached** ("they contribute to the MMD *value* but zero
   gradient" -- comment in the code); only fresh rows carry gradient to `x_t`.

The SD version is identical in structure but buffers CLIP embeddings (the
sprinter+VAE+CLIP forward is the expensive part there).

## Fractions used

* `run_reuse_momentum_grid.py`: `reuse_frac in {0.0, 0.1, ..., 0.9}` x
  `momentum in {0.0, 0.9}` (beta2=0.999 when momentum>0), `nsamples=250`,
  `num_x_t=3`, methods LGD / LGD-CM, 25 runs.
* `run_reuse_adamdps_grid.py`: same `reuse_frac` grid x `adamdps in {0,1}`.
* The "70% current / 30% previous" description corresponds to
  `reuse_frac = 0.3`.

## No results in the repo

No grid outputs for reuse are committed on any branch (only pre-existing
baseline result JSONs); the sanity-check notebook exercises the code path but
carries no comparative numbers. So the mechanism is implemented upstream but
**unscreened**; this campaign's screening is the first quantitative test.

## Relation to the generalisation built here (`tfg/replay.py`)

Ori's variant = our `subsample` mode with `depth=1`, `batch_total=n`,
deterministic "first rows" selection instead of tape-keyed subsampling, and
replay fraction `p` (=`reuse_frac`); our `replay30` candidate reproduces it
(fresh `n - round(0.3 n)`, replay `round(0.3 n)`) up to the row-selection rule
(we subsample the previous step's cache via the NoiseTape instead of taking a
prefix -- for depth 1 with a fresh-only buffer the two have the same
distribution, since the buffer rows are i.i.d.). The generalisation adds
geometric depth-k buffers, a weighted-V-statistic mode, and the augment
(`_aug`) arms where fresh n is kept and replay grows the batch.
