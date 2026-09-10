# Stage 2 plan — androgynous-scribble experiment on the SD pipeline (design only)

How the mean-blindness / witness-selection hypothesis would run on this branch's
Stable-Diffusion pipeline (`SD_cond_SD_controlnet/scripts/run_mlgd_f.py` +
`SD_cond_SD_controlnet/src/`). **No code changes are made here** — this file lists
precisely what would need adding.

## Target sets (pipeline's existing prompt-group mechanism — no code needed)

`run_mlgd_f.py --mode gender` builds targets from `--target_prompts` specs
(`name:prompt:n` triples, parsed in `build_targets_gender`; defaults to a 10+10
Man/Woman split). The two conditions:

- **Bimodal (existing M/F)** — the default groups, or explicitly:
  `--target_prompts "Man:a superrealistic portrait photograph of a man, studio lighting:16" "Woman:a superrealistic portrait photograph of a woman, studio lighting:16"`
- **Concentrated unimodal (androgynous)** — a SINGLE group:
  `--target_prompts "Androgynous:a superrealistic portrait photograph of an androgynous person, ambiguous gender, gender-neutral appearance, studio lighting:32"`
  (equal total n so MMD estimator variance matches; one caveat: `build_targets_gender`
  fits its diagnostic PCA on the first and last group — with one group both anchors are
  the same set, which is fine for a single-mode target but worth noting in plots).

Both conditions share (approximately) the same CLIP-space mean along the gender axis;
the androgynous set is the concentrated-midpoint analogue of the toy's N(0, s²).

## Arms

1. **mean-loss** — guidance loss `||mean(gen CLIP) − mean(target CLIP)||²`.
2. **MMD** — existing `--loss_fn mmd` (`src/metrics.py:compute_mmd`).
3. **MMD + witness back-selection** — MMD value on all `num_variations` embeddings,
   gradient through only k of them chosen ∝ |witness score| (+0.3 uniform floor),
   exactly `simulations/src/witness_utils.py`'s construction.
   (A 4th arm, MMD + uniform back-selection, is the natural control, same as Stage 1.)

## What would need adding to `run_mlgd_f.py` / `src` (precisely; NOT implemented)

This branch's `src/generation.py:run_dps_step_clip` has `reuse_frac` but **no
back-selection machinery**, and `src/metrics.py` has **no `compute_witness_scores`**
(both verified by grep on this branch). Required additions:

1. `src/metrics.py`: add `compute_mean_loss(x, y)` returning
   `((x.mean(0) − y.mean(0))**2).sum()`; register `"mean": compute_mean_loss` in
   `run_mlgd_f.py`'s `LOSS_FNS` dict and extend `--loss_fn` choices to
   `["mmd", "swd", "mean"]`. (`run_dps_step_clip` already takes `loss_fn` as a
   callable `loss_fn(generated, targets) -> scalar`, so nothing else changes.)
2. `src/metrics.py` (or a new `src/witness.py`): port `compute_witness_scores`,
   `select_backsel_mask`, and `apply_backsel` from `simulations/src/witness_utils.py`,
   swapping the toy `LossFunctions.RBF` kernel for the same generalized RBF that
   `compute_mmd` uses (respecting `bandwidth_scale` / `kernel_alpha`), on GPU tensors.
3. `src/generation.py:run_dps_step_clip`: after the fresh variation CLIP embeddings
   are assembled (and any `reuse_frac` rows appended), insert the backsel step —
   score all `num_variations` embeddings against `all_clip_embeddings` under
   `no_grad`, then build the loss batch with non-selected rows `.detach()`ed
   (loss VALUE still uses all rows; gradient flows through k). New parameters:
   `backsel_k=None`, `backsel_rule="uniform"`, `witness_floor=0.3`,
   `backsel_generator=None`. Note the interaction with `reuse_frac`: reused rows are
   already detached, so backsel should select among the `n_new` fresh rows only
   (or count reused rows as never-selectable) — this needs an explicit decision.
   Memory note: with per-variation `torch.utils.checkpoint` batches of 1, detaching
   a row also lets its Sprinter/VAE graph be freed — backsel gives a real speed/memory
   win here, unlike the toy where the forward cost is unavoidable.
4. `run_mlgd_f.py`: CLI flags `--backsel_k`, `--backsel_rule {uniform,witness}`,
   `--witness_floor`, threaded into the `run_dps_step_clip` call and into the wandb
   config dict; a per-run `torch.Generator` seeded from `--seed` for reproducible
   selection.
5. Submit scripts: one per (condition × arm), cloned from the existing
   `submit_dps_*.sh` pattern (salmon L40S, `scribble_env`, **no pip installs**),
   500 steps / 40% noise / alpha=2 per the completed-runs table.

## Evaluation

- **Fresh-sample MMD**: existing `src/metrics.py:evaluate_distribution_mmd` on
  `n_eval` fresh Sprinter photos from the final (guided) scribble vs the run's own
  target embeddings — already wired into `run_mlgd_f.py`'s final-eval section.
  Additionally, cross-evaluate each run against BOTH target sets (bimodal and
  androgynous): the mean-loss arm should score near-identically against the two
  (mean-blindness), the MMD arms should not. This cross-eval is offline
  (`metrics.json` + saved `npy/photos_*.npy` embeddings), no pipeline change.
- **Gender-axis projection histogram (bimodality check)**: project eval-photo CLIP
  embeddings onto a text-defined gender axis (normalize
  `text("a photo of a man") − text("a photo of a woman")`), histogram the scalar
  projections per arm. `scripts/plot_semantic_axes.py` is **not present on this
  branch** (verified); it exists on `experiment-future-projects` and would be the
  reference implementation (including its tokenizer fix:
  `processor.tokenizer`, not `processor`, for `get_text_features`). Expected:
  M/F-target runs → bimodal histogram; androgynous-target MMD runs → concentrated
  unimodal at the midpoint; mean-loss runs on the androgynous target → wide and/or
  bimodal despite a correct mean (P2's image-space analogue). Report a bimodality
  statistic (Ashman's D between 2-component GMM fits, or the dip test) alongside
  the histogram.
- **Witness-vs-uniform contrast**: paired across seeds at matched
  (`num_variations`=n, `backsel_k`=k), final fresh-sample MMD — the prediction is
  the witness gain is larger for the androgynous (concentrated) target than for M/F.
