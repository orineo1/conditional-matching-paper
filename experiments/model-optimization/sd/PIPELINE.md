# SD guided step — what runs, what it costs, where the two ideas go

Agent S, CDM performance campaign, 2026-08-24. Entry point on this branch:
`SD_cond_SD_controlnet/scripts/run_mlgd_f.py` (+ `src/generation.py`,
`metrics.py`, `clip_utils.py`, `models.py`, `visualization.py`). File:line
references are to the state AFTER the opt-in flags of this task were added;
every default reproduces the original behaviour (audited in
`systems/AUDIT.md` sec 3 against the pre-patch line numbers).

## 0. Notation

* `x_t` architect latent, fp16, shape `[1,4,64,64]` (`numel = 16384`),
  DDIM (`eta = 0`), `n_steps` timesteps, guidance runs from `start_step` to the
  end (`n_steps - start_step` guided steps). SDEdit init: HED scribble ->
  architect VAE encode -> noised to `t_start` with `abar_{t_start}`.
* `N = num_variations` sprinter samples per step, `m = n_targets` target CLIP
  embeddings (`[m,768]`, unit-norm, **fixed for the whole run**, detached).
* Sprinter = SDXL-Turbo UNet + ControlNet-Scribble (fp16), `num_inference_steps=2`,
  `guidance_scale=0` (so diffusers runs batch 1, no CFG), `controlnet_conditioning_scale=0.8`
  (hard-coded in `run_dps_step_clip`, independent of `--controlnet_scale`, which
  is used for target generation / baseline vis only), scheduler = the model's
  default (EulerAncestral for sdxl-turbo: one `randn` initial latent + one
  ancestral noise draw per step, both from `generator`).
* Loss `L = sqrt(|MMD2_U(E, S)| + 1e-8) * loss_scale` with the *unbiased* U-statistic
  (`metrics.compute_mmd`), generalised RBF `exp(-(||a-b||^2/2bw^2)^alpha)`, single
  bandwidth = median heuristic on the current `E x S` distances (recomputed every
  step, detached) times `bandwidth_scale`.

## 1. One guided outer step (`run_mlgd_f.py` main loop)

| # | what | tensors | grad graph | cost unit |
|---|---|---|---|---|
| 1 | architect UNet on `x_t` (guided path) — `predict_noise_cfg` | prompt embeds + `added_cond_kwargs` **fixed** (encoded once before the loop) | yes (block-level checkpointing) | 1 UNet fwd at batch 2 (CFG double batch even at `gs=0`) + recompute + bwd |
| 2 | architect UNet on `x_t^reg` (unguided comparison path) | same | no | 1 UNet fwd at batch 2 |
| 3 | Tweedie `x0_hat = (x_t - sqrt(1-abar_t) eps) / sqrt(abar_t)` (`compute_pred_x0_direct`) | `abar_t` from `alphas_cumprod` | yes | O(numel) |
| 4 | architect VAE decode of `x0_hat / sf` (fp32, 512^2), clamp to [0,1] -> `pixel_x0_norm` = the ControlNet image | | yes (one `checkpoint` block) | 1 VAE dec |
| 5 | **variation loop** `run_dps_step_clip`: for each chunk of `variation_batch_size` (default 1): sprinter(`ctrl = pixel_x0_norm`) -> latent -> sprinter VAE decode fp32 -> clamp -> CLIP ViT-L/14 fp32 224^2 -> unit-norm `e_i` | ControlNet image is the ONLY input that depends on `x_t`; prompt is re-encoded every call (text encoders, fixed value) | yes: the whole closure is wrapped in a non-reentrant `torch.utils.checkpoint` AND the sprinter UNet/ControlNet have block-level checkpointing | per variation: 2 sprinter steps x (UNet+CN) fwd; VAE dec; CLIP fwd; + recompute (x2 outer, x3 inner UNet) + backward |
| 6 | MMD: `L(E, S)` | `S` fixed | yes | O(N^2 + N m) kernel |
| 7 | adaptive zeta `zeta_i = base_zeta / L.detach()` | | — | |
| 8 | `grad = dL/dx_t` (`autograd.grad`, single VJP) | | backward through 5 (all N), 4, 3, 1 | the dominant cost |
| 9 | `correction = -zeta_i * grad`; DDIM step `x_{t_prev} = DDIM(x_t, eps) + correction` (`denoise_step`) | `abar_{t_prev}` | — | |
| 10 | intermediate eval every `eval_interval` steps: `evaluate_distribution_mmd(x0_hat^reg)` — VAE decode -> PIL -> `n_eval` sprinter photos (batch 2) -> CLIP -> MMD | | no | `n_eval` sprinter calls |
| 11 | `visualize_step`: 4 architect VAE decodes + **5 extra sprinter photos** + 2 full sprinter-VAE dtype casts + matplotlib | | no | 5 sprinter calls + 4 VAE dec |

Only `x_t` (and hence `pixel_x0_norm`) changes between steps; everything else the
sprinter sees is fixed (prompt, CN scale, seeds if `--seeded_rng`). Note the sqrt
transform: `dL/dE = (1/(2L)) dMMD2/dE` so with `zeta_i = base_zeta/L` the applied
step is `-(base_zeta / (2 L^2)) dMMD2/dx` — the adaptive zeta and the sqrt together
give a step of size `base_zeta * ||dMMD2/dx|| / (2 L^2)`; it grows as the loss shrinks.

### Cost expression (per guided step, `A` = one sprinter UNet+CN pass, batch 1, 1 sprinter step)

    C_step = 2 * C_arch(batch 2) [1 guided (+recompute +bwd) + 1 regular]
           + C_VAE_arch (fwd + recompute + bwd)
           + N * [ 2 steps * A * (1 fwd + 2 recompute) + 2 steps * A_bwd
                   + 2 * (C_VAEdec + C_CLIP) + C_VAEdec_bwd + C_CLIP_bwd ]
           + C_MMD(N, m)
           + [vis] 5 * (2 A + C_VAEdec) + 4 C_VAEdec
           + [eval, every eval_interval] n_eval * (2 A + C_VAEdec) + C_CLIP

THEORY.md sec 6 accounting: each in-graph variation costs ~5 A-units per sprinter
step (3 forward-equivalents + 2 backward) vs 1 A-unit for a no-grad forward.
Memory: because of the outer per-variation checkpoint the N variation graphs are
NOT held simultaneously — peak ~= fixed graph (architect UNet ckpt + fp32 VAE
decode) + one variation's recompute workspace. `--profile` measures this.

## 2. Where the trust region goes (`--trust_noise TAU`)

Location: `run_mlgd_f.py`, right after `correction = -zeta_i * grad` and before
`denoise_step` (step 9 above). The correction is added to `x_{t_prev}`, whose
per-element noise std under the VP process is `sqrt(1 - abar_{t_prev})`, with

    t_prev            = t - num_train_timesteps // num_inference_steps
    abar_{t_prev}     = alphas_cumprod[t_prev]  (t_prev >= 0)  else final_alpha_cumprod

exactly as `DDIMScheduler.step` computes it (`src/trust.py::prev_alpha_bar`,
tested against a real `DDIMScheduler`). Definition:

    cap_t  = TAU * sqrt(1 - abar_{t_prev}) * sqrt(numel)      (numel = 16384)
    Delta  = correction * min(1, cap_t / ||correction||_2)    (direction preserved)

Justification. The synthetic promoted rule is `||Delta|| <= tau sqrt(1-abar_t)` on a
d-dim vector with d = 2..10. There are two ways to carry it to d = 16384: (i) keep
the raw norm cap (per-element RMS of the step <= `sqrt(1-abar)/sqrt(d)`), which is a
dimension-dependent and, at d=16384, ~128x tighter constraint than "one noise std
per element"; (ii) cap the **per-element RMS** of the step at TAU noise stds, which
is dimension-free and is what "the guidance may never move the sample further
than the current noise amplitude" means for an isotropic noise vector whose norm is
`sqrt(1-abar) sqrt(numel)`. We use (ii); synthetic `tau=1` corresponds to
`TAU = 1/sqrt(d)` under (ii). TAU is the free knob. To calibrate it without extra
runs, every run (trust on or off) logs per step `correction_norm_raw`,
`trust_cap_tau1 = sqrt(1-abar_{t_prev}) sqrt(numel)` and `abar_prev` in
`metrics.json["steps"]`; the baseline arm therefore tells directly at which TAU the
cap would bind and how often. Cost: O(numel), no extra model calls.

Assumption: the architect latent is the unit-scale VP latent (`scaling_factor`
applied, DDIM `eta=0`), so `sqrt(1-abar)` is the actual per-element noise std.
The last guided step lands on `final_alpha_cumprod = 1` (set_alpha_to_one), which
would give `cap = 0`; `abar_prev` is therefore floored at `alphas_cumprod[0]`
(the t=0 noise level, `sqrt(1-abar) ~ 0.03`), so the final correction is capped
at ~3% of a unit-noise step rather than disabled or unbounded. Logged per step.

## 3. Where back-selection goes (`--backsel K --backsel_rule {uniform,is,kcenter}`)

Location: `src/generation.py::run_dps_step_clip` (step 5-8 above). It needs
per-variation seeds (`--seeded_rng`, forced on by `--backsel`):

1. **Pass 1, no graphs**: for every variation `i` run the sprinter+VAE+CLIP closure
   under `torch.no_grad()` (NO checkpoint wrapper, no recompute) with
   `generator = torch.Generator(device).manual_seed(seed_{t,i})` -> `E in R^{N x 768}`.
   Cost N x 1 forward-unit.
2. **Full-batch MMD geometry**: `L = loss_fn(E_leaf, S)`, `g = dL/dE` (one autograd call
   through the kernel only, `[N,768]`), `zeta_i = base_zeta / L`. The loss VALUE,
   zeta and the logged embeddings are the full-batch ones.
3. **Select** `S` (k rows) and vectors `G_i` (`src/backsel.py`, CPU generator seeded
   from `(seed, step)`):
   * `uniform`: k of N without replacement, `G_i = (N/k) g_i` (Horvitz-Thompson,
     inclusion prob k/N) — `E[G_hat] = sum_i (k/N)(N/k) J_i^T g_i = dL/dx`. Unbiased.
   * `is`: k iid draws from `p_i = 0.75 ||g_i|| / sum ||g|| + 0.25/N`, de-duplicated,
     `G_i = c_i/(k p_i) g_i` — `E[G_hat] = (1/k) sum_m E[h_{d_m}/p_{d_m}] = sum_i h_i`.
     Unbiased; floor bounds every weight by `N/(0.25 k)`.
   * `kcenter`: greedy k-center on `E`, `G_{r_c} = sum_{i in C_c} g_i`. Biased
     (Jacobian substitution within a cluster), zero selection variance, keeps 100%
     of the output-gradient mass; exact at k = N.
   Unbiasedness of the first two is verified statistically in
   `tests/test_sd_flags.py::test_unbiased` on a toy differentiable sampler
   (4000 selection draws, max |z| < 4.5), and `k >= N` reproduces the full gradient
   to 1e-12 for all three rules.
4. **Pass 2, graphs for k rows only**: regenerate exactly the selected variations
   with the checkpointed closure and the SAME seeds; the generator is created from
   its seed *inside* the closure, so the checkpoint recompute in backward (which
   re-runs the closure) sees identical noise — `checkpoint(preserve_rng_state=True)`
   restores only the global RNG, not user generators, so this is required for
   correctness. `regen_max_abs_err = max |e_sel - E[S]|` is logged per step
   (expected ~1e-6 fp16 round-off at equal batch size, exactly 0 at
   `variation_batch_size=1` if kernels are deterministic).
5. **Backward through the surrogate** `s = sum_{i in S} <G_i, e_i>` (`G_i` detached):
   `ds/dx = sum_{i in S} J_i^T G_i = G_hat`. One VJP through k variation graphs +
   the fixed architect graph.

Cost currencies (reported separately in `profile.json` and `metrics.json["steps"][*]["backsel"]`):
`n_forward = N + k` sprinter forwards (UP by k/N), `n_differentiated = k` (DOWN from N);
in A-units per sprinter step `(N + 5k)/(5N)` vs 1 (THEORY.md sec 6). Peak memory:
unchanged in principle (the outer checkpoint already serialises variation graphs),
so on SD the win is compute, not memory — the profile will say.

## 4. The other flags

* `--no_vis`: skips `visualize_step` (per step and the baseline figure). Exact for
  the guidance path. Without `--seeded_rng` the 5+ sprinter calls in vis consume the
  global RNG, so `novis` vs `baseline` would differ in sprinter noise; with
  `--seeded_rng` (used by every dev arm) they are RNG-identical.
* `--arch_single_batch`: at `guidance_scale == 0`, `predict_noise_cfg` runs the UNet
  on the unconditional row only (`encoder_states[:1]`, `added_cond[:1]`) instead of
  the `cat([x]*2)` CFG batch; the blend `np_u + 0*(np_t - np_u)` equals `np_u`
  exactly (modulo `0*inf`, never observed), so the result is identical up to
  fp16 batch-dimension kernel round-off. Ignored when `gs != 0`. Applies to the
  guided, regular and baseline-vis calls. Halves the architect share (forward,
  checkpoint recompute AND backward).
* `--seeded_rng`: init noise from `Generator(seed*7919+1)`, variation `i` at guided
  step `s` from `seed*1000003 + s*10000 + i`, selection generator `... + 9999`,
  intermediate eval `... + 5000 + j`, final eval `seed*1000003 + 7000000 + j`. Arms
  with the same seed see the same noise everywhere. Target generation (before the
  loop) still uses the global RNG seeded by `--seed`; `--target_cache DIR` makes the
  targets/scribble byte-identical across arms (first arm builds, others load).
* `--variation_batch_size B`: chunk size for the variation loop (exact w.r.t. the
  per-variation noise because of per-sample generators; fp16 batch round-off only).
* `--profile`: `StepProfiler` — CUDA-synchronised sections `architect` (UNet x2,
  x0, VAE decode), `sprinter_fwd` / `vae` / `clip` (forward of the closure),
  `nograd_sprinter_fwd` / `nograd_vae` / `nograd_clip` (backsel pass 1), `mmd`, `select`, `backward` (the VJP incl.
  checkpoint recomputes), `eval_intermediate`, `vis`, `denoise`; per-step
  `max_memory_allocated_mb`; JSON at `<output_dir>/profile.json` (rewritten every
  step; `summary` = per-section means; `meta.final_eval` = final MMDs).
* `--eval_n` (final fresh-sample eval count) and `--eval_n_intermediate` (default 10
  = the repo's hard-coded `n_eval`; NOTE the brief said 2000 — that is the default of
  `run_dps_synthetic_targets.py` on another branch; here the code default stays 10
  and the dev scripts pass `--eval_n 2000`), `--eval_batch_size` (repo: 2).

## 5. Dev configuration (faithful, reduced)

Original zeta5 run (memory): architect SDXL-base, prompt "", DDIM 250 steps from
125, base_zeta 5, 50 man + 50 woman targets (`--target_prompts`), sprinter default
prompt, seed 1, eval 100. Reduced: `n_steps 100 / start_step 50` (50 guided
steps, same strength 0.5), targets 50+50, `N = 32`, `eval 2000` (batch 8),
`kernel_alpha 1, bandwidth_scale 1, loss_scale 1` (the run_mlgd_f defaults).
Arms (`submit_sd_dev.sh ARM`), all with `--seeded_rng --profile --seed 1 --target_cache`:

| arm | flags |
|---|---|
| baseline | (vis on, CFG double batch) |
| novis | `--no_vis --arch_single_batch` |
| trust | novis + `--trust_noise $TRUST_TAU` (default 1.0; recalibrate from baseline's `trust_cap_tau1`) |
| backsel_k8_kcenter | novis + `--backsel 8 --backsel_rule kcenter` |
| backsel_k8_uniform | novis + `--backsel 8 --backsel_rule uniform` |
| trust_backsel | novis + trust + `--backsel 8 --backsel_rule kcenter` |

Outputs: `output/sd_perf/<arm>_seed<seed>/{metrics.json, profile.json, final_*.png, photos_*}`.
`metrics.json`: `final_mlgd_f_mmd`, `final_regular_mmd`, `mmd_delta`, per-step
records (grad norm, zeta, correction norms, trust cap/scale, backsel selection),
`profile_summary`.

## 6. Post-mortem 2026-08-24: final-eval OOM and persistence

* `evaluate_distribution_mmd` generated photos in batches but CLIP-encoded ALL
  `n_eval` photos in one batch (2000x3x512x512 fp32 = 6.3 GB on GPU + ~2 GB per ViT
  activation) -> OOM on the L40S. Fixed in `src/metrics.py`: photos stay on CPU,
  CLIP runs in chunks of `clip_batch_size=32`. Exact up to batch round-off.
* The final latent/scribble was only written AFTER the final eval. Now
  (`run_mlgd_f.py` step 9b) `final_latents.pt`, `final_scribble_{mlgd_f,regular}.png`,
  `target_clip_embeddings.pt` and `metrics_partial.json` are written BEFORE it, and
  `sd/eval_final.py` (+ `submit_sd_eval.sh`) re-evaluates any run dir standalone
  (sprinter + CLIP only; eval seeds `seed*1000003 + 7000000 + j`, disjoint from
  guidance seeds and identical across arms). Runs from before this fix cannot be
  evaluated post hoc.

## 7. Shared framework (2026-08-25)

* `src/trust.py` and `src/backsel.py` are adapters over `simulations/src/tfg/trust.py` and
  `simulations/src/tfg/backsel.py` (sec 2-3 above describe the same math; the rules are now
  implemented once). `src/_tfg_path.py` appends `simulations/src` to `sys.path`.
* `src/generation.py::variation_objective` returns the loss / backsel surrogate WITH graph;
  `run_dps_step_clip` (legacy loop) differentiates it w.r.t. `latents_step`; the
  `--engine tfg` path (`src/tfg_engine_path.py`) hands `-zeta_i * loss` to
  `tfg.engine.GeneralizedTFG` as `log_f` and lets the engine differentiate, DDIM-step and
  trust-clip (`TemporalConfig(step_clip="noise_prev_rms", step_tau=TAU,
  step_min_noise=sqrt(1-alphas_cumprod[0]))`, `guidance_scaling="raw"`, `rho=1`,
  `n_schedule(constant, n_max=N)` so `log_f` receives the eta keys that seed the sprinter).
  Differences from the legacy loop: module docstring of `tfg_engine_path.py`.
