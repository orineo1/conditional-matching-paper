"""
tfg_engine_path.py — run the SD architect loop through the shared generalised
TFG engine (``simulations/src/tfg/engine.py::GeneralizedTFG``), opt-in via
``run_mlgd_f.py --engine tfg``.

What is mapped onto the engine (Algorithm 1 with every extension off except
the ones listed):

  eps_theta(x_t, t)  = architect UNet via ``predict_noise_cfg`` (engine step t in
                       T..1 -> diffusers timestep ``timesteps_to_run[T - t]``)
  x0                 = (x_t - sqrt(1-abar_t) eps) / sqrt(abar_t)     (engine line)
  log_f(x0, n_t, eta_keys)
                     = -zeta_i * L_for_grad, where L_for_grad comes from
                       ``generation.variation_objective`` (architect VAE decode
                       under checkpoint -> N sprinter+VAE+CLIP variations -> sqrt
                       MMD vs the fixed CLIP targets; with backsel: full-batch
                       value + selected-subset surrogate) and zeta_i =
                       base_zeta / L.detach()  -- the adaptive zeta is folded into
                       log_f so the engine's rho can stay 1 (its value is the
                       constant -base_zeta; only its gradient matters).
  schedule           = ``SDSchedule``: the diffusers DDIM alphas_cumprod restricted
                       to the guided timesteps, alphabar[0] = final_alpha_cumprod.
  noise              = NoiseTape(seed): variation seeds are the tape's key hashes
                       of the engine's eta keys ("eta", t, i); the selection
                       generator is seeded from ("backsel", t); the SDEdit init
                       noise comes from ("x_T",).
  trust              = TemporalConfig(step_clip="noise_prev_rms", step_tau=TAU,
                       step_min_noise=sqrt(1-alphas_cumprod[0]))  -- the engine
                       applies ``tfg.trust`` with the SD cap convention.
  guidance           = config.guidance_scaling="raw", rho_scalar=1:
                       x_{t-1} = DDIM(x_t) + clip(Delta_t),  Delta_t = -zeta_i dL/dx_t.

DIFFERENCES from the legacy loop (honest list, see README "engine tfg"):
  1. DDIM arithmetic: the engine computes sqrt(abar)*x0 + sqrt(1-abar)*eps with
     float32 scalars from ``SDSchedule`` on fp16 tensors; diffusers'
     ``DDIMScheduler.step`` does the same operations from its own float32 table.
     Same formula, not guaranteed bit-identical (fp16 round-off; the loop is
     chaotic, so trajectories diverge like the measured GPU-nondeterminism floor).
  2. The engine calls log_f on ``x0 + 0 * delta`` (gamma_bar = 0, n_mc = 1): one
     extra tape draw and a no-op add per step.
  3. No per-step visualisation and no intermediate unguided eval inside the
     engine loop (the unguided twin trajectory is run separately, unchanged).
  4. The trust cap is applied by the engine (``_step_clip``) to Delta_t before
     it is added -- identical math to the legacy ``apply_trust`` (both call
     ``tfg.trust``), including the abar floor via ``step_min_noise``.
  5. Backsel selection randomness: legacy uses ``Generator(seed*1000003 +
     step*10000 + 9999)``; here ``Generator(tape.key_seed(("backsel", t)))``.
     Variation seeds likewise come from the tape, not the legacy formula, so
     the two paths draw different (but equally reproducible) sprinter noise.
  6. Everything else -- sprinter call, CLIP encode, MMD (SD's unbiased,
     median-bandwidth, generalised-RBF ``compute_mmd``; NOT tfg's
     DistributionalLoss / fast_mmd), zeta rule, sqrt loss, checkpointing,
     regeneration -- is the same code (``variation_objective``).
"""

import math
import time

import torch

import _tfg_path  # noqa: F401
from tfg.config import BackselConfig, NScheduleConfig, TFGConfig, TemporalConfig
from tfg.engine import GeneralizedTFG
from tfg.noise_tape import NoiseTape
from tfg.schedule import DiffusionSchedule

from generation import denoise_step, predict_noise_cfg, variation_objective, compute_pred_x0_direct


class SDSchedule(DiffusionSchedule):
    """``DiffusionSchedule`` interface over the diffusers DDIM alphas_cumprod
    restricted to the guided timesteps: ``alphabar[T-i] = ac[timesteps[i]]``
    (i = 0 the noisiest guided step), ``alphabar[0]`` = the abar the last guided
    DDIM step lands on (= final_alpha_cumprod when the list runs to the end)."""

    def __init__(self, ddim_scheduler, timesteps_to_run, device, dtype=torch.float32):
        c = ddim_scheduler.config
        if getattr(c, "clip_sample", False) or getattr(c, "thresholding", False) \
                or getattr(c, "prediction_type", "epsilon") != "epsilon":
            raise ValueError("--engine tfg assumes an epsilon-prediction DDIM without "
                             "clip_sample/thresholding (SDXL-base config); got "
                             f"{dict(c)}")
        ac = ddim_scheduler.alphas_cumprod
        T = len(timesteps_to_run)
        ab = torch.empty(T + 1, dtype=torch.float64)
        # abar the LAST guided step lands on, exactly as DDIMScheduler.step computes
        # it (final_alpha_cumprod when the guided list runs to the end of the schedule)
        from trust import prev_alpha_bar
        ab[0] = prev_alpha_bar(ddim_scheduler, timesteps_to_run[-1])
        for i, ts in enumerate(timesteps_to_run):
            ab[T - i] = float(ac[int(ts)])
        self.T = T
        self.dtype = dtype
        self.device = torch.device(device)
        self.alphabar = ab.to(device=device, dtype=dtype)
        self.min_alphabar = None
        self.betas = 1.0 - self.alphabar[1:] / self.alphabar[:-1]
        self.timesteps = [int(ts) for ts in timesteps_to_run]

    def diffusers_t(self, t):
        """Engine step t (T..1) -> diffusers timestep."""
        return self.timesteps[self.T - int(t)]


def build_config(T, num_variations, args, step_min_noise):
    temporal = TemporalConfig(
        step_clip=("noise_prev_rms" if args.trust_noise > 0 else "none"),
        step_tau=(args.trust_noise if args.trust_noise > 0 else 1.0),
        step_min_noise=step_min_noise)
    backsel = BackselConfig(
        enabled=args.backsel > 0, k=max(1, args.backsel),
        rule={"uniform": "uniform", "is": "importance", "kcenter": "kcenter",
              "strat": "stratified_balanced"}[args.backsel_rule],
        weighting=("soft" if args.backsel_weighting == "soft" else "hard"),
        tau_mult=args.backsel_soft_tau_scale, tau_mode=args.backsel_soft_tau_mode)
    return TFGConfig(
        T=T, N_recur=1, N_iter=0, gamma_bar=0.0, n_mc=1,
        rho_scalar=1.0, mu_scalar=0.0, rho_structure="constant",
        init="randn", guidance_scaling="raw",
        temporal=temporal, backsel=backsel,
        distributional_tfg_enabled=True,
        n_schedule=NScheduleConfig(enabled=True, type="constant", n_max=num_variations),
    ).validate()


def run_engine_loop(*, args, architect, sprinter, clip_model, clip_processor, loss_fn,
                    all_clip_embeddings, latents_init, latents_regular_init,
                    cfg_encoder_states, added_cond_kwargs, timesteps_to_run,
                    scheduler_regular, prof, device, wandb_log=None):
    """Guided trajectory through GeneralizedTFG + the unguided twin through the
    legacy DDIM loop.  Returns (latents, latents_regular, step_gradients)."""
    sch = SDSchedule(architect.scheduler, timesteps_to_run, device)
    T = sch.T
    tape = NoiseTape(args.seed, device=device, dtype=torch.float16)
    step_min_noise = math.sqrt(1.0 - float(architect.scheduler.alphas_cumprod[0]))
    cfg = build_config(T, args.num_variations, args, step_min_noise)
    numel = latents_init.numel()

    records = {}          # engine t -> per-step dict
    cur = {"t": None, "pixel": None}

    def eps_theta(x_t, t):
        prof.begin_step(T - int(t) + 1, sch.diffusers_t(t))
        cur["t"] = int(t)
        with prof.section("architect"):
            eps = predict_noise_cfg(architect.unet, architect.scheduler, x_t,
                                    torch.tensor(sch.diffusers_t(t), device=device),
                                    cfg_encoder_states, added_cond_kwargs,
                                    args.guidance_scale, single_batch=args.arch_single_batch)
        return eps

    def log_f(x0, n_t=None, eta_keys=None):
        t = cur["t"]
        with prof.section("architect"):
            def vae_decode_checkpoint(lat):
                return architect.vae.decode(lat.to(architect.vae.dtype)).sample
            pixel_x0 = torch.utils.checkpoint.checkpoint(
                vae_decode_checkpoint, x0 / architect.vae.config.scaling_factor, use_reentrant=False)
            pixel_x0_norm = torch.clamp((pixel_x0 + 1.0) / 2.0, 0.0, 1.0)
        seeds = [tape._key_seed(tuple(k)) % (2**31 - 1) for k in eta_keys]
        gen = torch.Generator().manual_seed(tape._key_seed(("backsel", int(t))) % (2**31 - 1))
        loss_for_grad, loss_scaled, zeta_i, loss_norm, vl_clip_flat, info = variation_objective(
            latents=None, latents_step=None, noise_pred=None, pixel_x0_norm=pixel_x0_norm,
            sprinter=sprinter, all_clip_embeddings=all_clip_embeddings,
            num_variations=args.num_variations, variation_batch_size=args.variation_batch_size,
            base_zeta_prime=args.base_zeta, clip_model=clip_model, clip_processor=clip_processor,
            vae=sprinter.vae, vae_scaling_factor=sprinter.vae.config.scaling_factor,
            variation_prompt=args.sprinter_variation_prompt, loss_fn=loss_fn,
            loss_scale=args.loss_scale, variation_seeds=seeds,
            backsel_k=args.backsel, backsel_rule=args.backsel_rule, backsel_generator=gen,
            backsel_weighting=args.backsel_weighting,
            backsel_soft_tau_scale=args.backsel_soft_tau_scale,
            backsel_soft_tau_mode=args.backsel_soft_tau_mode, profiler=prof)
        zeta_val = float(zeta_i)
        records[t] = {"step": T - t + 1, "timestep": sch.diffusers_t(t),
                      "mmd_loss": float(loss_scaled), "zeta_i": zeta_val,
                      "loss_norm": float(loss_norm), "variation_seeds": seeds,
                      "backsel": ({k: v for k, v in info.items() if k != "p"} if info else None),
                      "n_differentiated": (info["n_differentiated"] if info else args.num_variations)}
        # adaptive zeta folded in: d/dx (-zeta L) = -zeta dL/dx  (zeta detached)
        prof.in_backward = True          # the engine's autograd.grad follows immediately
        return -(zeta_val * loss_for_grad)

    def trace(name, t, r, k, tensor):
        if name == "Delta_t":
            rec = records[int(t)]
            rec["gradient_norm"] = float(tensor.norm()) / max(rec["zeta_i"], 1e-30)
            rec["correction_norm_raw"] = float(tensor.norm())
            rec["correction_norm"] = rec["correction_norm_raw"]
            ab_prev = float(sch.ab(int(t) - 1))
            rec["abar_prev"] = ab_prev
            rec["trust_cap_tau1"] = max(math.sqrt(max(1 - ab_prev, 0.0)), step_min_noise) * math.sqrt(numel)
            rec["trust_cap"] = (args.trust_noise * rec["trust_cap_tau1"] if args.trust_noise > 0 else None)
        elif name == "x_ddim":
            cur["x_ddim"] = tensor.detach()
        elif name == "x_prev":
            rec = records[int(t)]
            applied = float((tensor - cur["x_ddim"]).float().norm())
            rec["correction_norm_applied"] = applied
            rec["trust_scale"] = (applied / rec["correction_norm_raw"] if rec["correction_norm_raw"] > 0 else 1.0)
            prof.in_backward = False
            prof.end_step(extra={"mmd_loss": rec["mmd_loss"], "grad_norm": rec["gradient_norm"],
                                 "trust_scale": rec["trust_scale"],
                                 "n_differentiated": rec["n_differentiated"]})
            print(f"  [tfg] step {rec['step']}/{T} t={rec['timestep']} MMD={rec['mmd_loss']:.6f} "
                  f"zeta={rec['zeta_i']:.3f} ||corr||={rec['correction_norm_raw']:.3f} "
                  f"applied={applied:.3f}", flush=True)
            if wandb_log is not None:
                wandb_log({"step": rec["step"], "mmd_loss": rec["mmd_loss"], "zeta": rec["zeta_i"],
                           "correction_norm": rec["correction_norm_raw"]})

    engine = GeneralizedTFG(eps_theta, log_f, sch, tape, cfg)
    latents = engine.run(tuple(latents_init.shape), trace=trace, x_init=latents_init).to(latents_init.dtype)

    # unguided twin: the legacy DDIM loop, unchanged
    latents_regular = latents_regular_init.detach().clone()
    with torch.no_grad():
        for t in timesteps_to_run:
            eps_r = predict_noise_cfg(architect.unet, scheduler_regular, latents_regular, t,
                                      cfg_encoder_states, added_cond_kwargs, args.guidance_scale,
                                      single_batch=args.arch_single_batch)
            latents_regular = denoise_step(scheduler_regular, eps_r, t, latents_regular)

    step_gradients = [records[t] for t in sorted(records, reverse=True)]
    counts = engine.counter.as_dict()
    print(f"[tfg] engine done: denoiser_calls={counts['denoiser_calls']} "
          f"predictor_evals={counts['predictor_evals']}", flush=True)
    return latents, latents_regular, step_gradients
