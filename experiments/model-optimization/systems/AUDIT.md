# Agent 5 — systems / autograd efficiency audit (static)

Campaign: `experiments/model-optimization/` (brief FABLE_CDM_PERFORMANCE_ORCHESTRATION, 2026-08-23).
Commit audited: `6af2081` (branch `tfg-generalization-v2`, working tree with uncommitted
`simulations/experiments/_guided.py`). Nothing under `simulations/`, `MNIST/`,
`SD_cond_SD_controlnet/` was modified; every runnable check lives in
`experiments/model-optimization/systems/` (see `BENCH.md`, `bench_rows.csv`,
`runners.py`, `bench.py`, `microbatch_mmd.py`).

Legend for every finding: **EXACT** = bit-identical or float32 round-off only
(caching, `requires_grad_(False)`, batching with the same RNG draws);
**REORDER** = same arithmetic, different float32 reduction order (per-step agreement
~1e-6, but see the chaos note in §1.0 — end-to-end trajectories can still differ);
**NUMERICS** = changes the computed quantities (mixed precision, truncation).

---

## 1. Synthetic path (runnable): `_guided.run` + `ConsistencyModels` + `Diffusion` + `LossFunctions` + `tfg/engine.py`

### 1.0 Two facts that frame every "equivalence" claim below

1. **The loop is float32 throughout, not mixed.** `torch.get_default_dtype()` is float32
   in every experiment except `exp1_delta_target_equivalence.py:77`; the checkpoints are
   float32; `target_set` casts to `.float()` (`_common.py:55-58`); `x = torch.zeros(1, d)`
   (`_guided.py:86`) is float32; `betas/baralphas` are float32 (`Diffusion.py:28-32`).
   `oracle.load_params` returns float64 *parameters* (`tfg/oracle.py:53`) but they only
   enter the loop through `target_set(...).float()`. The only casts per step are no-ops
   (`Diffusion.py:348` `x_start.to(dtype=float32)`, `:345` `condition_x.to(device)`,
   `LossFunctions.py:21,36-37` `.to('cpu')`). So "(vi) float32 vs mixed" is moot; we
   benchmarked float64-throughout instead (2x slower, §BENCH).
2. **The guided trajectory is chaotic at float32 resolution.** Running the *identical*
   algorithm in float64 (`bench.py`, `float64_throughout`) moves the final `x` by
   0.04 / 0.22 / 0.12 / 1.1 (2D, restart 0, the four cells) and flips which mode some
   restarts land in (`lgd none` restart 2: x32 = +5.67, x64 = -4.87). A 1-ulp change in
   the per-step gradient is therefore enough to change the *final* `x` by O(1) in a
   fraction of restarts. Consequently a "same inputs -> same outputs to <= 1e-6"
   test end-to-end is passable **only** by bit-identical variants (those marked EXACT
   below; they pass with 0.0). Any REORDER variant must be judged by (a) teacher-forced
   per-step gradient agreement (we measure 1e-7..1e-4 absolute, i.e. float32 round-off
   of a gradient of magnitude 1e-2..1) and (b) the distribution of `L2` over restarts.
   The repo's own `_guided.run` is bit-reproducible across repeats and across
   `torch.set_num_threads(1|4)` (checked), so the reference itself is a fixed point; it is
   just not a *stable* one.

### 1.1 Frozen parameters with `requires_grad=True`
* Both checkpoints load with `requires_grad=True` on all 66,689 (CM) + 200,065 (denoiser)
  parameters (`_models.py:79-83, 112-116` — `load_state_dict` + `.eval()`, no
  `requires_grad_(False)`). `_guided.py:118` and `engine.py:184,208` use
  `torch.autograd.grad(loss, x)`, so **no `.grad` buffers are populated and the engine
  prunes the weight-gradient branches** (only nodes on the path to `x` execute). What it
  *does* cost: every `Linear` saves its input for a weight-grad that is never computed
  (`AddmmBackward` saves `mat1`), and the graph holds `~2x` the nodes. With the 128-wide
  MLPs this is invisible in wall time: `frozen_params` vs reference is within timing noise
  (`bench_rows.csv`, `frozen_params` rows: 0.26/0.20, 0.29/0.30, 0.91/0.74, 0.20/0.19 s
  — the machine was shared with other agents' benchmarks; see BENCH.md). **EXACT** (0.0
  end-to-end diff, all four cells). Still recommended as hygiene
  (`for p in m.parameters(): p.requires_grad_(False)` after load in `_models.py:82,115`)
  because it is what makes the MNIST/SD graphs smaller (there the saved activations are
  large).

### 1.2 Score network evaluated more than once per outer step?
* `_guided.run`: **once** (`_guided.py:89-90` -> `Diffusion.sample_ddim_step:354`). The
  Tweedie `pred_x0` is returned by the same call (`Diffusion.py:368-380`). Good.
* `tfg/engine.py`: `eps_theta` once per recurrence (`engine.py:174`); but the **mu branch
  (`engine.py:205-210`) re-evaluates the predictor `log_f` `N_iter` times**, each of which
  is a *fresh conditional sampler batch of `n_t` rows* (`_call_log_f`, `:103-109`), and
  `_log_f_tilde` (`:111-120`) runs the `n_mc` smoothing perturbations sequentially. With
  the campaign defaults (`config.py:165-172`: `N_recur=1, N_iter=0, n_mc=1`) none of this
  fires; if Exp 6-style mu-guidance is run through the engine, the mu branch costs
  `N_iter` extra conditional batches per step with no denoiser call. Batching the `n_mc`
  perturbations is the same transformation as (ii) below (**EXACT** given the tape keys
  `("delta", t, j)` are drawn from `NoiseTape`, `noise_tape.py:74`, independent of batch).

### 1.3 Uncached work per step (synthetic)
| what | where | cost (measured, 4-thread M-series CPU) | fix / exactness |
|---|---|---|---|
| `self.to(device)` on the 200k-param denoiser **every** DDIM step | `Diffusion.py:335` | 75 us of the 266 us step (28%); `ddim_step_lean` in `runners.py` is 182 us | hoist; **EXACT** (`batched_restarts+lean_ddim` rows = identical x) |
| `next(self.parameters()).dtype`, `x_start.to(...)`, `torch.full(...)` int64 `t_batch` | `Diffusion.py:338,348,351` | ~5 us | cache `t_batch` per t; EXACT |
| `TimeEmbedding` recomputes `arange/exp` sinusoid table every forward (6x per `sample`) | `ConsistencyModels.py:70-77`; `NN_utils.py:99-101` | 13 us x 6 per sampler call | register as buffer; EXACT |
| `torch.tensor([t] * B)` Python list per CM forward | `ConsistencyModels.py:196` | 2.4 us at B=8, **210 us at B=3072** (batched restarts) | `torch.full((B,1), t)`; EXACT |
| `condition_x.to(device)`, `x.to(device)` | `ConsistencyModels.py:252,255` | ~2 us | drop; EXACT |
| `hasattr(self,'input_norm')`/`'output_norm'` per forward | `ConsistencyModels.py:181,206` | negligible | — |
| `torch.manual_seed(key_seed(...))` per perturbation (global reseed also reseeds CUDA/MPS/XPU/MTIA generators, `torch/random.py:_manual_seed_impl`) | `_guided.py:101` | **99 us** each (0.1 ms x M per step; ~5% of a no-LGD step) | `torch.Generator().manual_seed(seed)` + `randn(..., generator=g)` — **bit-identical stream** (verified: `generator_seed` rows, 0.0) |
| MMD: full `(n+m)^2` stacked kernel, 5 exps, every call | `LossFunctions.py:38-44` | 460 us at n=8 of which the constant `YY` block is **93.9%** of entries (78.6% at n=32, 24% at n=256) | compute `XX`, `XY` only, cache `YY` once per run: 92 us (5x) — `BatchedMMD` in `runners.py`; **REORDER** (kernel entries bit-identical, `.mean()` order differs -> per-step grad diff <= 1.3e-6) |
| `X.to(self.device)`, `torch.vstack([X, Y])` copy of the 250-row target every call | `LossFunctions.py:36-38` | included above | same fix |
| `cur_var = model_uncond.betas[t].to(device)`; `r_t` recomputed | `_guided.py:91-92` | negligible | — |

### 1.4 Graph retention
* `_guided.py:100-108`: the `M=3` LGD perturbation graphs are built sequentially and **all
  three are kept resident** in `terms` until `autograd.grad` at `:118`. Peak graph memory
  is therefore `3x` one ladder of `n` rows — identical to the batched form (one ladder of
  `3n` rows), so batching LGD saves launches, not memory (peak RSS unchanged in
  `bench_rows.csv`). No `retain_graph=True` anywhere in the synthetic path
  (`engine.py:184` explicitly `retain_graph=False`). Graph is released every step
  (`x.detach()` at `:88`).

### 1.5 Python loops where a batched form exists
| loop | where | batched form | measured (BENCH.md) |
|---|---|---|---|
| M=3 perturbations sequential | `_guided.py:100-107` | one CM call on `3n` rows (`run_single(batched_lgd=True)`) with the three seeds' noise pre-drawn | **EXACT** (0.0) at n=8; 0.74->0.54 s |
| M separate MMDs | `_guided.py:107` | one batched `cdist` + cached YY (`BatchedMMD`) | REORDER, 1.0e-6 per-step; 0.54->0.17 s (together with the YY cache this is the 4x) |
| restarts sequential (exp2/3/4/6/7 loop `for r in range(restarts)`) | e.g. `exp2_lgd_vs_adam.py:57` | `run_batched_restarts` — denoiser is unconditional, CM rows independent, Adam element-wise, per-row divergence mask | REORDER; throughput 5 -> 70-90 restarts/s at B=16-32 (n=8), saturating ~25-30 r/s at n=32 on 4 CPU threads |
| 6-step CM ladder | `ConsistencyModels.py:257-260` | inherently sequential (each step feeds the next); only the time-embedding table is cacheable | — |
| `n` conditional rows | already batched (`cond.repeat(n,1)`) | — | — |

### 1.6 dtype / autocast / compile / device
* No autocast anywhere in the synthetic path (all float32, §1.0). `torch.compile` on the
  two MLP forwards: see BENCH.md (`torch.compile[default]` row — compile overhead reported
  separately; REORDER at best because inductor fuses reductions).
* MPS: `ConsistencyModeliCT.__init__:114` and `DiffusionModel.__init__:20` pick CUDA-or-CPU;
  `sample_ddim_step:335` moves the model to the *argument* device each call (so the repo
  loop would work on MPS with `device="mps"` in float32 only — MPS has no float64, and
  `RBF(device='cpu')` in `_guided.py:73` would pull every `y` back to CPU for the MMD,
  keeping autograd but adding two device hops per perturbation). See BENCH.md for the
  MPS row.

### 1.7 Where a scalar-loss VJP could avoid materialising activations
* In the synthetic path the only large tensor is the `(n+m)^2 x 5` kernel slab
  (`LossFunctions.py:24-25`, 333k floats at n=8, m=250, 5 bandwidths; 1.4M at n=256). Its
  backward materialises the same slab again. `microbatch_mmd.py` (part C) shows the MMD
  gradient w.r.t. the rows of `X` decomposes over row-chunks **exactly** for a fixed
  bandwidth — peak `chunk x (n+m) x 5` instead of `(n+m)^2 x 5` — and then **one** VJP
  through the sampler (`autograd.grad(X, theta, grad_outputs=g_X)`). Verified to 3.4e-16
  relative (float64) against an exact broadcast kernel, 3e-9 against the repo's
  `torch.cdist`-based `MMDLoss` (that residual is `cdist`'s own mm-path rounding,
  `torch.cdist` uses `x^2 + y^2 - 2xy` for >25 rows). Relevant for n >= 1e3 or image
  embeddings (SD: 768-D CLIP, n up to 100) rather than for the current 2D runs.

---

## 2. MNIST path (static; not runnable locally) — `MNIST/run_mlgdf.py` + `MNIST/src/*`

Audit text below was produced by a read-only sub-audit and checked for file:line consistency; nothing was run.

### Efficiency audit — MNIST guided-sampling path (`MNIST/run_mlgdf.py` + `MNIST/src/*`)

Scope: static read of `/Users/stolk/github/conditional-matching-paper/MNIST/run_mlgdf.py` (731 lines) and `MNIST/src/cond_model.py`, `uncond_model.py`, `classifier.py`, `dataset.py`. No files modified.

**Hot path:** `optimize_LGD()` at `run_mlgdf.py:304-393`, called once per seed from `run_and_save()` (`run_mlgdf.py:409-433`, 15 seeds sequential). Per outer DDIM step (`run_mlgdf.py:321-361`): 1 UNet forward → `num_x_t` sequential conditional-CM sampling ladders (6-node `ts`, 5 network evals each) → SWD loss per perturbation → `logsumexp` → `torch.autograd.grad(log_me, x_t, retain_graph=True)`.

Important framing: everything here is tiny (UNet on 1×1×28×28, CM is a 128-wide MLP on `nsamples`=1500 rows). The workload is **kernel-launch / Python-overhead / host-sync bound**, not FLOP- or memory-bound. So the biggest wins are (a) removing host↔device syncs, (b) collapsing repeated small evaluations, and (c) batching perturbations and seeds — not mixed precision or checkpointing.

Note: `classifier.py` and `dataset.py` are **not imported** by `run_mlgdf.py` (it defines its own `ImprovedCNN`/`train_classifier`, lines 99-209, and imports only `cond_model`/`uncond_model`, line 47-48). They are training/data-prep utilities and are outside the guided loop; brief notes at the end.

---

### 1. Frozen parameters still `requires_grad=True`; `autograd.grad` vs `backward`

**Finding.** Neither generative model is frozen. Only `.eval()` is called:

- `run_mlgdf.py:680-685` — `cond_model = CircularAngleConsistencyModel(...)`; `cond_model.load_state_dict(...)`; `cond_model.eval()` — no `requires_grad_(False)`.
- `run_mlgdf.py:687-690` — `uncond_model = UnconditionalUnet().to(device)`; `uncond_model.eval()` — no `requires_grad_(False)`.
- `run_mlgdf.py:696` — classifier via `load_or_train_classifier` (lines 201-209): `.eval()` only, but it is only used under `torch.no_grad()` (`run_mlgdf.py:215`), so it is harmless.
- No `torch.inference_mode` anywhere. `torch.no_grad` appears only at `run_mlgdf.py:356` (the x update), `:364` (final DDIM step), `:215` (classifier).

The loop uses `torch.autograd.grad(log_me, x_t, retain_graph=True)[0]` (`run_mlgdf.py:354`), **not** `loss.backward()`, so no `.grad` buffers are populated on parameters and the engine skips the weight-gradient kernels. But because params have `requires_grad=True`:
  - Autograd still **saves the input activations of every Conv/Linear/attention projection** in both the UNet forward (`run_mlgdf.py:323`) and each of the 5 × `num_x_t` CM forwards + `cond_embed` forwards (`cond_model.py:176, 186-187`) because they are needed for the (never-computed) weight gradients. With frozen weights only what's needed for the input-gradient path is saved.
  - The final evaluation at `run_mlgdf.py:373-388` (`model_cond_cm.sample(nsamples=final_n, condition_x=x_final...)`, SWD) builds a full graph through the CM parameters even though `x_final` is detached (`:370`), and that graph stays alive until `loss` is rebound on the next seed (`run_and_save`, `:412/:428`).

**Patch.**
```python
#run_mlgdf.py after line 685 and after line 690
cond_model.requires_grad_(False)
uncond_model.requires_grad_(False)
#run_mlgdf.py: wrap lines 373-388 in `with torch.no_grad():` (or torch.inference_mode())
```
Optionally `classifier.requires_grad_(False)` after `:696` (cosmetic).

**Savings.** Memory: removes the saved-for-weight-grad activations in every forward in the graph (UNet: one 28×28 batch-1 pass — small; CM: ~7 Linear inputs of `nsamples×128` × 5 ladder steps × `num_x_t` — a few tens of MB at `nsamples=1500, num_x_t=3`; larger for bimodal `num_x_t=10`). Compute: small (a few fewer saves/frees per op). The final-eval `no_grad` removes one whole retained CM graph (4×`nsamples` rows × 5 steps) per seed. **EXACT** (forward values unchanged; the gradient w.r.t. `x_t` is identical).
**Test.** Same seed, compare `x_final` and `grad` at every step: `max|diff| == 0` (bitwise) expected; accept ≤1e-6.

---

### 2. Score-network (UNet) evaluations per outer step

**Finding.** Already minimal: exactly **one** UNet forward per step (`run_mlgdf.py:323` `residual = model_uncond(x_t, torch.tensor([t], device=device))`), and `pred_x0` and `x_{t-1}` are derived algebraically from that same `residual` (`:328-329`); no separate Tweedie re-eval, no CFG (unconditional model, `uncond_model.py:33-34`), no restarts/recurrence. One extra forward under `no_grad` for the final step (`:366`). Total = `num_inference_steps` forwards + (`num_inference_steps-1`) backwards through the UNet.

The redundant evaluation is in the **conditional CM**, not the UNet:
- `cond_model.py:230-233` — the ladder `for t in ts[1:]: ... x = self(x, t, cond=condition_x)` runs 5 forwards per `sample()` call, and every forward re-runs the **conditioning CNN encoder** `self.cond_embed(cond)` (`cond_model.py:176`) on the identical `cond` (batch 1, 784-d). So `cond_embed` (4 convs + GroupNorms + Linear 3136→128) is evaluated **5 × num_x_t** times per outer step, each with its own autograd subgraph, when 1 × `num_x_t` (or 1, see §6) would do.

**Patch** (`cond_model.py`):
```python
#forward(): add kwarg cond_emb=None
def forward(self, x, t, cond=None, cond_emb=None):
    ...
    if cond_emb is None and cond is not None:
        cond_emb = self.cond_embed(cond.view(cond.size(0), -1))
    if cond_emb is not None:
        x = x + cond_emb
#sample(): compute once before the ladder
cond_emb = self.cond_embed(condition_x) if condition_x is not None else None
for t in ts[1:]:
    ...
    x = self(x, t, cond_emb=cond_emb)
```
**Savings.** 5→1 encoder forwards (and backward subgraphs) per `sample()` call: ~80% of the encoder cost; the encoder is the only conv path in the CM and the only route for gradient to `x0_sample`, so this is the dominant CM cost at batch 1 (launch-bound: ~15 kernels fwd + ~25 bwd each). **EXACT.**
**Test.** Same seed: `max|x_final diff| ≤ 1e-6`; `max|grad diff|/|grad| ≤ 1e-6` per step.

---

### 3. Uncached / recomputed quantities

| # | Location | What | Patch | Class |
|---|---|---|---|---|
| 3a | `run_mlgdf.py:344-345` (in the `num_x_t` loop), also `:381` | `generate_mog_samples(...)` rebuilds `Categorical`, `MultivariateNormal` (Cholesky of the 1×1 covs), `MixtureSameFamily`, and does `means_t.to(device)`, `covs_t.to(device)`, `weights.to(device)` **every perturbation of every step** (`:249-257`). Default `device='cpu'` is used (not passed at `:344`), so samples are drawn in **float64 on CPU**. | Build the distribution once in `optimize_LGD` (before `:321`) — or once per run in `run_and_save` — and call `dist.sample((nsamples,))`. Because the CPU generator is independent of the CUDA generator and sizes/order are unchanged, the draws are identical. | **EXACT** (same RNG stream). |
| 3b | `run_mlgdf.py:337-340` then `:348` | Round-trip `angles_to_circular(circular_to_angles(cm_out))` — the CM output is already unit-norm (cos,sin) (`cond_model.py:193`), so `atan2 → %360 → deg2rad → cos/sin` is an identity (and a 5-kernel detour in forward and backward, per perturbation). Same at `:373-385` for `final_ang`. | Use `model_cond_cm.sample(...)[0]` directly as `X` in the SWD. Keep `angles_to_circular(ref_ang)` (refs are angles). | **EXACT** up to fp32 round-off (~1e-7). |
| 3c | `run_mlgdf.py:323`, `:366` | `torch.tensor([t], device=device)` per step: builds a tensor from a 0-dim CPU tensor → pageable H2D copy → **stream-synchronising** (`copy_` with `non_blocking=False` on CUDA syncs). | `ts_dev = ddim.timesteps.to(device)` once after `:317`; use `ts_dev[i:i+1]` / `ts_dev[-1:]`. | **EXACT.** |
| 3d | `cond_model.py:178-179` | `torch.tensor([t] * x.shape[0], dtype=float32, device=x.device)` — builds a **1500-element Python list** and a synchronous H2D copy on every CM forward (5 × `num_x_t` per step). | `t = torch.full((x.shape[0], 1), float(t), device=x.device)` (device-side fill, no sync); or compute `c_skip/c_out` as Python floats since `t` is scalar. | **EXACT** (`torch.full` with the same fp32 value). |
| 3e | `cond_model.py:73-78` `TimeEmbedding.forward` | `torch.exp(torch.arange(half_dim)*-emb)` recomputed per forward; and `t[:,None]*emb` computes an `N×dim` embedding although `t` is constant across the batch. | Register `emb` as a buffer in `__init__`; in `sample()` compute the `(1,dim)` embedding once per ladder step and broadcast. | **EXACT** (identical elementwise ops). |
| 3f | `run_mlgdf.py:324-332` | `ddim.alphas_cumprod[t]`, `alpha_t_prev`, `beta_t`, `r_t`, `compute_step_size(...)` are per-step CPU 0-dim tensor ops (cheap) and `torch.tensor(1.0)` at `:326`. | Precompute lists `alpha_t[i], alpha_prev[i], r_t[i], step_size[i]` once per `optimize_LGD` (or once per run, outside the seed loop: `ddim` is rebuilt per seed at `:315-316`). | **EXACT** if kept as fp32 tensors (identical values); negligible speed, mainly tidiness. |
| 3g | `run_mlgdf.py:260-267` SWD | `proj` redrawn per call — that is intended randomness (projections are part of the MC estimator), leave as is. Note `X.to(device).float()`/`Y.to(device).float()` — `Y` for MoG targets is a CPU fp32 tensor → one H2D sync per perturbation (follows from 3a). | See 3a/§4. | — |

There is no MMD/kernel-bandwidth computation in this file (the loss is sliced-Wasserstein); nothing to cache there.

**Savings (3a+3c+3d):** removes ≈ `1 + num_x_t×(5+1)` host-sync points per outer step (≈19 for `num_x_t=3`, ≈61 for `num_x_t=10`). Each sync drains the GPU queue and kills CPU/GPU overlap; on a launch-bound loop this is plausibly a 1.5-3× wall-clock win by itself. 3b removes ~10 small kernels fwd+bwd per perturbation.
**Test.** Same seed: `max|x_final diff| ≤ 1e-6` over 3 seeds (3a/3c/3d/3e/3f bitwise; 3b round-off).

---

### 4. Per-call `.to(device)` moves and dtype churn in the loop

- `run_mlgdf.py:249-254` (`weights.to(device)`, `means_t...to(device)`, `covs_t...to(device)`) every perturbation — but `device='cpu'` so these are CPU no-ops; the real cost is the CPU-side sampling + the later H2D of the (1500,2) ref set in SWD (`:261-262`). Fix as in 3a; if you additionally want the refs on GPU, pre-draw **all** reference sets for the run in one go on CPU in the same order (`dist.sample((n_steps*num_x_t, nsamples))` is *not* guaranteed identical to sequential draws; a loop of `dist.sample((nsamples,))` in the same order **is**) then a single `.to(device)` — EXACT; or sample directly on GPU (`component = torch.multinomial(w)`, `mean + sqrt(var)*randn`) — **CHANGES NUMERICS** (different generator stream, statistically identical).
- dtype: `mog_means`/`mog_variances`/`weights` are **float64** (`run_mlgdf.py:609-620`) → MoG samples fp64 on CPU → `angles_to_circular` casts with `.float()` (`cond_model.py:29`). `alphas_cumprod` fp32 CPU. No autocast. Recommend building the MoG in fp32 on `device` (value-equivalent for these scales; **CHANGES NUMERICS** at the 1e-7 level and changes the RNG stream if moved to GPU).
- `cond_model.py:223` `condition_x.to(device)` — same device, no-op.
- `run_mlgdf.py:322` and `:359` — `.detach().clone()` twice per step: the `clone()`s are unnecessary (`x_t.detach().requires_grad_(True)` suffices; `x_t_minus_1` is already a fresh tensor). EXACT, trivial.

---

### 5. Graph retention

- `run_mlgdf.py:354` — `torch.autograd.grad(log_me, x_t, retain_graph=True)`. The graph is **never reused** (next iteration re-detaches `x_t` at `:322`; `x_t_minus_1` is detached at `:359`). With `retain_graph=True` all saved tensors of the step-`i` graph (UNet + `num_x_t` CM ladders + `cond_embed`s) stay alive until the Python names are rebound: `residual` (`:323`, *after* the step-`i+1` UNet forward has already allocated), `losses` (`:334`), `log_me` (`:353`). So peak memory ≈ **2× one step's graph**. Patch: drop `retain_graph=True` (and optionally `del losses, log_me` after `:354`). **EXACT.**
- `run_mlgdf.py:334-351` — all `num_x_t` perturbation graphs are kept simultaneously in `losses` (needed because `logsumexp` couples them at `:353`). At MNIST scale this is modest (~tens of MB), but if it ever matters: per-perturbation `autograd.grad(loss_k, pred_x0, grad_outputs=-w_k)` with `w = softmax(stack(losses).detach())`, accumulate into `g_x0`, then one `autograd.grad(pred_x0, x_t, grad_outputs=g_x0)` through the UNet — peak memory = 1 graph, UNet backward still once. **EXACT** up to summation order (≤1e-6). Better alternative: batch them (§6).
- `run_mlgdf.py:373-388` — final CM sample + SWD not under `no_grad`: graph through CM params retained until next seed (see §1).
- No lists of non-detached tensors persist across steps (`results.append(x_final.squeeze().cpu().numpy())` at `:429` is detached). Good.

---

### 6. Python loops that could be batched

**6a. Perturbations (`num_x_t`)** — `run_mlgdf.py:335-351` runs `num_x_t` sequential `model_cond_cm.sample()` ladders (each: 1 encoder ×5 + 5 MLP forwards on 1500×128) + SWD. Batch: `x0 = pred_x0 + r_t**2*randn(num_x_t,1,28,28)`; `cond_emb = cond_embed(x0.view(num_x_t,784))` → `(num_x_t,128)` → `repeat_interleave(nsamples,0)`; run the ladder once on `(num_x_t*nsamples, 2)`; SWD on `(num_x_t, nsamples, 2)` with `torch.sort(dim=1)` and projections `(num_x_t,50,2)` (or shared). Everything in the CM is per-row (LayerNorm over features, GroupNorm per sample), so values are identical per perturbation. **CHANGES RNG realization** (one `randn(M*N,2)` is not guaranteed to equal `M` consecutive `randn(N,2)` draws on either CPU or CUDA), i.e. statistically equivalent, not bit-equivalent. Savings: `num_x_t`× fewer launches for the whole CM+SWD block (~`num_x_t`× on that block since it is launch-bound: 3× for unimodal/uniform, 10× for bimodal). Test: over ≥15 seeds, mean/std of `final_loss` and of per-step `‖grad‖` within noise (e.g., Welch t-test p>0.05), plus a **one-off exactness check** by pre-drawing the noise sequentially in the original order and feeding it to the batched path (`max|x_final diff| ≤ 1e-5`).

**6b. Seeds (15 sequential `optimize_LGD` calls)** — `run_mlgdf.py:409-433`. Each seed is an independent chain with batch-1 tensors. Batch all `S` seeds: `x_t: (S,1,28,28)`; UNet batch `S` costs ≈ the same as batch 1 (UNet2DModel with GroupNorm/attention is per-sample, no cross-batch coupling); CM batch `S*num_x_t*nsamples` rows still trivial; `log_me` per seed, `autograd.grad(log_me.sum(), x_t)` gives exact per-seed gradients (no coupling). Expected **~5-15× wall-clock** on the whole run. **CHANGES NUMERICS** (per-seed `set_seed(seed)` streams can't be reproduced; use `set_seed(GLOBAL_SEED)` once, or per-seed `torch.Generator`s for the initial noise). Test: distribution of `final_loss` / classifier hit-rate over seeds matches the sequential run statistically; optionally seed-0 exactness by running batch size 1.

**6c. CM ladder** (`cond_model.py:230-233`, 5 sequential steps) — inherently sequential; nothing to batch except the encoder/time-embedding hoisting (§2, 3e).

**6d. `classify_generated_images`** (`run_mlgdf.py:216-226`) — batch-1 loop with two `.item()`s per image; only 15 images, negligible; batching is **EXACT** (BatchNorm in eval).

**6e. Uniform target: wasted inner loop for `t<250`** — `run_mlgdf.py:357-358` sets `step_size = 0` *after* the full CM sampling + SWD + backward have already been computed. Move the check above `:334` and skip the inner loop / `autograd.grad` when `mog_means is None and t < 250`. Saves ~25% of uniform runs (`num_inference_steps=290` → ~72 tail steps). **EXACT for `x_final`** (gradient is multiplied by 0 anyway); `final_loss` realization changes because the skipped CUDA RNG draws shift the stream for the final MC evaluation. Test: `x_final` bitwise/≤1e-6; `final_loss` only statistically.

---

### 7. Mixed precision / checkpointing

- No autocast/AMP anywhere; all fp32 (refs fp64 on CPU). Given batch-1 28×28 UNet and a 128-wide MLP, AMP/bf16/TF32 would **not** help (launch-bound, not FLOP-bound) and **CHANGES NUMERICS** — not recommended.
- Gradient checkpointing: not warranted; graph memory is tens of MB. Recommend against.
- Optional: `torch.compile(uncond_model, mode="reduce-overhead")` / CUDA graphs for the UNet fwd+bwd to attack launch overhead — **CHANGES NUMERICS** slightly (kernel fusion/rounding); test `max|x_final diff| ≤ 1e-4` and SWD statistics. Note `make_deterministic` (`run_mlgdf.py:73-80`) sets `cudnn.deterministic=True`, which also constrains algorithm choice; harmless at this size.

---

### 8. Other waste

- `run_mlgdf.py:390-391` — `torch.cuda.empty_cache()` after **every seed** forces the caching allocator to release and re-`cudaMalloc` on the next seed. Remove. **EXACT.**
- `run_mlgdf.py:315-317` — `DDIMScheduler.from_config` + `set_timesteps` per seed; hoist to `run_and_save`. **EXACT**, negligible.
- W&B / plotting / pickling happen only after all seeds (`run_mlgdf.py:443-461`, `:471-575`): no per-step logging, no image saves in the hot loop. Good.
- `.item()` in the hot loop: none (only `loss.item()` per seed at `:428`); `if t < 250` at `:357` is on a CPU tensor (no sync). The **implicit** syncs are the `torch.tensor(..., device=cuda)` constructions (3c/3d) and the CPU→GPU ref-set copies (3a) — those are the real sync sources.
- `compute_step_size` with `t` a CPU `int64` 0-dim tensor (`:332`, `:285` `5*t/1000`) — fine, but folds into 3f.
- Out of loop, FYI only: `classifier.py`/`dataset.py` not used by this script; `dataset.py:189-212` `AugmentedMNISTDataset` runs 60k×3 PIL rotations and batch-1 classifier calls with `.item()` (data-prep only); `train_classifier` in `run_mlgdf.py:162-165` uses `shuffle=False` with a generator and `num_workers=0` (training-time only; `shuffle=False` is a correctness smell rather than a perf one).

---

### Prioritised patch list

| Prio | Change | Lines | Est. effect | Class | Test |
|---|---|---|---|---|---|
| 1 | Batch seeds (`S` chains in one batch) | `run_mlgdf.py:304-433` | ~5-15× wall-clock | CHANGES NUMERICS (RNG) | per-seed-loss distribution matches |
| 2 | Batch `num_x_t` perturbations into one CM call + batched SWD | `:334-351`, `cond_model.py:219-235` | ~`num_x_t`× on CM+SWD block | CHANGES NUMERICS (RNG); exact with pre-drawn noise | noise-injected exactness ≤1e-5; stats |
| 3 | Hoist `cond_embed` out of the ladder | `cond_model.py:176, 230-233` | 5→1 encoder evals per sample() (~80% of encoder fwd+bwd) | EXACT | `x_final` ≤1e-6, 3 seeds |
| 4 | Kill host syncs: `ts_dev` index, `torch.full` for `t`, cache MoG dist / pre-draw refs | `:323,:366`; `cond_model.py:179`; `:249-257,:344` | removes ~`1+6·num_x_t` syncs/step; plausibly 1.5-3× | EXACT | `x_final` bitwise / ≤1e-6 |
| 5 | `requires_grad_(False)` on both models; `no_grad` around final eval | after `:685,:690`; `:373-388` | memory (saved activations; one retained graph/seed) | EXACT | bitwise |
| 6 | Drop `retain_graph=True`; `del` graph refs | `:354` | ~2× lower peak memory | EXACT | bitwise |
| 7 | Remove circular↔angle round-trip on CM output | `:337-340,:348,:373-385` | ~10 kernels fwd+bwd per perturbation | EXACT (round-off) | ≤1e-6 |
| 8 | Skip inner loop when `step_size==0` (uniform, `t<250`) | `:357-358` → before `:334` | ~25% of uniform runs | EXACT for `x_final` | `x_final` ≤1e-6; `final_loss` stats |
| 9 | Remove per-seed `empty_cache`; hoist scheduler; cache time-embedding buffer; precompute per-step scalars | `:390-391,:315-317`; `cond_model.py:73-78`; `:324-332` | small | EXACT | bitwise |
| — | AMP / checkpointing / compile | — | not recommended (launch-bound); compile optional | CHANGES NUMERICS | — |

---

## 3. Stable Diffusion path (static; not runnable locally) — `SD_cond_SD_controlnet/scripts/run_mlgd_f.py` + `src/*`

Note: `SD_cond_SD_controlnet/run_dps_synthetic_targets.py` named in the brief does not exist on branch `tfg-generalization-v2` (it lives on `experiment-future-projects` / `discrete-x-smc`); `scripts/run_mlgd_f.py` is the equivalent entry point here and shares `src/generation.py::run_dps_step_clip`, `src/models.py`, `src/clip_utils.py`, `src/metrics.py`. Audit text below was produced by a read-only sub-audit and checked for file:line consistency; nothing was run.

### Static efficiency audit — SD guided-diffusion path (`SD_cond_SD_controlnet/`, branch `tfg-generalization-v2`)

Scope read fully: `scripts/run_mlgd_f.py` (913 l), `src/generation.py`, `src/models.py`, `src/clip_utils.py`, `src/metrics.py`, `src/image_utils.py`; `src/visualization.py::visualize_step` (called inside the hot loop, l.60-130); `scripts/eval_baselines.py` helpers (l.105-200); `notebooks/measure_dps_step_memory.ipynb` and `notebooks/results/vjp_results_lightning.csv` (grep only). No files modified.

Defaults that matter (`run_mlgd_f.py` l.73-92): `n_steps=30`, `start_step=15`, `guidance_scale=0.0` (architect CFG scale), `num_variations=6`, `controlnet_scale=0.5`; hot loop hard-codes `variation_batch_size=1` (l.697), sprinter `num_inference_steps=2`, sprinter `guidance_scale=0.0`, `controlnet_conditioning_scale=0.8` (`generation.py` l.264-272). Note `n_eval=10` and `visualize_step(num_cond=5)` are also executed inside the loop.

### 0. What one outer step actually executes (baseline accounting)

Per outer step `i` (`run_mlgd_f.py` l.655-780), with N = `num_variations`:

| Work | Where | In autograd graph? |
|---|---|---|
| Architect UNet, batch **2** (uncond+cond) | l.663 → `predict_noise_cfg` (`generation.py` l.91-99) | yes (gradient-checkpointed) |
| Architect UNet, batch 2, regular/unguided path | l.667-671 | no (`no_grad`) |
| Architect VAE decode fp32 512² | l.684-686 (outer `checkpoint`) | yes |
| N × sprinter call: `encode_prompt` (2 text encoders), 2 × (UNet + ControlNet), latent out | `generation.py` l.264-272 | yes — wrapped in `checkpoint` l.281-283 |
| N × sprinter VAE decode **fp32** 512² | l.273-275 | yes (inside same checkpoint) |
| N × CLIP ViT-L/14 fp32 on 224² | l.278-279 | yes |
| MMD (vectorised) + `autograd.grad` | l.298-305 | — |
| `visualize_step`: 5 architect VAE decodes + **5 extra sprinter calls** (PIL, VAE decode each) + 2 full-weight dtype casts of sprinter VAE + matplotlib savefig | `run_mlgd_f.py` l.765 → `visualization.py` l.70-107 | no |
| every `eval_interval` (~3 steps by default): `evaluate_distribution_mmd`: 1 VAE decode + 10 sprinter calls + CLIP, plus **CLIP model moved GPU→CPU** | l.736-742 → `metrics.py` l.201-227 | no |
| `gc.collect(); torch.cuda.empty_cache()` | l.780 (and `generation.py` l.287) | — |

Because of the outer per-variation `checkpoint` (non-reentrant) **plus** `enable_gradient_checkpointing()` on sprinter UNet/ControlNet (`models.py` l.143-145), each sprinter UNet/ControlNet block is computed **3×** per variation (forward-no-save, recompute in backward, then block-level recompute inside that), and sprinter VAE-decode / CLIP / text encoders **2×**. So in UNet-forward units per step with N=6: architect 2 (grad, ×~2 for ckpt) + 2 (regular) ; sprinter 6×2×3 = 36 grad-path UNet+CN passes + 10 (visualize) + ~7 (eval amortised) ≈ **~50 UNet(+CN) passes/step**, of which ~36 are for guidance and ~17 are visualisation/eval overhead.

The CSV `notebooks/results/vjp_results_lightning.csv` is a different experiment (ε_s / ε_g Jacobian-gap via random VJPs, SDXL-Base vs Lightning; columns `eps_s, eps_g, vjp_00..09`) — **not** a memory measurement, don't cite it for memory. `measure_dps_step_memory.ipynb` is a per-stage peak-memory harness (`report('Stage 1 — architect UNet forward')`, `Stage 3 — VAE decode [END OF FIXED GRAPH]`, `Stage 4 — after run_dps_step_clip`, decomposes "fixed graph" vs "variation graph (N=100, K=2)" and projects K=30) but **the saved notebook has no executed outputs** — no numbers to cite; it can be re-run to validate the patches below.

---

### 1. Frozen parameters / grad mode

**Findings**
- `models.py` l.138-153 `setup_gradient_checkpointing` freezes `architect.unet, architect.vae, sprinter.unet, sprinter.controlnet, sprinter.vae`; `clip_utils.py` l.23-24 freezes CLIP. Called at `run_mlgd_f.py` l.545 before the loop. Good.
- **Not frozen**: `sprinter.text_encoder`, `sprinter.text_encoder_2` (and `architect.text_encoder*`, unused in loop). This matters because `models.py` l.122-127 deliberately **strips the `@torch.no_grad()` decorator** from `StableDiffusionXLControlNetPipeline.__call__` (`original_call.__wrapped__`), so the sprinter's `encode_prompt` (CLIP-L + OpenCLIP-bigG, ~0.8B params) runs with grad-enabled params **inside every variation call** and is recomputed in backward (2× per variation). Activations are not retained (outer non-reentrant checkpoint discards them), but the forward FLOPs are wasted and the graph is built.
- Gradient is taken with `torch.autograd.grad(loss_scaled, latents_step, retain_graph=False, create_graph=False)` (`generation.py` l.303-305) — correct; no `.backward()`, no `.grad` accumulation on parameters.
- No `torch.inference_mode()` anywhere; `no_grad` is used for all non-guided work (l.547, 572, 602, 667, 674, 751, 772; `metrics.py` l.201/207/225; `visualization.py` l.70). `clip_model.eval()` is set (`clip_utils.py` l.22); UNets/VAE are left in whatever mode `from_pretrained` gives (eval) — fine.

**Patches**
- (EXACT) `models.py` l.146-152: add `sprinter.text_encoder, sprinter.text_encoder_2, architect.text_encoder, architect.text_encoder_2` to the freeze list. Combined with caching prompt embeds (§3) this removes 2 text-encoder forwards × 2 (recompute) × N per step.
- (EXACT) Wrap `predict_noise_cfg` regular path, `visualize_step`, `evaluate_distribution_mmd` in `torch.inference_mode()` instead of `no_grad` (small allocator/version-counter savings). Test: bitwise-identical outputs.

### 2. Score-network evaluations per step / CFG

- `predict_noise_cfg` (`generation.py` l.91-99) **always** does `torch.cat([latents_in]*2)` and returns `np_u + gs*(np_t-np_u)`. With the default `--guidance_scale 0.0` (l.79: "0.0 = unconditional") the conditional half is multiplied by 0 — pure waste, and it sits **inside the differentiated graph** (backward traverses both halves; activations for batch 2 are recomputed in the architect's checkpointed blocks). Same doubling for the regular path (l.667-671) and baseline (l.603). `architect.encode_prompt(..., do_classifier_free_guidance=True)` at l.553-556 and `added_cond_kwargs` with `.repeat(2,1)` (l.565-568) are built for CFG unconditionally.
- Sprinter: `guidance_scale=0.0` everywhere (`generation.py` l.268, `metrics.py` l.215, `visualization.py` l.101, `run_mlgd_f.py` l.621 uses `args.guidance_scale`=0) → diffusers sets `do_classifier_free_guidance=False` (scale ≤ 1), so sprinter runs batch 1 — no waste there.
- Count per step (N=6, defaults): architect UNet fwd: 1 (batch 2, grad) + 1 (batch 2, no-grad). Sprinter UNet+CN: 12 in graph (×3 due to nested checkpoint = 36 compute-equivalents), + 10 (visualize) + 20/eval_interval (eval).

**Patches**
- (EXACT when gs==0 or gs==1) `generation.py` l.84-99: `if gs == 0: run unet on latents_in only with encoder_states[:1] / added_cond sliced to first row, return np_u`; analogously build single-row `added_cond_kwargs`/`cfg_encoder_states` in `run_mlgd_f.py` l.562-569 when `args.guidance_scale == 0`. Halves architect UNet cost (forward + checkpoint recompute + backward) — ~2× on the architect share. Test: `torch.allclose(noise_pred_new, noise_pred_old)` (identical up to fp16 batch-kernel nondeterminism; with `gs=0` the formula is exactly `np_u`).
- (EXACT) Fuse guided + regular forwards into one batch-2 call (`torch.cat([latents_step, latents_step_regular])`) — the regular half then sits in the graph but its grad path is dead; saves one kernel launch sequence, not FLOPs. Lower priority than the gs==0 fix.

### 3. Uncached work

| Item | Location | Status |
|---|---|---|
| Architect prompt embeds / `add_time_ids` / `added_cond_kwargs` | `run_mlgd_f.py` l.547-569 | cached once — OK |
| **Sprinter prompt embeds** | `generation.py` l.264-272 passes `prompt=[variation_prompt]*bs` every call; `visualization.py` l.98; `metrics.py` l.211 | **re-encoded by both SDXL text encoders on every call** (N + 5 per step + 10 per eval) |
| Target CLIP embeddings | computed once l.299-306 / l.393-400; passed detached | OK |
| ControlNet cond image | tensor `pixel_x0_norm` passed directly; diffusers `prepare_image` casts fp32→fp16 per call | OK, differentiable; one cast per variation (cheap) |
| VAE encode/decode | architect VAE decode of `pred_x0` done **3×** per step: l.684 (grad), `visualization.py` l.83-86 (`latent_to_pil(pred_x0)`) **and** l.89-92 (`architect.vae.decode(pred_x0_dev)`) — the last two are duplicates of each other and of the first | waste: 2 extra fp32 512² VAE decodes/step |
| Sprinter VAE dtype | `sprinter.vae.to(float16)` / `.to(float32)` toggled twice per `visualize_step` (l.95/107), per `evaluate_distribution_mmd` (`metrics.py` l.205/220), baseline l.615/627 | **full 84M-param cast ×2 per step** (~335 MB copy each way) |
| CLIP preprocessing | `clip_utils.py` l.44-55: tensor `F.interpolate` + normalise on GPU | OK (differentiable); `mean/std` tensors re-created per call (trivial) — hoist to module constants |
| Scheduler tensors | `compute_pred_x0_direct` l.126: `scheduler.alphas_cumprod[t.long().cpu()].to(device)` — CPU-resident table, `.cpu()` on a CUDA `t` = **host sync + H2D per call** (2 calls/step + `scheduler.step` internals) | minor; `run_mlgd_f.py` l.579 already has `alphas_cumprod.to(device)` but doesn't reuse it |
| CLIP model | `clip_model.to(device)` at `generation.py` l.255 every step; `evaluate_distribution_mmd` ends with `clip_model.to("cpu")` (`metrics.py` l.227) → **full ~1.7 GB fp32 CLIPModel (incl. unused text tower) shuttled GPU↔CPU every eval interval** | waste |
| Sprinter per-call pipeline overhead | `check_inputs`, `scheduler.set_timesteps(2)`, `_get_add_time_ids`, `prepare_image` per call | small but N+5+10 times/step |

**Patches**
- (EXACT) Pre-encode once under `no_grad`: `pe, _, pooled, _ = sprinter.encode_prompt(variation_prompt, device, 1, False)`; pass `prompt_embeds=pe, pooled_prompt_embeds=pooled` (and eval prompt likewise) to every `sprinter(...)` call in `generation.py` l.264, `visualization.py` l.97, `metrics.py` l.210, `run_mlgd_f.py` l.617. Saves 2 text-encoder forwards (×2 recompute in graph) per call. Test: `torch.equal(prompt_embeds_cached, prompt_embeds_recomputed)` (deterministic), final images identical under same seed.
- (EXACT) Pass `pixel_x0_norm.detach()` into `visualize_step` and reuse for both `img_x0_dps` and `px_norm`; remove decodes at `visualization.py` l.83-92. Saves 2 VAE decodes/step.
- (EXACT) Keep `sprinter.vae` permanently fp32 *or* keep a second fp16 copy on GPU for the no-grad decodes (84M params, ~170 MB fp16); drop all `.to(dtype=...)` toggles (`visualization.py` l.95/107, `metrics.py` l.205/220, `run_mlgd_f.py` l.615/627, `generation.py` l.56-80). Test: none needed for the "keep fp32" variant (numerics identical to current grad path; no-grad PIL outputs change slightly vs the fp16 decode — only visualisation).
- (EXACT) Keep CLIP resident on GPU: delete `clip_model.to("cpu")` in `metrics.py` l.227 and `run_mlgd_f.py` l.305/399/636 (it's already reloaded to GPU at l.255 each step anyway). Optionally load only `CLIPVisionModelWithProjection` (drops ~0.4 GB text tower).
- (EXACT) `compute_pred_x0_direct`: accept a device-resident `alphas_cumprod` (from l.579) and index with `t` directly (no `.cpu()`); precompute `sqrt(alpha)`, `sqrt(1-alpha)` per timestep once.

### 4. dtype / autocast

- UNets, ControlNet loaded fp16 (`models.py` l.86-104); both VAEs forced fp32 (`models.py` l.117-118, `run_mlgd_f.py` l.544); CLIP loaded **fp32** (no `torch_dtype` in `clip_utils.py` l.21).
- The known CLIP-autocast bug is handled at `generation.py` l.277-279: `with torch.amp.autocast("cuda", enabled=False): encode_images_clip(var_pixels.float(), ...)`. Note that in `run_dps_step_clip` **no autocast is ever enabled** (the old `run_dps_step` used `torch.cuda.amp.autocast()` at l.188; the CLIP variant doesn't), so the guard is currently a no-op — harmless, keep as defence.
- Per-step dtype traffic: `latents` kept fp16 (l.582); `pred_x0` fp16 → `.to(vae.dtype)` fp32 (l.682); `pixel_x0_norm` fp32 → diffusers casts to fp16 for ControlNet; sprinter latents fp16 → `.float()` → fp32 VAE decode (l.274) → `.float()` CLIP. Casts are cheap; the cost is that **two fp32 SDXL VAE decoders at 512² are in the differentiated path per variation** (the single largest activation tensor set; decoder fp32 at 512² is O(GB) when materialised during recompute).

**Patches**
- (CHANGES NUMERICS) Use `madebyollin/sdxl-vae-fp16-fix` in fp16 for `sprinter.vae` (and optionally architect VAE) and drop the fp32 forcing (`models.py` l.117-118, `run_mlgd_f.py` l.544): ~2× less VAE memory/time; gradients differ slightly. Test: compare per-step `grad` cosine similarity and `‖grad‖` vs fp32 run for same seed (expect cos > 0.99), and final MMD within run-to-run noise.
- (CHANGES NUMERICS) Run CLIP vision tower in bf16 under autocast (remove the `enabled=False` guard; the original bug was fp16 overflow/mismatch, bf16 avoids it). ~2× CLIP; test as above.
- (EXACT) Keep fp32 but hoist the CLIP `mean/std` to buffers (`clip_utils.py` l.49-54).

### 5. Graph retention / checkpointing structure

- `retain_graph=False, create_graph=False` — good (`generation.py` l.304).
- N variations are **sequential with `variation_batch_size=1`** (`run_mlgd_f.py` l.697; loop `generation.py` l.258-284). Each variation is wrapped in `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` so its activations are **not** held simultaneously; memory ≈ fixed graph (architect UNet ckpt + fp32 VAE decode) + max over one variation's recompute. So memory is *not* N×sprinter activations — the price is compute: **3× UNet/CN** (outer ckpt × inner `enable_gradient_checkpointing`), **2× sprinter-VAE + CLIP + text encoders** per variation. Backward traverses: CLIP ← VAE dec ← (UNet+CN)×2 steps ← ControlNet cond image ← architect VAE dec ← architect UNet, for all N.
- Stored non-detached tensors: none problematic — `vl_clip_flat` is `.detach().cpu().numpy()` (l.307); `sd` dict is built under `no_grad` with `.detach().cpu()` (l.751-763); `step_vis_data` holds only small CPU tensors. `noise_pred`/`latents_step` survive to `denoise_step` (l.769) then are `del`'d (l.777-782). OK.
- `preserve_rng_state=True` (default) in the checkpoints correctly replays the sprinter's `randn` initial latents during recompute (the sprinter draws from global RNG, `generator=None`).

**Patches**
- (EXACT up to nondeterminism) Remove the nested checkpointing: either (a) drop `sprinter.unet/controlnet.enable_gradient_checkpointing()` (`models.py` l.144-145) and keep the outer per-variation checkpoint → UNet 3×→2×, memory for one variation's full fp16 UNet activations at batch 1 (a few GB — should fit on L40S 48GB, check on L4 22GB with the memory notebook); or (b) keep inner and shrink the outer checkpoint to wrap only `vae.decode + CLIP` so UNet sees exactly 2× and the fp32 VAE/CLIP activations are still not kept across variations. Test: grad `allclose` to current (rtol ~1e-3 fp16), same seed.
- (EXACT) Batch variations (§6) keeps the single-checkpoint-per-batch structure so peak ≈ one batched variation's recompute.

### 6. Python loops that should be batched

- Variations: `for start_idx in range(0, num_variations, 1)` (`generation.py` l.258) with `variation_batch_size=1` hard-coded at `run_mlgd_f.py` l.697. Batch all N (or N/2) in one sprinter call: `ctrl_batch = pixel_x0_norm.repeat(N,1,1,1)`, one UNet+CN pass per sprinter step at batch N, one fp32 VAE decode at batch N, one CLIP call at batch N. Expect 1.5-3× on sprinter throughput (CLIP at 224² batch 1 is badly under-utilised; SDXL UNet at 64² latent batch 1 is ~50-70% utilised on L4/L40S). Memory: activations scale with N inside the recompute — with inner UNet checkpointing only block inputs scale; the fp32 VAE decode at batch N is the constraint (consider `vae.enable_slicing()` which decodes per-sample but keeps one graph, or fp16 VAE per §4).
  - EXACTNESS: batching changes the RNG stream (one `randn([N,...])` vs N `randn([1,...])` draws give different noise). To make it exact pass `generator=[torch.Generator(device).manual_seed(s_k) for k in range(N)]` in both old and new code — diffusers draws per-generator per sample, yielding identical per-variation noise. Test: per-variation CLIP embeddings `allclose` and `grad` `allclose` between batched and sequential with per-sample generators.
- `visualize_step` l.96-106 runs 5 sprinter calls of batch 1 with a list comprehension → one call with `num_images_per_prompt=5` or batched prompt list (EXACT with per-sample generators, otherwise only RNG differs).
- `evaluate_distribution_mmd` batches by 2 (`metrics.py` l.208) → batch 10.
- CLIP encode is already batched per call; MMD is fully vectorised (`metrics.py` l.46-77), no per-pair loop. `K_yy` (targets×targets, no grad) is recomputed every step — negligible (100×100), could be cached since `y` and bandwidth... bandwidth is median-heuristic on x (l.52-64) so `K_yy` changes with bandwidth; leave.
- CPU-side: `TF.to_tensor(img)` on PIL lists (`metrics.py` l.222, `run_mlgd_f.py` l.629) — eval only; use `output_type="pt"` from the sprinter to skip PIL round trip (EXACT up to uint8 quantisation — current path quantises to 8-bit PNG before CLIP, so changing it *does* change eval numerics slightly; keep if you want eval parity with offline analysis).

### 7. Checkpointing / VJP opportunities

- Already checkpointed: architect UNet (block-level), sprinter UNet+CN (block-level), architect VAE decode (one block, l.684), whole per-variation sprinter+VAE+CLIP (l.281). Over-checkpointed rather than under (see §5).
- Scalar loss ⇒ `autograd.grad` is already a single VJP; there is no way to avoid materialising activations other than recompute (checkpointing) or reduced precision. Unexploited levers:
  - (CHANGES NUMERICS) Truncated backprop through the sprinter: detach the output of sprinter step 1 of 2 (or use the last-step-only Jacobian) — halves sprinter backward + recompute; a method change, must be validated on ΔMMD.
  - (CHANGES NUMERICS) Single-step sprinter for the guidance path (`num_inference_steps=1`) while keeping 2 for eval.
  - (EXACT) Sprinter VAE decode does not need the fp32 *encoder*; nothing to gain there. `vae.enable_slicing()` is exact and bounds VAE activation peak when batching.
- Measurement: `notebooks/measure_dps_step_memory.ipynb` has the right stage breakdown (baseline / after architect UNet / after pred_x0 / after VAE decode = "fixed graph" / after `run_dps_step_clip`) and a `NUM_VARIATIONS=100, TURBO_STEPS=2` config, but no saved outputs — re-run it before/after patches to put numbers in the paper. `vjp_results_lightning.csv` is unrelated (Jacobian-gap experiment).

### 8. Other hot-loop waste

1. **`visualize_step` every step** (`run_mlgd_f.py` l.765): 5 fp32 VAE decodes, 5 sprinter generations (each with text encoding + VAE decode to PIL), 2 full-VAE dtype casts, PCA transform, matplotlib figure → for N=6 this is ≈ 45% of all sprinter forwards in the loop. Patch: gate on `i % eval_interval == 0` or `--vis_every`, reuse `pixel_x0_norm`, batch the 5 gens. Savings: ~25-35% wall-clock at N=6. EXACT for the guidance path (no effect on gradients).
2. **`gc.collect(); torch.cuda.empty_cache()` every step** (l.780) and `empty_cache()` inside `run_dps_step_clip` (`generation.py` l.287): empty_cache forces cudaFree/cudaMalloc churn (and the allocator re-growth next step); gc.collect is a ~10-50 ms Python pause. Remove; rely on `del`. EXACT.
3. **Host syncs before backward**: print at `generation.py` l.289-296 does `.isnan().sum().item()`, `.min().item()`, `.max().item()` (3 syncs) before `autograd.grad`; l.708-710 and l.719-725 `.item()`s after (fine). Move the debug print behind a flag. Minor.
4. **CLIP GPU↔CPU shuttling** (§3) and **sprinter VAE dtype casts** (§3) — ~0.5-2 GB of PCIe/device copies per step.
5. `compute_pred_x0_direct` `.cpu()` sync ×2/step; `scheduler.step` DDIM also indexes CPU `alphas_cumprod` with a CUDA timestep (sync). Minor.
6. `wandb.log(..., commit=False)` every step (l.749) and never `commit=True` until the final `wandb.log` at l.832 — all per-step scalars collapse into one wandb step (logging bug, not perf; use `step=i` or `commit=True`).
7. Image saving: `visualize_step` savefig per step (disk I/O, ~1-2 s of matplotlib per step); per-step wandb image logging is **not** done in the loop (good). `step_vis_data` accumulates small CPU tensors only.
8. Pipelines are constructed once; no model device moves except CLIP. The `StableDiffusionXLControlNetPipeline.__call__` monkey-patch (`models.py` l.122-127) is process-global — fine.
9. Full `CLIPModel` loaded (text tower unused in loop) in fp32 — ~0.4 GB of VRAM wasted; load `CLIPVisionModelWithProjection` (EXACT).

### Priority list (estimated wall-clock at defaults N=6, 15 steps, L40S)

| # | Patch | Exactness | Est. saving |
|---|---|---|---|
| 1 | Gate/trim `visualize_step` (reuse `pixel_x0_norm`, batch 5 gens, vis every k steps) | EXACT | 25-35% |
| 2 | Un-nest checkpointing (UNet 3×→2×) | EXACT (nondet.) | 15-25% of guidance compute |
| 3 | Batch variations (`variation_batch_size=N`, per-sample generators) + batch CLIP | EXACT w/ generators | 1.5-2× on sprinter+CLIP path |
| 4 | `gs==0` → single-batch architect UNet | EXACT | ~2× architect share (small absolute) |
| 5 | Cache sprinter prompt embeds + freeze text encoders | EXACT | 2 TE fwd ×2 × (N+15)/step |
| 6 | Keep CLIP on GPU; stop VAE dtype toggling; drop `empty_cache/gc` | EXACT | ~0.5-1 s/step |
| 7 | fp16-fix VAE / bf16 CLIP | CHANGES NUMERICS | ~2× on VAE/CLIP, large memory drop |

Equivalence harness: fix `torch.manual_seed`, use explicit per-variation generators in both branches, and assert per-step `torch.allclose(grad_new, grad_old, rtol=1e-3, atol=1e-5)` (fp16 path) plus identical `mmd_loss`; for numerics-changing patches report grad cosine similarity per step and final ΔMMD across ≥3 seeds.

---

## 4. Cross-path summary (what to do first)

| path | finding | exactness | est. saving | test |
|---|---|---|---|---|
| synthetic | batch restarts of a cell through denoiser+CM+MMD (`runners.run_batched_restarts`) | REORDER | 10-25x CPU throughput (BENCH.md) | teacher-forced per-step grad <= 1e-4; L2 distribution over restarts |
| synthetic | MMD: XX/XY blocks + cached YY instead of the (n+m)^2 stacked kernel (`LossFunctions.py:38-44`) | REORDER (entries bit-identical) | 1.5-4.5x per restart | per-step grad <= 1.3e-6 (measured) |
| synthetic | batch the M=3 LGD perturbations, pre-draw noise with `torch.Generator` (`_guided.py:100-107`) | EXACT (0.0) | 1.4x on LGD cells | `torch.equal` on x_final (passes) |
| synthetic | hoist `self.to(device)` out of `sample_ddim_step` (`Diffusion.py:335`), cache time-embedding table, `torch.full` for t | EXACT | 2-5% | `torch.equal` |
| MNIST | see §2: `.item()`/host syncs and per-perturbation work in `optimize_LGD` (`run_mlgdf.py:304-393`), sequential perturbations and seeds, `retain_graph=True` | EXACT / REORDER | launch-bound workload: batching perturbations x seeds is the lever | per-step grad allclose(rtol 1e-5) same seed |
| SD | `visualize_step` every step (5 VAE decodes + 5 sprinter gens + 2 full-VAE dtype casts), `gs==0` still runs the CFG double batch inside the graph, nested gradient checkpointing (UNet 3x), sequential variations, uncached sprinter prompt embeds, CLIP shuttled GPU<->CPU | mostly EXACT | 25-35% (vis) + 15-25% (ckpt) + 1.5-2x sprinter path (batching) | per-step grad allclose(rtol 1e-3, fp16) with per-sample generators; delta-MMD over >= 3 seeds for NUMERICS changes |

Numerics-changing proposals (fp16-fix VAE, bf16 CLIP, truncated sprinter backprop, 1-step
sprinter) are listed in §3.4/§3.7 and flagged as such; none of the EXACT items above
change what is computed, only how often / in what order.
