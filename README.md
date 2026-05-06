# Inverse Design for Conditional Distribution Matching

Official code for the paper **"Inverse Design for Conditional Distribution Matching"** (NeurIPS 2026 submission).

> **Anonymous submission.** Author information has been omitted for double-blind review.
> Anonymous repository mirror: [anonymous.4open.science/r/conditional-matching-paper-AE20](https://anonymous.4open.science/r/conditional-matching-paper-AE20)

---

## Overview

We introduce **Conditional Distribution Matching (CDM)**, a new inverse-design problem class: given a frozen generative pipeline P(Y | X) and a target distribution G(Y), find an input x\* whose induced conditional P(Y | X = x\*) matches G. This goes beyond standard inverse design, which targets a single point output y\*.

To solve CDM, we propose **MLGD-F** (*Matching-Loss Guided Diffusion with a Fast inner sampler*) — a plug-and-play, inference-time algorithm that combines:

- A **pretrained score-based diffusion model** as a prior over X
- A **pretrained fast conditional sampler** (consistency model / distilled diffusion) for P(Y | X)

No additional training or fine-tuning is required. Single-step conditional sampling keeps gradient computation memory-efficient and tractable even at the scale of Stable Diffusion.

---

## Repository Structure

```
conditional-matching-paper/
├── simulations/                # Synthetic MoG experiments (Section 4.1)
│   ├── src/                    # Diffusion, consistency model, loss, utils
│   ├── notebooks/              # Experiment notebooks (2D, 5D, 10D, β-sweep)
│   ├── params/                 # Pretrained GMM parameters (.pt)
│   ├── results/                # Output files
│   └── requirements.txt
│
├── MNIST/                      # MNIST rotation task (Section 4.2)
│   ├── src/                    # Conditional iCT model, DDPM, classifier, dataset
│   ├── train/                  # Training scripts (unconditional + conditional)
│   ├── run_mlgdf.py            # Main MLGD-F inference script
│   ├── run_mlgdf.sh            # SLURM job script
│   ├── visualize.py            # Evaluation & polar-histogram visualization
│   └── requirements.txt
│
├── SD_cond_SD_controlnet/      # Stable Diffusion image editing (Section 4.3)
│   ├── src/                    # Models, generation, metrics, CLIP utils, viz
│   ├── scripts/                # run_mlgd_f.py, eval_baselines.py
│   ├── notebooks/              # Evaluation notebooks & ablation
│   ├── experiments/            # Per-scenario scribbles and result JSONs
│   └── requirements.txt
│
└── requirements.txt            # Top-level consolidated dependencies
```

---

## Method

MLGD-F has two components:

1. **Outer loop** — standard LGD-style reverse diffusion guided by a distributional loss L(x).
2. **Inner estimator** — draws n_cond samples from the fast conditional sampler f_φ(x, ·), computes a distributional distance (MMD or SWD) against a fixed set of target samples from G, and backpropagates through f_φ.

Because f_φ is a **single-step** sampler, the gradient computation is shallow (no K-step unrolled chain), keeping peak VRAM feasible (43 GB vs. ~375 GB projected for a 30-step SDXL-Base inner sampler on an A100 80 GB).

---

## Experiments

### 1 · Synthetic Simulations (MoG)

Mixture-of-Gaussians experiments in 2D, 5D, and 10D input space with 1D output. Compares MLGD-F against a slow (multi-step) inner sampler across 25 optimization runs.

**Models used:**

| Role | Model |
|---|---|
| Prior over X (Architect) | Pretrained DDPM over Gaussian mixtures (provided in `params/`) |
| Fast conditional sampler f_φ (Sprinter) | Consistency model trained on the same MoG (provided in `params/`) |

Pretrained checkpoints for 2D, 5D, and 10D are hosted on HuggingFace:
> [huggingface.co/anon-submission-cdm/cdm-inverse-design](https://huggingface.co/anon-submission-cdm/cdm-inverse-design)

```bash
cd simulations
pip install -r requirements.txt
# Open notebooks/Exp_2D_cond_1D.ipynb (or 5D / 10D variants)
# β-sweep demo: notebooks/toy_example_with_beta_sweep.ipynb
```

See [`simulations/README.md`](simulations/README.md) for full details.

---

### 2 · MNIST Rotation Task

Find a digit image x\* ∈ R^784 such that P(rotation angle | X = x\*) matches a user-specified target G (unimodal, bimodal, or uniform over rotation angles).

**Models used:**

| Role | Model |
|---|---|
| Prior over X (Architect) | Unconditional DDPM over MNIST images |
| Fast conditional sampler f_φ (Sprinter) | Conditional improved Consistency Training (iCT) model for P(angle \| digit image) |

Pretrained checkpoints are hosted on HuggingFace:
> [huggingface.co/anon-submission-cdm/cdm-inverse-design](https://huggingface.co/anon-submission-cdm/cdm-inverse-design)

#### Quick start (pretrained checkpoints)

```bash
cd MNIST
pip install -r requirements.txt

# Run MLGD-F (bimodal target, 15 seeds)
python run_mlgdf.py --target bimodal --num_runs 15

# Visualize results
python visualize.py --results_dir results/
```

#### Train from scratch

```bash
# Unconditional DDPM
python train/train_uncond.py

# Conditional iCT
python train/train_conditional.py
```

See [`MNIST/README.md`](MNIST/README.md) for hyperparameter details and SLURM scripts.

---

### 3 · Stable Diffusion Image Editing

Optimize a portrait scribble x\* so that images generated from it via ControlNet-Scribble satisfy a distributional target G in CLIP embedding space. Targets include discrete gender mixtures and continuous age/gender interpolations.

**Models used (all auto-downloaded from HuggingFace on first run):**

| Role | Model |
|---|---|
| Prior over X (Architect) | `stabilityai/stable-diffusion-xl-base-1.0` |
| Fast conditional sampler f_φ (Sprinter) | `stabilityai/sdxl-turbo` |
| Conditioning | `xinsir/controlnet-scribble-sdxl-1.0` |
| Embedding | `openai/clip-vit-large-patch14` |

#### Requirements

```bash
cd SD_cond_SD_controlnet
pip install -r requirements.txt
```

**Compute:** MLGD-F optimization runs on a single NVIDIA L40S 48 GB GPU (salmon partition). The full N=2000 baseline evaluation (notebooks) runs on the same hardware and takes several hours per experiment due to image generation at scale.

#### Run MLGD-F

```bash
export ENV_PATH=/path/to/your/env
sbatch SD_cond_SD_controlnet/gender_submit_mlgd_f.sh   # gender / interpolation target
sbatch SD_cond_SD_controlnet/age_submit_mlgd_f.sh      # age sweep target
```

#### Run baselines

```bash
sbatch SD_cond_SD_controlnet/slurm/run_eval_baselines.sh SkewedTarget 241
sbatch SD_cond_SD_controlnet/slurm/run_eval_baselines.sh BalancedTarget 241
sbatch SD_cond_SD_controlnet/slurm/run_eval_baselines.sh GenderInterpolation 241
sbatch SD_cond_SD_controlnet/slurm/run_eval_baselines.sh AgeInterpolation 177
```

The time argument (e.g. `241`) is the MLGD-F runtime in minutes, used to give SDEdit the same wall-clock budget.

#### Evaluate results (N=2000)

```bash
# Full evaluation across all experiments and methods — opens as Jupyter notebook
# SD_cond_SD_controlnet/notebooks/eval_all_experiments.ipynb

# Cached results already available at:
# SD_cond_SD_controlnet/experiments/eval_all_results.json
```

#### Notebooks

| Notebook | Paper section                                                                                                                                                                                                                                    |
|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `eval_all_experiments.ipynb` | N=2000 MMD + gender classification across all experiments                                                                                                                                                                                        |
| `eval_scribbl_interpolation.ipynb` | Figures — gender interpolation evaluation + PCA                                                                                                                                                                                                    |
| `eval_scribbl_interpolation_age.ipynb` | Figure — age interpolation evaluation                                                                                                                                                                                                            |
| `gender_saliency_eval.ipynb` | Section: *What Does MLGD-F Actually Change in the Scribble?* — pixel diff heatmaps, CLIP gender saliency maps, Spearman correlation between saliency and edit magnitude                                                                          |
| `measure_dps_step_memory.ipynb` | Ablation (Section: *Necessity of Distilled Models*) — peak VRAM measurement for SDXL-Turbo vs. projected SDXL-Base cost                                                                                                                          |
| `eps_g_experiment.ipynb` | Theory (Section: *Theoretical Analysis*) — empirical diagnostic of ε_g, the Jacobian fidelity gap between the distilled sampler f_φ and the teacher, measured as relative VJP-norm discrepancy between SDXL-Lightning and SDXL-Base across 100 prompts |

See [`SD_cond_SD_controlnet/README.md`](SD_cond_SD_controlnet/README.md) for full argument reference.

---

## Installation

Each sub-module has its own `requirements.txt`. A consolidated top-level file covers all three:

```bash
pip install -r requirements.txt
```

GPU requirements vary by experiment:
- **Simulations (2D):** any modern GPU
- **Simulations (5D, 10D) and MNIST:** NVIDIA L40S 48 GB
- **Stable Diffusion optimization:** NVIDIA L40S 48 GB (salmon partition)
- **Stable Diffusion evaluation (N=2000, `eval_all_experiments`):** NVIDIA A100 80 GB or equivalent
- **`eval_scribbl_interpolation` / `eval_scribbl_interpolation_age`:** NVIDIA L40S 48 GB
- **`gender_saliency_eval`:** NVIDIA A100 80 GB (backpropagation through full pipeline)
- **`measure_dps_step_memory`:** NVIDIA A100 80 GB (memory profiling requires full VRAM headroom)
- **`eps_g_experiment`:** NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB)

---

## Pretrained Checkpoints

MNIST checkpoints and synthetic simulation checkpoints (2D, 5D, 10D GMM) are hosted on HuggingFace:

> [huggingface.co/anon-submission-cdm/cdm-inverse-design](https://huggingface.co/anon-submission-cdm/cdm-inverse-design)

Stable Diffusion models are downloaded automatically from the HuggingFace Hub on first run. No SD checkpoints are hosted separately.

---

## Citation

Anonymous submission — citation will be added after the review period.
