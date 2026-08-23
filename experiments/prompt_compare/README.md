# Prompt Comparison Experiment

## Goal
Test whether prompt wording affects the gender distribution when generating portraits from the same DPS-guided scribble.

## Setup
- **Model**: SDXL Turbo + ControlNet-Scribble (`xinsir/controlnet-scribble-sdxl-1.0`)
- **Scribble**: Final guided scribble from best zeta=5 DPS run (`scripts/assets/zeta5_final_guided.png`)
- **Seed**: 1
- **ControlNet scale**: 0.5
- **Inference steps**: 2 (SDXL Turbo default)
- **N images per prompt**: 100

## Prompts
- **Prompt A** (original): `"a superrealistic professional photograph of"`
- **Prompt B** (gender-neutral): `"in a world without gender norms, a superrealistic professional photograph of"`

## Results

| Prompt | Male | Female |
|--------|------|--------|
| A — original | 60 | 40 |
| B — gender-neutral | 1 | 99 |

The "gender-neutral" prompt dramatically shifts the output to almost entirely female. The scribble is identical — only the text conditioning changes.

## Additional: Extreme Prompts (Single Image)
Generated one image each with strongly gendered prompts from the same scribble:
- `"a superrealistic photo of a very manly man"` → clearly male portrait
- `"a superrealistic photo of a very feminine woman"` → clearly female portrait

These two images (`manly_man.png`, `feminine_woman.png`) serve as anchor points for CLIP embeddings in the bimodal target experiment.

## Files
- `SD_cond_SD_controlnet/run_prompt_compare.py` — generation script (cluster)
- `SD_cond_SD_controlnet/submit_prompt_compare.sh` — SLURM submit script
- `scripts/build_prompt_compare_pdf.py` — local PDF builder (gender eval + PCA)
- `SD_cond_SD_controlnet/output/prompt_compare_44369230/` — results (200 images + PDF)
- `manly_man.png`, `feminine_woman.png` — extreme prompt single images

## Key Takeaway
The text prompt has a massive effect on the gender distribution even when the structural conditioning (scribble) is held constant. This motivates using distribution-level guidance (DPS with MMD) rather than relying on prompt engineering alone, since prompt effects are unpredictable and can overshoot.
