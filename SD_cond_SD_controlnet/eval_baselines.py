"""
eval_baselines.py — Build candidate scribbles for all methods, save to disk.
Runs on SLURM cluster (no N=2000 eval — that happens in Colab).

Usage:
    python eval_baselines.py \
        --experiment SkewedTarget \
        --lgd_cm_minutes 241
"""

import os, sys, copy, argparse, json, time
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# ── EXPERIMENT CONFIGS ────────────────────────────────────────────────────────

EXPERIMENT_CONFIGS = {

    'SkewedTarget': dict(
        best_jid         = 44430631,
        runs_root        = '/sci/labs/orzuk/ori_m/gdrive/conditional-matching/runs',
        seed             = 5,
        n_eval_search    = 100,
        sdedit_start     = 125,
        n_steps          = 250,
        sdedit_cfg       = 7.5,
        sdedit_prompt    = 'a face scribble sketch, mostly feminine features sometimes with masculine features',
        neutral_prompt   = 'a superrealistic professional photograph of',
        controlnet_scale = 0.5,
        mode             = 'binary',
        groups = [
            dict(label='Man',   prompt='a superrealistic portrait photograph of a man, studio lighting',   frac=0.25),
            dict(label='Woman', prompt='a superrealistic portrait photograph of a woman, studio lighting', frac=0.75),
        ],
    ),

    'BalancedTarget': dict(
        best_jid         = 44424933,
        runs_root        = '/sci/labs/orzuk/ori_m/gdrive/conditional-matching/runs',
        seed             = 5,
        n_eval_search    = 100,
        sdedit_start     = 125,
        n_steps          = 250,
        sdedit_cfg       = 7.5,
        sdedit_prompt    = 'a face scribble sketch of man or woman',
        neutral_prompt   = 'a superrealistic professional photograph of',
        controlnet_scale = 0.5,
        mode             = 'binary',
        groups = [
            dict(label='Man',   prompt='a superrealistic portrait photograph of a man, studio lighting',   frac=0.5),
            dict(label='Woman', prompt='a superrealistic portrait photograph of a woman, studio lighting', frac=0.5),
        ],
    ),

    'GenderInterpolation': dict(
        best_jid         = 44432053,
        runs_root        = '/sci/labs/orzuk/ori_m/gdrive/conditional-matching/runs',
        seed             = 5,
        n_eval_search    = 100,
        sdedit_start     = 125,
        n_steps          = 250,
        sdedit_cfg       = 7.5,
        sdedit_prompt    = 'a face scribble sketch, with a range of feminine to masculine features',
        neutral_prompt   = 'a superrealistic professional photograph of',
        controlnet_scale = 0.5,
        mode             = 'multiclass',
        groups = [
            dict(label='Woman',                  prompt='superrealistic portrait photograph of a woman, extremely feminine features, studio lighting',                                                               frac=0.25),
            dict(label='Woman w/ masc features', prompt='a superrealistic portrait photograph of a woman with masculine features, heavy brow ridge, studio lighting',                                               frac=0.25),
            dict(label='Man w/ fem features',    prompt='a superrealistic portrait photograph of a man with extremely feminine feminine features, soft delicate face, high cheekbones, studio lighting',            frac=0.25),
            dict(label='Man',                    prompt='a superrealistic portrait photograph of a man, extremely masculine features, studio lighting',                                                             frac=0.25),
        ],
    ),

    'AgeInterpolation': dict(
        best_jid         = 44492374,
        runs_root        = '/sci/labs/orzuk/ori_m/gdrive/conditional-matching/runs/InterpolationMenWomen',
        seed             = 42,
        n_eval_search    = 120,
        sdedit_start     = 125,
        n_steps          = 250,
        sdedit_cfg       = 7.5,
        sdedit_prompt    = 'a face scribble sketch of a man between 40 and 79 years old',
        neutral_prompt   = 'a superrealistic professional photograph of',
        controlnet_scale = 0.5,
        mode             = 'age',
        age_min          = 40,
        age_max          = 79,
        age_step         = 1,
        n_per_age        = 3,
    ),
}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def gen_images(sprinter, scribble_pil, prompt, n, controlnet_scale, seed=None):
    sprinter.vae.to(dtype=torch.float16)
    generator = torch.Generator(device=sprinter.device).manual_seed(seed) if seed is not None else None
    imgs = []
    with torch.no_grad():
        for start in range(0, n, 2):
            bs = min(2, n - start)
            imgs.extend(sprinter(
                prompt=[prompt] * bs,
                image=[scribble_pil] * bs,
                num_inference_steps=2,
                guidance_scale=0.0,
                controlnet_conditioning_scale=controlnet_scale,
                output_type='pil',
                generator=generator,
            ).images)
    sprinter.vae.to(dtype=torch.float32)
    return imgs


def clip_embed(images, clip_model, clip_processor, device):
    from clip_utils import encode_images_clip
    tensors = torch.cat([TF.to_tensor(img).unsqueeze(0) for img in images]).to(device)
    clip_model.to(device)
    with torch.no_grad():
        embs = encode_images_clip(tensors, clip_model, clip_processor)
    clip_model.to('cpu')
    return embs


def pil_to_tensor(pil_list, device):
    return torch.cat([TF.to_tensor(img).unsqueeze(0) for img in pil_list], dim=0).to(device)


# ── SDEdit ────────────────────────────────────────────────────────────────────

def run_sdedit(scr_pil, seed, architect, device, cfg_scale, n_steps_total, sdedit_start, prompt=''):
    architect.vae.to(dtype=torch.float32)
    architect.scheduler.set_timesteps(n_steps_total, device=device)
    timesteps = architect.scheduler.timesteps

    with torch.no_grad():
        t_in   = TF.to_tensor(scr_pil.convert('RGB')).unsqueeze(0).to(device).float()
        latent = architect.vae.encode((t_in * 2) - 1).latent_dist.mean * architect.vae.config.scaling_factor

    alpha   = architect.scheduler.alphas_cumprod.to(device)[timesteps[sdedit_start].long()].float()
    noise   = torch.randn(latent.shape,
                          generator=torch.Generator(device=device).manual_seed(seed),
                          device=device, dtype=torch.float32)
    latents = ((alpha ** 0.5) * latent + ((1 - alpha) ** 0.5) * noise).half()
    sched   = copy.deepcopy(architect.scheduler)

    if cfg_scale > 0:
        with torch.no_grad():
            pe, ne, pooled, neg_pooled = architect.encode_prompt(
                prompt=prompt, negative_prompt='', device=device,
                do_classifier_free_guidance=True, num_images_per_prompt=1)
        add_ids = torch.tensor([[512, 512, 0, 0, 512, 512]], dtype=pe.dtype, device=device)
        added   = {'text_embeds': torch.cat([neg_pooled, pooled]), 'time_ids': add_ids.repeat(2, 1)}
        cfg_st  = torch.cat([ne, pe])
        with torch.no_grad():
            for t in timesteps[sdedit_start:]:
                lmi     = sched.scale_model_input(torch.cat([latents] * 2), t)
                out     = architect.unet(lmi, t, encoder_hidden_states=cfg_st,
                                         added_cond_kwargs=added, return_dict=False)[0]
                nu, nc  = out.chunk(2)
                latents = sched.step(nu + cfg_scale * (nc - nu), t, latents).prev_sample
    else:
        with torch.no_grad():
            pe, _, pooled, _ = architect.encode_prompt(
                prompt='', negative_prompt='', device=device,
                do_classifier_free_guidance=False, num_images_per_prompt=1)
        add_ids = torch.tensor([[512, 512, 0, 0, 512, 512]], dtype=pe.dtype, device=device)
        added   = {'text_embeds': pooled, 'time_ids': add_ids}
        with torch.no_grad():
            for t in timesteps[sdedit_start:]:
                lmi     = sched.scale_model_input(latents, t)
                out     = architect.unet(lmi, t, encoder_hidden_states=pe,
                                         added_cond_kwargs=added, return_dict=False)[0]
                latents = sched.step(out, t, latents).prev_sample

    with torch.no_grad():
        dec = architect.vae.decode(
            (latents.float() / architect.vae.config.scaling_factor).to(architect.vae.dtype)
        ).sample
    return T.ToPILImage()(torch.clamp((dec.float() + 1) / 2, 0, 1).squeeze(0).cpu())


# ── BUILD TARGET EMBEDDINGS ───────────────────────────────────────────────────

def build_target_embeddings(cfg, source_scribble, sprinter, clip_model, clip_processor, device):
    from clip_utils import encode_images_clip
    SEED     = cfg['seed']
    cn_scale = cfg['controlnet_scale']
    mode     = cfg['mode']
    group_imgs = {}

    if mode in ('binary', 'multiclass'):
        n_target = 500
        all_imgs = []
        for i, g in enumerate(cfg['groups']):
            n_i  = max(1, int(n_target * g['frac']))
            print(f"  [{g['label']}] n={n_i}...")
            imgs = gen_images(sprinter, source_scribble, g['prompt'], n_i, cn_scale, seed=SEED + i * 1000)
            all_imgs.extend(imgs)
            group_imgs[g['label']] = imgs
        target_clip = clip_embed(all_imgs, clip_model, clip_processor, device)

    elif mode == 'age':
        ages      = list(range(cfg['age_min'], cfg['age_max'], cfg['age_step']))
        n_per_age = cfg['n_per_age']
        age_embs  = {}
        clip_model.to(device)
        with torch.no_grad():
            for age in tqdm(ages, desc='Building age target'):
                prompt = (f'a superrealistic portrait photograph of a {age}-year-old man, '
                          'studio lighting, sharp focus, photographic')
                imgs = gen_images(sprinter, source_scribble, prompt, n_per_age, cn_scale, seed=SEED + age * 7)
                embs = encode_images_clip(pil_to_tensor(imgs, device), clip_model, clip_processor)
                age_embs[age] = embs.cpu()
                group_imgs[str(age)] = imgs
        clip_model.to('cpu')
        target_clip = torch.cat([age_embs[a] for a in ages], dim=0).to(device)

    return target_clip, group_imgs


# ── AVG SCRIBBLE ──────────────────────────────────────────────────────────────

def build_avg_scribble(cfg, source_scribble, sprinter, hed, device):
    SEED     = cfg['seed']
    cn_scale = cfg['controlnet_scale']
    mode     = cfg['mode']

    if mode == 'age':
        ages = list(range(cfg['age_min'], cfg['age_max'] , cfg['age_step']))
        prompts = [
            f'a superrealistic portrait photograph of a {age}-year-old man, studio lighting, sharp focus, photographic'
            for age in ages]
        fracs = [1.0 / len(ages)] * len(ages)
    else:
        prompts = [g['prompt'] for g in cfg['groups']]
        fracs   = [g['frac']   for g in cfg['groups']]

    sprinter.vae.to(dtype=torch.float16)
    hed_scribbles = []
    for i, (prompt, frac) in enumerate(zip(prompts, fracs)):
        gen = torch.Generator(device=sprinter.device).manual_seed(SEED + i)
        with torch.no_grad():
            portrait = sprinter(
                prompt=[prompt], image=[source_scribble],
                num_inference_steps=2, guidance_scale=0.0,
                controlnet_conditioning_scale=cn_scale,
                output_type='pil', generator=gen,
            ).images[0]
        hed_scribbles.append(np.array(hed(portrait, scribble=True)).astype(np.float32))
    sprinter.vae.to(dtype=torch.float32)

    avg_np = sum(f * s for f, s in zip(fracs, hed_scribbles)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(avg_np)


# ── SAVE HELPERS ──────────────────────────────────────────────────────────────

def save_scribble_grid(scribbles_dict, out_dir):
    names  = list(scribbles_dict.keys())
    images = list(scribbles_dict.values())
    n      = len(names)

    for name, img in zip(names, images):
        img.save(out_dir / f'scribble_{name}.png')
        print(f'  Saved scribble_{name}.png')

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1: axes = [axes]
    for ax, img, name in zip(axes, images, names):
        ax.imshow(img, cmap='gray'); ax.set_title(name); ax.axis('off')
    plt.suptitle('All Scribbles', fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / 'scribbles_all.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved scribbles_all.png')


def save_target_previews(group_imgs, out_dir, n_preview=5):
    for label, imgs in group_imgs.items():
        safe = label.replace(' ', '_').replace('/', '_')
        n_cols = min(n_preview, len(imgs))
        fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3))
        if n_cols == 1: axes = [axes]
        for ax, img in zip(axes, imgs[:n_cols]):
            ax.imshow(img); ax.axis('off')
        fig.suptitle(f'Target: {label}', fontweight='bold')
        plt.tight_layout()
        fig.savefig(out_dir / f'target_{safe}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved target_{safe}.png')


def save_sdedit_search_plot(candidate_mmds, best_idx, out_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(candidate_mmds, color='steelblue', alpha=0.7, linewidth=1)
    ax.scatter([best_idx], [candidate_mmds[best_idx]], color='red', zorder=5,
               label=f'Best (i={best_idx}, MMD={candidate_mmds[best_idx]:.5f})')
    ax.set_xlabel('Candidate index'); ax.set_ylabel('MMD (search)')
    ax.set_title('SDEdit unguided candidate search')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / 'sdedit_search.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved sdedit_search.png')


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment',     required=True, choices=list(EXPERIMENT_CONFIGS.keys()))
    parser.add_argument('--lgd_cm_minutes', type=float, required=True,
                        help='LGD-CM runtime in minutes — sets N_SDEDIT_CANDIDATES')
    parser.add_argument('--repo_path',      default='/sci/labs/orzuk/ori_m/conditional-matching-paper/SD_cond_SD_controlnet')
    parser.add_argument('--gdrive_root',    default='gdrive:conditional-matching/runs')
    args = parser.parse_args()

    if args.repo_path not in sys.path:
        sys.path.insert(0, args.repo_path)

    cfg     = EXPERIMENT_CONFIGS[args.experiment]
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED    = cfg['seed']
    jid     = cfg['best_jid']
    run_dir = Path(args.repo_path) / 'experiments' / args.experiment
    out_dir = run_dir / 'baselines'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n=== {args.experiment} | jid={jid} | measuring candidate time... ===\n')

    # ── W&B init ──
    import wandb
    wandb.init(
        project='eval-baselines',
        entity='conditional-matching',
        name=f'{args.experiment}',
        config=dict(
            experiment=args.experiment,
            lgd_cm_minutes=args.lgd_cm_minutes,
            **{k: v for k, v in cfg.items() if k != 'groups'},
        ),
    )

    # ── Load models ──
    from models         import load_models
    from clip_utils     import load_clip_model
    from controlnet_aux import HEDdetector

    architect, sprinter        = load_models(device)
    clip_model, clip_processor = load_clip_model(device)
    hed                        = HEDdetector.from_pretrained('lllyasviel/Annotators')
    print('Models loaded.')

    # ── Load scribbles ──
    source_scribble = Image.open(run_dir / 'scribble.png')
    lgd_scribble = Image.open(run_dir / 'mlgdd.png')
    print('Scribbles loaded.')

    # ── Build target (small, for search only) ──
    print('\nBuilding search target...')
    target_clip, group_imgs = build_target_embeddings(
        cfg, source_scribble, sprinter, clip_model, clip_processor, device)
    print(f'Target CLIP: {target_clip.shape}')
    save_target_previews(group_imgs, out_dir)

    # ── Avg scribble ──
    print('\nBuilding avg scribble...')
    avg_scribble = build_avg_scribble(cfg, source_scribble, sprinter, hed, device)

    # ── Guided SDEdit scribble ──
    print('\nBuilding guided SDEdit scribble...')
    sdedit_scribble = run_sdedit(
        source_scribble, seed=SEED,
        architect=architect, device=device,
        cfg_scale=cfg['sdedit_cfg'], n_steps_total=cfg['n_steps'],
        sdedit_start=cfg['sdedit_start'], prompt=cfg['sdedit_prompt'],
    )

    # ── Measure first candidate time, then compute budget ──
    print('\nMeasuring first SDEdit candidate...')
    from metrics import compute_mmd
    t0 = time.time()
    first_cand = run_sdedit(
        source_scribble, seed=SEED,
        architect=architect, device=device,
        cfg_scale=0.0, n_steps_total=cfg['n_steps'],
        sdedit_start=cfg['sdedit_start'], prompt='',
    )
    first_imgs = gen_images(sprinter, first_cand, cfg['neutral_prompt'],
                            cfg['n_eval_search'], cfg['controlnet_scale'], seed=SEED)
    first_embs = clip_embed(first_imgs, clip_model, clip_processor, device)
    first_mmd  = compute_mmd(first_embs, target_clip).item()
    sec_per_candidate = time.time() - t0

    n_candidates = max(1, int(args.lgd_cm_minutes * 60 / sec_per_candidate))
    print(f'  First candidate: {sec_per_candidate:.1f}s → budget={args.lgd_cm_minutes}min → {n_candidates} candidates')
    wandb.log({'sec_per_candidate': sec_per_candidate, 'n_candidates': n_candidates})

    # ── SDEdit best search ──
    print('\nSearching SDEdit best...')
    candidate_mmds      = [first_mmd]
    candidate_scribbles = [first_cand]
    tqdm.write(f'  [1/{n_candidates}] seed={SEED}  MMD={first_mmd:.5f}')
    wandb.log({'candidate_mmd': first_mmd, 'candidate_idx': 1})

    for i in tqdm(range(1, n_candidates), desc='SDEdit candidates'):
        cand = run_sdedit(
            source_scribble, seed=SEED + i,
            architect=architect, device=device,
            cfg_scale=0.0, n_steps_total=cfg['n_steps'],
            sdedit_start=cfg['sdedit_start'], prompt='',
        )
        imgs = gen_images(sprinter, cand, cfg['neutral_prompt'],
                          cfg['n_eval_search'], cfg['controlnet_scale'], seed=SEED)
        embs = clip_embed(imgs, clip_model, clip_processor, device)
        mmd  = compute_mmd(embs, target_clip).item()
        candidate_mmds.append(mmd)
        candidate_scribbles.append(cand)
        tqdm.write(f'  [{i+1}/{n_candidates}] seed={SEED+i}  MMD={mmd:.5f}')
        wandb.log({'candidate_mmd': mmd, 'candidate_idx': i + 1})

    best_idx    = int(np.argmin(candidate_mmds))
    sdedit_best = candidate_scribbles[best_idx]
    print(f'\nBest: seed={SEED + best_idx}  MMD={candidate_mmds[best_idx]:.5f}')

    # ── Save everything ──
    print('\nSaving outputs...')
    scribbles_dict = {
        'source':      source_scribble,
        'avg':         avg_scribble,
        'sdedit':      sdedit_scribble,
        'sdedit_best': sdedit_best,
        'lgd_cm':      lgd_scribble,
    }
    save_scribble_grid(scribbles_dict, out_dir)
    save_sdedit_search_plot(candidate_mmds, best_idx, out_dir)

    # ── Log final results to W&B ──
    wandb.log({
        'best_candidate_idx': best_idx,
        'best_candidate_mmd': candidate_mmds[best_idx],
        'scribble/source':      wandb.Image(source_scribble),
        'scribble/avg':         wandb.Image(avg_scribble),
        'scribble/sdedit':      wandb.Image(sdedit_scribble),
        'scribble/sdedit_best': wandb.Image(sdedit_best),
        'scribble/lgd_cm':      wandb.Image(lgd_scribble),
        'search_curve': wandb.plot.line_series(
            xs    = list(range(len(candidate_mmds))),
            ys    = [candidate_mmds],
            keys  = ['MMD'],
            title = 'SDEdit candidate search',
            xname = 'candidate',
        ),
    })

    meta = dict(
        experiment          = args.experiment,
        best_jid            = jid,
        lgd_cm_minutes      = args.lgd_cm_minutes,
        n_candidates        = n_candidates,
        sec_per_candidate   = round(sec_per_candidate, 2),
        best_candidate_idx  = best_idx,
        best_candidate_seed = SEED + best_idx,
        best_candidate_mmd  = candidate_mmds[best_idx],
        all_candidate_mmds  = candidate_mmds,
    )
    with open(out_dir / 'baselines_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print('  Saved baselines_meta.json')

    # ── Sync to GDrive ──
    gdrive_dest = f"{args.gdrive_root}/{args.experiment}/baselines"

    print(f'\nSyncing to {gdrive_dest}...')
    os.system(f'rclone copy "{out_dir}" "{gdrive_dest}" --tpslimit 10 --transfers 4')

    wandb.finish()
    print('\n✅ Done.')


if __name__ == '__main__':
    main()
