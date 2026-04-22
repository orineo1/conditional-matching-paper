"""
run_experiment.py  —  Scribble Conditioning Comparison (cluster version)
Usage:
    python run_experiment.py --exp 50_50
    python run_experiment.py --exp 25_75
    python run_experiment.py --exp gender_interp
    python run_experiment.py --exp age_interp
    python run_experiment.py --exp 50_50 --sanity_check   # quick sanity check only
"""
import argparse
import sys
import os
import copy
import json
import time

import matplotlib
matplotlib.use('Agg')   # non-interactive backend — must be before pyplot import
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter

# ── Cluster repo paths ─────────────────────────────────────────────────────
REPO_PATH = '/sci/labs/orzuk/ori_m/conditional-matching-paper'
sys.path.insert(0, REPO_PATH)
sys.path.insert(0, os.path.join(REPO_PATH, 'SD_cond_SD_controlnet'))

from models      import load_models
from image_utils import sobel_proxy, build_base_image
from clip_utils  import load_clip_model, encode_images_clip
from run_dps     import compute_clip_softmax
from metrics     import compute_mmd, compute_swd

import wandb

# ── Args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, required=True,
                    choices=['50_50', '25_75', 'gender_interp', 'age_interp'])
parser.add_argument('--sanity_check', action='store_true',
                    help='Run quick sanity check (n=4) instead of full experiment')
parser.add_argument('--wandb_project', type=str, default='scribble_conditioning')
parser.add_argument('--wandb_entity',  type=str, default=None)
args = parser.parse_args()

# ===========================================================================
# 3. Global config
# ===========================================================================

EXPERIMENT_NAME  = f'scribble_v3_{args.exp}'
LAB_ROOT         = '/sci/labs/orzuk/ori_m'
SAVE_DIR         = os.path.join(LAB_ROOT, 'scribble_results', args.exp)
CHECKPOINT_DIR   = os.path.join(SAVE_DIR, 'checkpoints')
BEST_DIR         = os.path.join(SAVE_DIR, 'best_scribbles')
for d in [SAVE_DIR, CHECKPOINT_DIR, BEST_DIR]:
    os.makedirs(d, exist_ok=True)

N_RUNS           = 10
N_FINAL          = 2000
N_RANDOM         = 50
N_MAN_POOL       = 100   # number of real man portraits to search as HED scribbles
BATCH_SIZE       = 2
N_PROXY_DEFAULT  = 100
CONTROLNET_SCALE = 0.5
N_STEPS_FAST     = 2
SDEDIT_START     = 125
N_STEPS_ARCH     = 250
SDEDIT_CFG       = 7.5

NEUTRAL_PROMPT = 'a superrealistic professional photograph of'
MAN_PROMPT     = 'a superrealistic portrait photograph of a man, studio lighting'
WOMAN_PROMPT   = 'a superrealistic portrait photograph of a woman, studio lighting'

SDEDIT_PROMPT_BALANCED = 'a face scribble sketch of man or woman'
SDEDIT_PROMPT_SKEWED   = 'a face scribble sketch, mostly feminine features sometimes with masculine features'
SDEDIT_PROMPT_INTERP   = 'a face scribble sketch, with a range of feminine to masculine features'
SDEDIT_PROMPT_AGE      = 'a face scribble sketch of a person at various ages between 40 to 79'

AGE_MIN            = 40
AGE_MAX            = 79
AGE_IMAGES_PER_AGE = 3
AGE_LIST           = list(range(AGE_MIN, AGE_MAX + 1))
N_AGES             = len(AGE_LIST)
N_PROXY_AGE        = N_AGES * AGE_IMAGES_PER_AGE   # 120

AGE_PROMPT_FN = lambda age: (
    f'a superrealistic portrait photograph of a {age}-year-old man, '
    'studio lighting, sharp focus, photographic'
)

GENDER_INTERP_CLASSES = [
    ('Woman',
     'a superrealistic portrait photograph of a woman, extremely feminine features, studio lighting',
     0.25),
    ('Woman+masc',
     'a superrealistic portrait photograph of a woman with masculine features, heavy brow ridge, studio lighting',
     0.25),
    ('Man+fem',
     'a superrealistic portrait photograph of a man with extremely feminine features, soft delicate face, high cheekbones, studio lighting',
     0.25),
    ('Man',
     'a superrealistic portrait photograph of a man, extremely masculine features, studio lighting',
     0.25),
]
assert abs(sum(f for _, _, f in GENDER_INTERP_CLASSES) - 1.0) < 1e-6

CONDITION_NAMES = ['avg_scribble', 'sdedit_scribble', 'random_best_scribble', 'man_pool_best_scribble']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
print(f'Experiment: {args.exp}')
print(f'Save dir:   {SAVE_DIR}')

# ===========================================================================
# 4. W&B init
# ===========================================================================

run = wandb.init(
    project = args.wandb_project,
    entity  = args.wandb_entity,
    name    = f'{args.exp}_{"sanity" if args.sanity_check else "full"}',
    config  = {
        'exp':               args.exp,
        'N_RUNS':            N_RUNS,
        'N_FINAL':           N_FINAL,
        'N_RANDOM':          N_RANDOM,
        'N_MAN_POOL':        N_MAN_POOL,
        'N_PROXY_DEFAULT':   N_PROXY_DEFAULT,
        'N_PROXY_AGE':       N_PROXY_AGE,
        'CONTROLNET_SCALE':  CONTROLNET_SCALE,
        'SDEDIT_START':      SDEDIT_START,
        'SDEDIT_CFG':        SDEDIT_CFG,
        'sanity_check':      args.sanity_check,
    }
)
print(f'W&B run: {run.url}')

# ===========================================================================
# 5. Load models
# ===========================================================================

architect, sprinter        = load_models(device)
clip_model, clip_processor = load_clip_model(device)
print('Models loaded.')

# ===========================================================================
# 6. Base oval + Sobel
# ===========================================================================

from controlnet_aux import HEDdetector
hed = HEDdetector.from_pretrained('lllyasviel/Annotators')

base_image_pil, base_tensor = build_base_image(device)
with torch.no_grad():
    sobel_pil = T.ToPILImage()(sobel_proxy(base_tensor, device).squeeze(0).cpu())

# Log base images to W&B once
wandb.log({
    'setup/base_oval':  wandb.Image(base_image_pil,           caption='Base oval'),
    'setup/sobel_cond': wandb.Image(sobel_pil,                caption='Sobel conditioning'),
})
print('Sobel bootstrapping image ready.')

# ===========================================================================
# 7. Helpers
# ===========================================================================

def generate_images(pipe, prompt, cond_pil, n, seed=None):
    pipe.vae.to(dtype=torch.float16)
    gen = torch.Generator(device=pipe.device).manual_seed(seed) if seed is not None else None
    images = []
    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            bs = min(BATCH_SIZE, n - start)
            result = pipe(
                prompt=[prompt] * bs,
                image=[cond_pil] * bs,
                num_inference_steps=N_STEPS_FAST,
                guidance_scale=0.0,
                controlnet_conditioning_scale=CONTROLNET_SCALE,
                output_type='pil',
                generator=gen,
            )
            images.extend(result.images)
            print(f'  {len(images)}/{n}', end='\r')
    pipe.vae.to(dtype=torch.float32)
    print()
    return images


def clip_embed(images):
    tensors = torch.cat([TF.to_tensor(img).unsqueeze(0) for img in images]).to(device)
    clip_model.to(device)
    with torch.no_grad():
        embs = encode_images_clip(tensors, clip_model, clip_processor)
    clip_model.to('cpu')
    return embs


def encode_text_prompts(prompts):
    clip_model.to(device)
    inputs = clip_processor(text=prompts, return_tensors='pt', padding=True).to(device)
    with torch.no_grad():
        out       = clip_model.text_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        text_embs = clip_model.text_projection(out.pooler_output)
    clip_model.to('cpu')
    return F.normalize(text_embs, dim=-1)


def classify_binary_gender(images):
    res, _ = compute_clip_softmax(images, clip_model, clip_processor,
                                   MAN_PROMPT, WOMAN_PROMPT, device)
    n_male = sum(1 for r in res if r['label'] == 'male')
    return n_male, n_male / len(res) * 100


def classify_multinomial(image_embs, text_embs):
    img_n  = F.normalize(image_embs.float(), dim=-1)
    logits = img_n @ text_embs.T.float()
    probs  = torch.softmax(logits * 100, dim=-1).cpu().numpy()
    labels = probs.argmax(axis=1).tolist()
    props  = np.bincount(labels, minlength=text_embs.shape[0]) / len(labels)
    return labels, probs, props


def binomial_ci(n_success, n_total, z=1.96):
    p  = n_success / n_total
    se = np.sqrt(p * (1 - p) / n_total)
    return float(p), float(p - z * se), float(p + z * se)


def multinomial_ci(counts, n_total, z=1.96):
    p  = counts / n_total
    se = np.sqrt(p * (1 - p) / n_total)
    return p, p - z * se, p + z * se


def bootstrap_hed(prompt, seed):
    imgs = generate_images(sprinter, prompt, sobel_pil, 1, seed=seed)
    scr  = hed(imgs[0], scribble=True)
    return imgs[0], scr


def run_sdedit(scr_pil, seed, sdedit_prompt):
    architect.vae.to(dtype=torch.float32)
    architect.scheduler.set_timesteps(N_STEPS_ARCH, device=device)
    timesteps = architect.scheduler.timesteps

    with torch.no_grad():
        t      = TF.to_tensor(scr_pil.convert('RGB')).unsqueeze(0).to(device).float()
        t      = (t * 2.0) - 1.0
        latent = architect.vae.encode(t).latent_dist.mean * architect.vae.config.scaling_factor

    t_start = timesteps[SDEDIT_START]
    alpha   = architect.scheduler.alphas_cumprod.to(device)[t_start.long()].float()
    gen     = torch.Generator(device=device).manual_seed(seed)
    noise   = torch.randn(latent.shape, generator=gen, device=device, dtype=torch.float32)
    latents = ((alpha ** 0.5) * latent + ((1 - alpha) ** 0.5) * noise).half()

    with torch.no_grad():
        prompt_embeds, neg_embeds, pooled, neg_pooled = architect.encode_prompt(
            prompt=sdedit_prompt, negative_prompt='', device=device,
            do_classifier_free_guidance=True, num_images_per_prompt=1,
        )
    add_ids = torch.tensor([[512, 512, 0, 0, 512, 512]], dtype=prompt_embeds.dtype, device=device)
    added   = {'text_embeds': torch.cat([neg_pooled, pooled]),
               'time_ids':    add_ids.repeat(2, 1)}
    cfg_st  = torch.cat([neg_embeds, prompt_embeds])

    sched = copy.deepcopy(architect.scheduler)
    with torch.no_grad():
        for t in timesteps[SDEDIT_START:]:
            lmi     = sched.scale_model_input(torch.cat([latents] * 2), t)
            out     = architect.unet(lmi, t, encoder_hidden_states=cfg_st,
                                     added_cond_kwargs=added, return_dict=False)[0]
            nu, nc  = out.chunk(2)
            latents = sched.step(nu + SDEDIT_CFG * (nc - nu), t, latents).prev_sample

    with torch.no_grad():
        dec = architect.vae.decode(
            (latents.float() / architect.vae.config.scaling_factor).to(architect.vae.dtype)
        ).sample
        pil = T.ToPILImage()(
            torch.clamp((dec.float() + 1.0) / 2.0, 0.0, 1.0).squeeze(0).cpu()
        )
    return pil


def perturb_scribble(base_pil, rng_seed, noise_std=25, blur_range=(0.0, 2.0)):
    rng = np.random.RandomState(rng_seed)
    arr = np.array(base_pil).astype(np.float32)
    arr = (arr + rng.normal(0, noise_std, arr.shape)).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    blur = rng.uniform(*blur_range)
    if blur > 0.3:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=blur))
    return pil


def search_random_best(base_scr, proxy_target, gen_seed_base, n_proxy):
    best_scr, best_mmd, all_mmds = None, float('inf'), []
    for i in range(N_RANDOM):
        cand = perturb_scribble(base_scr, rng_seed=gen_seed_base + i)
        imgs = generate_images(sprinter, NEUTRAL_PROMPT, cand, n_proxy,
                               seed=gen_seed_base + i)
        embs = clip_embed(imgs)
        mmd  = compute_mmd(embs, proxy_target.detach()).item()
        all_mmds.append(float(mmd))
        if mmd < best_mmd:
            best_mmd = mmd
            best_scr = cand.copy()
        print(f'    rand {i+1:3d}/{N_RANDOM}  mmd={mmd:.5f}  best={best_mmd:.5f}', end='\r')
    print()
    return best_scr, float(best_mmd), all_mmds



def search_man_pool_best(seed, proxy_target, n_proxy, n_pool=None):
    """Generate n_pool real man portraits via Sobel→HED (each a different face),
    evaluate each scribble with n_proxy neutral images, return the one
    with lowest MMD vs proxy_target.
    Unlike random_best which perturbs one anchor, this searches over
    genuinely different man face geometries."""
    if n_pool is None:
        n_pool = N_MAN_POOL
    best_scr, best_mmd, all_mmds = None, float('inf'), []
    for i in range(n_pool):
        # Different seed per portrait → genuinely different man face each time
        _, scr = bootstrap_hed(MAN_PROMPT, seed=seed * 200000 + i)
        imgs   = generate_images(sprinter, NEUTRAL_PROMPT, scr, n_proxy,
                                 seed=seed * 200000 + i)
        embs   = clip_embed(imgs)
        mmd    = compute_mmd(embs, proxy_target.detach()).item()
        all_mmds.append(float(mmd))
        if mmd < best_mmd:
            best_mmd = mmd
            best_scr = scr.copy()
        print(f'    pool {i+1:3d}/{n_pool}  mmd={mmd:.5f}  best={best_mmd:.5f}', end='\r')
    print()
    return best_scr, float(best_mmd), all_mmds


def eval_scribble(scribble_pil, target_clip, n, seed=None):
    imgs = generate_images(sprinter, NEUTRAL_PROMPT, scribble_pil, n, seed=seed)
    embs = clip_embed(imgs)
    mmd  = compute_mmd(embs, target_clip.detach()).item()
    swd  = compute_swd(embs, target_clip.detach()).item()
    return imgs, embs, float(mmd), float(swd)


def enrich_binary_gender(result, imgs, n_total):
    n_male, _ = classify_binary_gender(imgs)
    p, lo, hi = binomial_ci(n_male, n_total)
    result.update({'n_male': int(n_male), 'n_female': int(n_total - n_male),
                   'p_male': p, 'ci_lo_male': lo, 'ci_hi_male': hi})


def enrich_gender_interp(result, embs, n_total, text_embs, labels):
    lbs, _, _  = classify_multinomial(embs, text_embs)
    counts     = np.bincount(lbs, minlength=len(labels))
    p, lo, hi  = multinomial_ci(counts, n_total)
    result.update({'class_labels': labels, 'class_counts': counts.tolist(),
                   'class_proportions': p.tolist(),
                   'class_ci_lo': lo.tolist(), 'class_ci_hi': hi.tolist()})


def enrich_age(result, embs, n_total):
    lbs, _, _  = classify_multinomial(embs, age_text_embs)
    counts     = np.bincount(lbs, minlength=N_AGES)
    p, lo, hi  = multinomial_ci(counts, n_total)
    result.update({'class_labels': age_labels, 'class_counts': counts.tolist(),
                   'class_proportions': p.tolist(),
                   'class_ci_lo': lo.tolist(), 'class_ci_hi': hi.tolist()})


def fmt(s):
    h, r = divmod(int(s), 3600)
    m, s = divmod(r, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def to_json(obj):
    _skip = {'scribble', 'portrait', 'config', 'build_target_fn',
             'build_source_fn', 'text_embs'}
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items() if k not in _skip}
    if isinstance(obj, list):
        return [to_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    return path


def wandb_images(imgs, captions=None, max_imgs=8):
    """Convert a list of PIL images to wandb.Image objects."""
    imgs = imgs[:max_imgs]
    if captions is None:
        captions = [f'img_{i}' for i in range(len(imgs))]
    return [wandb.Image(img, caption=cap) for img, cap in zip(imgs, captions)]


# ===========================================================================
# 8. Text embeddings
# ===========================================================================

gender_interp_labels    = [l for l, _, _ in GENDER_INTERP_CLASSES]
gender_interp_prompts   = [p for _, p, _ in GENDER_INTERP_CLASSES]
gender_interp_fracs     = [f for _, _, f in GENDER_INTERP_CLASSES]
gender_interp_text_embs = encode_text_prompts(gender_interp_prompts)

age_prompts   = [AGE_PROMPT_FN(a) for a in AGE_LIST]
age_labels    = [f'age_{a}' for a in AGE_LIST]
age_fracs     = [1.0 / N_AGES] * N_AGES
age_text_embs = encode_text_prompts(age_prompts)

print(f'Gender-interp text embs : {gender_interp_text_embs.shape}')
print(f'Age text embs           : {age_text_embs.shape}')

# ===========================================================================
# 9. Target builders + source builders
# ===========================================================================

def build_target_gender_ratio(hed_anchor, seed_offset, n, man_frac):
    n_man   = int(n * man_frac)
    n_woman = n - n_man
    man_imgs   = generate_images(sprinter, MAN_PROMPT,   hed_anchor, n_man,   seed=seed_offset)
    woman_imgs = generate_images(sprinter, WOMAN_PROMPT, hed_anchor, n_woman, seed=seed_offset + 500)
    return clip_embed(man_imgs + woman_imgs)


def build_target_gender_interp(hed_anchor, seed_offset, n):
    all_imgs = []
    for i, (label, prompt, frac) in enumerate(GENDER_INTERP_CLASSES):
        n_i  = int(n * frac)
        imgs = generate_images(sprinter, prompt, hed_anchor, n_i, seed=seed_offset + i * 300)
        all_imgs.extend(imgs)
    return clip_embed(all_imgs)


def build_target_age(hed_anchor, seed_offset):
    all_imgs = []
    for i, (age, prompt) in enumerate(zip(AGE_LIST, age_prompts)):
        imgs = generate_images(sprinter, prompt, hed_anchor,
                               AGE_IMAGES_PER_AGE, seed=seed_offset + i * 10)
        all_imgs.extend(imgs)
    return clip_embed(all_imgs)


def build_source_gender_ratio(seed, man_frac):
    _, scr_man   = bootstrap_hed(MAN_PROMPT,   seed=seed)
    _, scr_woman = bootstrap_hed(WOMAN_PROMPT, seed=seed + 1)
    woman_frac = 1.0 - man_frac
    avg_np = (
        man_frac   * np.array(scr_man).astype(np.float32) +
        woman_frac * np.array(scr_woman).astype(np.float32)
    ).clip(0, 255).astype(np.uint8)
    return {
        'anchor_scr':            scr_man,
        'anchor_hed_for_target': scr_man,
        'avg_scribble':          Image.fromarray(avg_np),
        '_scr_man':              scr_man,
        '_scr_woman':            scr_woman,
    }


def build_source_gender_interp(seed):
    class_scrs = []
    for i, (label, prompt, frac) in enumerate(GENDER_INTERP_CLASSES):
        _, scr = bootstrap_hed(prompt, seed=seed + i)
        class_scrs.append((scr, frac))
    avg_np = sum(
        f * np.array(s).astype(np.float32) for s, f in class_scrs
    ).clip(0, 255).astype(np.uint8)
    anchor_scr = class_scrs[-1][0]
    return {
        'anchor_scr':            anchor_scr,
        'anchor_hed_for_target': anchor_scr,
        'avg_scribble':          Image.fromarray(avg_np),
        '_class_scrs':           [s for s, _ in class_scrs],
    }


def build_source_age(seed):
    age_scrs = []
    for i, (age, prompt) in enumerate(zip(AGE_LIST, age_prompts)):
        _, scr = bootstrap_hed(prompt, seed=seed + i)
        age_scrs.append(scr)
    avg_np = (
        sum(np.array(s).astype(np.float32) for s in age_scrs) / N_AGES
    ).clip(0, 255).astype(np.uint8)
    anchor_scr = age_scrs[0]
    return {
        'anchor_scr':            anchor_scr,
        'anchor_hed_for_target': anchor_scr,
        'avg_scribble':          Image.fromarray(avg_np),
        '_age_scrs':             age_scrs,
    }


# ===========================================================================
# 10. Experiment registry — only the requested experiment
# ===========================================================================

ALL_EXPERIMENTS_FULL = [
    {
        'name':          '50_50',
        'description':   '50% man / 50% woman',
        'n_proxy':       N_PROXY_DEFAULT,
        'eval_type':     'binary_gender',
        'class_labels':  ['man', 'woman'],
        'class_fracs':   [0.5, 0.5],
        'sdedit_prompt': SDEDIT_PROMPT_BALANCED,
        'build_source_fn':  lambda seed: build_source_gender_ratio(seed, man_frac=0.5),
        'build_target_fn':  lambda anc, so, n: build_target_gender_ratio(anc, so, n, man_frac=0.5),
    },
    {
        'name':          '25_75',
        'description':   '25% man / 75% woman',
        'n_proxy':       N_PROXY_DEFAULT,
        'eval_type':     'binary_gender',
        'class_labels':  ['man', 'woman'],
        'class_fracs':   [0.25, 0.75],
        'sdedit_prompt': SDEDIT_PROMPT_SKEWED,
        'build_source_fn':  lambda seed: build_source_gender_ratio(seed, man_frac=0.25),
        'build_target_fn':  lambda anc, so, n: build_target_gender_ratio(anc, so, n, man_frac=0.25),
    },
    {
        'name':          'gender_interp',
        'description':   '4-class gender spectrum (each 25%)',
        'n_proxy':       N_PROXY_DEFAULT,
        'eval_type':     'gender_interp',
        'class_labels':  gender_interp_labels,
        'class_fracs':   gender_interp_fracs,
        'sdedit_prompt': SDEDIT_PROMPT_INTERP,
        'build_source_fn':  build_source_gender_interp,
        'build_target_fn':  lambda anc, so, n: build_target_gender_interp(anc, so, n),
    },
    {
        'name':          'age_interp',
        'description':   f'Uniform ages {AGE_MIN}–{AGE_MAX} men ({N_PROXY_AGE} proxy images)',
        'n_proxy':       N_PROXY_AGE,
        'eval_type':     'age',
        'class_labels':  age_labels,
        'class_fracs':   age_fracs,
        'sdedit_prompt': SDEDIT_PROMPT_AGE,
        'build_source_fn':  build_source_age,
        'build_target_fn':  lambda anc, so, n: build_target_age(anc, so),
    },
]

exp = next(e for e in ALL_EXPERIMENTS_FULL if e['name'] == args.exp)
n_proxy = exp['n_proxy']
print(f'\nRunning: {exp["name"]}  —  {exp["description"]}')
print(f'SDEdit prompt: {exp["sdedit_prompt"]}')

# ===========================================================================
# 11. SANITY CHECK (--sanity_check flag)
# ===========================================================================

if args.sanity_check:
    print('\n=== SANITY CHECK MODE (n=4) ===')
    n_tiny    = 4
    seed_test = 1
    n_rand_test = 5

    torch.manual_seed(seed_test)
    np.random.seed(seed_test)

    # [A] Source scribbles
    print('\n[A] Source scribbles...')
    src_test   = exp['build_source_fn'](seed_test)
    anchor_scr = src_test['anchor_scr']
    avg_scr    = src_test['avg_scribble']

    scr_log = {}
    scr_log['anchor'] = wandb.Image(anchor_scr,  caption='anchor HED')
    scr_log['avg']    = wandb.Image(avg_scr,      caption='avg scribble')
    if '_scr_man' in src_test:
        scr_log['man']   = wandb.Image(src_test['_scr_man'],   caption='man HED')
        scr_log['woman'] = wandb.Image(src_test['_scr_woman'], caption='woman HED')
    elif '_class_scrs' in src_test:
        for i, (lbl, _, _) in enumerate(GENDER_INTERP_CLASSES):
            scr_log[lbl] = wandb.Image(src_test['_class_scrs'][i], caption=lbl)
    elif '_age_scrs' in src_test:
        for idx in [0, N_AGES // 4, N_AGES // 2, 3 * N_AGES // 4, -1]:
            scr_log[f'age_{AGE_LIST[idx]}'] = wandb.Image(
                src_test['_age_scrs'][idx], caption=f'age {AGE_LIST[idx]}')
    wandb.log({f'sanity/scribbles/{k}': v for k, v in scr_log.items()})
    print('    [A] logged to W&B')

    # [B] Proxy target samples
    print('\n[B] Proxy target samples...')
    proxy_test = exp['build_target_fn'](anchor_scr, seed_test * 1000, n_tiny)
    print(f'    proxy shape: {proxy_test.shape}')

    if exp['eval_type'] == 'binary_gender':
        target_vis    = (generate_images(sprinter, MAN_PROMPT,   anchor_scr, 2, seed=1) +
                         generate_images(sprinter, WOMAN_PROMPT, anchor_scr, 2, seed=2))
        target_titles = ['man', 'man', 'woman', 'woman']
    elif exp['eval_type'] == 'gender_interp':
        target_vis, target_titles = [], []
        for lbl, prompt, _ in GENDER_INTERP_CLASSES:
            target_vis  += generate_images(sprinter, prompt, anchor_scr, 1, seed=1)
            target_titles.append(lbl)
    else:  # age
        target_vis, target_titles = [], []
        for idx in [0, N_AGES // 4, N_AGES // 2, 3 * N_AGES // 4, -1]:
            target_vis  += generate_images(sprinter, age_prompts[idx], anchor_scr, 1, seed=1)
            target_titles.append(f'age {AGE_LIST[idx]}')

    wandb.log({'sanity/target_samples': wandb_images(target_vis, target_titles)})
    print('    [B] logged to W&B')

    # [C] SDEdit
    print(f'\n[C] SDEdit  prompt="{exp["sdedit_prompt"]}"')
    scr_sdedit_test = run_sdedit(anchor_scr, seed_test, exp['sdedit_prompt'])
    wandb.log({
        'sanity/sdedit/anchor': wandb.Image(anchor_scr,        caption='anchor (input)'),
        'sanity/sdedit/output': wandb.Image(scr_sdedit_test,   caption='SDEdit output'),
    })
    print('    [C] logged to W&B')

    # [D] Random search (5 candidates)
    print(f'\n[D] Random search ({n_rand_test} candidates, n={n_tiny} each)...')
    rand_scrs, rand_mmds = [], []
    for i in range(n_rand_test):
        cand = perturb_scribble(anchor_scr, rng_seed=seed_test * 100000 + i)
        imgs = generate_images(sprinter, NEUTRAL_PROMPT, cand, n_tiny,
                               seed=seed_test * 100000 + i)
        embs = clip_embed(imgs)
        mmd  = compute_mmd(embs, proxy_test.detach()).item()
        rand_scrs.append(cand)
        rand_mmds.append(mmd)
        print(f'    candidate {i+1}  MMD={mmd:.5f}')

    best_idx  = int(np.argmin(rand_mmds))
    worst_idx = int(np.argmax(rand_mmds))

    # Log all candidate scribbles
    wandb.log({
        'sanity/random_candidates/scribbles': [
            wandb.Image(scr, caption=f'cand {i+1}  MMD={m:.4f}')
            for i, (scr, m) in enumerate(zip(rand_scrs, rand_mmds))
        ]
    })

    # Log generated images from best vs worst
    best_imgs  = generate_images(sprinter, NEUTRAL_PROMPT, rand_scrs[best_idx],  n_tiny, seed=1)
    worst_imgs = generate_images(sprinter, NEUTRAL_PROMPT, rand_scrs[worst_idx], n_tiny, seed=1)
    wandb.log({
        'sanity/random_candidates/best_images':  wandb_images(
            best_imgs,  [f'best cand {best_idx+1}'] * n_tiny),
        'sanity/random_candidates/worst_images': wandb_images(
            worst_imgs, [f'worst cand {worst_idx+1}'] * n_tiny),
        'sanity/random_candidates/best_mmd':  rand_mmds[best_idx],
        'sanity/random_candidates/worst_mmd': rand_mmds[worst_idx],
    })
    print('    [D] logged to W&B')

    # [E] Evaluate each condition
    print('\n[E] Evaluating conditions...')
    # [D2] Man pool (sanity: just 3 portraits)
    print('\n[D2] Man pool sanity (3 portraits)...')
    scr_man_pool_test, man_pool_mmd_test, _ = search_man_pool_best(
        seed_test, proxy_test, n_tiny, n_pool=3
    )
    wandb.log({
        'sanity/man_pool/best_scribble': wandb.Image(scr_man_pool_test, caption='man pool best'),
        'sanity/man_pool/best_mmd':      man_pool_mmd_test,
    })
    print(f'    [D2] man_pool_best_mmd={man_pool_mmd_test:.5f}  logged to W&B')

    test_conditions = {
        'avg_scribble':           avg_scr,
        'sdedit_scribble':        scr_sdedit_test,
        'random_best_scribble':   rand_scrs[best_idx],
        'man_pool_best_scribble': scr_man_pool_test,
        'anchor_scribble':        anchor_scr,
    }
    for cond_name, scribble in test_conditions.items():
        imgs_test, embs_test, mmd_test, swd_test = eval_scribble(
            scribble, proxy_test, n_tiny, seed=seed_test
        )
        print(f'    [{cond_name}]  MMD={mmd_test:.5f}  SWD={swd_test:.5f}')
        wandb.log({
            f'sanity/conditions/{cond_name}/mmd': mmd_test,
            f'sanity/conditions/{cond_name}/swd': swd_test,
            f'sanity/conditions/{cond_name}/images': wandb_images(
                imgs_test, [f'{cond_name} img {i+1}' for i in range(len(imgs_test))]
            ),
            f'sanity/conditions/{cond_name}/scribble': wandb.Image(
                scribble, caption=cond_name),
        })

    print('\n=== SANITY CHECK COMPLETE — check W&B dashboard ===')
    print(f'W&B run: {run.url}')
    wandb.finish()
    sys.exit(0)


# ===========================================================================
# 12. Main experiment loop
# ===========================================================================

exp_dir = os.path.join(CHECKPOINT_DIR, exp['name'])
os.makedirs(exp_dir, exist_ok=True)

all_records     = []
seed_times      = {}
best_per_method = {c: {'seed': None, 'mmd': float('inf'), 'scribble': None}
                   for c in CONDITION_NAMES}

for seed in range(1, N_RUNS + 1):
    t_seed_start = time.time()
    print(f"\n{'='*55}")
    print(f'  {exp["name"]}  |  SEED {seed}/{N_RUNS}')
    print(f"{'='*55}")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    seed_dir = os.path.join(exp_dir, f'seed_{seed:02d}')
    os.makedirs(seed_dir, exist_ok=True)

    # [A] Source scribbles
    t_a = time.time()
    print('  [A] Source scribbles (Sobel→portrait→HED)...')
    src        = exp['build_source_fn'](seed)
    anchor_scr = src['anchor_scr']
    hed_anchor = src['anchor_hed_for_target']
    print(f'      done in {fmt(time.time() - t_a)}')

    anchor_scr.save(os.path.join(seed_dir, 'scribble_anchor.png'))
    src['avg_scribble'].save(os.path.join(seed_dir, 'scribble_avg.png'))
    if '_scr_man' in src:
        src['_scr_man'].save(os.path.join(seed_dir, 'scribble_man.png'))
        src['_scr_woman'].save(os.path.join(seed_dir, 'scribble_woman.png'))
    if '_class_scrs' in src:
        for i, (lbl, _, _) in enumerate(GENDER_INTERP_CLASSES):
            safe = lbl.lower().replace(' ', '_').replace('+', '_')
            src['_class_scrs'][i].save(os.path.join(seed_dir, f'scribble_{safe}.png'))
    if '_age_scrs' in src:
        for idx in [0, N_AGES // 2, -1]:
            src['_age_scrs'][idx].save(
                os.path.join(seed_dir, f'scribble_age_{AGE_LIST[idx]}.png'))

    # Log seed scribbles to W&B
    wandb.log({
        f'seed_{seed}/scribbles/anchor': wandb.Image(anchor_scr, caption='anchor'),
        f'seed_{seed}/scribbles/avg':    wandb.Image(src['avg_scribble'], caption='avg'),
    }, step=seed)

    # [B] Proxy target
    t_b = time.time()
    print('  [B] Proxy target...')
    proxy_target = exp['build_target_fn'](hed_anchor, seed * 1000, n_proxy)
    print(f'      shape={proxy_target.shape}  done in {fmt(time.time() - t_b)}')

    # [C] SDEdit
    t_c = time.time()
    print(f'  [C] SDEdit...')
    scr_sdedit = run_sdedit(anchor_scr, seed, exp['sdedit_prompt'])
    scr_sdedit.save(os.path.join(seed_dir, 'scribble_sdedit.png'))
    print(f'      done in {fmt(time.time() - t_c)}')

    wandb.log({
        f'seed_{seed}/scribbles/sdedit': wandb.Image(scr_sdedit, caption='sdedit'),
    }, step=seed)

    # [D] Random best search
    t_d = time.time()
    print(f'  [D] Random best ({N_RANDOM} candidates)...')
    scr_rand, rand_best_mmd, rand_all_mmds = search_random_best(
        anchor_scr, proxy_target,
        gen_seed_base=seed * 100000,
        n_proxy=n_proxy,
    )
    scr_rand.save(os.path.join(seed_dir, 'scribble_random_best.png'))
    print(f'      best_mmd={rand_best_mmd:.5f}  done in {fmt(time.time() - t_d)}')

    wandb.log({
        f'seed_{seed}/scribbles/random_best':     wandb.Image(scr_rand, caption='random best'),
        f'seed_{seed}/random_search/best_mmd':    rand_best_mmd,
        f'seed_{seed}/random_search/all_mmds':    wandb.Histogram(rand_all_mmds),
    }, step=seed)

    # [D2] Man pool best search
    t_d2 = time.time()
    print(f'  [D2] Man pool best ({N_MAN_POOL} real man portraits → HED)...')
    scr_man_pool, man_pool_best_mmd, man_pool_all_mmds = search_man_pool_best(
        seed, proxy_target, n_proxy
    )
    scr_man_pool.save(os.path.join(seed_dir, 'scribble_man_pool_best.png'))
    print(f'      best_mmd={man_pool_best_mmd:.5f}  done in {fmt(time.time() - t_d2)}')

    wandb.log({
        f'seed_{seed}/scribbles/man_pool_best':      wandb.Image(scr_man_pool, caption='man pool best'),
        f'seed_{seed}/man_pool_search/best_mmd':     man_pool_best_mmd,
        f'seed_{seed}/man_pool_search/all_mmds':     wandb.Histogram(man_pool_all_mmds),
    }, step=seed)

    # [E] Evaluate all conditions
    run_conditions = {
        'avg_scribble':           src['avg_scribble'],
        'sdedit_scribble':        scr_sdedit,
        'random_best_scribble':   scr_rand,
        'man_pool_best_scribble': scr_man_pool,
        'anchor_scribble':        anchor_scr,
    }

    seed_results  = {'rand_all_mmds': rand_all_mmds}
    seed_wandb_log = {}

    for cond_name, scribble in run_conditions.items():
        t_e = time.time()
        print(f'  [E] [{cond_name}]...')
        imgs, embs, mmd, _ = eval_scribble(scribble, proxy_target, n_proxy, seed=seed)
        seed_results[cond_name] = {'mmd': mmd, 'time_s': round(time.time() - t_e, 1)}

        record = {'seed': seed, 'condition': cond_name, 'mmd': mmd}
        if exp['eval_type'] == 'binary_gender':
            n_male, pct = classify_binary_gender(imgs)
            record['pct_male'] = pct
            print(f'      MMD={mmd:.5f}  male={pct:.1f}%')
            seed_wandb_log[f'seed_{seed}/{cond_name}/pct_male'] = pct
        else:
            print(f'      MMD={mmd:.5f}')

        seed_wandb_log[f'seed_{seed}/{cond_name}/mmd'] = mmd

        # Log a sample of generated images every 5 seeds
        if seed % 5 == 1:
            seed_wandb_log[f'seed_{seed}/{cond_name}/sample_images'] = wandb_images(
                imgs[:4], [f'{cond_name} {i+1}' for i in range(4)]
            )

        all_records.append(record)

        if cond_name in CONDITION_NAMES and mmd < best_per_method[cond_name]['mmd']:
            best_per_method[cond_name].update(
                {'mmd': mmd, 'seed': seed, 'scribble': scribble.copy()}
            )

    # Timing
    t_total = time.time() - t_seed_start
    seed_results['total_time_s'] = round(t_total, 1)
    seed_times[seed]             = t_total
    seed_wandb_log[f'seed_{seed}/time_min'] = t_total / 60
    print(f'  SEED {seed} total: {fmt(t_total)}')

    wandb.log(seed_wandb_log, step=seed)

    with open(os.path.join(seed_dir, 'results.json'), 'w') as f:
        json.dump(to_json(seed_results), f, indent=2)

print(f'\nAll seeds done.  Mean: {fmt(np.mean(list(seed_times.values())))}')

# ===========================================================================
# 13. Save best scribbles
# ===========================================================================

best_dir_exp = os.path.join(BEST_DIR, exp['name'])
os.makedirs(best_dir_exp, exist_ok=True)

print('\nBest scribbles:')
best_scr_wandb = {}
for cond, info in best_per_method.items():
    if info['scribble'] is not None:
        path = os.path.join(best_dir_exp, f'best_{cond}.png')
        info['scribble'].save(path)
        best_scr_wandb[f'best_scribbles/{cond}'] = wandb.Image(
            info['scribble'],
            caption=f'{cond}  seed={info["seed"]}  proxy_mmd={info["mmd"]:.5f}'
        )
        print(f'  {cond:<28}  seed={info["seed"]}  proxy_mmd={info["mmd"]:.5f}')
wandb.log(best_scr_wandb)

# ===========================================================================
# 14. Final evaluation (N=2000)
# ===========================================================================

FINAL_SEED = 9999
print(f'\n{"="*60}')
print(f'  FINAL EVAL : {exp["name"]}')
print(f'{"="*60}')

src_final    = exp['build_source_fn'](FINAL_SEED)
final_anchor = src_final['anchor_hed_for_target']
final_anchor.save(os.path.join(best_dir_exp, 'final_anchor_scribble.png'))

print(f'  Building final target ({N_FINAL} images)...')
final_target = exp['build_target_fn'](final_anchor, FINAL_SEED, N_FINAL)
print(f'  Target shape: {final_target.shape}')

final_results_exp = {}
final_wandb_log   = {}

for cond in CONDITION_NAMES:
    info = best_per_method[cond]
    if info['scribble'] is None:
        print(f'  [{cond}] skipped')
        continue

    t0 = time.time()
    print(f'\n  [{cond}]  generating {N_FINAL} images...')
    imgs, embs, mmd, swd = eval_scribble(info['scribble'], final_target, N_FINAL, seed=FINAL_SEED)

    result = {
        'mmd': mmd, 'swd': swd,
        'best_seed': info['seed'],
        'time_s': round(time.time() - t0, 1),
    }

    final_wandb_log[f'final/{cond}/mmd'] = mmd
    final_wandb_log[f'final/{cond}/swd'] = swd
    final_wandb_log[f'final/{cond}/sample_images'] = wandb_images(
        imgs[:8], [f'{cond} {i+1}' for i in range(8)]
    )

    if exp['eval_type'] == 'binary_gender':
        enrich_binary_gender(result, imgs, N_FINAL)
        print(f'  MMD={mmd:.5f}  SWD={swd:.5f}  p(male)={result["p_male"]:.3f}  '
              f'95%CI=[{result["ci_lo_male"]:.3f},{result["ci_hi_male"]:.3f}]')
        final_wandb_log[f'final/{cond}/p_male']    = result['p_male']
        final_wandb_log[f'final/{cond}/ci_lo_male'] = result['ci_lo_male']
        final_wandb_log[f'final/{cond}/ci_hi_male'] = result['ci_hi_male']
    elif exp['eval_type'] == 'gender_interp':
        enrich_gender_interp(result, embs, N_FINAL, gender_interp_text_embs, gender_interp_labels)
        print(f'  MMD={mmd:.5f}  SWD={swd:.5f}')
        for lbl, p, lo, hi in zip(result['class_labels'], result['class_proportions'],
                                   result['class_ci_lo'], result['class_ci_hi']):
            print(f'    {lbl:<20} p={p:.3f}  95%CI=[{lo:.3f},{hi:.3f}]')
            final_wandb_log[f'final/{cond}/class_{lbl}/proportion'] = p
    elif exp['eval_type'] == 'age':
        enrich_age(result, embs, N_FINAL)
        print(f'  MMD={mmd:.5f}  SWD={swd:.5f}')
        for idx in [0, N_AGES // 2, -1]:
            lbl = result['class_labels'][idx]
            p   = result['class_proportions'][idx]
            print(f'    {lbl:<12} p={p:.3f}')
            final_wandb_log[f'final/{cond}/class_{lbl}/proportion'] = p

    # Base anchor comparison
    print(f'  Evaluating base anchor (seed {info["seed"]})...')
    base_anchor_path = os.path.join(
        CHECKPOINT_DIR, exp['name'], f'seed_{info["seed"]:02d}', 'scribble_anchor.png'
    )
    base_scr = Image.open(base_anchor_path)
    base_scr.save(os.path.join(best_dir_exp, f'base_anchor_for_{cond}.png'))

    base_imgs, base_embs, base_mmd, base_swd = eval_scribble(
        base_scr, final_target, N_FINAL, seed=FINAL_SEED
    )
    result['base_mmd']            = base_mmd
    result['base_swd']            = base_swd
    result['mmd_improvement']     = float(base_mmd - mmd)
    result['mmd_improvement_pct'] = float((base_mmd - mmd) / max(base_mmd, 1e-9) * 100)

    final_wandb_log[f'final/{cond}/base_mmd']            = base_mmd
    final_wandb_log[f'final/{cond}/mmd_improvement']     = result['mmd_improvement']
    final_wandb_log[f'final/{cond}/mmd_improvement_pct'] = result['mmd_improvement_pct']

    if exp['eval_type'] == 'binary_gender':
        base_res = {}
        enrich_binary_gender(base_res, base_imgs, N_FINAL)
        result['base_p_male']     = base_res['p_male']
        result['base_ci_lo_male'] = base_res['ci_lo_male']
        result['base_ci_hi_male'] = base_res['ci_hi_male']
        final_wandb_log[f'final/{cond}/base_p_male'] = base_res['p_male']

    print(f'  base_mmd={base_mmd:.5f}  '
          f'improvement={result["mmd_improvement"]:.5f} ({result["mmd_improvement_pct"]:.1f}%)')

    final_results_exp[cond] = result

wandb.log(final_wandb_log)

# ===========================================================================
# 15. Save JSON results
# ===========================================================================

per_seed_path     = os.path.join(SAVE_DIR, 'all_run_records.json')
final_path        = os.path.join(SAVE_DIR, 'final_results.json')
best_summary_path = os.path.join(SAVE_DIR, 'best_seeds_summary.json')
timing_path       = os.path.join(SAVE_DIR, 'timing.json')

with open(per_seed_path, 'w') as f:
    json.dump(to_json(all_records), f, indent=2)

with open(final_path, 'w') as f:
    json.dump(to_json(final_results_exp), f, indent=2)

with open(best_summary_path, 'w') as f:
    json.dump(
        {cond: {'seed': info['seed'], 'proxy_mmd': info['mmd']}
         for cond, info in best_per_method.items()}, f, indent=2)

with open(timing_path, 'w') as f:
    json.dump({
        'seed_times_s': seed_times,
        'mean_s':  float(np.mean(list(seed_times.values()))),
        'total_s': float(np.sum( list(seed_times.values()))),
    }, f, indent=2)

# Upload JSON to W&B as artifacts
artifact = wandb.Artifact(f'results_{args.exp}', type='results')
for p in [per_seed_path, final_path, best_summary_path, timing_path]:
    artifact.add_file(p)
    print(f'Saved → {p}')
wandb.log_artifact(artifact)

print(f'\nW&B run: {run.url}')
wandb.finish()
print('Done.')
