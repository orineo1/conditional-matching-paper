"""
eval_all_experiments.py — N-sample evaluation of every scribble method, on the cluster.

Scripted port of notebooks/eval_all_experiments.ipynb. For one experiment (or all):
    1. loads the method scribbles (source / avg / avg_latent / avg_prompt /
       sdedit_best / lgd_cm) written by scripts/eval_baselines.py
    2. builds the target CLIP distribution (N images from the source scribble
       with the group prompts, frac-weighted)
    3. generates N neutral-prompt images from every method scribble and scores
       MMD, root-MMD, SWD and (binary / multiclass) p(male) with a 95% CI
    4. saves previews (scribble grid, target samples, K images per method),
       a results table and a JSON of all metrics.

Run AFTER eval_baselines.py for the same experiment:

    python scripts/eval_all_experiments.py --experiment BalancedTarget --n_eval 2000
    python scripts/eval_all_experiments.py --experiment all --n_eval 2000

Scribbles are looked up in experiments/<Exp>/baselines/ first (where
eval_baselines.py writes them), then experiments/<Exp>/. Outputs go to
experiments/<Exp>/eval/ unless --output_dir is given.
"""

import os, sys, re, json, argparse
import numpy as np
import torch
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Make src/ and scripts/ importable when called from repo root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR    = os.path.join(os.path.dirname(_SCRIPT_DIR), "src")
for _d in (_SRC_DIR, _SCRIPT_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from eval_baselines import EXPERIMENT_CONFIGS  # same configs as the baseline builder

# ── Constants (match the notebook) ────────────────────────────────────────────

NEUTRAL_PROMPT = 'a superrealistic professional photograph of'
MAN_PROMPT     = 'a superrealistic portrait photograph of a man, studio lighting'
WOMAN_PROMPT   = 'a superrealistic portrait photograph of a woman, studio lighting'

SCRIBBLE_FILES = {
    'source':      'scribble_source.png',
    'avg':         'scribble_avg.png',
    'avg_latent':  'scribble_avg_latent.png',
    'avg_prompt':  'scribble_avg_prompt.png',
    'sdedit':      'scribble_sdedit.png',
    'sdedit_best': 'scribble_sdedit_best.png',
    'lgd_cm':      'scribble_mlgdd.png',
}
DEFAULT_METHODS = ['source', 'avg', 'avg_latent', 'avg_prompt', 'sdedit_best', 'lgd_cm']


def age_prompt(age):
    return (f'a superrealistic portrait photograph of a {age}-year-old man, '
            'studio lighting, sharp focus, photographic')


# ── Helpers ───────────────────────────────────────────────────────────────────

def gen_images(sprinter, scribble_pil, prompt, n, controlnet_scale, seed=None):
    sprinter.vae.to(dtype=torch.float16)
    generator = torch.Generator(device=sprinter.device).manual_seed(seed) if seed is not None else None
    imgs = []
    with torch.no_grad():
        for start in range(0, n, 2):
            bs = min(2, n - start)
            imgs.extend(sprinter(
                prompt=[prompt] * bs, image=[scribble_pil] * bs,
                num_inference_steps=2, guidance_scale=0.0,
                controlnet_conditioning_scale=controlnet_scale,
                output_type='pil', generator=generator,
            ).images)
    sprinter.vae.to(dtype=torch.float32)
    return imgs


def clip_embed(images, clip_model, clip_processor, device, batch_size=32):
    from clip_utils import encode_images_clip
    clip_model.to(device)
    all_embs = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch   = images[start:start + batch_size]
            tensors = torch.cat([TF.to_tensor(img).unsqueeze(0) for img in batch]).to(device)
            all_embs.append(encode_images_clip(tensors, clip_model, clip_processor).cpu())
            del tensors
    clip_model.to('cpu')
    if device == 'cuda':
        torch.cuda.empty_cache()
    return torch.cat(all_embs, dim=0).to(device)


def binomial_ci(n_success, n_total, z=1.96):
    p  = n_success / n_total
    se = np.sqrt(p * (1 - p) / n_total)
    return p, p - z * se, p + z * se


def _as_float(v):
    """mmd_report's value as a number (works whether it returns a number or a string with one)."""
    if isinstance(v, (int, float)):
        return float(v)
    if torch.is_tensor(v):
        return float(v.item())
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', str(v))
    return float(m.group()) if m else float('nan')


def to_json(obj):
    if isinstance(obj, dict):  return {k: to_json(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [to_json(v) for v in obj]
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    return obj


def find_scribble(run_dir, name):
    """Prefer experiments/<Exp>/baselines/ (eval_baselines.py output), then experiments/<Exp>/."""
    fname = SCRIBBLE_FILES[name]
    for d in (run_dir / 'baselines', run_dir):
        p = d / fname
        if p.exists():
            return p
    return None


def save_image_row(images, title, out_path, n=None):
    n_show = min(n or len(images), len(images))
    fig, axes = plt.subplots(1, n_show, figsize=(3 * n_show, 3))
    if n_show == 1: axes = [axes]
    for ax, img in zip(axes, images[:n_show]):
        ax.imshow(img); ax.axis('off')
    fig.suptitle(title, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out_path.name}')


def save_scribble_grid(scribbles, title, out_path):
    fig, axes = plt.subplots(1, len(scribbles), figsize=(4 * len(scribbles), 4))
    if len(scribbles) == 1: axes = [axes]
    for ax, (name, scr) in zip(axes, scribbles.items()):
        ax.imshow(scr, cmap='gray'); ax.set_title(name); ax.axis('off')
    plt.suptitle(title, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out_path.name}')


# ── One experiment ────────────────────────────────────────────────────────────

def evaluate_experiment(exp_name, args, models, wandb_run=None):
    from metrics import compute_mmd, compute_swd, mmd_report
    from run_mlgd_f import compute_clip_softmax

    sprinter, clip_model, clip_processor, device = models
    cfg    = EXPERIMENT_CONFIGS[exp_name]
    mode   = cfg['mode']
    SEED   = args.seed
    N_EVAL = args.n_eval
    cn     = args.controlnet_scale

    repo_path = Path(args.repo_path).resolve()
    run_dir   = repo_path / 'experiments' / exp_name
    out_dir   = Path(args.output_dir) if args.output_dir and args.experiment != 'all' \
                else run_dir / 'eval'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"=" * 60}\n  EXPERIMENT: {exp_name}  (N={N_EVAL}, seed={SEED})\n{"=" * 60}')
    torch.manual_seed(SEED); np.random.seed(SEED)   # makes SWD's random projections repeatable

    # ── Load scribbles ──
    scribbles, scribble_paths = {}, {}
    for name in args.methods:
        p = find_scribble(run_dir, name)
        if p is None:
            msg = f'{exp_name}: scribble for "{name}" not found ({SCRIBBLE_FILES[name]} under {run_dir}/baselines or {run_dir})'
            if args.skip_missing:
                print(f'  ⚠️  {msg} -- skipping method'); continue
            raise FileNotFoundError(msg)
        scribbles[name] = Image.open(p)
        scribble_paths[name] = str(p)
    if 'source' not in scribbles:
        raise FileNotFoundError(f'{exp_name}: the "source" scribble is required')
    source_scribble = scribbles['source']
    print('Scribbles: ' + ', '.join(f'{k} <- {Path(v).relative_to(repo_path)}' for k, v in scribble_paths.items()))
    save_scribble_grid(scribbles, f'{exp_name} — scribbles', out_dir / 'scribbles_all.png')

    # ── Target distribution (N_EVAL) ──
    print(f'\nBuilding target distribution (N={N_EVAL})...')
    target_preview = {}
    if mode in ('binary', 'multiclass'):
        all_target_imgs = []
        for i, g in enumerate(cfg['groups']):
            n_i = max(1, int(N_EVAL * g['frac']))
            print(f"  [{g['label']}] n={n_i}...")
            imgs = gen_images(sprinter, source_scribble, g['prompt'], n_i, cn, seed=SEED + i * 1000)
            all_target_imgs.extend(imgs)
            target_preview[g['label']] = imgs[:args.n_preview]
        target_clip = clip_embed(all_target_imgs, clip_model, clip_processor, device)
        del all_target_imgs
    elif mode == 'age':
        from clip_utils import encode_images_clip
        ages      = list(range(cfg['age_min'], cfg['age_max'] + 1, cfg['age_step']))
        n_per_age = max(1, N_EVAL // len(ages))
        age_embs  = {}
        clip_model.to(device)
        for age in tqdm(ages, desc='Age target'):
            imgs    = gen_images(sprinter, source_scribble, age_prompt(age), n_per_age, cn, seed=SEED + age * 7)
            tensors = torch.cat([TF.to_tensor(img).unsqueeze(0) for img in imgs]).to(device)
            with torch.no_grad():
                age_embs[age] = encode_images_clip(tensors, clip_model, clip_processor).cpu()
            del tensors
        clip_model.to('cpu')
        target_clip = torch.cat([age_embs[a] for a in ages], dim=0).to(device)
        # preview row: a few anchor ages, same seed so only the age changes
        n_anchor = min(args.n_preview, len(ages))
        anchors  = [ages[round(j * (len(ages) - 1) / max(1, n_anchor - 1))] for j in range(n_anchor)]
        target_preview['ages_' + '_'.join(map(str, anchors))] = [
            gen_images(sprinter, source_scribble, age_prompt(a), 1, cn, seed=SEED)[0] for a in anchors]
    else:
        raise ValueError(f'unknown mode {mode}')
    print(f'Target CLIP: {tuple(target_clip.shape)}')

    for label, imgs in target_preview.items():
        safe = label.replace(' ', '_').replace('/', '_')
        save_image_row(imgs, f'{exp_name} — target: {label}', out_dir / f'target_{safe}.png')

    # ── Evaluate each method ──
    print('\nEvaluating methods...')
    exp_results = {}
    for method_name, scribble in scribbles.items():
        print(f'  [{method_name}] generating {N_EVAL} images...')
        imgs = gen_images(sprinter, scribble, NEUTRAL_PROMPT, N_EVAL, cn, seed=SEED)
        save_image_row(imgs[:args.n_preview], f'{exp_name} — {method_name} (seed={SEED})',
                       out_dir / f'preview_{method_name}.png')
        embs     = clip_embed(imgs, clip_model, clip_processor, device)
        mmd      = compute_mmd(embs, target_clip).item()
        root_mmd = _as_float(mmd_report(mmd))          # square root of the MMD
        swd      = compute_swd(embs, target_clip).item()  # independent divergence, not the objective
        result   = {'mmd': mmd, 'root_mmd': root_mmd, 'swd': swd, 'scribble_path': scribble_paths[method_name]}

        if mode in ('binary', 'multiclass'):
            sr, _ = compute_clip_softmax(imgs, clip_model, clip_processor, MAN_PROMPT, WOMAN_PROMPT, device)
            n_male = sum(1 for r in sr if r['label'] == 'male')
            p, lo, hi = binomial_ci(n_male, N_EVAL)
            result.update(dict(p_male=p, ci_lo=lo, ci_hi=hi, n_male=n_male, n_female=N_EVAL - n_male))
            print(f'    MMD={mmd:.5f}  root-MMD={root_mmd:.5f}  SWD={swd:.5f}  '
                  f'p(male)={p:.3f}  95%CI=[{lo:.3f},{hi:.3f}]')
        else:
            print(f'    MMD={mmd:.5f}  root-MMD={root_mmd:.5f}  SWD={swd:.5f}')

        exp_results[method_name] = result
        del imgs, embs
        if device == 'cuda':
            torch.cuda.empty_cache()

    # ── Improvement vs source ──
    src = exp_results['source']
    for r in exp_results.values():
        for key in ('mmd', 'root_mmd', 'swd'):
            r[f'{key}_improvement']     = src[key] - r[key]
            r[f'{key}_improvement_pct'] = (src[key] - r[key]) / src[key] * 100

    # ── Results table ──
    rows = {}
    for name, r in exp_results.items():
        row = {
            'MMD':                  f"{r['mmd']:.5f}",
            'Δ MMD vs source':      f"{r['mmd_improvement']:+.5f}",
            'Δ MMD (%)':            f"{r['mmd_improvement_pct']:+.1f}%",
            'root-MMD':             f"{r['root_mmd']:.5f}",
            'Δ root-MMD vs source': f"{r['root_mmd_improvement']:+.5f}",
            'Δ root-MMD (%)':       f"{r['root_mmd_improvement_pct']:+.1f}%",
            'SWD':                  f"{r['swd']:.5f}",
            'Δ SWD vs source':      f"{r['swd_improvement']:+.5f}",
            'Δ SWD (%)':            f"{r['swd_improvement_pct']:+.1f}%",
        }
        if mode in ('binary', 'multiclass'):
            row['p(male)'] = f"{r['p_male']:.3f}"
            row['95% CI']  = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
            row['n_male / n_female'] = f"{r['n_male']} / {r['n_female']}"
        rows[name] = row
    df = pd.DataFrame(rows).T
    print(f'\n{exp_name} Results:\n{df.to_string()}')
    df.to_csv(out_dir / 'results_table.csv')

    # ── Save JSON ──
    meta = dict(experiment=exp_name, mode=mode, n_eval=N_EVAL, seed=SEED,
                controlnet_scale=cn, neutral_prompt=NEUTRAL_PROMPT,
                methods=list(exp_results.keys()), results=exp_results)
    with open(out_dir / 'eval_results.json', 'w') as f:
        json.dump(to_json(meta), f, indent=2)
    print(f'  Saved eval_results.json  ->  {out_dir}')

    # ── W&B ──
    if wandb_run is not None:
        import wandb
        log = {f'{exp_name}/scribble_{k}': wandb.Image(v) for k, v in scribbles.items()}
        for name, r in exp_results.items():
            for key in ('mmd', 'root_mmd', 'swd', 'p_male', 'mmd_improvement_pct', 'swd_improvement_pct'):
                if key in r:
                    log[f'{exp_name}/{name}/{key}'] = r[key]
        for png in sorted(out_dir.glob('*.png')):
            log[f'{exp_name}/fig/{png.stem}'] = wandb.Image(str(png))
        log[f'{exp_name}/results_table'] = wandb.Table(dataframe=df.reset_index().rename(columns={'index': 'method'}))
        wandb.log(log)

    return exp_results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', required=True, choices=list(EXPERIMENT_CONFIGS.keys()) + ['all'])
    parser.add_argument('--n_eval',     type=int, default=2000, help='images per method and for the target')
    parser.add_argument('--n_preview',  type=int, default=5,    help='images per preview row')
    parser.add_argument('--seed',       type=int, default=2025, help='eval seed (notebook default)')
    parser.add_argument('--controlnet_scale', type=float, default=0.5,
                        help='must match the scale used by eval_baselines.py / run_mlgd_f.py')
    parser.add_argument('--methods',    nargs='+', default=DEFAULT_METHODS, choices=list(SCRIBBLE_FILES.keys()),
                        help='which scribbles to evaluate ("source" is required)')
    parser.add_argument('--skip_missing', action='store_true',
                        help='skip a method whose scribble PNG is missing instead of failing')
    parser.add_argument('--repo_path',  type=str, default='.', help='repo root containing experiments/')
    parser.add_argument('--output_dir', type=str, default='',
                        help='where to write outputs (single experiment only; default experiments/<Exp>/eval)')
    parser.add_argument('--wandb_project', type=str, default='', help='leave empty to disable W&B')
    parser.add_argument('--wandb_entity',  type=str, default='')
    args = parser.parse_args()

    if 'source' not in args.methods:
        args.methods = ['source'] + list(args.methods)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    experiments = list(EXPERIMENT_CONFIGS.keys()) if args.experiment == 'all' else [args.experiment]

    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity or None,
                               name=f'eval_{args.experiment}_N{args.n_eval}', config=vars(args))

    from models     import load_models
    from clip_utils import load_clip_model
    _, sprinter                = load_models(device)
    clip_model, clip_processor = load_clip_model(device)
    print('Models loaded.')
    models = (sprinter, clip_model, clip_processor, device)

    all_results = {}
    for exp_name in experiments:
        all_results[exp_name] = evaluate_experiment(exp_name, args, models, wandb_run)

    # ── Summary + aggregate JSON (same layout as the notebook's eval_all_results.json) ──
    print('\n=== SUMMARY ===')
    for exp_name, exp_results in all_results.items():
        print(f'\n{exp_name}:')
        for method, r in exp_results.items():
            line = (f'  {method:<12} MMD={r["mmd"]:.5f} (Δ={r["mmd_improvement_pct"]:+.1f}%)'
                    f'  SWD={r["swd"]:.5f} (Δ={r["swd_improvement_pct"]:+.1f}%)')
            if 'p_male' in r:
                line += f'  p(male)={r["p_male"]:.3f}'
            print(line)

    if args.experiment == 'all':
        agg = Path(args.repo_path).resolve() / 'experiments' / 'eval_all_results.json'
        with open(agg, 'w') as f:
            json.dump(to_json(all_results), f, indent=2)
        print(f'\nAggregate results saved to {agg}')

    if wandb_run is not None:
        wandb_run.finish()
    print('\n✅ All experiments done.')


if __name__ == '__main__':
    main()
