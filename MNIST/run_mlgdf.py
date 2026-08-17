"""
run_mlgdf.py
==============
Single run for unimodal / bimodal / uniform experiments.
All hyperparameters passed via CLI — no grid search.

Usage:
    python run_mlgdf.py --experiment unimodal \
        --unimodal_var 515 --num_inference_steps 130 \
        --step_size_mode double --num_x_t 3 --nsamples 1500 --clamp

    python run_mlgdf.py --experiment bimodal \
        --bimodal_var 252 --num_inference_steps 125 \
        --step_size_mode original --num_x_t 10 --nsamples 1500 --clamp

    python run_mlgdf.py --experiment uniform \
        --num_inference_steps 290 \
        --step_size_mode original --num_x_t 3 --nsamples 600 --clamp

Train classifier only:
    python run_mlgdf.py --train_classifier_only
"""

import os, sys, math, argparse, random, time, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.distributions import Categorical, MultivariateNormal, MixtureSameFamily
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT = os.environ.get("REPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MNIST_SRC = os.path.join(REPO_ROOT, "MNIST", "src")
for p in [REPO_ROOT, MNIST_SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

from cond_model   import CircularAngleConsistencyModel, angles_to_circular, circular_to_angles
from uncond_model import UnconditionalUnet
from diffusers    import DDPMScheduler, DDIMScheduler
from huggingface_hub import hf_hub_download

# ─────────────────────────────────────────────────────────────────────────────
# Fixed
# ─────────────────────────────────────────────────────────────────────────────
N_SEEDS     = 15
GLOBAL_SEED = 42

# HF_TOKEN is read from the environment variable HF_TOKEN (set in your shell or .env).
# Do NOT hardcode tokens here.
HF_TOKEN   = os.environ.get("HF_TOKEN", "")
HF_REPO_ID = "anon-submission-cdm/cdm-inverse-design"

CLF_PATH  = os.path.join(REPO_ROOT, "MNIST", "checkpoints", "robust_classifier.pth")
NORM_MEAN = 0.1307
NORM_STD  = 0.3081

UNIMODAL_DIGITS_OF_INTEREST = [2, 3, 5, 7]
BIMODAL_DIGITS_OF_INTEREST  = [1, 6, 8, 9]

# ─────────────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────────────
def make_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────
class AddGaussianNoise:
    def __init__(self, mean=0., std=0.15):
        self.mean = mean
        self.std  = std
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dropout  = nn.Dropout2d(0.1)
        self.conv1          = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1            = nn.BatchNorm2d(32)
        self.dropout_mid1   = nn.Dropout2d(0.1)
        self.conv2          = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2            = nn.BatchNorm2d(32)
        self.pool1          = nn.MaxPool2d(2)
        self.dropout1       = nn.Dropout2d(0.25)
        self.conv3          = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3            = nn.BatchNorm2d(64)
        self.dropout_mid2   = nn.Dropout2d(0.1)
        self.conv4          = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4            = nn.BatchNorm2d(64)
        self.pool2          = nn.MaxPool2d(2)
        self.dropout2       = nn.Dropout2d(0.25)
        self.conv5          = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5            = nn.BatchNorm2d(128)
        self.pool3          = nn.MaxPool2d(2)
        self.dropout3       = nn.Dropout2d(0.25)
        self.fc1            = nn.Linear(128 * 3 * 3, 256)
        self.bn_fc1         = nn.BatchNorm1d(256)
        self.dropout_fc1    = nn.Dropout(0.5)
        self.fc2            = nn.Linear(256, 128)
        self.bn_fc2         = nn.BatchNorm1d(128)
        self.dropout_fc2    = nn.Dropout(0.5)
        self.fc3            = nn.Linear(128, 10)

    def forward(self, x):
        x = self.input_dropout(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout_mid1(x)
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout_mid2(x)
        x = self.pool2(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout2(x)
        x = self.pool3(F.relu(self.bn5(self.conv5(x))))
        x = self.dropout3(x)
        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout_fc1(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout_fc2(F.relu(self.bn_fc2(self.fc2(x))))
        return self.fc3(x)


def train_classifier(device):
    make_deterministic(42)
    g = torch.Generator()
    g.manual_seed(42)
    train_tf = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        AddGaussianNoise(0., 0.15),
        transforms.Normalize((NORM_MEAN,), (NORM_STD,)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((NORM_MEAN,), (NORM_STD,)),
    ])
    train_loader = DataLoader(
        datasets.MNIST('./data', train=True,  download=True, transform=train_tf),
        batch_size=128, shuffle=False, num_workers=0, generator=g,
    )
    test_loader = DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=test_tf),
        batch_size=128, shuffle=False, num_workers=0,
    )
    model     = ImprovedCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc  = 0.0
    print(f"Training classifier ({sum(p.numel() for p in model.parameters()):,} params)...")
    for epoch in range(1, 11):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            criterion(model(data), target).backward()
            optimizer.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                correct += model(data).argmax(1).eq(target).sum().item()
        acc = 100. * correct / len(test_loader.dataset)
        print(f"  Epoch {epoch:2d}/10 | Test {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            os.makedirs(os.path.dirname(CLF_PATH), exist_ok=True)
            torch.save(model.state_dict(), CLF_PATH)
            print(f"    → Saved ({best_acc:.2f}%)")
    model.load_state_dict(torch.load(CLF_PATH, map_location=device))
    model.eval()
    print(f"Classifier ready — best: {best_acc:.2f}%")
    return model


def load_or_train_classifier(device):
    model = ImprovedCNN().to(device)
    if os.path.exists(CLF_PATH):
        print(f"Loading classifier from {CLF_PATH}")
        model.load_state_dict(torch.load(CLF_PATH, map_location=device))
        model.eval()
        return model
    print(f"No classifier found at {CLF_PATH} — training...")
    return train_classifier(device)


def classify_generated_images(images_np, classifier, device, threshold=0.7):
    classifier.eval()
    preds, confs = [], []
    with torch.no_grad():
        for img in images_np:
            t      = torch.tensor(img, dtype=torch.float32).flatten()
            lo, hi = t.min(), t.max()
            t      = (t - lo) / (hi - lo + 1e-8)
            t_norm = (t - NORM_MEAN) / NORM_STD
            inp    = t_norm.reshape(1, 1, 28, 28).to(device)
            probs  = F.softmax(classifier(inp), dim=1).squeeze()
            conf, pred = probs.max(0)
            conf_val = conf.item()
            confs.append(conf_val)
            preds.append(pred.item() if conf_val > threshold else None)
    return preds, confs

# ─────────────────────────────────────────────────────────────────────────────
# MoG helpers
# ─────────────────────────────────────────────────────────────────────────────
def mog_pdf(x, means, variances, weights=None):
    components = len(means)
    if weights is None:
        weights = torch.ones(components) / components
    pdf = torch.zeros_like(x)
    for mean, var, weight in zip(means, variances, weights):
        var_t  = torch.tensor(var,  dtype=torch.float32)
        mean_t = torch.tensor(mean, dtype=torch.float32)
        diff   = (x - mean_t + 180) % 360 - 180
        pdf   += weight * torch.exp(-0.5 * diff**2 / var_t) / (
            torch.sqrt(2 * torch.pi * var_t))
    return pdf


def generate_mog_samples(num_samples, means, variances, weights=None, device='cpu'):
    components = len(means)
    if weights is None:
        weights = torch.ones(components, device=device) / components
    else:
        weights = weights.to(device)
    weights = weights / weights.sum()
    means_t = torch.stack([m.flatten() for m in means]).to(device)
    covs_t  = torch.stack([torch.diag(v.flatten()) for v in variances]).to(device)
    mix  = Categorical(weights)
    comp = MultivariateNormal(means_t, covs_t)
    return MixtureSameFamily(mix, comp).sample((num_samples,))


def sliced_wasserstein_distance(X, Y, n_projections=50, device='cpu'):
    X    = X.to(device).float()
    Y    = Y.to(device).float()
    proj = torch.randn(n_projections, X.shape[1], device=device)
    proj = proj / torch.norm(proj, dim=1, keepdim=True)
    return torch.mean(torch.abs(
        torch.sort(X @ proj.T, dim=0)[0] - torch.sort(Y @ proj.T, dim=0)[0]
    ))


def _build_target_mog(mog_means, mog_variances, weights):
    x_range    = torch.linspace(0, 360, 200)
    target_pdf = mog_pdf(
        x_range,
        [m.item() for m in mog_means],
        [v.squeeze().item() for v in mog_variances],
        weights,
    )
    return x_range.numpy(), target_pdf.numpy()

# ─────────────────────────────────────────────────────────────────────────────
# Step-size helper
# ─────────────────────────────────────────────────────────────────────────────
def compute_step_size(r_t, t, mode):
    if mode == 'original':
        return r_t / (1 + r_t**2) + 5 * t / 1000
    elif mode == 'dps':
        return 1.0 / (r_t + 1e-8) * 0.1
    elif mode == 'half':
        return 0.5 * (r_t / (1 + r_t**2) + 5 * t / 1000)
    elif mode == 'double':
        return 2.0 * (r_t / (1 + r_t**2) + 5 * t / 1000)
    elif mode == 'tripleLinear':
        return r_t / (1 + r_t**2) + 15 * t / 1000
    elif mode == 'doubleLinear':
        return r_t / (1 + r_t**2) + 10 * t / 1000
    elif mode == 'no_linear':
        return r_t / (1 + r_t**2)
    else:
        raise ValueError(f"Unknown step_size_mode: {mode}")

# ─────────────────────────────────────────────────────────────────────────────
# Core optimization loop  (shared by all three experiments)
# ─────────────────────────────────────────────────────────────────────────────
def optimize_LGD(model_uncond, model_cond_cm, noise_scheduler,
                 nsamples, num_x_t, num_inference_steps,
                 step_size_mode, device, seed=None, clamp=False,
                 mog_means=None, mog_variances=None, weights=None,
                 tfg_n_iter=0, tfg_mu=0.0, use_variance_guidance=True):
    """
    Unified optimization loop for uniform, unimodal, and bimodal targets.
    If mog_means is None, uses uniform target.

    tfg_n_iter, tfg_mu: TFG (Algorithm 1) Mean Guidance hyper-parameters
    (N_iter, mu). Defaults (0, 0.0) reproduce plain LGD exactly.

    use_variance_guidance: if False, disables Line 7 (Delta_t = rho_t *
    grad_{x_t} log f(x0|t)) entirely — no gradient is taken w.r.t. x_t,
    and only TFG Mean Guidance (Delta_0) drives the correction.
    """
    if seed is not None:
        set_seed(seed)

    ddim = DDIMScheduler.from_config(noise_scheduler.config)
    ddim.set_timesteps(num_inference_steps=num_inference_steps)
    timesteps = ddim.timesteps

    x_t = torch.randn(1, 1, 28, 28, device=device, requires_grad=True)

    for i, t in enumerate(timesteps[:-1]):
        x_t      = x_t.detach().clone().requires_grad_(use_variance_guidance)
        residual = model_uncond(x_t, torch.tensor([t], device=device))
        alpha_t      = ddim.alphas_cumprod[t]
        alpha_t_prev = (ddim.alphas_cumprod[timesteps[i+1]]
                        if i < len(timesteps) - 2 else torch.tensor(1.0))
        beta_t       = 1 - alpha_t
        pred_x0      = (x_t - beta_t**0.5 * residual) / alpha_t**0.5
        x_t_minus_1  = alpha_t_prev**0.5 * pred_x0 + (1 - alpha_t_prev)**0.5 * residual

        r_t       = torch.sqrt(beta_t)
        step_size = compute_step_size(r_t, t, step_size_mode)

        grad = torch.zeros_like(x_t)
        if use_variance_guidance:
            losses = []
            for _ in range(num_x_t):
                x0_sample     = pred_x0 + r_t**2 * torch.randn_like(pred_x0)
                target_angles = circular_to_angles(
                    model_cond_cm.sample(nsamples=nsamples, condition_x=x0_sample,
                                         ts=[150., 50., 20., 10., 5., 1.])[0]
                )
                if mog_means is None:
                    ref_ang = torch.rand(nsamples, device=device) * 360.0
                else:
                    ref_ang = generate_mog_samples(
                        nsamples, mog_means, mog_variances, weights).squeeze()

                losses.append(-sliced_wasserstein_distance(
                    angles_to_circular(target_angles),
                    angles_to_circular(ref_ang),
                    n_projections=50, device=device,
                ))

            log_me = -torch.logsumexp(torch.stack(losses), dim=0) + math.log(num_x_t)
            grad   = torch.autograd.grad(log_me, x_t, retain_graph=True)[0]

        # TFG Mean Guidance (Algorithm 1, Line 8): Delta_0 = Delta_0 + mu_t * grad_{x0|t} log f(x0|t + Delta_0)
        # Re-samples y ~ p(y | x0|t + Delta_0) at every inner iteration, since the
        # conditioning point moves as Delta_0 is updated.
        delta0 = torch.zeros_like(pred_x0)
        for _ in range(tfg_n_iter):
            delta0 = delta0.detach().requires_grad_(True)
            x0_plus_delta = pred_x0.detach() + delta0

            target_angles = circular_to_angles(
                model_cond_cm.sample(nsamples=nsamples, condition_x=x0_plus_delta,
                                     ts=[150., 50., 20., 10., 5., 1.])[0]
            )
            if mog_means is None:
                ref_ang = torch.rand(nsamples, device=device) * 360.0
            else:
                ref_ang = generate_mog_samples(
                    nsamples, mog_means, mog_variances, weights).squeeze()

            loss_iter = -sliced_wasserstein_distance(
                angles_to_circular(target_angles),
                angles_to_circular(ref_ang),
                n_projections=50, device=device,
            )
            grad_delta = torch.autograd.grad(loss_iter, delta0)[0]
            delta0 = delta0.detach() - tfg_mu * grad_delta

        with torch.no_grad():
            if mog_means is None and t < 250:
                step_size = 0
            x_t = x_t_minus_1.detach().clone() - step_size * grad - alpha_t_prev**0.5 * delta0
            if clamp:
                x_t = x_t.clamp(-1.0, 1.0)

    # Final DDIM step
    with torch.no_grad():
        last_t = timesteps[-1]
        res    = model_uncond(x_t, torch.tensor([last_t], device=device))
        a      = ddim.alphas_cumprod[last_t]
        x_t    = (x_t - (1 - a)**0.5 * res) / a**0.5

    x_final = x_t.detach().clone()
    final_n = nsamples * 4

    final_ang = circular_to_angles(
        model_cond_cm.sample(nsamples=final_n,
                             condition_x=x_final.view(1, 28, 28),
                             ts=[150., 50., 20., 10., 5., 1.])[0]
    )
    if mog_means is None:
        ref_ang = torch.rand(final_n, device=device) * 360.0
    else:
        ref_ang = generate_mog_samples(
            final_n, mog_means, mog_variances, weights).squeeze()

    final_loss = sliced_wasserstein_distance(
        angles_to_circular(final_ang),
        angles_to_circular(ref_ang),
        n_projections=50, device=device,
    )

    if str(device) == 'cuda':
        torch.cuda.empty_cache()

    return x_final, final_loss

# ─────────────────────────────────────────────────────────────────────────────
# run_and_save
# ─────────────────────────────────────────────────────────────────────────────
def run_and_save(model_uncond, model_cond, noise_scheduler,
                 nsamples, num_x_t, num_inference_steps, step_size_mode,
                 experiment_name, save_dir,
                 clamp=False, seeds=range(15), device='cuda',
                 mog_means=None, mog_variances=None, weights=None,
                 tfg_n_iter=0, tfg_mu=0.0, use_variance_guidance=True):

    os.makedirs(save_dir, exist_ok=True)
    print(f'\n{"="*60}\n  EXPERIMENT: {experiment_name}\n{"="*60}')

    results, loss_log, seed_log, time_log = [], [], [], []

    for seed in seeds:
        print(f'[Seed {seed:2d}] optimizing...', end='  ', flush=True)
        t0 = time.time()
        x_final, loss = optimize_LGD(
            model_uncond        = model_uncond,
            model_cond_cm       = model_cond,
            noise_scheduler     = noise_scheduler,
            nsamples            = nsamples,
            num_x_t             = num_x_t,
            num_inference_steps = num_inference_steps,
            step_size_mode      = step_size_mode,
            device              = device,
            seed                = seed,
            clamp               = clamp,
            mog_means           = mog_means,
            mog_variances       = mog_variances,
            weights             = weights,
            tfg_n_iter          = tfg_n_iter,
            tfg_mu              = tfg_mu,
            use_variance_guidance = use_variance_guidance,
        )
        elapsed  = time.time() - t0
        loss_val = loss.item()
        results.append(x_final.squeeze().cpu().numpy())
        loss_log.append(loss_val)
        seed_log.append(seed)
        time_log.append(elapsed)
        print(f'loss = {loss_val:.4f}  time = {elapsed:.1f}s')

    if mog_means is not None:
        x_range_np, target_pdf_np = _build_target_mog(mog_means, mog_variances, weights)
        use_uniform = False
    else:
        x_range_np    = np.linspace(0, 360, 200)
        target_pdf_np = np.ones(200) / 360.0
        use_uniform   = True

    payload = {
        'experiment_name':     experiment_name,
        'nsamples':            nsamples,
        'num_x_t':             num_x_t,
        'num_inference_steps': num_inference_steps,
        'step_size_mode':      step_size_mode,
        'clamp':               clamp,
        'tfg_n_iter':          tfg_n_iter,
        'tfg_mu':              tfg_mu,
        'use_variance_guidance': use_variance_guidance,
        'results':             results,
        'loss_log':            loss_log,
        'seed_log':            seed_log,
        'time_log':            time_log,
        'x_range':             x_range_np,
        'target_pdf':          target_pdf_np,
        'use_uniform':         use_uniform,
    }

    save_path = os.path.join(save_dir, f'{experiment_name}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(payload, f)

    best_i = int(np.argmin(loss_log))
    print(f'Saved  → {save_path}')
    print(f'Best   → seed {seed_log[best_i]}, loss {loss_log[best_i]:.4f}')
    return save_path, payload

# ─────────────────────────────────────────────────────────────────────────────
# W&B logging + figures
# ─────────────────────────────────────────────────────────────────────────────
def log_results(payload, preds, confs, digits_of_interest, experiment, args):
    losses  = np.array(payload['loss_log'])
    top_ix  = np.argsort(losses)[:5]
    n       = len(payload['results'])

    nsamples            = payload['nsamples']
    num_inference_steps = payload['num_inference_steps']
    step_size_mode      = payload['step_size_mode']
    num_x_t             = payload['num_x_t']
    experiment_name     = payload['experiment_name']

    digit_counts     = {d: sum(1 for p in preds if p == d) for d in digits_of_interest}
    n_classified     = sum(1 for p in preds if p is not None)
    top_preds        = [preds[i] for i in top_ix]
    top_confs        = [confs[i] for i in top_ix]
    n_top_clf        = sum(1 for p in top_preds if p is not None)
    top5_conf_mean   = float(np.mean(top_confs))
    top_digit_counts = {d: sum(1 for p in top_preds if p == d) for d in digits_of_interest}

    print(f"\n--- Classification Summary ---")
    print(f"  Total classified  : {n_classified}/{n}")
    for d in digits_of_interest:
        print(f"  Digit {d}           : {digit_counts[d]}/{n}")
    print(f"  Top-5 classified  : {n_top_clf}/5")
    print(f"  Top-5 mean conf   : {top5_conf_mean:.3f}")

    # all seeds figure
    ncols = min(5, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 3))
    axes = np.array(axes).reshape(nrows, ncols)
    for i, (img, loss_val, seed, pred, conf) in enumerate(
            zip(payload['results'], payload['loss_log'],
                payload['seed_log'], preds, confs)):
        r, c = divmod(i, ncols)
        axes[r, c].imshow(img.reshape(28, 28), cmap='gray')
        pred_str = str(pred) if pred is not None else "?"
        axes[r, c].set_title(
            f's{seed} | swd={loss_val:.3f}\npred={pred_str} ({conf:.2f})',
            fontsize=6)
        axes[r, c].axis('off')
    for i in range(n, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].axis('off')
    plt.suptitle(f'{experiment} | {experiment_name}', fontsize=7)
    plt.tight_layout()
    fig_all = fig

    # top-5 figure
    k = len(top_ix)
    fig_top, axes_top = plt.subplots(1, k, figsize=(k * 2.5, 3))
    axes_top = np.array(axes_top).reshape(k)
    for rank, idx in enumerate(top_ix):
        axes_top[rank].imshow(payload['results'][idx].reshape(28, 28), cmap='gray')
        pred_str = str(preds[idx]) if preds[idx] is not None else "?"
        axes_top[rank].set_title(
            f'#{rank+1} | s{payload["seed_log"][idx]}\n'
            f'swd={losses[idx]:.3f} | pred={pred_str} ({confs[idx]:.2f})',
            fontsize=7)
        axes_top[rank].axis('off')
    plt.suptitle(f'{experiment} Top-5 | {experiment_name}', fontsize=7)
    plt.tight_layout()
    fig_top5 = fig_top

    # SWD per seed
    fig_loss, ax_loss = plt.subplots(figsize=(8, 4))
    ax_loss.plot(payload['seed_log'], payload['loss_log'],
                 marker='o', linewidth=1.5, color='steelblue')
    ax_loss.axhline(losses.mean(), color='red', linestyle='--', linewidth=1,
                    label=f'mean={losses.mean():.4f}')
    ax_loss.set_xlabel('Seed')
    ax_loss.set_ylabel('SWD loss')
    ax_loss.set_title(f'{experiment} SWD per seed | {experiment_name}')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    plt.tight_layout()

    # W&B
    log_dict = {
        "images/top5":            wandb.Image(fig_top5),
        "images/all_seeds":       wandb.Image(fig_all),
        "images/swd_per_seed":    wandb.Image(fig_loss),
        "swd/mean":               float(losses.mean()),
        "swd/std":                float(losses.std()),
        "swd/min":                float(losses.min()),
        "swd/top5_mean":          float(losses[top_ix].mean()),
        "clf/n_classified":       n_classified,
        "clf/pct_classified":     100. * n_classified / n,
        "clf/n_top5_classified":  n_top_clf,
        "clf/top5_conf_mean":     top5_conf_mean,
        "config/experiment":          experiment,
        "config/nsamples":            nsamples,
        "config/num_inference_steps": num_inference_steps,
        "config/step_size_mode":      step_size_mode,
        "config/num_x_t":             num_x_t,
        "config/clamp":               int(payload['clamp']),
        "config/tfg_n_iter":          payload['tfg_n_iter'],
        "config/tfg_mu":              payload['tfg_mu'],
        "config/use_variance_guidance": int(payload['use_variance_guidance']),
    }
    for d in digits_of_interest:
        log_dict[f"clf/digit_{d}"]      = digit_counts[d]
        log_dict[f"clf/top5_digit_{d}"] = top_digit_counts[d]

    wandb.log(log_dict)
    plt.close(fig_all)
    plt.close(fig_top5)
    plt.close(fig_loss)

# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", type=str, default="unimodal",
                   choices=["unimodal", "bimodal", "uniform"])
    p.add_argument("--num_inference_steps", type=int,   default=130)
    p.add_argument("--step_size_mode",      type=str,   default="double")
    p.add_argument("--num_x_t",             type=int,   default=3)
    p.add_argument("--nsamples",            type=int,   default=1500)
    p.add_argument("--clamp",               action="store_true", default=False)
    p.add_argument("--tfg_n_iter",          type=int,   default=0,
                   help="TFG Mean Guidance N_iter (Algorithm 1, Line 8). 0 = plain LGD.")
    p.add_argument("--tfg_mu",              type=float, default=0.0,
                   help="TFG Mean Guidance step size mu (Algorithm 1, Line 8).")
    p.add_argument("--disable_variance_guidance", action="store_true", default=False,
                   help="Disable Line 7 (Delta_t = rho_t grad_xt log f(x0|t)) entirely, "
                        "avoiding any gradient w.r.t. x_t; only Delta_0 (Mean Guidance) is used.")
    p.add_argument("--n_seeds",             type=int,   default=15)
    p.add_argument("--smoke_test",          action="store_true")
    p.add_argument("--unimodal_var",        type=float, default=515)
    p.add_argument("--bimodal_var",         type=float, default=252)
    p.add_argument("--wandb_entity",        type=str,   default="")
    p.add_argument("--wandb_mode",          type=str,   default="online",
                   choices=["online", "offline", "disabled"],
                   help="Set to 'disabled' to skip W&B logging entirely")
    p.add_argument("--train_classifier_only", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    make_deterministic(GLOBAL_SEED)

    if args.train_classifier_only:
        train_classifier(device)
        sys.exit(0)

    experiment = args.experiment
    if experiment == "unimodal":
        mog_means     = [torch.tensor([360], dtype=torch.float64)]
        mog_variances = [torch.tensor([[args.unimodal_var]], dtype=torch.float64)]
        weights       = torch.tensor([1.0], dtype=torch.float64)
        digits_of_interest = UNIMODAL_DIGITS_OF_INTEREST
        var_str = f"_var{int(args.unimodal_var)}"

    elif experiment == "bimodal":
        mog_means     = [torch.tensor([0],   dtype=torch.float64),
                         torch.tensor([180], dtype=torch.float64)]
        mog_variances = [torch.tensor([[args.bimodal_var]], dtype=torch.float64),
                         torch.tensor([[args.bimodal_var]], dtype=torch.float64)]
        weights       = torch.tensor([0.5, 0.5], dtype=torch.float64)
        digits_of_interest = BIMODAL_DIGITS_OF_INTEREST
        var_str = f"_var{int(args.bimodal_var)}"

    else:
        mog_means     = None
        mog_variances = None
        weights       = None
        digits_of_interest = list(range(10))
        var_str = ""

    clamp_str = "_cl1" if args.clamp else "_cl0"
    run_name  = (f"{experiment}{var_str}"
                 f"_st{args.num_inference_steps}"
                 f"_ss{args.step_size_mode}"
                 f"_xt{args.num_x_t}"
                 f"_ns{args.nsamples}"
                 f"{clamp_str}"
                 f"_tfgN{args.tfg_n_iter}"
                 f"_tfgMu{args.tfg_mu}"
                 f"_vg{0 if args.disable_variance_guidance else 1}")

    save_dir = os.path.join(REPO_ROOT, "MNIST", "results", f"{experiment}_run", run_name)

    print(f"\n{'='*65}")
    print(f"Experiment : {experiment}")
    print(f"Run name   : {run_name}")
    print(f"Device     : {device}")
    print(f"{'='*65}\n")

    os.environ["WANDB_MODE"] = args.wandb_mode
    wandb.init(
        project = f"mnist_{experiment}_run",
        entity  = args.wandb_entity or None,
        config  = {
            "experiment":          experiment,
            "num_inference_steps": args.num_inference_steps,
            "step_size_mode":      args.step_size_mode,
            "num_x_t":             args.num_x_t,
            "nsamples":            args.nsamples,
            "clamp":               args.clamp,
            "tfg_n_iter":          args.tfg_n_iter,
            "tfg_mu":              args.tfg_mu,
            "use_variance_guidance": not args.disable_variance_guidance,
            "n_seeds":             args.n_seeds,
            "global_seed":         GLOBAL_SEED,
            "smoke_test":          args.smoke_test,
            **({"unimodal_var": args.unimodal_var} if experiment == "unimodal" else {}),
            **({"bimodal_var":  args.bimodal_var}  if experiment == "bimodal"  else {}),
        },
        name   = run_name,
        tags   = ["mnist", experiment, "single-run"],
        reinit = "finish_previous",
    )

    # Load pretrained models from HuggingFace
    # Set HF_TOKEN in your environment: export HF_TOKEN=<your_token>
    print("Downloading conditional model...")
    cond_path = hf_hub_download(
        repo_id=HF_REPO_ID, filename="MnistConditional500Epoch.pt",
        subfolder="MNIST", token=HF_TOKEN or None)
    print("Downloading unconditional model...")
    uncond_path = hf_hub_download(
        repo_id=HF_REPO_ID, filename="MnistUncond100Epoch.pth",
        subfolder="MNIST", token=HF_TOKEN or None)

    cond_model = CircularAngleConsistencyModel(
        nfeatures=2, img_features=784, eps=0.002,
        nunits=128, depth=5, device=str(device))
    ckpt = torch.load(cond_path, map_location=device)
    cond_model.load_state_dict(ckpt['model_state_dict'])
    cond_model.eval()

    uncond_model = UnconditionalUnet().to(device)
    ckpt_u = torch.load(uncond_path, map_location=device)
    uncond_model.load_state_dict(ckpt_u['model_state_dict'])
    uncond_model.eval()

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
    print("Generative models ready ✓")

    classifier = load_or_train_classifier(device)

    seeds = range(2) if args.smoke_test else range(args.n_seeds)

    save_path, payload = run_and_save(
        model_uncond        = uncond_model,
        model_cond          = cond_model,
        noise_scheduler     = noise_scheduler,
        nsamples            = args.nsamples,
        num_x_t             = args.num_x_t,
        num_inference_steps = args.num_inference_steps,
        step_size_mode      = args.step_size_mode,
        experiment_name     = run_name,
        save_dir            = save_dir,
        clamp               = args.clamp,
        seeds               = seeds,
        device              = str(device),
        mog_means           = mog_means,
        mog_variances       = mog_variances,
        weights             = weights,
        tfg_n_iter          = args.tfg_n_iter,
        tfg_mu              = args.tfg_mu,
        use_variance_guidance = not args.disable_variance_guidance,
    )

    preds, confs = classify_generated_images(payload['results'], classifier, device)
    log_results(payload, preds, confs, digits_of_interest, experiment, args)

    artifact = wandb.Artifact(f'{experiment}_run_{run_name}', type='results')
    artifact.add_file(save_path)
    wandb.log_artifact(artifact)

    wandb.finish()
    print(f"\nDone: {run_name}")


if __name__ == "__main__":
    main()
