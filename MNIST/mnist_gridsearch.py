"""
mnist_gridsearch.py
===================
Grid search over num_inference_steps × num_x_t × nsamples × variance
for Bimodal and Unimodal MNIST rotation experiments.

Launch:
    sbatch mnist_gridsearch.sh
Smoke test:
    python mnist_gridsearch.py --config_id 0 --smoke_test
List configs:
    python mnist_gridsearch.py --list_configs
"""

import os, sys, math, json, argparse, itertools, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.distributions import Categorical, MultivariateNormal, MixtureSameFamily
import wandb

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT = os.environ.get(
    "REPO_ROOT",
    "/sci/labs/orzuk/ori_m/conditional-matching-paper"
)
MNIST_SRC = os.path.join(REPO_ROOT, "MNIST", "src")
for p in [REPO_ROOT, MNIST_SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

from cond_model   import CircularAngleConsistencyModel, angles_to_circular, circular_to_angles
from uncond_model import UnconditionalUnet
from diffusers    import DDPMScheduler, DDIMScheduler

# ─────────────────────────────────────────────────────────────────────────────
# Grid
# ─────────────────────────────────────────────────────────────────────────────
GRID = {
    "num_inference_steps": [100, 200, 300],
    "num_x_t":             [5, 10, 15],
    "nsamples":            [300, 600, 900],
    "bimodal_var":         [200, 245, 300],
    "unimodal_var":        [500, 620, 750],
}

N_SEEDS     = 15
TOP_K       = 5
GLOBAL_SEED = 42

# Base classifier threshold (strict)
BASE_THRESHOLD     = 0.85
# Improved classifier threshold (permissive)
IMPROVED_THRESHOLD = 0.50
# TTA rotations for improved classifier
TTA_ANGLES = [0, 90, 180, 270]

RELEVANT_DIGITS = {
    "Bimodal":  {0, 1, 6, 8, 9},
    "Unimodal": {2, 3, 6, 7},   # added 6 — rotated 6 looks like valid upright
}


def all_configs():
    keys, values = zip(*GRID.items())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


# ─────────────────────────────────────────────────────────────────────────────
# Seed
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Classifier architectures
# ─────────────────────────────────────────────────────────────────────────────
class BaselineCNN(nn.Module):
    """Original three-block CNN — unchanged from notebook."""
    def __init__(self):
        super().__init__()
        self.conv1       = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1         = nn.BatchNorm2d(32)
        self.conv2       = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2         = nn.BatchNorm2d(32)
        self.pool1       = nn.MaxPool2d(2)
        self.dropout1    = nn.Dropout2d(0.25)
        self.conv3       = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3         = nn.BatchNorm2d(64)
        self.conv4       = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4         = nn.BatchNorm2d(64)
        self.pool2       = nn.MaxPool2d(2)
        self.dropout2    = nn.Dropout2d(0.25)
        self.conv5       = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5         = nn.BatchNorm2d(128)
        self.pool3       = nn.MaxPool2d(2)
        self.dropout3    = nn.Dropout2d(0.25)
        self.fc1         = nn.Linear(128 * 3 * 3, 256)
        self.bn_fc1      = nn.BatchNorm1d(256)
        self.dropout_fc1 = nn.Dropout(0.5)
        self.fc2         = nn.Linear(256, 128)
        self.bn_fc2      = nn.BatchNorm1d(128)
        self.dropout_fc2 = nn.Dropout(0.5)
        self.fc3         = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))))
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(x)))))))
        x = self.dropout2(x)
        x = self.pool3(F.relu(self.bn5(self.conv5(x))))
        x = self.dropout3(x)
        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout_fc1(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout_fc2(F.relu(self.bn_fc2(self.fc2(x))))
        return self.fc3(x)


class ImprovedCNN(nn.Module):
    """
    Deeper CNN trained with heavy augmentation:
    - random rotation up to 180 degrees
    - elastic distortion
    - cutout / random erasing
    - brightness / contrast jitter
    Handles rotated, noisy, and partially occluded digits.
    """
    def __init__(self):
        super().__init__()
        # Block 1
        self.conv1    = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1      = nn.BatchNorm2d(32)
        self.conv1b   = nn.Conv2d(32, 32, 3, padding=1)
        self.bn1b     = nn.BatchNorm2d(32)
        self.pool1    = nn.MaxPool2d(2)
        self.drop1    = nn.Dropout2d(0.25)
        # Block 2
        self.conv2    = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2      = nn.BatchNorm2d(64)
        self.conv2b   = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2b     = nn.BatchNorm2d(64)
        self.pool2    = nn.MaxPool2d(2)
        self.drop2    = nn.Dropout2d(0.25)
        # Block 3
        self.conv3    = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3      = nn.BatchNorm2d(128)
        self.conv3b   = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3b     = nn.BatchNorm2d(128)
        self.pool3    = nn.MaxPool2d(2)
        self.drop3    = nn.Dropout2d(0.25)
        # FC
        self.fc1      = nn.Linear(128 * 3 * 3, 512)
        self.bn_fc1   = nn.BatchNorm1d(512)
        self.drop_fc1 = nn.Dropout(0.5)
        self.fc2      = nn.Linear(512, 256)
        self.bn_fc2   = nn.BatchNorm1d(256)
        self.drop_fc2 = nn.Dropout(0.5)
        self.fc3      = nn.Linear(256, 10)

    def forward(self, x):
        x = self.drop1(self.pool1(F.relu(self.bn1b(self.conv1b(
            F.relu(self.bn1(self.conv1(x))))))))
        x = self.drop2(self.pool2(F.relu(self.bn2b(self.conv2b(
            F.relu(self.bn2(self.conv2(x))))))))
        x = self.drop3(self.pool3(F.relu(self.bn3b(self.conv3b(
            F.relu(self.bn3(self.conv3(x))))))))
        x = x.view(-1, 128 * 3 * 3)
        x = self.drop_fc1(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.drop_fc2(F.relu(self.bn_fc2(self.fc2(x))))
        return self.fc3(x)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
from torchvision import datasets, transforms
import torch.optim as optim
from torch.utils.data import DataLoader

NORM_MEAN = 0.1307
NORM_STD  = 0.3081


def _get_loaders(augment_heavy: bool, batch_size: int):
    if augment_heavy:
        # Heavy augmentation for ImprovedCNN
        train_tf = transforms.Compose([
            transforms.RandomRotation(180),           # full rotation range
            transforms.RandomAffine(
                degrees=0,
                translate=(0.15, 0.15),
                scale=(0.85, 1.15),
                shear=10,
            ),
            transforms.ElasticTransform(alpha=50.0, sigma=5.0),
            transforms.ColorJitter(brightness=0.4, contrast=0.4),
            transforms.ToTensor(),
            transforms.Normalize((NORM_MEAN,), (NORM_STD,)),
            transforms.RandomErasing(p=0.4, scale=(0.02, 0.20)),  # cutout
        ])
    else:
        # Light augmentation for BaselineCNN (same as notebook)
        train_tf = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize((NORM_MEAN,), (NORM_STD,)),
        ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((NORM_MEAN,), (NORM_STD,)),
    ])

    train_loader = DataLoader(
        datasets.MNIST('./data', train=True,  download=True, transform=train_tf),
        batch_size=batch_size, shuffle=True, num_workers=2,
    )
    test_loader = DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=test_tf),
        batch_size=batch_size, shuffle=False, num_workers=2,
    )
    return train_loader, test_loader


def train_model(model, augment_heavy, save_path, device,
                epochs=20, batch_size=128, lr=1e-3):
    train_loader, test_loader = _get_loaders(augment_heavy, batch_size)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    best_acc  = 0.0

    label = "ImprovedCNN" if augment_heavy else "BaselineCNN"
    print(f"Training {label} ({sum(p.numel() for p in model.parameters()):,} params)...")

    for epoch in range(1, epochs + 1):
        model.train()
        correct, total = 0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out  = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            correct += out.argmax(1).eq(target).sum().item()
            total   += target.size(0)

        model.eval()
        test_loss, test_correct, test_total = 0, 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                out           = model(data)
                test_loss    += criterion(out, target).item()
                test_correct += out.argmax(1).eq(target).sum().item()
                test_total   += target.size(0)

        test_acc = 100. * test_correct / test_total
        scheduler.step()
        print(f"  Epoch {epoch:2d}/{epochs} | "
              f"Train {100.*correct/total:.2f}% | Test {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"    → Saved ({best_acc:.2f}%)")

    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    print(f"{label} ready — best: {best_acc:.2f}%")
    return model


def load_or_train_classifier(model, save_path, augment_heavy, device,
                              epochs=20, batch_size=128):
    if os.path.exists(save_path):
        print(f"Loading from {save_path}")
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.eval()
        return model
    print(f"No checkpoint at {save_path} — training...")
    return train_model(model, augment_heavy, save_path, device, epochs, batch_size)


# ─────────────────────────────────────────────────────────────────────────────
# Classification helpers — baseline (strict) and improved (TTA + permissive)
# ─────────────────────────────────────────────────────────────────────────────
def _to_tensor(img_np, device):
    """28x28 numpy array → normalised tensor on device."""
    t = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    # already in normalised space from the optimizer
    return t.to(device)


def classify_baseline(images_np, model, device, threshold=BASE_THRESHOLD):
    """Single-pass, high-threshold — same as original notebook."""
    model.eval()
    preds = []
    with torch.no_grad():
        for img in images_np:
            t     = _to_tensor(img, device)
            probs = torch.softmax(model(t), dim=1)
            conf, pred = probs.max(dim=1)
            preds.append(pred.item() if conf.item() > threshold else None)
    return preds


def classify_improved_tta(images_np, model, device, threshold=IMPROVED_THRESHOLD):
    """
    TTA over 4 rotations (0/90/180/270 deg).
    Average softmax probabilities, then apply permissive threshold.
    Helps catch rotated 7s, partial 6s, etc.
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for img in images_np:
            t    = _to_tensor(img, device)           # (1,1,28,28)
            probs_sum = torch.zeros(10, device=device)
            for angle in TTA_ANGLES:
                rotated = TF.rotate(t.squeeze(0), angle).unsqueeze(0)
                probs_sum += torch.softmax(model(rotated), dim=1).squeeze()
            probs_avg  = probs_sum / len(TTA_ANGLES)
            conf, pred = probs_avg.max(dim=0)
            preds.append(pred.item() if conf.item() > threshold else None)
    return preds


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
            torch.sqrt(2 * torch.pi * var_t)
        )
    return pdf


def generate_mog_samples(num_samples, means, variances, weights=None, device='cpu'):
    components = len(means)
    if weights is None:
        weights = torch.ones(components, device=device) / components
    else:
        weights = weights.to(device)
    weights   = weights / weights.sum()
    means_t   = torch.stack([m.flatten() for m in means]).to(device)
    covs_t    = torch.stack([torch.diag(v.flatten()) for v in variances]).to(device)
    mix       = Categorical(weights)
    comp      = MultivariateNormal(means_t, covs_t)
    return MixtureSameFamily(mix, comp).sample((num_samples,))


def sliced_wasserstein_distance(X, Y, n_projections=50, device='cpu'):
    X    = X.to(device).float()
    Y    = Y.to(device).float()
    dim  = X.shape[1]
    proj = torch.randn(n_projections, dim, device=device)
    proj = proj / torch.norm(proj, dim=1, keepdim=True)
    X_s  = torch.sort(X @ proj.T, dim=0)[0]
    Y_s  = torch.sort(Y @ proj.T, dim=0)[0]
    return torch.mean(torch.abs(X_s - Y_s))


# ─────────────────────────────────────────────────────────────────────────────
# LGD core
# ─────────────────────────────────────────────────────────────────────────────
def optimize_LGD(model_uncond, model_cond_cm, noise_scheduler,
                 mog_means, mog_variances, weights,
                 nsamples, num_x_t, num_inference_steps,
                 device, use_uniform=False, seed=None):

    if seed is not None:
        set_seed(seed)

    ddim = DDIMScheduler.from_config(noise_scheduler.config)
    ddim.set_timesteps(num_inference_steps=num_inference_steps)
    timesteps = ddim.timesteps

    x_t = torch.randn(1, 1, 28, 28, device=device, requires_grad=True)

    for i, t in enumerate(timesteps[:-1]):
        x_t         = x_t.detach().clone().requires_grad_(True)
        residual    = model_uncond(x_t, torch.tensor([t], device=device))
        alpha_t     = ddim.alphas_cumprod[t]
        alpha_t_prev = (ddim.alphas_cumprod[timesteps[i+1]]
                        if i < len(timesteps)-2 else torch.tensor(1.0))
        beta_t      = 1 - alpha_t
        pred_x0     = (x_t - beta_t**0.5 * residual) / alpha_t**0.5
        x_t_minus_1 = alpha_t_prev**0.5 * pred_x0 + (1 - alpha_t_prev)**0.5 * residual

        r_t       = torch.sqrt(beta_t)
        step_size = r_t / (1 + r_t**2) + 5 * t / 1000

        losses = []
        for _ in range(num_x_t):
            x0_sample     = pred_x0 + r_t**2 * torch.randn_like(pred_x0)
            target_angles = circular_to_angles(
                model_cond_cm.sample(nsamples=nsamples, condition_x=x0_sample,
                                     ts=[150., 50., 20., 10., 5., 1.])[0]
            )
            target_circ = angles_to_circular(target_angles)
            if use_uniform:
                mog_circ = angles_to_circular(torch.rand(nsamples, device=device) * 360)
            else:
                mog_ang  = generate_mog_samples(nsamples, mog_means,
                                                mog_variances, weights).squeeze()
                mog_circ = angles_to_circular(mog_ang)
            losses.append(-sliced_wasserstein_distance(target_circ, mog_circ,
                                                       n_projections=50, device=device))

        log_me = -torch.logsumexp(torch.stack(losses), dim=0) + math.log(num_x_t)
        grad   = torch.autograd.grad(log_me, x_t, retain_graph=True)[0]
        with torch.no_grad():
            x_t = x_t_minus_1.detach().clone() - step_size * grad

    # Final DDIM step
    with torch.no_grad():
        last_t = timesteps[-1]
        res    = model_uncond(x_t, torch.tensor([last_t], device=device))
        a      = ddim.alphas_cumprod[last_t]
        x_t    = (x_t - (1 - a)**0.5 * res) / a**0.5

    x_final    = x_t.detach().clone()
    final_n    = nsamples * 4
    final_ang  = circular_to_angles(
        model_cond_cm.sample(nsamples=final_n,
                             condition_x=x_final.view(1, 28, 28),
                             ts=[150., 50., 20., 10., 5., 1.])[0]
    )
    final_circ = angles_to_circular(final_ang)
    if use_uniform:
        ref_circ = angles_to_circular(torch.rand(final_n, device=device) * 360)
    else:
        ref_ang  = generate_mog_samples(final_n, mog_means,
                                        mog_variances, weights).squeeze()
        ref_circ = angles_to_circular(ref_ang)

    final_loss = sliced_wasserstein_distance(final_circ, ref_circ,
                                             n_projections=50, device=device)
    if device == 'cuda':
        torch.cuda.empty_cache()
    return x_final, final_loss


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(losses, preds_base, preds_imp, exp_name, top_k=TOP_K):
    losses_arr = np.array(losses)
    top_ix     = np.argsort(losses_arr)[:top_k]
    n          = len(losses)
    relevant   = RELEVANT_DIGITS[exp_name]

    def _stats(preds, ix=None):
        p = [preds[i] for i in ix] if ix is not None else preds
        n_clf = sum(1 for x in p if x is not None)
        n_rel = sum(1 for x in p if x in relevant)
        digit_counts = {str(d): sum(1 for x in p if x == d) for d in range(10)}
        total = len(p)
        return {
            "pct_classified": 100.0 * n_clf / total,
            "pct_relevant":   100.0 * n_rel / total,
            "digit_counts":   digit_counts,
        }

    return {
        "swd_all_mean":  float(losses_arr.mean()),
        "swd_all_std":   float(losses_arr.std()),
        "swd_top_mean":  float(losses_arr[top_ix].mean()),
        "swd_top_std":   float(losses_arr[top_ix].std()),
        "top_ix":        top_ix.tolist(),
        # baseline stats
        "base_all":  _stats(preds_base),
        "base_top":  _stats(preds_base, top_ix),
        # improved stats
        "imp_all":   _stats(preds_imp),
        "imp_top":   _stats(preds_imp, top_ix),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def _label(pred):
    return str(pred) if pred is not None else "None"


def make_strip_fig(images_np, losses, preds_base, preds_imp, seeds, title=""):
    n     = len(images_np)
    ncols = min(5, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.5, nrows * 3.2),
                             gridspec_kw=dict(wspace=0.05, hspace=0.45))
    axes = np.array(axes).reshape(nrows, ncols)
    for idx in range(n):
        r, c = divmod(idx, ncols)
        axes[r, c].imshow(images_np[idx], cmap='gray')
        axes[r, c].set_title(
            f"s{seeds[idx]} | {losses[idx]:.3f}\n"
            f"Base: {_label(preds_base[idx])}\n"
            f"Imp:  {_label(preds_imp[idx])}",
            fontsize=6
        )
        axes[r, c].axis('off')
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis('off')
    fig.suptitle(title, fontsize=8)
    plt.tight_layout()
    return fig


def make_top5_fig(images_np, losses, preds_base, preds_imp, seeds, top_ix, title=""):
    k    = len(top_ix)
    fig, axes = plt.subplots(1, k, figsize=(k * 2.8, 3.2),
                             gridspec_kw=dict(wspace=0.05))
    axes = np.array(axes).reshape(k)
    for rank, idx in enumerate(top_ix):
        axes[rank].imshow(images_np[idx], cmap='gray')
        axes[rank].set_title(
            f"#{rank+1} | {losses[idx]:.3f}\n"
            f"Base: {_label(preds_base[idx])}\n"
            f"Imp:  {_label(preds_imp[idx])}",
            fontsize=7
        )
        axes[rank].axis('off')
    fig.suptitle(title, fontsize=8)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Run one experiment
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment(exp_name, mog_means, mog_variances, weights,
                   cfg, model_uncond, model_cond, noise_scheduler,
                   clf_base, clf_imp, device, n_seeds, smoke_test=False):

    results, losses, seeds_used, times = [], [], [], []
    n = 2 if smoke_test else n_seeds

    for seed in range(n):
        t0 = time.time()
        x_final, loss = optimize_LGD(
            model_uncond        = model_uncond,
            model_cond_cm       = model_cond,
            noise_scheduler     = noise_scheduler,
            mog_means           = mog_means,
            mog_variances       = mog_variances,
            weights             = weights,
            nsamples            = cfg["nsamples"],
            num_x_t             = cfg["num_x_t"],
            num_inference_steps = cfg["num_inference_steps"],
            device              = device,
            seed                = seed + GLOBAL_SEED,
        )
        elapsed = time.time() - t0
        results.append(x_final.squeeze().cpu().numpy())
        losses.append(loss.item())
        seeds_used.append(seed)
        times.append(elapsed)
        print(f"  [{exp_name}] seed={seed} loss={loss.item():.4f} t={elapsed:.1f}s")

    # classify with both classifiers
    preds_base = classify_baseline(results, clf_base, device)
    preds_imp  = classify_improved_tta(results, clf_imp, device)

    metrics = compute_metrics(losses, preds_base, preds_imp, exp_name)
    top_ix  = metrics["top_ix"]

    fig_all = make_strip_fig(results, losses, preds_base, preds_imp,
                             seeds_used, title=f"{exp_name} | all {n} seeds")
    fig_top = make_top5_fig(results, losses, preds_base, preds_imp,
                            seeds_used, top_ix,
                            title=f"{exp_name} | top-{TOP_K}")

    wandb.log({
        f"{exp_name}/swd_all_mean":          metrics["swd_all_mean"],
        f"{exp_name}/swd_all_std":           metrics["swd_all_std"],
        f"{exp_name}/swd_top_mean":          metrics["swd_top_mean"],
        f"{exp_name}/swd_top_std":           metrics["swd_top_std"],
        f"{exp_name}/time_mean":             float(np.mean(times)),
        f"{exp_name}/time_std":              float(np.std(times)),
        # baseline
        f"{exp_name}/base_pct_classified_all": metrics["base_all"]["pct_classified"],
        f"{exp_name}/base_pct_relevant_all":   metrics["base_all"]["pct_relevant"],
        f"{exp_name}/base_pct_classified_top": metrics["base_top"]["pct_classified"],
        f"{exp_name}/base_pct_relevant_top":   metrics["base_top"]["pct_relevant"],
        # improved
        f"{exp_name}/imp_pct_classified_all":  metrics["imp_all"]["pct_classified"],
        f"{exp_name}/imp_pct_relevant_all":    metrics["imp_all"]["pct_relevant"],
        f"{exp_name}/imp_pct_classified_top":  metrics["imp_top"]["pct_classified"],
        f"{exp_name}/imp_pct_relevant_top":    metrics["imp_top"]["pct_relevant"],
        # images
        f"{exp_name}/img_all":               wandb.Image(fig_all),
        f"{exp_name}/img_top5":              wandb.Image(fig_top),
    })
    plt.close(fig_all)
    plt.close(fig_top)

    return metrics, results, losses, preds_base, preds_imp, seeds_used, times


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run_config(cfg, args, smoke_test=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(GLOBAL_SEED)

    print(f"\n{'='*65}\nConfig: {cfg}\nDevice: {device}\n{'='*65}\n")

    run_name = (
        f"steps{cfg['num_inference_steps']}_"
        f"xt{cfg['num_x_t']}_"
        f"ns{cfg['nsamples']}_"
        f"bv{cfg['bimodal_var']}_"
        f"uv{cfg['unimodal_var']}"
    )
    wandb.init(
        project = args.wandb_project,
        entity  = args.wandb_entity or None,
        config  = {
            **cfg,
            "n_seeds":           N_SEEDS,
            "top_k":             TOP_K,
            "global_seed":       GLOBAL_SEED,
            "base_threshold":    BASE_THRESHOLD,
            "improved_threshold": IMPROVED_THRESHOLD,
            "tta_angles":        TTA_ANGLES,
            "smoke_test":        smoke_test,
        },
        name   = run_name,
        tags   = ["mnist", "gridsearch", "dual-classifier"],
        reinit = True,
    )

    # ── load models ───────────────────────────────────────────────────────
    ckpt_dir    = os.path.join(REPO_ROOT, "MNIST", "checkpoints")
    cond_path   = os.path.join(ckpt_dir, "MnistConditional500Epoch.pt")
    uncond_path = os.path.join(ckpt_dir, "MnistUncond100Epoch.pth")
    base_clf_path = os.path.join(ckpt_dir, "baseline_classifier.pth")
    imp_clf_path  = os.path.join(ckpt_dir, "improved_classifier.pth")

    cond_model = CircularAngleConsistencyModel(
        nfeatures=2, img_features=784, eps=0.002,
        nunits=128, depth=5, device=str(device),
    )
    cond_model.load_state_dict(
        torch.load(cond_path, map_location=device)['model_state_dict']
    )
    cond_model.eval()

    uncond_model = UnconditionalUnet().to(device)
    uncond_model.load_state_dict(
        torch.load(uncond_path, map_location=device)['model_state_dict']
    )
    uncond_model.eval()

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2'
    )

    # baseline classifier — light augmentation
    clf_base = load_or_train_classifier(
        BaselineCNN().to(device), base_clf_path,
        augment_heavy=False, device=device, epochs=15
    )
    # improved classifier — heavy augmentation + deeper net
    clf_imp = load_or_train_classifier(
        ImprovedCNN().to(device), imp_clf_path,
        augment_heavy=True, device=device, epochs=25
    )

    # ── experiments ───────────────────────────────────────────────────────
    experiments = {
        "Bimodal": {
            "mog_means":     [torch.tensor([180], dtype=torch.float64),
                              torch.tensor([360], dtype=torch.float64)],
            "mog_variances": [torch.tensor([[cfg["bimodal_var"]]],
                                           dtype=torch.float64)] * 2,
            "weights":       torch.tensor([0.5, 0.5], dtype=torch.float64),
        },
        "Unimodal": {
            "mog_means":     [torch.tensor([360], dtype=torch.float64)],
            "mog_variances": [torch.tensor([[cfg["unimodal_var"]]],
                                           dtype=torch.float64)],
            "weights":       torch.tensor([1.0], dtype=torch.float64),
        },
    }

    all_results = {}
    for exp_name, exp_cfg in experiments.items():
        print(f"\n--- Running {exp_name} ---")
        metrics, images, losses, pb, pi, seeds, times = run_experiment(
            exp_name        = exp_name,
            mog_means       = exp_cfg["mog_means"],
            mog_variances   = exp_cfg["mog_variances"],
            weights         = exp_cfg["weights"],
            cfg             = cfg,
            model_uncond    = uncond_model,
            model_cond      = cond_model,
            noise_scheduler = noise_scheduler,
            clf_base        = clf_base,
            clf_imp         = clf_imp,
            device          = str(device),
            n_seeds         = N_SEEDS,
            smoke_test      = smoke_test,
        )
        all_results[exp_name] = dict(
            metrics=metrics, losses=losses,
            preds_base=pb, preds_imp=pi,
            seeds=seeds, times=times
        )

    # combined objective — improved classifier, top-k relevant
    combined_score = np.mean([
        all_results["Bimodal"]["metrics"]["imp_top"]["pct_relevant"],
        all_results["Unimodal"]["metrics"]["imp_top"]["pct_relevant"],
    ])
    wandb.log({"combined_imp_top_relevant": combined_score})
    print(f"\n>>> Combined improved top-relevant: {combined_score:.1f}%")

    # save json
    save_dir  = os.path.join(REPO_ROOT, "MNIST", "results", "gridsearch")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{run_name}.json")
    with open(save_path, "w") as f:
        json.dump({
            "config":    cfg,
            "run_name":  run_name,
            "Bimodal":   {k: v for k, v in all_results["Bimodal"].items()
                          if k != "metrics"},
            "Unimodal":  {k: v for k, v in all_results["Unimodal"].items()
                          if k != "metrics"},
            "Bimodal_metrics":  all_results["Bimodal"]["metrics"],
            "Unimodal_metrics": all_results["Unimodal"]["metrics"],
            "combined_imp_top_relevant": combined_score,
        }, f, indent=2, default=str)
    wandb.save(save_path)
    wandb.finish()
    return combined_score


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config_id",     type=int, default=0)
    p.add_argument("--wandb_project", type=str, default="mnist-gridsearch")
    p.add_argument("--wandb_entity",  type=str, default="")
    p.add_argument("--list_configs",  action="store_true")
    p.add_argument("--smoke_test",    action="store_true")
    args = p.parse_args()

    configs = all_configs()
    if args.list_configs:
        print(f"Total configs: {len(configs)}")
        for i, c in enumerate(configs):
            print(f"  [{i:3d}] {c}")
        sys.exit(0)

    print(f"Config {args.config_id} / {len(configs)-1}")
    run_config(configs[args.config_id], args, smoke_test=args.smoke_test)