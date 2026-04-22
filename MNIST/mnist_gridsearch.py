"""
mnist_variance_sweep.py
=======================
Paired variance sweep for Bimodal + Unimodal experiments.
Each SLURM array task = one (bimodal_var, unimodal_var) pair.

Launch:
    sbatch mnist_variance_sweep.sh
Smoke test:
    python mnist_variance_sweep.py --config_id 0 --smoke_test
List configs:
    python mnist_variance_sweep.py --list_configs
Train classifier only:
    python mnist_variance_sweep.py --train_classifier_only
"""

import os, sys, math, json, argparse, random, time, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
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
from huggingface_hub import hf_hub_download, login

# ─────────────────────────────────────────────────────────────────────────────
# Fixed hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
NUM_INFERENCE_STEPS = 250
NUM_X_T             = 10
NSAMPLES            = 1500
N_SEEDS             = 15
GLOBAL_SEED         = 42

HF_TOKEN   = os.environ.get("HF_TOKEN", "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")
HF_REPO_ID = "Orineo/conditional-matching-paper"

CLF_PATH = os.path.join(REPO_ROOT, "MNIST", "checkpoints", "robust_classifier.pth")

NORM_MEAN = 0.1307
NORM_STD  = 0.3081

# ─────────────────────────────────────────────────────────────────────────────
# Paired variance sweep — 20 pairs
# Bimodal:  200 → 390  step 10 (with extras at edges/between for coverage)
# Unimodal: 400 → 590  step 10  (same length, advance together)
# ─────────────────────────────────────────────────────────────────────────────
BIMODAL_VARS  = list(range(200, 350, 5))
UNIMODAL_VARS = list(range(400, 550, 5))


CONFIGS = [
    {"bimodal_var": b, "unimodal_var": u}
    for b, u in zip(BIMODAL_VARS, UNIMODAL_VARS)
]

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


def classify_images(images_np, model, device, threshold=0.7):
    """Min-max normalize each image then classify. Returns list of int-or-None."""
    model.eval()
    preds = []
    with torch.no_grad():
        for img in images_np:
            t      = torch.tensor(img, dtype=torch.float32).flatten()
            lo, hi = t.min(), t.max()
            t      = (t - lo) / (hi - lo + 1e-8)
            t      = (t - NORM_MEAN) / NORM_STD
            t      = t.reshape(1, 1, 28, 28).to(device)
            probs  = F.softmax(model(t), dim=1).squeeze()
            conf, pred = probs.max(0)
            preds.append(pred.item() if conf.item() >= threshold else None)
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
            torch.sqrt(2 * torch.pi * var_t))
    return pdf


def generate_mog_samples(num_samples, means, variances, weights=None, device='cpu'):
    components = len(means)
    if weights is None:
        weights = torch.ones(components, device=device) / components
    else:
        weights = weights.to(device)
    weights  = weights / weights.sum()
    means_t  = torch.stack([m.flatten() for m in means]).to(device)
    covs_t   = torch.stack([torch.diag(v.flatten()) for v in variances]).to(device)
    mix      = Categorical(weights)
    comp     = MultivariateNormal(means_t, covs_t)
    return MixtureSameFamily(mix, comp).sample((num_samples,))


def sliced_wasserstein_distance(X, Y, n_projections=50, device='cpu'):
    X    = X.to(device).float()
    Y    = Y.to(device).float()
    proj = torch.randn(n_projections, X.shape[1], device=device)
    proj = proj / torch.norm(proj, dim=1, keepdim=True)
    return torch.mean(torch.abs(
        torch.sort(X @ proj.T, dim=0)[0] - torch.sort(Y @ proj.T, dim=0)[0]
    ))


def _build_target(mog_means, mog_variances, weights):
    x_range    = torch.linspace(0, 360, 200)
    target_pdf = mog_pdf(
        x_range,
        [m.item() for m in mog_means],
        [v.squeeze().item() for v in mog_variances],
        weights,
    )
    return x_range.numpy(), target_pdf.numpy()

# ─────────────────────────────────────────────────────────────────────────────
# LGD core
# ─────────────────────────────────────────────────────────────────────────────
def optimize_LGD(model_uncond, model_cond_cm, noise_scheduler,
                 mog_means, mog_variances, weights,
                 nsamples, num_x_t, num_inference_steps,
                 device, seed=None):

    if seed is not None:
        set_seed(seed)

    ddim = DDIMScheduler.from_config(noise_scheduler.config)
    ddim.set_timesteps(num_inference_steps=num_inference_steps)
    timesteps = ddim.timesteps

    x_t = torch.randn(1, 1, 28, 28, device=device, requires_grad=True)

    for i, t in enumerate(timesteps[:-1]):
        x_t      = x_t.detach().clone().requires_grad_(True)
        residual = model_uncond(x_t, torch.tensor([t], device=device))
        alpha_t  = ddim.alphas_cumprod[t]
        alpha_t_prev = (ddim.alphas_cumprod[timesteps[i+1]]
                        if i < len(timesteps)-2 else torch.tensor(1.0))
        beta_t   = 1 - alpha_t
        pred_x0  = (x_t - beta_t**0.5 * residual) / alpha_t**0.5
        x_t_minus_1 = alpha_t_prev**0.5 * pred_x0 + (1 - alpha_t_prev)**0.5 * residual

        r_t       = torch.sqrt(beta_t)
        step_size = r_t / (1 + r_t**2) + 5 * t / 1000

        losses = []
        for _ in range(num_x_t):
            x0_sample    = pred_x0 + r_t**2 * torch.randn_like(pred_x0)
            target_angles = circular_to_angles(
                model_cond_cm.sample(nsamples=nsamples, condition_x=x0_sample,
                                     ts=[150., 50., 20., 10., 5., 1.])[0]
            )
            mog_ang  = generate_mog_samples(nsamples, mog_means, mog_variances,
                                            weights).squeeze()
            losses.append(-sliced_wasserstein_distance(
                angles_to_circular(target_angles),
                angles_to_circular(mog_ang),
                n_projections=50, device=device,
            ))

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

    x_final   = x_t.detach().clone()
    final_n   = nsamples * 4
    final_ang = circular_to_angles(
        model_cond_cm.sample(nsamples=final_n,
                             condition_x=x_final.view(1, 28, 28),
                             ts=[150., 50., 20., 10., 5., 1.])[0]
    )
    ref_ang  = generate_mog_samples(final_n, mog_means, mog_variances, weights).squeeze()
    final_loss = sliced_wasserstein_distance(
        angles_to_circular(final_ang),
        angles_to_circular(ref_ang),
        n_projections=50, device=device,
    )

    if str(device) == 'cuda':
        torch.cuda.empty_cache()

    return x_final, final_loss

# ─────────────────────────────────────────────────────────────────────────────
# run_and_save — exact notebook PKL format
# ─────────────────────────────────────────────────────────────────────────────
def run_and_save(model_uncond, model_cond, noise_scheduler,
                 mog_means, mog_variances, weights,
                 experiment_name, save_dir,
                 seeds=range(15),
                 nsamples=1500, num_x_t=10, num_inference_steps=100,
                 device='cuda'):

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
            mog_means           = mog_means,
            mog_variances       = mog_variances,
            weights             = weights,
            nsamples            = nsamples,
            num_x_t             = num_x_t,
            num_inference_steps = num_inference_steps,
            device              = device,
            seed                = seed,
        )
        elapsed  = time.time() - t0
        loss_val = loss.item()
        results.append(x_final.squeeze().cpu().numpy())
        loss_log.append(loss_val)
        seed_log.append(seed)
        time_log.append(elapsed)
        print(f'loss = {loss_val:.4f}  time = {elapsed:.1f}s')

    x_range_np, target_pdf_np = _build_target(mog_means, mog_variances, weights)

    # exact same payload format as the notebook
    payload = {
        'experiment_name': experiment_name,
        'results':         results,
        'loss_log':        loss_log,
        'seed_log':        seed_log,
        'time_log':        time_log,
        'x_range':         x_range_np,
        'target_pdf':      target_pdf_np,
        'use_uniform':     False,
    }

    save_path = os.path.join(save_dir, f'{experiment_name}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(payload, f)

    best_i = int(np.argmin(loss_log))
    print(f'Saved  → {save_path}')
    print(f'Best   → seed {seed_log[best_i]}, loss {loss_log[best_i]:.4f}')
    return save_path, payload

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run_config(cfg, args, smoke_test=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    make_deterministic(GLOBAL_SEED)

    bimodal_var  = cfg["bimodal_var"]
    unimodal_var = cfg["unimodal_var"]

    print(f"\n{'='*65}")
    print(f"Config: bimodal_var={bimodal_var}  unimodal_var={unimodal_var}")
    print(f"Device: {device}\n{'='*65}\n")

    run_name = f"bi{bimodal_var}_uni{unimodal_var}"
    save_dir = os.path.join(REPO_ROOT, "MNIST", "results", "variance_sweep", run_name)

    wandb.init(
        project = "grid_mnist_orgniezd",
        entity  = args.wandb_entity or None,
        config  = {
            **cfg,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "num_x_t":             NUM_X_T,
            "nsamples":            NSAMPLES,
            "n_seeds":             N_SEEDS,
            "global_seed":         GLOBAL_SEED,
            "smoke_test":          smoke_test,
        },
        name  = run_name,
        tags  = ["mnist", "variance_sweep"],
        reinit = True,
    )

    # ── load generative models ────────────────────────────────────────────
    login(token=HF_TOKEN, add_to_git_credential=False)

    print("Downloading conditional model...")
    cond_path = hf_hub_download(
        repo_id=HF_REPO_ID, filename="MNIST/MnistConditional500Epoch.pt", token=HF_TOKEN)
    print("Downloading unconditional model...")
    uncond_path = hf_hub_download(
        repo_id=HF_REPO_ID, filename="MNIST/MnistUncond100Epoch.pth", token=HF_TOKEN)

    cond_model = CircularAngleConsistencyModel(
        nfeatures=2, img_features=784, eps=0.002, nunits=128, depth=5, device=str(device))
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

    # ── load classifier (shared checkpoint, trained once) ─────────────────
    classifier = load_or_train_classifier(device)

    # ── experiments ───────────────────────────────────────────────────────
    seeds = range(2) if smoke_test else range(N_SEEDS)

    experiments = {
        "Bimodal": {
            "mog_means":     [torch.tensor([180], dtype=torch.float64),
                              torch.tensor([360], dtype=torch.float64)],
            "mog_variances": [torch.tensor([[bimodal_var]], dtype=torch.float64),
                              torch.tensor([[bimodal_var]], dtype=torch.float64)],
            "weights":       torch.tensor([0.5, 0.5], dtype=torch.float64),
        },
        "Unimodal": {
            "mog_means":     [torch.tensor([360], dtype=torch.float64)],
            "mog_variances": [torch.tensor([[unimodal_var]], dtype=torch.float64)],
            "weights":       torch.tensor([1.0], dtype=torch.float64),
        },
    }

    for exp_name, exp_cfg in experiments.items():
        save_path, payload = run_and_save(
            model_uncond        = uncond_model,
            model_cond          = cond_model,
            noise_scheduler     = noise_scheduler,
            mog_means           = exp_cfg["mog_means"],
            mog_variances       = exp_cfg["mog_variances"],
            weights             = exp_cfg["weights"],
            experiment_name     = exp_name,
            save_dir            = save_dir,
            seeds               = seeds,
            nsamples            = NSAMPLES,
            num_x_t             = NUM_X_T,
            num_inference_steps = NUM_INFERENCE_STEPS,
            device              = str(device),
        )

        # classify and log
        preds    = classify_images(payload['results'], classifier, device)
        losses   = np.array(payload['loss_log'])
        top_ix   = np.argsort(losses)[:5]
        top_preds = [preds[i] for i in top_ix]

        n_classified = sum(1 for p in preds if p is not None)
        n_top_clf    = sum(1 for p in top_preds if p is not None)

        # strip figure for wandb
        n     = len(payload['results'])
        ncols = min(5, n)
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2.5, nrows*3))
        axes = np.array(axes).reshape(nrows, ncols)
        for i, (img, loss, seed, pred) in enumerate(
                zip(payload['results'], payload['loss_log'], payload['seed_log'], preds)):
            r, c = divmod(i, ncols)
            axes[r, c].imshow(img.reshape(28, 28), cmap='gray')
            axes[r, c].set_title(
                f's{seed} | {loss:.3f}\n{pred if pred is not None else "None"}',
                fontsize=6)
            axes[r, c].axis('off')
        for i in range(n, nrows*ncols):
            r, c = divmod(i, ncols)
            axes[r, c].axis('off')
        plt.suptitle(f'{exp_name} | bi={bimodal_var} uni={unimodal_var}', fontsize=8)
        plt.tight_layout()

        # top-5 figure
        top_ix = np.argsort(losses)[:5]
        k = len(top_ix)
        fig_top, axes_top = plt.subplots(1, k, figsize=(k * 2.5, 3))
        axes_top = np.array(axes_top).reshape(k)
        for rank, idx in enumerate(top_ix):
            axes_top[rank].imshow(payload['results'][idx].reshape(28, 28), cmap='gray')
            axes_top[rank].set_title(
                f'#{rank + 1} | s{payload["seed_log"][idx]}\n'
                f'loss={losses[idx]:.3f} | {preds[idx] if preds[idx] is not None else "None"}',
                fontsize=7
            )
            axes_top[rank].axis('off')
        plt.suptitle(f'{exp_name} Top-5 | bi={bimodal_var} uni={unimodal_var}', fontsize=8)
        plt.tight_layout()

        wandb.log({
            f"{exp_name}/swd_mean": float(losses.mean()),
            f"{exp_name}/swd_std": float(losses.std()),
            f"{exp_name}/swd_top5_mean": float(losses[top_ix].mean()),
            f"{exp_name}/pct_classified": 100. * n_classified / n,
            f"{exp_name}/pct_top5_clf": 100. * n_top_clf / len(top_ix),
            f"{exp_name}/images_all": wandb.Image(fig),
            f"{exp_name}/images_top5": wandb.Image(fig_top),
            f"{exp_name}/bimodal_var": bimodal_var,
            f"{exp_name}/unimodal_var": unimodal_var,
        })
        plt.close(fig)
        plt.close(fig_top)


        print(f"[{exp_name}] classified {n_classified}/{n} | top-5 clf {n_top_clf}/5")

    wandb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config_id",             type=int,  default=0)
    p.add_argument("--wandb_entity",          type=str,  default="")
    p.add_argument("--list_configs",          action="store_true")
    p.add_argument("--smoke_test",            action="store_true")
    p.add_argument("--train_classifier_only", action="store_true")
    args = p.parse_args()

    if args.list_configs:
        print(f"Total configs: {len(CONFIGS)}")
        for i, c in enumerate(CONFIGS):
            print(f"  [{i:2d}] bimodal_var={c['bimodal_var']:3d}  unimodal_var={c['unimodal_var']:3d}")
        sys.exit(0)

    if args.train_classifier_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        make_deterministic(GLOBAL_SEED)
        train_classifier(device)
        sys.exit(0)

    print(f"Config {args.config_id} / {len(CONFIGS)-1}")
    run_config(CONFIGS[args.config_id], args, smoke_test=args.smoke_test)