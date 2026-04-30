"""
mnist_uniform_gridsearch.py
===========================
Uniform experiment — grid search over:
  - NSAMPLES              : [500, 1000, 1500, 2000]
  - NUM_INFERENCE_STEPS   : [100, 200, 300, 400, 500]
  - STEP_SIZE_MODE        : ['original', 'dps', 'half', 'double', 'no_linear']
  - NUM_X_T               : [3, 5, 10, 15, 20]

W&B project: mnist_uniform_gridsearch

Launch:
    sbatch mnist_uniform_gridsearch.sh
Smoke test:
    python mnist_uniform_gridsearch.py --config_id 0 --smoke_test
List configs:
    python mnist_uniform_gridsearch.py --list_configs
Train classifier only:
    python mnist_uniform_gridsearch.py --train_classifier_only
"""

import os, sys, math, argparse, random, time, pickle, itertools
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
# Fixed
# ─────────────────────────────────────────────────────────────────────────────
N_SEEDS     = 15
GLOBAL_SEED = 42

HF_TOKEN   = os.environ.get("HF_TOKEN", "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")
HF_REPO_ID = "Orineo/conditional-matching-paper"

CLF_PATH  = os.path.join(REPO_ROOT, "MNIST", "checkpoints", "robust_classifier.pth")
NORM_MEAN = 0.1307
NORM_STD  = 0.3081

# ─────────────────────────────────────────────────────────────────────────────
# Step-size modes
# ─────────────────────────────────────────────────────────────────────────────
# original  : r_t / (1 + r_t^2) + 5*t/1000         (current code)
# dps       : 1 / (r_t * sqrt(2*pi))                (DPS-style: 1/sigma * scale)
# half      : 0.5 * (r_t / (1+r_t^2) + 5*t/1000)
# double    : 2.0 * (r_t / (1+r_t^2) + 5*t/1000)
# no_linear : r_t / (1 + r_t^2)                     (drop the t-linear term)

STEP_SIZE_MODES = ['original', 'dps', 'half', 'double', 'no_linear',"multiply2"]

NSAMPLES_LIST           = [ 1500]
NUM_INFERENCE_STEPS_LIST = [90,110,290,310]
NUM_X_T_LIST            = [10,]

CONFIGS = [
    {
        "nsamples":            ns,
        "num_inference_steps": nsteps,
        "step_size_mode":      ssm,
        "num_x_t":             nxt,
    }
    for ns    in NSAMPLES_LIST
    for nsteps in NUM_INFERENCE_STEPS_LIST
    for ssm   in STEP_SIZE_MODES
    for nxt   in NUM_X_T_LIST
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


def classify_generated_images(images_np, classifier, device, threshold=0.7):
    """Classify 28x28 normalised numpy arrays. Returns list of (pred_int_or_None, conf_float)."""
    classifier.eval()
    preds = []
    confs = []
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
# Step-size helper
# ─────────────────────────────────────────────────────────────────────────────
def compute_step_size(r_t, t, mode):
    """
    r_t   : torch scalar, sqrt(beta_t)
    t     : int timestep (0–999)
    mode  : one of STEP_SIZE_MODES
    """
    if mode == 'multiply2':
        return 2*(r_t / (1 + r_t**2) + 5 * t / 1000)
    if mode == 'original':
        return r_t / (1 + r_t**2) + 5 * t / 1000
    elif mode == 'dps':
        # DPS-style: step proportional to 1/sigma  (sigma = r_t)
        return 1.0 / (r_t + 1e-8) * 0.1
    elif mode == 'half':
        return 0.5 * (r_t / (1 + r_t**2) + 5 * t / 1000)
    elif mode == 'double':
        return 2.0 * (r_t / (1 + r_t**2) + 5 * t / 1000)
    elif mode == 'no_linear':
        # drop the t-linear term that was causing step size blow-up
        return r_t / (1 + r_t**2)
    else:
        raise ValueError(f"Unknown step_size_mode: {mode}")

# ─────────────────────────────────────────────────────────────────────────────
# MoG / Uniform helpers
# ─────────────────────────────────────────────────────────────────────────────
def sliced_wasserstein_distance(X, Y, n_projections=50, device='cpu'):
    X    = X.to(device).float()
    Y    = Y.to(device).float()
    proj = torch.randn(n_projections, X.shape[1], device=device)
    proj = proj / torch.norm(proj, dim=1, keepdim=True)
    return torch.mean(torch.abs(
        torch.sort(X @ proj.T, dim=0)[0] - torch.sort(Y @ proj.T, dim=0)[0]
    ))

# ─────────────────────────────────────────────────────────────────────────────
# LGD core — uniform target
# ─────────────────────────────────────────────────────────────────────────────
def optimize_LGD_uniform(model_uncond, model_cond_cm, noise_scheduler,
                         nsamples, num_x_t, num_inference_steps,
                         step_size_mode, device, seed=None):

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
                        if i < len(timesteps) - 2 else torch.tensor(1.0))
        beta_t       = 1 - alpha_t
        pred_x0      = (x_t - beta_t**0.5 * residual) / alpha_t**0.5
        x_t_minus_1  = alpha_t_prev**0.5 * pred_x0 + (1 - alpha_t_prev)**0.5 * residual

        r_t       = torch.sqrt(beta_t)
        step_size = compute_step_size(r_t, t, step_size_mode)

        losses = []
        for _ in range(num_x_t):
            x0_sample     = pred_x0 + r_t**2 * torch.randn_like(pred_x0)
            target_angles = circular_to_angles(
                model_cond_cm.sample(nsamples=nsamples, condition_x=x0_sample,
                                     ts=[150., 50., 20., 10., 5., 1.])[0]
            )
            uniform_ang = torch.rand(nsamples, device=device) * 360.0
            losses.append(-sliced_wasserstein_distance(
                angles_to_circular(target_angles),
                angles_to_circular(uniform_ang),
                n_projections=50, device=device,
            ))

        log_me = -torch.logsumexp(torch.stack(losses), dim=0) + math.log(num_x_t)
        grad   = torch.autograd.grad(log_me, x_t, retain_graph=True)[0]
        with torch.no_grad():
            if t < 250:
                step_size = 0
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
    ref_ang    = torch.rand(final_n, device=device) * 360.0
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
                 seeds=range(15), device='cuda'):

    os.makedirs(save_dir, exist_ok=True)
    print(f'\n{"="*60}\n  EXPERIMENT: {experiment_name}\n{"="*60}')

    results, loss_log, seed_log, time_log = [], [], [], []

    for seed in seeds:
        print(f'[Seed {seed:2d}] optimizing...', end='  ', flush=True)
        t0 = time.time()
        x_final, loss = optimize_LGD_uniform(
            model_uncond        = model_uncond,
            model_cond_cm       = model_cond,
            noise_scheduler     = noise_scheduler,
            nsamples            = nsamples,
            num_x_t             = num_x_t,
            num_inference_steps = num_inference_steps,
            step_size_mode      = step_size_mode,
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

    # uniform target PDF
    x_range_np    = np.linspace(0, 360, 200)
    target_pdf_np = np.ones(200) / 360.0

    payload = {
        'experiment_name':     experiment_name,
        'nsamples':            nsamples,
        'num_x_t':             num_x_t,
        'num_inference_steps': num_inference_steps,
        'step_size_mode':      step_size_mode,
        'results':             results,
        'loss_log':            loss_log,
        'seed_log':            seed_log,
        'time_log':            time_log,
        'x_range':             x_range_np,
        'target_pdf':          target_pdf_np,
        'use_uniform':         True,
    }

    save_path = os.path.join(save_dir, f'{experiment_name}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(payload, f)

    best_i = int(np.argmin(loss_log))
    print(f'Saved  → {save_path}')
    print(f'Best   → seed {seed_log[best_i]}, loss {loss_log[best_i]:.4f}')
    return save_path, payload

# ─────────────────────────────────────────────────────────────────────────────
# Main run_config
# ─────────────────────────────────────────────────────────────────────────────
def run_config(cfg, args, smoke_test=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    make_deterministic(GLOBAL_SEED)

    nsamples            = cfg["nsamples"]
    num_inference_steps = cfg["num_inference_steps"]
    step_size_mode      = cfg["step_size_mode"]
    num_x_t             = cfg["num_x_t"]

    print(f"\n{'='*65}")
    print(f"Config: nsamples={nsamples}  steps={num_inference_steps}  "
          f"ss_mode={step_size_mode}  num_x_t={num_x_t}")
    print(f"Device: {device}\n{'='*65}\n")

    run_name = f"unif_ns{nsamples}_st{num_inference_steps}_ss{step_size_mode}_xt{num_x_t}"
    save_dir = os.path.join(
        REPO_ROOT, "MNIST", "results", "uniform_gridsearch", run_name)

    wandb.init(
        project = "mnist_uniform_gridsearch",
        entity  = args.wandb_entity or None,
        config  = {
            "nsamples":            nsamples,
            "num_inference_steps": num_inference_steps,
            "step_size_mode":      step_size_mode,
            "num_x_t":             num_x_t,
            "n_seeds":             N_SEEDS,
            "global_seed":         GLOBAL_SEED,
            "smoke_test":          smoke_test,
            "experiment":          "uniform",
        },
        name   = run_name,
        tags   = ["mnist", "uniform", "gridsearch"],
        reinit = True,
    )

    # ── load models ───────────────────────────────────────────────────────
    login(token=HF_TOKEN, add_to_git_credential=False)

    print("Downloading conditional model...")
    cond_path = hf_hub_download(
        repo_id=HF_REPO_ID, filename="MNIST/MnistConditional500Epoch.pt", token=HF_TOKEN)
    print("Downloading unconditional model...")
    uncond_path = hf_hub_download(
        repo_id=HF_REPO_ID, filename="MNIST/MnistUncond100Epoch.pth", token=HF_TOKEN)

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

    # ── classifier ────────────────────────────────────────────────────────
    classifier = load_or_train_classifier(device)

    # ── run ───────────────────────────────────────────────────────────────
    seeds           = range(2) if smoke_test else range(N_SEEDS)
    experiment_name = run_name

    save_path, payload = run_and_save(
        model_uncond        = uncond_model,
        model_cond          = cond_model,
        noise_scheduler     = noise_scheduler,
        nsamples            = nsamples,
        num_x_t             = num_x_t,
        num_inference_steps = num_inference_steps,
        step_size_mode      = step_size_mode,
        experiment_name     = experiment_name,
        save_dir            = save_dir,
        seeds               = seeds,
        device              = str(device),
    )

    # ── classify ──────────────────────────────────────────────────────────
    preds, confs = classify_generated_images(payload['results'], classifier, device)
    losses       = np.array(payload['loss_log'])
    top_ix       = np.argsort(losses)[:5]
    n            = len(payload['results'])

    # --- summary stats ---
    n_classified      = sum(1 for p in preds if p is not None)
    n_classified_as_0 = sum(1 for p in preds if p == 0)

    top_preds         = [preds[i] for i in top_ix]
    top_confs         = [confs[i] for i in top_ix]
    n_top_clf         = sum(1 for p in top_preds if p is not None)
    n_top_clf_as_0    = sum(1 for p in top_preds if p == 0)
    top5_conf_mean    = float(np.mean([c for c in top_confs]))

    print(f"\n--- Classification Summary ---")
    print(f"  Total classified     : {n_classified}/{n}")
    print(f"  Classified as 0      : {n_classified_as_0}/{n}")
    print(f"  Top-5 classified     : {n_top_clf}/5")
    print(f"  Top-5 classified as 0: {n_top_clf_as_0}/5")
    print(f"  Top-5 mean conf      : {top5_conf_mean:.3f}")

    # ── all seeds figure ──────────────────────────────────────────────────
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
    plt.suptitle(
        f'Uniform | ns={nsamples} steps={num_inference_steps} '
        f'ss={step_size_mode} xt={num_x_t}',
        fontsize=7)
    plt.tight_layout()
    fig_all = fig

    # ── top-5 figure ──────────────────────────────────────────────────────
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
    plt.suptitle(
        f'Uniform Top-5 | ns={nsamples} steps={num_inference_steps} '
        f'ss={step_size_mode} xt={num_x_t}',
        fontsize=7)
    plt.tight_layout()
    fig_top5 = fig_top

    # ── SWD per seed plot ─────────────────────────────────────────────────
    fig_loss, ax_loss = plt.subplots(figsize=(8, 4))
    ax_loss.plot(payload['seed_log'], payload['loss_log'],
                 marker='o', linewidth=1.5, color='steelblue')
    ax_loss.axhline(losses.mean(), color='red', linestyle='--', linewidth=1,
                    label=f'mean={losses.mean():.4f}')
    ax_loss.set_xlabel('Seed')
    ax_loss.set_ylabel('SWD loss')
    ax_loss.set_title(
        f'Uniform SWD per seed | ns={nsamples} steps={num_inference_steps} '
        f'ss={step_size_mode} xt={num_x_t}')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    plt.tight_layout()

    # ── W&B log — images FIRST so they appear at top of run page ──────────
    wandb.log({
        # images first → appear at top of W&B run page
        "images/top5":              wandb.Image(fig_top5),
        "images/all_seeds":         wandb.Image(fig_all),
        "images/swd_per_seed":      wandb.Image(fig_loss),
        # core metrics
        "swd/mean":                 float(losses.mean()),
        "swd/std":                  float(losses.std()),
        "swd/min":                  float(losses.min()),
        "swd/top5_mean":            float(losses[top_ix].mean()),
        # classification metrics
        "clf/n_classified":         n_classified,
        "clf/n_classified_as_0":    n_classified_as_0,
        "clf/pct_classified":       100. * n_classified / n,
        "clf/pct_classified_as_0":  100. * n_classified_as_0 / n,
        "clf/n_top5_classified":    n_top_clf,
        "clf/n_top5_as_0":          n_top_clf_as_0,
        "clf/top5_conf_mean":       top5_conf_mean,
        # config echoes for easy filtering
        "config/nsamples":            nsamples,
        "config/num_inference_steps": num_inference_steps,
        "config/step_size_mode":      step_size_mode,
        "config/num_x_t":             num_x_t,
    })
    plt.close(fig_all)
    plt.close(fig_top5)
    plt.close(fig_loss)

    # ── upload pkl as W&B artifact ────────────────────────────────────────
    artifact = wandb.Artifact(
        f'uniform_gridsearch_{run_name}', type='results'
    )
    artifact.add_file(save_path)
    wandb.log_artifact(artifact)

    wandb.finish()
    print(f"\nDone: {run_name}")


# ─────────────────────────────────────────────────────────────────────────────
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
            print(f"  [{i:3d}] nsamples={c['nsamples']:4d}  steps={c['num_inference_steps']:3d}  "
                  f"ss={c['step_size_mode']:12s}  num_x_t={c['num_x_t']:2d}")
        sys.exit(0)

    if args.train_classifier_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        make_deterministic(GLOBAL_SEED)
        train_classifier(device)
        sys.exit(0)

    print(f"Config {args.config_id} / {len(CONFIGS)-1}")
    run_config(CONFIGS[args.config_id], args, smoke_test=args.smoke_test)
