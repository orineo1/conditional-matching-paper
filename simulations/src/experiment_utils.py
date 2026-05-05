import os
import json
import random
import numpy as np
import torch
from datetime import datetime


# ============================================================
# SEED MANAGEMENT
# ============================================================

def set_global_seed(seed: int):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] All random seeds set to {seed}")


def set_run_seed(base_seed: int, run_idx: int):
    """
    Set seed for a specific optimization run.
    Call this right before each optimization attempt.
    base_seed + run_idx gives a unique but reproducible seed per run.
    """
    run_seed = base_seed + run_idx
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    torch.cuda.manual_seed(run_seed)
    torch.cuda.manual_seed_all(run_seed)
    return run_seed


# ============================================================
# ENVIRONMENT LOGGING
# ============================================================

def get_environment_info():
    """Capture package versions and hardware info."""
    try:
        import pkg_resources
        packages = ["torch", "numpy", "flow_matching", "POT",
                    "matplotlib", "pandas", "tqdm"]
        versions = {}
        for pkg in packages:
            try:
                versions[pkg] = pkg_resources.get_distribution(pkg).version
            except Exception:
                versions[pkg] = "not found"
    except Exception:
        versions = {}

    return {
        "timestamp": datetime.now().isoformat(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "packages": versions,

    }


def print_environment_info(env_info: dict):
    print("=" * 60)
    print("ENVIRONMENT INFO")
    print("=" * 60)
    for k, v in env_info.items():
        if k != "package_versions":
            print(f"  {k}: {v}")
    print("  packages:")
    for pkg, ver in env_info.get("package_versions", {}).items():
        print(f"    {pkg}: {ver}")
    print("=" * 60)


# ============================================================
# GMM PARAMETER SAVING / LOADING
# ============================================================

def save_gmm_params(mu_list, Sigma_list, alpha, mog_means, mog_variances,
                    weights, x_star, save_dir: str, experiment_name: str):
    """
    Save GMM parameters to disk so the exact same target distribution
    can be reloaded without regenerating.
    """
    path = os.path.join(save_dir, f"{experiment_name}_gmm_params.pt")
    torch.save({
        "mu_list":       mu_list,
        "Sigma_list":    Sigma_list,
        "alpha":         alpha,
        "mog_means":     mog_means,
        "mog_variances": mog_variances,
        "weights":       weights,
        "x_star":        x_star,
    }, path)
    print(f"[GMM] Parameters saved to {path}")
    return path


def load_gmm_params(save_dir: str, experiment_name: str):
    """Load previously saved GMM parameters."""
    path = os.path.join(save_dir, f"{experiment_name}_gmm_params.pt")
    if not os.path.exists(path):
        return None
    data = torch.load(path)
    print(f"[GMM] Parameters loaded from {path}")
    return (
        data["mu_list"], data["Sigma_list"], data["alpha"],
        data["mog_means"], data["mog_variances"],
        data["weights"], data["x_star"]
    )


# ============================================================
# MODEL CHECKPOINT SAVING / LOADING
# ============================================================

def save_model_checkpoint(model, model_name: str, save_dir: str,
                           experiment_name: str, seed: int):
    """Save a trained model checkpoint."""
    path = os.path.join(
        save_dir,
        f"{experiment_name}_{model_name}_seed{seed}.pt"
    )
    torch.save(model.state_dict(), path)
    print(f"[Checkpoint] {model_name} saved to {path}")
    return path


def load_model_checkpoint(model, model_name: str, save_dir: str,
                           experiment_name: str, seed: int, device):
    """
    Load a model checkpoint if it exists.
    Returns True if loaded, False if not found.
    """
    path = os.path.join(
        save_dir,
        f"{experiment_name}_{model_name}_seed{seed}.pt"
    )
    if not os.path.exists(path):
        print(f"[Checkpoint] No checkpoint found for {model_name} at {path}")
        return False
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"[Checkpoint] {model_name} loaded from {path}")
    return True


from huggingface_hub import hf_hub_download

HF_REPO_ID = "anon-submission-cdm/cdm-inverse-design"


def load_checkpoint_with_hf_fallback(model, model_name, checkpoint_dir, experiment_name, seed, device):
    """Load checkpoint locally, or download from HuggingFace if not found."""
    local_path = os.path.join(checkpoint_dir, f"{experiment_name}_{model_name}_seed{seed}.pt")

    if not os.path.exists(local_path):
        print(f"[Checkpoint] Not found locally, downloading from HuggingFace...")
        os.makedirs(checkpoint_dir, exist_ok=True)
        hf_path = f"simulations/checkpoints/{experiment_name}/{experiment_name}_{model_name}_seed{seed}.pt"
        try:
            downloaded = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=hf_path,
                local_dir=os.path.dirname(os.path.dirname(os.path.dirname(checkpoint_dir))),  # add one more dirname
                local_dir_use_symlinks=False,
            )
            print(f"[Checkpoint] Downloaded to {downloaded}")
        except Exception as e:
            print(f"[Checkpoint] HuggingFace download failed: {e}")
            return False

    return load_model_checkpoint(model, model_name, checkpoint_dir, experiment_name, seed, device)


# ============================================================
# RESULTS SUMMARY HELPERS
# ============================================================

def summary_row(name, l2_gmm, l2_x, times):
    return {
        "Method":        name,
        "L2 GMM mean":   f"{np.mean(l2_gmm):.4f}",
        "L2 GMM std":    f"{np.std(l2_gmm):.4f}",
        "L2 to x* mean": f"{np.mean(l2_x):.4f}",
        "L2 to x* std":  f"{np.std(l2_x):.4f}",
        "Time mean (s)": f"{np.mean(times):.2f}",
        "Time std (s)":  f"{np.std(times):.2f}",
    }


def top10_stats(name, final_loss, l2_gmm, l2_x, times):
    losses = [fl.item() if hasattr(fl, "item") else fl for fl in final_loss]
    k = min(10, len(losses))
    top10_idx  = np.argsort(losses)[:k]
    top10_loss = [losses[i]  for i in top10_idx]
    top10_gmm  = [l2_gmm[i] for i in top10_idx]
    top10_x    = [l2_x[i]   for i in top10_idx]
    top10_time = [times[i]   for i in top10_idx]
    return {
        "Method":         name,
        "Loss mean":      f"{np.mean(top10_loss):.4f}",
        "Loss std":       f"{np.std(top10_loss):.4f}",
        "L2 GMM mean":    f"{np.mean(top10_gmm):.4f}",
        "L2 GMM std":     f"{np.std(top10_gmm):.4f}",
        "L2 to x* mean":  f"{np.mean(top10_x):.4f}",
        "L2 to x* std":   f"{np.std(top10_x):.4f}",
        "Time mean (s)":  f"{np.mean(top10_time):.2f}",
        "Time std (s)":   f"{np.std(top10_time):.2f}",
        "Top-k selected": k,
    }


