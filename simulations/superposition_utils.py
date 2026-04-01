
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.notebook import trange


def to_circular_fixed_radius(all_data, radius=3.0, indices_to_transform=[0], x_min_max_dict=None):
   """
   Transforms selected dimensions of the data into circular coordinates.

   Args:
       all_data: Tensor of shape [N, D]
       radius: Radius of the circular base
       indices_to_transform: List of indices to convert into circular coordinates
       x_min_max_dict: Optional dict {index: (x_min, x_max)} for normalization

   Returns:
       Tensor of shape [N, 2 * len(indices_to_transform) + (D - len(indices_to_transform))]
   """
   N, D = all_data.shape
   transformed = []
   used_indices = set(indices_to_transform)

   for idx in indices_to_transform:
       x = all_data[:, idx]

       # Get min/max if provided, else compute from data
       if x_min_max_dict and idx in x_min_max_dict:
           x_min, x_max = x_min_max_dict[idx]
       else:
           x_min, x_max = torch.min(x), torch.max(x)

       # Normalize x into angle θ
       angles = 2 * torch.pi * (x - x_min) / (x_max - x_min + 1e-8)  # avoid div by zero
       x_circle = (radius * torch.cos(angles)).unsqueeze(1)
       y_circle = (radius * torch.sin(angles)).unsqueeze(1)

       transformed.append(x_circle)
       transformed.append(y_circle)

   # Add the rest of the dimensions not transformed
   remaining = [all_data[:, i].unsqueeze(1) for i in range(D) if i not in used_indices]
   return torch.cat(transformed + remaining, dim=1)


def vector_field(model, t, x, device='cpu',cond=None):
    """
    Compute vector field and its divergence using JVP (Jacobian-Vector Product)
    This replaces the JAX vector_field function
    """
    model.eval()

    # Ensure we can compute gradients
    if not x.requires_grad:
        x = x.clone().detach().requires_grad_(True)

    # Ensure t has proper format
    if hasattr(t, 'clone'):
        t_var = t.clone().detach()
    else:
        t_var = torch.tensor(t, device=device, dtype=torch.float32)

    # Make sure t_var has the right shape
    if t_var.dim() == 0:
        t_var = t_var.expand(x.shape[0], 1)
    elif t_var.shape[0] != x.shape[0]:
        t_var = t_var.expand(x.shape[0], -1)

    # Compute the score function
    sdlogdx_val = model(t_var, x, cond=cond)

    return sdlogdx_val.detach()

def get_score_vector_field(model, t, x,cond):
    """
    Get the vector field u_t for the reverse SDE from the score.
    u_t = -f_t + (g_t^2/2) * ∇log q_t
    """
    model.eval()
    with torch.no_grad():
        score = model(t, x,cond=cond)  # ∇log q_t
        f_t = model.dlog_alphadt(t) * x  # drift of forward process
        g_sq = 2 * torch.exp(model.log_sigma(t)) * model.beta_func(t)
        return -f_t + (g_sq / 2) * score



def compute_matrix_elements(models, x_t, current_t, dt, noise, device, cond_vec):
    """
    Generalized compute_matrix_elements for N models.

    Args:
        models: list of trained models
        x_t: current state (batch_size, ndim)
        current_t: current time (batch_size, 1)
        dt: time step
        noise: Brownian motion increment (batch_size, ndim)
        device: torch device
        cond_vec: list of conditioning vectors (same length as models)

    Returns:
        A: coefficient matrix (batch_size, N+1, N+1)
        b: right-hand side vector (batch_size, N+1)
    """
    N = len(models)
    batch_size, ndim = x_t.shape

    scores = []
    u_ts = []

    # Get scores and vector fields
    with torch.no_grad():
        for model, cond in zip(models, cond_vec):
            scores.append(model(current_t, x_t, cond=cond))
            u_ts.append(get_score_vector_field(model, current_t, x_t, cond=cond))

    # Build coefficient matrix A
    A = torch.zeros(batch_size, N + 1, N + 1, device=device)

    for i in range(N):
        for j in range(N):
            A[:, i, j] = dt * torch.sum(u_ts[j] * scores[i], dim=1)
        A[:, i, N] = -1  # last column: -d_log

    # Last row for constraint: sum(κ_i) = 1
    A[:, N, :N] = 1
    A[:, N, N] = 0

    # Compute b vector
    div_f = model.dlog_alphadt(current_t) * ndim  # scalar (batch_size, 1)
    f_t = model.dlog_alphadt(current_t) * x_t
    g_sq = 2 * torch.exp(model.log_sigma(current_t)) * model.beta_func(current_t)
    g = torch.sqrt(g_sq)

    b = torch.zeros(batch_size, N + 1, device=device)
    for i in range(N):
        score = scores[i]
        term2 = torch.sum((f_t - (g_sq / 2) * score) * score, dim=1)
        noise_term = 0  # Can be re-enabled if needed: torch.sum(g * noise * score, dim=1)
        b[:, i] = -(div_f.squeeze(-1) + term2 * dt + noise_term)

    b[:, N] = 1  # constraint: sum(κ_i) = 1

    return A, b


def solve_linear_system(A, b):
    """
    Solve the linear system A * κ = b.

    Returns:
        kappa: solution (batch_size, 3) containing [κ₁, κ₂, d_log]
    """
    try:
        kappa = torch.linalg.solve(A, b)
    except:
        # Fallback to least squares for numerical stability
        kappa = torch.linalg.lstsq(A, b, rcond=None).solution

    return kappa


def generate_isosurface_samples(model_cond, cond_l=None, bs=512, device='cpu',diffusion_steps=1e-3):
    """
    Generate samples from the isosurface (equal density of both models)
    """
    dt = diffusion_steps
    t = 1.0
    n = int(t / dt)
    # change
    ndim = model_cond.gen_features  # Use gen_features instead of nfeatures
    # changed

    # Initialize
    x_gen = torch.zeros(bs, n + 1, ndim, device=device)
    x_gen[:, 0, :] = torch.randn(bs, ndim, device=device)

    current_t = torch.full((bs, 1), t, device=device)

    if cond_l is not None:
        cond_list = [c.unsqueeze(0).repeat(bs, 1) for c in cond_l]
    else:
        cond_list = []

    model_cond.eval()

    # Don't use torch.no_grad() here because we need gradients for divergence computation
    for i in trange(n):
        x_t = x_gen[:, i, :].clone()  # Clone to avoid in-place operations

        # Get vector fields and divergences for both models
        sdlogdx_list = [vector_field(model_cond, current_t, x_t, device, cond=cond) for cond in
                        cond_list]
        batch_size, _ = x_t.shape

        # Generate noise for this step
        noise = torch.randn(batch_size, ndim, device=device)

        # Compute matrix elements
        models = (model_cond,) * len(cond_l)

        # cond_vec = (cond1,cond2)
        A, b =compute_matrix_elements(models, x_t, current_t, dt, noise, device, cond_list)

        # Compute kappa for interpolation
        kappa_solutions = solve_linear_system(A, b)

        # Compute interpolated dynamics
        with torch.no_grad():  # Only disable gradients for the update steps
            interpolated_sdlogdx = 0
            for idx in range(0, len(sdlogdx_list)):
                interpolated_sdlogdx += kappa_solutions[:, idx:idx + 1] * (sdlogdx_list[idx])
            dxdt = (model_cond.dlog_alphadt(current_t) * x_t - model_cond.beta_func(current_t) * (interpolated_sdlogdx))

            # Update position
            x_gen[:, i + 1, :] = x_t - dt * dxdt
            current_t -= dt

    # change
    # If conditioning is enabled, concatenate conditioning values back
    if model_cond.condition and cond_l:
        # Take the first conditioning value (assuming all samples use same conditioning)
        cond_expanded = cond_list[0].unsqueeze(1).expand(-1, n + 1, -1)  # [bs, n+1, condition_on]
        x_gen = torch.cat([cond_expanded, x_gen], dim=2)  # [bs, n+1, nfeatures]

    return x_gen, _

def compute_matrix_elements(models, x_t, current_t, dt, noise, device, cond_vec):
    """
    Generalized compute_matrix_elements for N models.

    Args:
        models: list of trained models
        x_t: current state (batch_size, ndim)
        current_t: current time (batch_size, 1)
        dt: time step
        noise: Brownian motion increment (batch_size, ndim)
        device: torch device
        cond_vec: list of conditioning vectors (same length as models)

    Returns:
        A: coefficient matrix (batch_size, N+1, N+1)
        b: right-hand side vector (batch_size, N+1)
    """
    N = len(models)
    batch_size, ndim = x_t.shape

    scores = []
    u_ts = []

    # Get scores and vector fields
    with torch.no_grad():
        for model, cond in zip(models, cond_vec):
            scores.append(model(current_t, x_t, cond=cond))
            u_ts.append(get_score_vector_field(model, current_t, x_t, cond=cond))

    # Build coefficient matrix A
    A = torch.zeros(batch_size, N + 1, N + 1, device=device)

    for i in range(N):
        for j in range(N):
            A[:, i, j] = dt * torch.sum(u_ts[j] * scores[i], dim=1)
        A[:, i, N] = -1  # last column: -d_log

    # Last row for constraint: sum(κ_i) = 1
    A[:, N, :N] = 1
    A[:, N, N] = 0

    # Compute b vector
    div_f = model.dlog_alphadt(current_t) * ndim  # scalar (batch_size, 1)
    f_t = model.dlog_alphadt(current_t) * x_t
    g_sq = 2 * torch.exp(model.log_sigma(current_t)) * model.beta_func(current_t)
    g = torch.sqrt(g_sq)

    b = torch.zeros(batch_size, N + 1, device=device)
    for i in range(N):
        score = scores[i]
        term2 = torch.sum((f_t - (g_sq / 2) * score) * score, dim=1)
        noise_term = 0  # Can be re-enabled if needed: torch.sum(g * noise * score, dim=1)
        b[:, i] = -(div_f.squeeze(-1) + term2 * dt + noise_term)

    b[:, N] = 1  # constraint: sum(κ_i) = 1

    return A, b


def solve_linear_system(A, b):
    """
    Solve the linear system A * κ = b.

    Returns:
        kappa: solution (batch_size, 3) containing [κ₁, κ₂, d_log]
    """
    try:
        kappa = torch.linalg.solve(A, b)
    except:
        # Fallback to least squares for numerical stability
        kappa = torch.linalg.lstsq(A, b, rcond=None).solution

    return kappa

#
# def generate_isosurface_samples(model_cond, cond_l=None,bs=512,device='cpu'):
#     """
#     Generate samples from the isosurface (equal density of both models)
#     """
#     dt = 1e-3
#     t = 1.0
#     n = int(t / dt)
#     ndim=model_cond.nfeatures
#
#     # Initialize
#     x_gen = torch.zeros(bs, n + 1, ndim, device=device)
#     x_gen[:, 0, :] = torch.randn(bs, ndim, device=device)
#
#     current_t = torch.full((bs, 1), t, device=device)
#
#
#     if cond_l is not None:
#
#         cond_list = [c.unsqueeze(0).repeat(bs, 1)  for c in cond_l]
#     else:
#         cond_list = []
#
#     model_cond.eval()
#
#     # Don't use torch.no_grad() here because we need gradients for divergence computation
#     for i in trange(n):
#         x_t = x_gen[:, i, :].clone()  # Clone to avoid in-place operations
#
#         # Get vector fields and divergences for both models
#         sdlogdx_list = [vector_field(model_cond, current_t, x_t, device, cond=cond) for cond in cond_list]
#         batch_size, _ = x_t.shape
#
#         # # Generate noise for this step
#         noise = torch.randn(batch_size, ndim, device=device)
#
#         # # Compute matrix elements
#         models = (model_cond,) * len(cond_l)
#
#
#         # cond_vec = (cond1,cond2)
#         A, b = compute_matrix_elements(models, x_t, current_t, dt, noise, device,cond_list)
#
#         # Compute kappa for interpolation
#         kappa_solutions=solve_linear_system(A, b)
#
#         # Compute interpolated dynamics
#         with torch.no_grad():  # Only disable gradients for the update steps
#             interpolated_sdlogdx=0
#             for idx in range(0, len(sdlogdx_list)):
#               interpolated_sdlogdx += kappa_solutions[:, idx:idx+1] * (sdlogdx_list[idx] )
#             dxdt = (model_cond.dlog_alphadt(current_t) * x_t -model_cond.beta_func(current_t) * (interpolated_sdlogdx ))
#
#             # Update position
#             x_gen[:, i + 1, :] = x_t - dt * dxdt
#             current_t -= dt
#
#     return x_gen, _