import torch
import numpy as np
from scipy.optimize import fsolve
import torch.nn.functional as F
from scipy.stats import qmc
from torch.distributions import Normal, MultivariateNormal
from tqdm import tqdm

########### Mixture of Gaussian's ###########
#### Sampling ####
def sample_univariate_gaussian(mean, variance):
    """
    Sample from univariate Gaussian using reparameterization trick

    Args:
        mean: tensor of shape [1] or scalar
        variance: tensor of shape [1] or scalar

    Returns:
        sample: tensor of shape [1]
    """
    std = torch.sqrt(variance).squeeze()
    epsilon = torch.randn(1, dtype=mean.dtype, device=mean.device)  # match device

    return mean.squeeze() + std * epsilon

def sample_multivariate_gaussian(mean, covariance):
    """
    Sample from multivariate Gaussian using reparameterization trick

    Args:
        mean: tensor of shape [D]
        covariance: tensor of shape [D, D]

    Returns:
        sample: tensor of shape [D]
    """
    # Ensure covariance matrix is positive definite
    eps = 1e-6 * torch.eye(covariance.size(0), dtype=covariance.dtype)
    L = torch.linalg.cholesky(covariance + eps)

    # Generate standard normal noise
    epsilon = torch.randn(mean.size(), dtype=mean.dtype, device=mean.device)  # match device

    # Apply reparameterization trick
    return mean + torch.matmul(L, epsilon)


def mog_covariance(means, variances, weights):
    """
    Compute covariance matrix of a mixture of Gaussians.

    means: list of (d,) tensors OR already stacked (K, d) tensor
    variances: list of (d, d) covariance matrices
    weights: list or tensor of scalars, length K
    """
    if isinstance(means, list):
        means = torch.stack(means)  # (K, d)

    if isinstance(weights, torch.Tensor):
        weights = weights.clone().to(dtype=means.dtype, device=means.device)
    else:
        weights = torch.tensor(weights, dtype=means.dtype, device=means.device)
    weights = weights / weights.sum()

    d = means[0].flatten().shape[0]

    # mixture mean
    means_flat = torch.stack([m.flatten() for m in means])  # (K, d)
    mu = torch.sum(weights[:, None] * means_flat, dim=0)

    # mixture covariance
    cov = torch.zeros(d, d, device=means.device, dtype=means.dtype)
    for k in range(len(weights)):
        Sigma_k = variances[k]

        diff = (means_flat[k] - mu).unsqueeze(1)
        cov += weights[k] * (Sigma_k + diff @ diff.T)

    return cov


def generate_mog_samples(n_samples, xi, S, beta, tau=7.5,device="cpu",kernel_func=None):
    """
    Generate samples from mixture of Gaussians using reparameterization trick

    Args:
        n_samples: number of samples to generate
        xi: list of means (each can be scalar or vector)
        S: list of variances/covariance matrices
        beta: mixture weights
        tau: temperature for Gumbel-Softmax
    """
    # Ensure xi and S are tensors
    if isinstance(xi, list):
        xi = torch.stack(xi)
    if isinstance(S, list):
        S = torch.stack(S)

    # Get dimensions
    is_multivariate = len(xi.shape) > 1 and xi.shape[-1] > 1
    device = xi.device  # Assume all inputs are on same device

    # Compute logits from beta
    logits = torch.log(beta.to(device))

    # Sample component assignments using Gumbel-Softmax
    component_weights = F.gumbel_softmax(logits, tau=tau, hard=False)

    # Sample component indices
    component_idx = torch.multinomial(component_weights, n_samples, replacement=True)
    means = xi[component_idx]
    vars_or_covs = S[component_idx]

    # Generate samples using appropriate sampling function
    samples = []
    for mean, var_or_cov in zip(means, vars_or_covs):
        if is_multivariate:
            sample = sample_multivariate_gaussian(mean, var_or_cov)
        else:
            sample = sample_univariate_gaussian(mean, var_or_cov)
        samples.append(sample)

    samples = torch.stack(samples)
    samples=samples.float()

    if kernel_func is not None:
      samples=kernel_func(samples)


    return samples



def generate_mog_samples_not_differentiable(num_samples, means, variances, weights=None, device="cpu",kernel_func=None):
    """
    Generate samples from a Mixture of Gaussians using torch.distributions - most efficient.
    """
    from torch.distributions import Categorical, MultivariateNormal, MixtureSameFamily

    components = len(means)
    if weights is None:
        weights = torch.ones(components, device=device) / components
    else:
        weights = weights.to(device)

    # Ensure weights sum to 1
    weights = weights / weights.sum()

    # Stack means and covariances for multivariate case
    flattened_means = [m.flatten() for m in means]
    dim = flattened_means[0].shape[0]  # Get actual dimensionality

    means_tensor = torch.stack(flattened_means).to(device)  # [components, dim]

    # Handle covariances based on input format
    if variances[0].numel() == dim * dim:  # Full covariance matrix
        covs_tensor = torch.stack([var.reshape(dim, dim) for var in variances]).to(device)
    else:  # Diagonal variances
        covs_tensor = torch.stack([torch.diag(var.flatten()) for var in variances]).to(device)

    # Create mixture distribution
    mix = Categorical(weights)
    comp = MultivariateNormal(means_tensor, covs_tensor)
    mixture = MixtureSameFamily(mix, comp)

    # Sample all at once
    samples = mixture.sample((num_samples,))
    if kernel_func is not None:
      samples=kernel_func(samples)

    return samples  # [num_samples, dim]


def mog_multivariate_pdf(x, means, covariances, weights=None, log=False):
        """
        Compute PDF or log-PDF of a Mixture of Gaussians (1D or multivariate).
    
        Args:
            x: Tensor of shape [num_points, dim]
            means: List of tensors, each shape [dim]
            covariances: List of tensors, shape [dim] for 1D or [dim, dim] for multivariate
            weights: Tensor of shape [num_components]
            log: Return log-probabilities if True
    
        Returns:
            Tensor of shape [num_points]
        """
        device = x.device
        num_points, dim = x.shape
        num_components = len(means)
    
        # Move everything to correct device
        means = [m.to(device) for m in means]
        covariances = [c.to(device) for c in covariances]
        if weights is None:
            weights = torch.ones(num_components, device=device) / num_components
        else:
            weights = weights.to(device)
    
        total_prob = torch.zeros(num_points, device=device)
    
        for i in range(num_components):
            if dim == 1:
                # Univariate case
                mean = means[i].view(-1)
                std = covariances[i].sqrt().view(-1)
                dist = Normal(mean, std)
                comp_log_prob = dist.log_prob(x.squeeze(-1))  # shape: [num_points]
            else:
                # Multivariate case
                dist = MultivariateNormal(means[i], covariances[i])
                comp_log_prob = dist.log_prob(x)  # shape: [num_points]
    
            total_prob += weights[i] * torch.exp(comp_log_prob)
    
        if log:
            return torch.log(total_prob + 1e-10)  # Avoid log(0)
        else:
            return total_prob




#### Compute conditionals of MoG ####

def split_mean_cov(mu: torch.Tensor, Sigma: torch.Tensor, x_cond: torch.Tensor, cond_indices: list = None) -> tuple:
    """
    Split the mean and covariance matrix for GMM conditioning.
    Args:
        mu (torch.Tensor): Mean vector of shape (d,)
        Sigma (torch.Tensor): Covariance matrix of shape (d, d)
        x_cond (torch.Tensor): Conditioning values
        cond_indices (list, optional): Indices for conditioning. Defaults to last n indices.
    Returns:
        tuple: (mu_1, mu_2, Sigma_11, Sigma_12, Sigma_21, Sigma_22)
    """
    # Get dimensions and device
    d = mu.shape[0]
    device = mu.device

    # Set default cond_indices if not provided
    if cond_indices is None:
        cond_indices = list(range(0, x_cond.shape[0]))

    # Convert indices to torch tensors for indexing, ensuring they're on the same device
    uncond_indices = torch.tensor([i for i in range(d) if i not in cond_indices], device=device)
    cond_indices = torch.tensor(cond_indices, device=device)

    # Split mean
    mu_1 = mu.index_select(0, uncond_indices)
    mu_2 = mu.index_select(0, cond_indices)

    # Split covariance using fancy indexing
    Sigma_11 = Sigma.index_select(0, uncond_indices).index_select(1, uncond_indices)
    Sigma_12 = Sigma.index_select(0, uncond_indices).index_select(1, cond_indices)
    Sigma_21 = Sigma.index_select(0, cond_indices).index_select(1, uncond_indices)
    Sigma_22 = Sigma.index_select(0, cond_indices).index_select(1, cond_indices)

    return mu_1, mu_2, Sigma_11, Sigma_12, Sigma_21, Sigma_22


def conditional_gaussian(mu: torch.Tensor, Sigma: torch.Tensor, x_cond: torch.Tensor,
                         cond_indices: list = None) -> tuple:
    """
    Calculate conditional Gaussian distribution given observations.
    Args:
        mu (torch.Tensor): Mean vector
        Sigma (torch.Tensor): Covariance matrix
        x_cond (torch.Tensor): Conditioning values
        cond_indices (list, optional): Indices for conditioning
    Returns:
        tuple: (conditional mean, conditional covariance)
    """
    # Ensure x_cond is on the same device as mu
    x_cond = x_cond.to(mu.device)
    mu_1, mu_2, Sigma_11, Sigma_12, Sigma_21, Sigma_22 = split_mean_cov(mu, Sigma, x_cond, cond_indices)
    # Compute conditional mean
    # Reshape for matrix multiplication
    mu_1 = mu_1.reshape(-1, 1)
    mu_2 = mu_2.reshape(-1, 1)
    x_cond = x_cond.reshape(-1, 1)
    # Calculate using matrix operations
    Sigma_22_inv = torch.inverse(Sigma_22)  # Keep original implementation
    mu_cond = mu_1 + Sigma_12 @ Sigma_22_inv @ (x_cond - mu_2)
    # Compute conditional covariance
    Sigma_cond = Sigma_11 - Sigma_12 @ Sigma_22_inv @ Sigma_21
    return mu_cond, Sigma_cond


def compute_conditionals(mu_list: list, Sigma_list: list, x_cond: torch.Tensor) -> tuple:
    """
    Calculate conditional distributions for all GMM components.
    Args:
        mu_list (list): List of mean vectors
        Sigma_list (list): List of covariance matrices
        x_cond (torch.Tensor): Conditioning values
    Returns:
        tuple: (conditional means, conditional covariances)
    """
    # Determine device from first mean tensor if available
    device = mu_list[0].device if len(mu_list) > 0 else x_cond.device
    # Ensure x_cond is on the right device
    x_cond = x_cond.to(device)
    condi_mu = []
    condi_sigma = []
    for mu, Sigma in zip(mu_list, Sigma_list):
        # Ensure tensors are on the same device
        mu = mu.to(device)
        Sigma = Sigma.to(device)
        mu_cond, Sigma_cond = conditional_gaussian(mu, Sigma, x_cond)
        condi_mu.append(mu_cond)
        condi_sigma.append(Sigma_cond)
    # Stack tensors along first dimension
    condi_mu = torch.stack(condi_mu)
    condi_sigma = torch.stack(condi_sigma)
    return condi_mu, condi_sigma



def compute_alpha(mu_list: list, Sigma_list: list, alpha: torch.Tensor, x_cond: torch.Tensor,
                  eps: float = 1e-10) -> torch.Tensor:
    """
    Calculate updated GMM weights given conditioning values, avoiding numerical underflow.

    Args:
        mu_list (list): List of mean vectors
        Sigma_list (list): List of covariance matrices
        alpha (torch.Tensor): Initial mixture weights
        x_cond (torch.Tensor): Conditioning values
        eps (float): Small value to prevent log(0)

    Returns:
        torch.Tensor: Updated mixture weights
    """
    # Get device from alpha tensor
    device = alpha.device

    # Ensure x_cond is on the same device
    x_cond = x_cond.to(device).flatten()

    num = torch.zeros(len(alpha), device=device)
    log_probs = torch.zeros(len(alpha), device=device)

    for i in range(len(alpha)):
        # Ensure the tensors are on the same device
        mu = mu_list[i].to(device)
        Sigma = Sigma_list[i].to(device)

        # Split components for the i-th Gaussian
        _, mu_2, _, _, _, Sigma_22 = split_mean_cov(mu, Sigma, x_cond)
        diff = x_cond - mu_2
        Sigma_22_inv = torch.linalg.inv(Sigma_22)

        # Exponent computation (avoid large negative values)
        exponent = -0.5 * diff.T @ Sigma_22_inv @ diff
        exponent = torch.clamp(exponent, min=-1000)  # Prevent extreme negative values

        # Determinant computation (avoid log(0))
        det_Sigma_22 = torch.linalg.det(Sigma_22)
        det_Sigma_22 = torch.clamp(det_Sigma_22, min=eps)  # Ensure non-zero determinant

        norm_const = -0.5 * len(x_cond) * torch.log(
            torch.tensor(2 * torch.pi, device=device)) - 0.5 * torch.log(det_Sigma_22)

        log_probs[i] = torch.log(torch.clamp(alpha[i], min=eps)) + norm_const + exponent

        if torch.isnan(log_probs[i]):
            raise ValueError(f"log_probs[{i}] is NaN! Stopping execution.")

    log_probs_max = torch.max(log_probs)
    log_probs = log_probs - log_probs_max  # Subtract max for numerical stability

    probs = torch.exp(log_probs)
    probs = probs / (probs.sum() + eps)  # Normalize
    return probs

def filter_and_normalize(mog_means, mog_variances, weights, threshold=0.02):
    # Step 1: Masking weights that are under the threshold
    mask = weights >= threshold

    # Filter means, variances, and weights based on the mask
    filtered_means = mog_means[mask]
    filtered_variances = mog_variances[mask]
    filtered_weights = weights[mask]

    # Step 2: Normalize the weights
    filtered_weights = filtered_weights / filtered_weights.sum()

    return filtered_means, filtered_variances, filtered_weights

def generate_gmm_parameters(d: int, num_components: int, seed: int = 42, epsilon: float = 1e-4,mean_scale: float = 5.0):
    """
    Generate random parameters for a Gaussian Mixture Model (GMM).

    Args:
        d (int): Number of dimensions.
        num_components (int): Number of Gaussian components.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: (mu_list, Sigma_list, alpha) where
            - mu_list: List of mean vectors.
            - Sigma_list: List of covariance matrices.
            - alpha: Mixture weights tensor.
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Generate random means for each component
    mu_list = [torch.randn(d, dtype=torch.float64) * mean_scale for _ in range(num_components)]

    # Generate random covariance matrices (positive semi-definite)
    Sigma_list = []
    for _ in range(num_components):
        A = torch.randn(d, d, dtype=torch.float64)  # Random matrix
        Sigma = A @ A.T  # Create a symmetric positive semi-definite covariance matrix
        # Add a small epsilon to the diagonal to make it positive-definite and improve conditioning
        Sigma += torch.eye(d, dtype=torch.float64) * epsilon  # Regularization term

        Sigma_list.append(Sigma)

    # Mixture weights (equal weights)
    alpha = torch.full((num_components,), 1 / num_components, dtype=torch.float64)

    return mu_list, Sigma_list, alpha
def generate_gmm_parameters_Distance(d: int, num_components: int, seed: int = 42, epsilon: float = 1e-4,
                                min_distance: float = 5.0):
    torch.manual_seed(seed)

    mu_list = []
    for i in range(num_components):
        max_attempts = 1000
        for attempt in range(max_attempts):
            # Generate a candidate mean
            candidate_mu = torch.randn(d, dtype=torch.float64) * 5

            # Check if it's far enough from existing means
            valid = True
            for existing_mu in mu_list:
                distance = torch.norm(candidate_mu - existing_mu).item()
                if distance < min_distance:
                    valid = False
                    break

            if valid or attempt == max_attempts - 1:
                mu_list.append(candidate_mu)
                break

    # Generate random covariance matrices (positive semi-definite)
    Sigma_list = []
    for _ in range(num_components):
        A = torch.randn(d, d, dtype=torch.float64)  # Random matrix
        Sigma = A @ A.T  # Create a symmetric positive semi-definite covariance matrix
        # Add a small epsilon to the diagonal to make it positive-definite and improve conditioning
        Sigma += torch.eye(d, dtype=torch.float64) * epsilon  # Regularization term

        Sigma_list.append(Sigma)

    # Mixture weights (equal weights)
    alpha = torch.full((num_components,), 1 / num_components, dtype=torch.float64)

    return mu_list, Sigma_list, alpha

# Example usage
# d = 5  # Number of dimensions
# num_components = 2  # Number of Gaussian components
# mu_list, Sigma_list, alpha = generate_gmm_parameters(d, num_components)
#
# # Print the generated parameters
# print("Means:", mu_list)
# print("Covariance Matrices:", Sigma_list)
# print("Mixture Weights:", alpha)

def get_param_mog_with_target(dim_data=3, num_components=2, device='cpu',
                              conditional_modes=2, uncond_dim=1,distanceOrScale="Distance"):
    """
    Generate GMM parameters with multi-modal conditional distribution.

    Args:
        conditional_modes: Number of modes for the conditional distribution
        condition_values: List of conditioning values to create modes at
    """
    if distanceOrScale=="Distance":
        mu_list, Sigma_list, alpha=generate_gmm_parameters_Distance(d=dim_data, num_components=num_components, seed=1)
    else:
        mu_list, Sigma_list, alpha = generate_gmm_parameters(d=dim_data, num_components=num_components, seed=1)

    if conditional_modes > 1:
        # change
        # Randomly select conditional_modes number of components from mu_list
        selected_indices = torch.randperm(len(mu_list))[:conditional_modes]

        # Get the conditioning part from the first selected component
        conditioning_value = mu_list[selected_indices[0]][:dim_data - uncond_dim].clone()

        # Set the selected conditional_modes components to have the same conditioning part
        for i in range(conditional_modes):
            idx = selected_indices[i]
            mu_list[idx][:dim_data - uncond_dim] = conditioning_value

        # Set x_star to be this conditioning value
        x_star = conditioning_value.float().to(device)
        # changed
    else:
        # Original behavior for conditional_modes <= 1
        samples = generate_mog_samples_not_differentiable(150, mu_list, Sigma_list, alpha).float().to(device)
        idx = torch.randint(0, 150, (1,))
        x_star = samples[idx, :dim_data - uncond_dim]

    # Compute the Y|X=x^*
    mog_means, mog_variances = compute_conditionals(mu_list, Sigma_list, x_star.reshape(-1, 1))
    weights = compute_alpha(mu_list, Sigma_list, alpha, x_star.reshape(-1, 1))

    return mu_list, Sigma_list, alpha, mog_means, mog_variances, weights, x_star


def gmm_l2_distance(mu_p, Sigma_p, w_p, mu_q, Sigma_q, w_q):
    """
    Exact L2 distance between two GMMs:
    ||p - q||^2 = <p,p> - 2<p,q> + <q,q>
    where <f,g> = integral f(x)g(x)dx, closed form for Gaussians.
    """
    import torch

    def gaussian_inner_product(mu1, S1, w1_list, mu2, S2, w2_list):
        # sum_{i,j} w1_i * w2_j * N(mu1_i; mu2_j, S1_i + S2_j)
        total = 0.0
        for mu_i, S_i, w_i in zip(mu1, S1, w1_list):
            for mu_j, S_j, w_j in zip(mu2, S2, w2_list):
                S_sum = S_i + S_j
                diff  = mu_i - mu_j
                d     = mu_i.shape[0]
                sign, logdet = torch.linalg.slogdet(S_sum)
                log_val = -0.5 * (d * torch.log(torch.tensor(2 * 3.14159265)) + logdet
                                  + diff @ torch.linalg.inv(S_sum) @ diff)
                total += w_i.item() * w_j.item() * torch.exp(log_val).item()
        return total

    pp = gaussian_inner_product(mu_p, Sigma_p, w_p, mu_p, Sigma_p, w_p)
    qq = gaussian_inner_product(mu_q, Sigma_q, w_q, mu_q, Sigma_q, w_q)
    pq = gaussian_inner_product(mu_p, Sigma_p, w_p, mu_q, Sigma_q, w_q)
    return pp - 2 * pq + qq