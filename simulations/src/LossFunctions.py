import torch
from torch import nn
import ot
import dist_utils
#### MMD Loss #####

class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None, device='cpu'):
        super().__init__()
        self.device = device
        self.bandwidth_multipliers = (mul_factor ** (torch.arange(n_kernels, device=device) - n_kernels // 2))
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.sum() / (n_samples ** 2 - n_samples)
        return self.bandwidth

    def forward(self, X):
        X = X.to(self.device)
        L2_distances = torch.cdist(X, X, p=2) ** 2
        bandwidth = self.get_bandwidth(L2_distances)
        scaled = L2_distances[None, ...] / (bandwidth * self.bandwidth_multipliers[:, None, None])
        return torch.exp(-scaled).sum(dim=0)


class EnergyKernel(nn.Module):
    """
    k(x, y) = -||x - y||_2. Plugging this into MMDLoss recovers (twice) the
    energy distance between the two distributions (Sejdinovic et al. 2013,
    "Equivalence of distance-based and RKHS-based statistics in hypothesis
    testing") -- a non-Gaussian alternative to RBF with no bandwidth to tune.
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

    def forward(self, X):
        X = X.to(self.device)
        return -torch.cdist(X, X, p=2)


class MMDLoss(nn.Module):
    def __init__(self, kernel=None, device='cpu'):
        super().__init__()
        self.device = device
        self.kernel = kernel if kernel else RBF(device=device)
        self.to(device)

    def forward(self, X, Y):
        X = X.to(self.device)
        Y = Y.to(self.device)
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY



