import torch
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from torch import nn, Tensor
from tqdm import tqdm
from torch.distributions import Independent, Normal
from FM_Solver_Extension_module import add_compute_likelihood_allow_grad
from NN_utils import GenericNN,TimeEmbedding


class FMModel(nn.Module):
    def __init__(self,
                 nfeatures: int,
                 condition_on: int,
                 nunits: int,
                 nblocks: int = 4,
                 device: torch.device = None):
        super().__init__()
        self.nfeatures = nfeatures
        self.nunits = nunits
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.condition_on = condition_on
        self.condition = False

        if condition_on>0:
            self.condition=True
            self.cond_embed = nn.Linear(condition_on, nunits)

        self.inblock = nn.Linear(nfeatures, nunits)
        self.time_embed = TimeEmbedding(nunits)

        self.vf = GenericNN(
            input_dim=nunits,
            hidden_dims=[nunits] * nblocks,  # depth controls number of residual layers
            output_dim=nfeatures,
        )


        # instantiate an affine path object - ensure it's using the right device
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.solver = ODESolver(velocity_model=self._wrapped_forward)

        # Move the entire model to the device
        self.to(self.device)

    def _wrapped_forward(self, t, x, y=None):
        return self.forward(x, t, x_cond=y)

    def forward(self,  x: Tensor, t: Tensor, x_cond: Tensor=None) -> Tensor:
        x_proj = self.inblock(x)
        t_emb = self.time_embed(t)

        if self.condition and x_cond is not None:
            cond_emb = self.cond_embed(x_cond)
            val = x_proj + cond_emb
        else:
            val = x_proj

        return self.vf(val, t_emb)


    def train_FM(
    self,
    lr=0.001,
    batch_size=1024,
    data_generator=None,
    nepochs=10000,   # total number of iterations
    guidance_dropout: float = 0.1,):
      assert data_generator is not None, "Need data_generator for training"

      optim_FM = torch.optim.Adam(self.parameters(), lr=lr)

      pbar = tqdm(range(nepochs))
      epoch_loss = 0.0
      num_batches = 0

      for step in pbar:
          optim_FM.zero_grad()

          # one fresh batch per iteration
          batch_x = data_generator(batch_size, device=self.device)


          y = batch_x[:, self.condition_on:].to(torch.float32).to(self.device)
          x_cond_full = batch_x[:, :self.condition_on].to(torch.float32).to(self.device)

          x_0 = torch.randn_like(y).to(self.device)
          t = torch.rand(y.shape[0], device=self.device)

          path_sample = self.path.sample(t=t, x_0=x_0, x_1=y)
          x_t = path_sample.x_t.to(self.device)
          dx_t = path_sample.dx_t.to(self.device)
          t = path_sample.t.to(self.device)

          ### Classifier-Free Guidance
          if self.condition and self.condition_on > 0:
              guidance_mask = torch.rand(y.shape[0], device=self.device) >= guidance_dropout
              predicted_velocity = torch.zeros_like(dx_t)

              if guidance_mask.any():
                  cond_output = self.forward(
                      x_t[guidance_mask],
                      t[guidance_mask],
                      x_cond_full[guidance_mask]
                  )
                  predicted_velocity[guidance_mask] = cond_output

              if (~guidance_mask).any():
                  uncond_output = self.forward(
                      x_t[~guidance_mask],
                      t[~guidance_mask],
                      None
                  )
                  predicted_velocity[~guidance_mask] = uncond_output
          else:
              predicted_velocity = self.forward(x_t, t)

          loss = torch.pow(predicted_velocity - dx_t, 2).mean()

          epoch_loss += loss.item()
          num_batches += 1

          loss.backward()
          optim_FM.step()

          avg_loss = epoch_loss / num_batches
          pbar.set_description(f"loss: {avg_loss :.6f}")

      self.solver = ODESolver(velocity_model=self._wrapped_forward)
      self.solver = add_compute_likelihood_allow_grad(self.solver)
    def sample(self, num_sample, y=None, return_intermediates=False, step_size=0.05, x_init=None):
        """
        Sample from the vector field model with conditioning.

        Args:
            num_sample: Number of samples to generate
            y: Conditioning value(s). Can be:
               - A scalar that will be expanded to all samples
               - A tensor of shape [num_sample, 1] with different values for each sample
               - A tensor of shape [1, 1] that will be expanded
            return_intermediates: Whether to return intermediate steps
            step_size: Step size for the solver
            x_init: Initial values (if None, will use random normal)

        Returns:
            Samples from the model
        """
        T = torch.linspace(0, 1, 10)


        if y is not None:
            # Check shape and expand if needed
            if y is not None:
                if y.numel() == 1:
                    # Case: scalar y → expand to [num_sample, 1]
                    y = y * torch.ones(num_sample, dtype=torch.float32, device=self.device).reshape(-1, 1)
                elif y.shape[0] == num_sample and (y.dim() == 1 or (y.dim() == 2 and y.shape[1] == 1)):
                    # Case: [num_sample] or [num_sample, 1] → reshape to [num_sample, 1]
                    y = y.reshape(num_sample, 1)

        # Initialize x values if needed
        if x_init is None:
            x_init = torch.randn((num_sample, self.nfeatures), dtype=torch.float32, device=self.device)

        # Ensure x_init is on the correct device
        x_init = x_init.to(device=self.device)

        # Call the solver with the prepared inputs
        return self.solver.sample(
            time_grid=T,
            x_init=x_init,
            method='midpoint',
            step_size=step_size,
            return_intermediates=return_intermediates,
            y=y,
            enable_grad=True
        )

    def compute_likelihood(self, Target, y, step_size=0.05, enable_grad=True,
                           Train_consistency=False, exact_divergence=False,
                           time_grid=None):

        gaussian_log_density = Independent(Normal(torch.zeros(self.nfeatures, device=self.device),
                                                  torch.ones(self.nfeatures, device=self.device)), 1).log_prob

        # If no time grid is provided, create one with multiple points
        if time_grid is None and Train_consistency:
            # Create a time grid from 1.0 to 0.0 with small steps
            time_grid = torch.linspace(1.0, 0.0, 101, device=self.device)  # 101 points for 100 intervals
        elif time_grid is None:
            time_grid = torch.tensor([1.0, 0.0], device=self.device)

        if exact_divergence:
            sol, log_det = self.solver.compute_likelihood_allow_grad(
                x_1=Target, method='midpoint', step_size=step_size,
                exact_divergence=exact_divergence, log_p0=gaussian_log_density,
                y=y, enable_grad=enable_grad,
                Train_consistency=Train_consistency,
                time_grid=time_grid
            )
            return sol, log_det

        # For non-exact, average over multiple samples (as in your original code)
        num_acc = 20
        log_p_acc = 0

        for i in range(num_acc):
            sol, log_p = self.solver.compute_likelihood_allow_grad(
                x_1=Target, method='midpoint', step_size=step_size,
                exact_divergence=False, log_p0=gaussian_log_density,
                y=y, enable_grad=enable_grad,
                Train_consistency=Train_consistency,
                time_grid=time_grid
            )
            log_p_acc += log_p

        log_p_acc /= num_acc

        if Train_consistency:
            return sol, log_p
        return log_p_acc



