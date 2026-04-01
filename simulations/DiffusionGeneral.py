# import torch
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import trange
# from NN_utils import GenericNN,TimeEmbedding  # Assuming you've defined this elsewhere
#
# class DiffusionModelIncludingCond(nn.Module):
#     def __init__(self, nfeatures: int, nunits: int = 128, condition: bool = False,
#                  condition_on: int = 1, diffusion_steps: int = 100, nblocks: int = 6, device=None):
#         super(DiffusionModel, self).__init__()
#         self.condition = condition
#         self.nfeatures = nfeatures
#         self.nunits = nunits
#         self.diffusion_steps = diffusion_steps
#         self.condition_on = condition_on
#         self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#         if condition:
#             self.cond_embed = nn.Linear(condition_on, nunits)
#
#         self.inblock = nn.Linear(nfeatures, nunits)
#         self.out = nn.Linear(nunits, nfeatures)
#         self.time_embed = TimeEmbedding(nunits)
#         self.net = GenericNN(nunits, [nunits] * nblocks, nunits).to(self.device)
#
#         self.beta_0 = 0.1
#         self.beta_1 = 20.0
#         self.to(self.device)
#
#     def log_alpha(self, t):
#         return -0.5 * t * self.beta_0 - 0.25 * t ** 2 * (self.beta_1 - self.beta_0)
#
#     def dlog_alphadt(self, t):
#         return -0.5 * self.beta_0 - 0.5 * t * (self.beta_1 - self.beta_0)
#
#     def log_sigma(self, t):
#         return torch.log(t)
#
#     def dlog_sigmadt(self, t):
#         return 1.0 / t
#
#     def beta_func(self, t):
#         return 1 + 0.5 * t * self.beta_0 + 0.5 * t ** 2 * (self.beta_1 - self.beta_0)
#
#     def q_t(self, data, t):
#         data = data.to(self.device)
#         t = t.to(self.device)
#         eps = torch.randn_like(data, device=self.device)
#         log_a = self.log_alpha(t).to(self.device)
#         log_s = self.log_sigma(t).to(self.device)
#         x_t = torch.exp(log_a) * data + torch.exp(log_s) * eps
#         return eps, x_t
#
#     def sm_loss(self, x, cond, t, eps, x_t):
#         pred_eps = self(t, x_t, cond)
#         return torch.mean((eps + pred_eps) ** 2)
#
#     def forward(self, t, x, cond=None):
#         x = x.to(self.device)
#         t = t.to(self.device)
#         t_emb = self.time_embed(t)
#
#         x_proj = self.inblock(x)
#         if self.condition and cond is not None:
#             cond_emb = self.cond_embed(cond.to(self.device))
#             val = x_proj + cond_emb
#         else:
#             val = x_proj
#
#         val = self.net(val, t_emb)
#         return self.out(val)
#         # return self.out(val) + x  # residual connection
#
#     def vector_field_SDE(self, t, x, cond=None,track_gradients=False):
#         context = torch.no_grad() if not track_gradients else torch.enable_grad()
#         with context:
#             out = self(t, x, cond)
#             dxdt = self.dlog_alphadt(t) * x - 2 * self.beta_func(t) * out
#         return dxdt
#
#
#     def sample(self, nsamples=512, cond_val=None, track_gradients=False):
#         self.eval()
#         dt = 1e-2
#         t = 1.0
#         n = int(t / dt)
#
#         x = torch.randn(nsamples, self.nfeatures, device=self.device, requires_grad=track_gradients)
#         samples = [x]
#
#         if self.condition and cond_val is not None:
#             if cond_val.dim() == 1:
#                 cond = cond_val.unsqueeze(0).repeat(nsamples, 1)
#             elif cond_val.dim() == 2 and cond_val.shape[0] == nsamples:
#                 cond = cond_val
#             else:
#                 raise ValueError(f"cond_val shape invalid: {cond_val.shape}")
#
#         current_t = torch.full((nsamples, 1), t, device=self.device)
#
#         context = torch.enable_grad() if track_gradients else torch.no_grad()
#         with context:
#             for _ in range(n):
#                 noise = torch.randn(nsamples, self.nfeatures, device=self.device)
#
#                 dx = (
#                         -dt * self.vector_field_SDE(current_t, samples[-1], cond)
#                         + torch.sqrt(2 * torch.exp(self.log_sigma(current_t)) * self.beta_func(current_t) * dt) * noise
#                 )
#                 x_next = samples[-1] + dx  # Not in-place
#                 samples.append(x_next)
#                 current_t -= dt
#
#         x_gen = torch.stack(samples, dim=1)  # [nsamples, n+1, nfeatures]
#         return x_gen
#
#     def train_model(self, data_generator=None, dataloader=None, num_iterations=5000):
#         assert data_generator or dataloader, "Need either data_generator or dataloader"
#
#         optimizer = optim.Adam(self.parameters(), lr=2e-4)
#         bs = 512
#         progress_bar = trange(num_iterations)
#         dataloader_iter = iter(dataloader) if dataloader else None
#
#         for step in progress_bar:
#             self.train()
#             optimizer.zero_grad()
#
#             if dataloader_iter:
#                 try:
#                     batch = next(dataloader_iter)
#                 except StopIteration:
#                     dataloader_iter = iter(dataloader)
#                     batch = next(dataloader_iter)
#
#                 x = batch[0].to(self.device).float()
#                 cond = batch[1].to(self.device).float() if self.condition and len(batch) > 1 else x[:, :self.condition_on]
#             else:
#                 x = data_generator(bs, device=self.device)
#                 cond = x[:, :self.condition_on] if self.condition else None
#
#             t = torch.rand(x.shape[0], 1, device=self.device)
#             eps, x_t = self.q_t(x, t)
#             loss = self.sm_loss(x, cond, t, eps, x_t)
#             loss.backward()
#             optimizer.step()
#             progress_bar.set_postfix(loss=loss.item())
#
#         return self
#
#     def sample_ddim_step(self, x_t, t, cond=None, dt=1e-2):
#
#         pred_eps = self(t, x_t, cond)
#
#         log_alpha_t = self.log_alpha(t)
#         log_sigma_t = self.log_sigma(t)
#
#         exp_log_alpha = torch.exp(log_alpha_t)
#         exp_log_sigma = torch.exp(log_sigma_t)
#
#         x0_hat = (x_t - exp_log_sigma * pred_eps) / exp_log_alpha
#         t_prev = t - dt
#         log_alpha_prev = self.log_alpha(t_prev)
#         log_sigma_prev = self.log_sigma(t_prev)
#
#         x_prev = (torch.exp(log_alpha_prev) * x0_hat +
#                   torch.exp(log_sigma_prev) * pred_eps)
#
#         return x_prev, x0_hat


#
# ########################### VERSION 2 - without the conditional #####################################
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from NN_utils import GenericNN, TimeEmbedding  # Assuming you've defined this elsewhere


class DiffusionModel(nn.Module):
    def __init__(self, nfeatures: int, nunits: int = 128, condition: bool = False,
                 condition_on: int = 1, diffusion_steps: int = 100, nblocks: int = 6, device=None):
        super(DiffusionModel, self).__init__()
        self.condition = condition
        self.nfeatures = nfeatures
        # change
        self.gen_features = nfeatures - condition_on if condition else nfeatures
        # changed
        self.nunits = nunits
        self.diffusion_steps = diffusion_steps
        self.condition_on = condition_on
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if condition:
            self.cond_embed = nn.Linear(condition_on, nunits)

        # change
        self.inblock = nn.Linear(self.gen_features, nunits)
        self.out = nn.Linear(nunits, self.gen_features)
        # changed
        self.time_embed = TimeEmbedding(nunits)
        self.net = GenericNN(nunits, [nunits] * nblocks, nunits).to(self.device)

        self.beta_0 = 0.1
        self.beta_1 = 20.0
        self.to(self.device)

    def log_alpha(self, t):
        return -0.5 * t * self.beta_0 - 0.25 * t ** 2 * (self.beta_1 - self.beta_0)

    def dlog_alphadt(self, t):
        return -0.5 * self.beta_0 - 0.5 * t * (self.beta_1 - self.beta_0)

    def sigma(self,t):
        return t

    def log_sigma(self, t):
        return torch.log(t)

    def dlog_sigmadt(self, t):
        return 1.0 / t

    def beta_func(self, t):
        return 1 + 0.5 * t * self.beta_0 + 0.5 * t ** 2 * (self.beta_1 - self.beta_0)

    def xai_func(self,t):
        return self.beta_func(t)* self.sigma(t)

    def q_t(self, data, t):
        data = data.to(self.device)
        t = t.to(self.device)
        # change
        if self.condition:
            # Only add noise to the non-conditioning features
            gen_data = data[:, self.condition_on:]
        else:
            gen_data = data

        eps = torch.randn_like(gen_data, device=self.device)
        # changed
        log_a = self.log_alpha(t).to(self.device)
        log_s = self.log_sigma(t).to(self.device)
        # change
        x_t = torch.exp(log_a) * gen_data + torch.exp(log_s) * eps
        # changed
        return eps, x_t

    def sm_loss(self, x, cond, t, eps, x_t):
        pred_eps = self(t, x_t, cond)
        return torch.mean((eps + pred_eps) ** 2)

    def forward(self, t, x, cond=None):
        x = x.to(self.device)
        t = t.to(self.device)
        t_emb = self.time_embed(t)

        x_proj = self.inblock(x)
        if self.condition and cond is not None:
            cond_emb = self.cond_embed(cond.to(self.device))
            val = x_proj + cond_emb
        else:
            val = x_proj

        val = self.net(val, t_emb)
        return self.out(val)
        # return self.out(val) + x  # residual connection

    def vector_field_SDE(self, t, x, cond=None, track_gradients=False):
        context = torch.no_grad() if not track_gradients else torch.enable_grad()
        with context:
            out = self(t, x, cond)
            score_coefficient = self.beta_func(t) + self.xai_func(t)
            dxdt = self.dlog_alphadt(t) * x - score_coefficient * out
        return dxdt

    def compute_log_likelihood_step(self, x_current, dx, current_t, cond=None, dt=1e-2):
        """
        Compute log likelihood change for a single step using Theorem 1

        Args:
            x_current: [nsamples, nfeatures] - current position
            dx: [nsamples, nfeatures] - step increment
            current_t: [nsamples, 1] - current time
            cond: conditioning values (optional)
            dt: time step size

        Returns:
            d_log_q: [nsamples] - log likelihood change for this step
        """
        # Get score function
        score = self(current_t, x_current, cond)

        # Compute terms from Theorem 1
        # f_{1-τ}(x_τ) = (d/dt log α_t) * x
        f_t = self.dlog_alphadt(current_t) * x_current

        # ∇ · f_{1-τ}(x_τ) = (d/dt log α_t) * ndim
        div_f = self.dlog_alphadt(current_t) * self.gen_features

        # g²_{1-τ} = 2 * β_t
        g_sq = 2 * self.beta_func(current_t)

        # Three terms from Theorem 1
        term1 = torch.sum(dx * score, dim=1)  # ⟨dx_τ, ∇log q_{1-τ}(x_τ)⟩
        term2 = div_f.squeeze(-1)  # ⟨∇, f_{1-τ}(x_τ)⟩
        term3 = (torch.sum(f_t * score, dim=1) -
                 (g_sq.squeeze(-1) / 2) * torch.sum(score * score, dim=1))

        # Apply dτ correctly
        d_log_q = term1 + (term2 + term3) * dt

        return d_log_q


    def sample(self, nsamples=512, cond_val=None, track_gradients=False, diffusion_steps=1e-2,
               compute_log_likelihood=True):
        self.eval()
        dt = diffusion_steps
        t = 1.0
        n = int(t / dt)

        # Initialize samples
        x = torch.randn(nsamples, self.gen_features, device=self.device, requires_grad=track_gradients)
        samples = [x]

        # Initialize log likelihood tracking
        if compute_log_likelihood:
            log_likelihoods = torch.zeros(nsamples, device=self.device)
            all_log_ll = [log_likelihoods.clone()]

        # Handle conditioning
        if self.condition and cond_val is not None:
            if cond_val.dim() == 1:
                cond = cond_val.unsqueeze(0).repeat(nsamples, 1)
            elif cond_val.dim() == 2 and cond_val.shape[0] == nsamples:
                cond = cond_val
            else:
                raise ValueError(f"cond_val shape invalid: {cond_val.shape}")
        else:
            cond = None

        current_t = torch.full((nsamples, 1), t, device=self.device)

        context = torch.enable_grad() if track_gradients else torch.no_grad()
        with context:
            for _ in range(n):
                x_current = samples[-1]

                # Generate noise
                noise = torch.randn(nsamples, self.gen_features, device=self.device)

                # Compute dx
                dx = (
                        -dt * self.vector_field_SDE(current_t, x_current, cond)
                        + torch.sqrt(2 * self.xai_func(current_t) * dt) * noise
                )

                # Compute log likelihood change for this step
                if compute_log_likelihood:
                    d_log_q = self.compute_log_likelihood_step(
                        x_current=x_current,
                        dx=dx,
                        current_t=current_t,
                        cond=cond,
                        dt=dt
                    )

                    # Update cumulative log likelihood
                    log_likelihoods += d_log_q
                    all_log_ll.append(log_likelihoods.clone())

                # Update trajectory
                x_next = x_current + dx
                samples.append(x_next)
                current_t -= dt

        x_gen = torch.stack(samples, dim=1)  # [nsamples, n+1, nfeatures]

        # Handle conditioning output
        if self.condition and cond_val is not None:
            cond_expanded = cond.unsqueeze(1).expand(-1, x_gen.shape[1], -1)
            x_gen = torch.cat([cond_expanded, x_gen], dim=2)

        # Return samples and optionally log likelihoods
        if compute_log_likelihood:
            log_ll_tensor = torch.stack(all_log_ll, dim=1)  # [nsamples, n+1]
            return x_gen, log_ll_tensor
        else:
            return x_gen

    def train_model(self, data_generator=None, dataloader=None, num_iterations=5000,batch_size=512):
        assert data_generator or dataloader, "Need either data_generator or dataloader"

        optimizer = optim.Adam(self.parameters(), lr=2e-4)
        progress_bar = trange(num_iterations)
        dataloader_iter = iter(dataloader) if dataloader else None

        for step in progress_bar:
            self.train()
            optimizer.zero_grad()

            if dataloader_iter:
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(dataloader)
                    batch = next(dataloader_iter)

                x = batch[0].to(self.device).float()
                cond = batch[1].to(self.device).float() if self.condition and len(batch) > 1 else x[:,
                                                                                                  :self.condition_on]
            else:
                x = data_generator(batch_size, device=self.device)
                cond = x[:, :self.condition_on] if self.condition else None

            t = torch.rand(x.shape[0], 1, device=self.device)
            eps, x_t = self.q_t(x, t)
            loss = self.sm_loss(x, cond, t, eps, x_t)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())

        return

    def sample_ddim_step(self, x_t, t, cond=None, dt=1e-2):

        pred_eps = self(t, x_t, cond)

        log_alpha_t = self.log_alpha(t)
        log_sigma_t = self.log_sigma(t)

        exp_log_alpha = torch.exp(log_alpha_t)
        exp_log_sigma = torch.exp(log_sigma_t)

        x0_hat = (x_t - exp_log_sigma * pred_eps) / exp_log_alpha
        t_prev = t - dt
        log_alpha_prev = self.log_alpha(t_prev)
        log_sigma_prev = self.log_sigma(t_prev)

        x_prev = (torch.exp(log_alpha_prev) * x0_hat +
                  torch.exp(log_sigma_prev) * pred_eps)

        return x_prev, x0_hat