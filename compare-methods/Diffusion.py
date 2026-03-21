import torch.optim as optim
import torch
import torch.nn as nn
from NN_utils import GenericNN, TimeEmbedding
from tqdm import tqdm


class DiffusionModel(nn.Module):
    def __init__(self, nfeatures: int, nunits: int = 128, condition: bool = False,
                 condition_on: int = 0, diffusion_steps: int = 100, nblocks: int = 6,
                 device=None):
        super(DiffusionModel, self).__init__()
        self.condition = condition
        self.nfeatures = nfeatures
        self.nunits = nunits
        self.diffusion_steps = diffusion_steps
        self.condition_on = condition_on
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timesteps = range(self.diffusion_steps - 1, 1, -1)

        if condition:
            self.cond_embed = nn.Linear(condition_on, nunits)

        s = 0.008
        timesteps = torch.tensor(range(0, self.diffusion_steps), dtype=torch.float32)
        schedule = torch.cos((timesteps / self.diffusion_steps + s) / (1 + s) * torch.pi / 2) ** 2
        self.baralphas = (schedule / schedule[0]).to(self.device)
        self.betas = (1 - self.baralphas / torch.cat([self.baralphas[0:1], self.baralphas[:-1]])).to(self.device)
        self.alphas = (1 - self.betas).to(self.device)
        self.inblock = nn.Linear(nfeatures, nunits)
        self.out = nn.Linear(nunits, nfeatures)
        self.time_embed = TimeEmbedding(nunits)
        self.scheduler = type('obj', (object,), {
            'alphas_cumprod': self.baralphas,
            'betas': self.betas,
            'alphas': self.alphas
        })()
        self.net = GenericNN(
            input_dim=nunits,
            hidden_dims=[nunits] * nblocks,
            output_dim=nunits,
        ).to(device)

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor, t: torch.Tensor, x_cond: torch.Tensor = None) -> torch.Tensor:
        x_original = x
        x_proj = self.inblock(x)
        t_emb = self.time_embed(t)
        if self.condition and x_cond is not None:
            cond_emb = self.cond_embed(x_cond)
            val = x_proj + cond_emb
        else:
            val = x_proj
        val = self.net(val, t_emb)
        val = self.out(val)
        return val + x_original

    def noise(self, Xbatch, t):
        device = Xbatch.device
        t = t.to(device=device)
        self.baralphas = self.baralphas.to(device)
        eps = torch.randn(size=Xbatch.shape, device=device)
        noised = (self.baralphas[t] ** 0.5).repeat(1, Xbatch.shape[1]) * Xbatch + \
                 ((1 - self.baralphas[t]) ** 0.5).repeat(1, Xbatch.shape[1]) * eps
        return noised, eps

    def train_model(self, X=None, data_generator=None, nepochs=100, batch_size=2048,
                    condition_on=0, guidance_dropout=0.2, device="cpu",
                    wandb_run=None, wandb_prefix="diffusion", log_every=100):
        """
        Train the diffusion model.

        Args:
            wandb_run:    active wandb run object (or None to skip logging)
            wandb_prefix: prefix for wandb metric keys, e.g. "diff_cond" or "diff_uncond"
            log_every:    log to wandb every this many epochs
        """
        assert X is not None or data_generator is not None, "Need either X or data_generator"

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=nepochs)

        self.to(device)
        losses = []
        pbar = tqdm(range(nepochs))

        for epoch in pbar:
            epoch_loss = 0
            steps = 0

            batches = ([None] if data_generator is not None
                       else range(0, len(X), batch_size))

            for batch_idx in batches:
                if data_generator is not None:
                    Xbatch = data_generator(batch_size, device=device)
                else:
                    Xbatch = X[batch_idx: batch_idx + batch_size]

                if self.condition:
                    x_cond = Xbatch[:, :condition_on]
                else:
                    x_cond = None

                timesteps_t = torch.randint(0, self.diffusion_steps, size=[len(Xbatch), 1])
                noised, eps = self.noise(Xbatch, timesteps_t)

                model_dtype = next(self.parameters()).dtype
                noised      = noised.to(device=device, dtype=model_dtype)
                timesteps_t = timesteps_t.to(device=device, dtype=model_dtype)
                eps         = eps.to(device=device, dtype=model_dtype)

                if self.condition and x_cond is not None:
                    x_cond = x_cond.to(device=device, dtype=model_dtype)
                    guidance_mask = torch.rand(len(Xbatch)) >= guidance_dropout
                    if guidance_mask.sum() < len(Xbatch):
                        cond_outputs  = self(noised[guidance_mask],  timesteps_t[guidance_mask],  x_cond[guidance_mask])
                        uncond_outputs= self(noised[~guidance_mask], timesteps_t[~guidance_mask], None)
                        predicted_noise = torch.zeros_like(noised)
                        predicted_noise[guidance_mask]  = cond_outputs
                        predicted_noise[~guidance_mask] = uncond_outputs
                    else:
                        predicted_noise = self(noised, timesteps_t, x_cond)
                else:
                    predicted_noise = self(noised, timesteps_t)

                loss = loss_fn(predicted_noise, eps)
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                steps += 1

            scheduler.step()
            avg = epoch_loss / steps
            pbar.set_description(f"loss: {avg:.6f}")

            # ── wandb logging ─────────────────────────────────────────────────
            if wandb_run is not None and (epoch % log_every == 0 or epoch == nepochs - 1):
                wandb_run.log({
                    f"{wandb_prefix}/loss": avg,
                    f"{wandb_prefix}/epoch": epoch,
                })

        return losses

    def sample(self, nsamples, condition_x=None, device="cpu", eta=0.0,
               t_start=None, t_end=None):
        self.to(device)
        model_dtype = next(self.parameters()).dtype
        if t_start is None: t_start = self.diffusion_steps - 1
        if t_end   is None: t_end   = 0
        if t_end > t_start:
            raise ValueError(f"t_end ({t_end}) > t_start ({t_start})")

        x = torch.randn(size=(nsamples, self.nfeatures), device=device, dtype=model_dtype)
        xt = [x.clone()]
        pred_x0_l = []

        if condition_x is not None:
            condition_x = condition_x.to(device=device, dtype=model_dtype)

        for t in range(t_start, t_end, -1):
            t_batch = torch.full([nsamples, 1], t, device=device, dtype=model_dtype)
            predicted_noise = self(x, t_batch, condition_x)

            alpha_bar_t    = self.baralphas[t]
            alpha_bar_prev = self.baralphas[t - 1] if t > t_end else torch.tensor(1.0, dtype=model_dtype, device=device)

            sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * \
                      torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)

            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
            dir_xt  = torch.sqrt(1 - alpha_bar_prev - sigma_t ** 2) * predicted_noise
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

            if eta > 0:
                x = x + sigma_t * torch.randn_like(x)

            xt.append(x.clone())
            pred_x0_l.append(pred_x0.clone())

        return x, xt, pred_x0_l

    def sample_ddim_step(self, x_start, t, i=None, condition_x=None, device="cpu", eta=0.0):
        self.to(device)
        model_dtype = next(self.parameters()).dtype

        if condition_x is not None:
            condition_x = condition_x.to(device=device)

        x = x_start.to(device=device, dtype=model_dtype)
        t_batch = torch.full([x.shape[0], 1], t, device=device)
        predicted_noise = self(x, t_batch, condition_x)

        alpha_bar_t    = self.baralphas[t]
        alpha_bar_prev = self.baralphas[t - 1] if t > 0 else torch.tensor(1.0, device=device)

        sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * \
                  torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)

        pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
        dir_xt  = torch.sqrt(1 - alpha_bar_prev - sigma_t ** 2) * predicted_noise
        x_next  = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

        if eta > 0:
            x_next = x_next + sigma_t * torch.randn_like(x)

        return x_next, pred_x0

    def sample_superdiff_and(self, nsamples, cond_pos_val, cond_neg_val, device="cpu", eta=0.0):
        self.eval()
        self.to(device)
        model_dtype = next(self.parameters()).dtype

        x = torch.randn(nsamples, self.nfeatures, device=device, dtype=model_dtype)
        xt = [x.clone()]
        pred_x0_l = []

        cond_pos = torch.full((nsamples, self.condition_on), cond_pos_val, device=device, dtype=model_dtype)
        cond_neg = torch.full((nsamples, self.condition_on), cond_neg_val, device=device, dtype=model_dtype)

        for t in reversed(range(self.diffusion_steps)):
            t_batch = torch.full((nsamples, 1), t, device=device, dtype=model_dtype)
            eps_pos = self(x, t_batch, cond_pos)
            eps_neg = self(x, t_batch, cond_neg)
            eps = 0.5 * (eps_pos + eps_neg)

            alpha_bar_t    = self.baralphas[t]
            alpha_bar_prev = self.baralphas[t - 1] if t > 0 else torch.tensor(1.0, device=device, dtype=model_dtype)
            sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * \
                      torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)

            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            dir_xt  = torch.sqrt(1 - alpha_bar_prev - sigma_t ** 2) * eps
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

            if eta > 0:
                x = x + sigma_t * torch.randn_like(x)

            xt.append(x.clone())
            pred_x0_l.append(pred_x0.clone())

        return x, xt