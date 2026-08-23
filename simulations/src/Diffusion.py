
import torch.optim as optim
import torch
import torch.nn as nn
from NN_utils import GenericNN,TimeEmbedding  # Assuming you've defined this elsewhere
from tqdm import tqdm




class DiffusionModel(nn.Module):
    def __init__(self, nfeatures: int, nunits: int = 128, condition: bool = False,
                 condition_on: int = 0, diffusion_steps: int = 100,nblocks: int = 6,device=None):
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

        # Noise schedule
        s = 0.008
        timesteps = torch.tensor(range(0, self.diffusion_steps), dtype=torch.float32)
        schedule = torch.cos((timesteps / self.diffusion_steps + s) / (1 + s) * torch.pi / 2) ** 2
        self.baralphas = (schedule / schedule[0]).to(self.device)
        self.betas = (1 - self.baralphas / torch.cat([self.baralphas[0:1], self.baralphas[:-1]])).to(self.device)
        self.alphas = (1 - self.betas).to(self.device)
        self.inblock = nn.Linear(nfeatures, nunits)
        self.out= nn.Linear( nunits,nfeatures)
        self.time_embed = TimeEmbedding(nunits)
        self.scheduler = type('obj', (object,), {
            'alphas_cumprod': self.baralphas,
            'betas': self.betas,
            'alphas': self.alphas
        })()
        self.net = GenericNN(
            input_dim=nunits,
            hidden_dims=[nunits] *nblocks,  # depth controls number of residual layers
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
        return val +x_original

    def noise(self, Xbatch, t):
        """
        Function that noises a data point following the diffusion process

        Args:
            Xbatch: Batch of data points to noise
            t: Timesteps for noise

        Returns:
            noised: Noised data
            eps: The added noise
        """
        device = Xbatch.device
        t = t.to(device=device)
        self.baralphas = self.baralphas.to(device)
        eps = torch.randn(size=Xbatch.shape, device=device)
        noised = (self.baralphas[t] ** 0.5).repeat(1, Xbatch.shape[1]) * Xbatch + \
                 ((1 - self.baralphas[t]) ** 0.5).repeat(1, Xbatch.shape[1]) * eps
        return noised, eps



    def train_model(self, X=None, data_generator=None, nepochs=100, batch_size=2048,condition_on=0,
                    guidance_dropout=0.2, device="cpu"):
        """
        Train the diffusion model with classifier-free guidance support.

        Args:
            X: Input data (optional if data_generator is provided)
            data_generator: Function that generates batches of data (optional if X is provided)
            nepochs: Number of training epochs
            batch_size: Batch size for training
            condition_on: Number of features to condition on (only used if self.condition is True)
            guidance_dropout: Probability of dropping the conditioning for classifier-free guidance
            device: Device to train on ("cpu" or "cuda")

        Returns:
            losses: List of training losses
        """
        assert X is not None or data_generator is not None, "Need either X or data_generator"

        # Define loss function and optimizers
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=nepochs)

        # Move model to device
        self.to(device)

        # Loss tracker
        losses = []
        pbar = tqdm(range(nepochs))
        for epoch in pbar:
            epoch_loss = 0
            steps = 0

            if data_generator is not None:
                # Use data generator
                Xbatch = data_generator(batch_size, device=device)
                # Extract conditioning input if needed
                if self.condition:
                    x_cond = Xbatch[:, :condition_on]
                else:
                    x_cond = None

                # Random timesteps
                timesteps = torch.randint(0, self.diffusion_steps, size=[len(Xbatch), 1])

                # Apply noise to Xbatch
                noised, eps = self.noise(Xbatch, timesteps)

                # Get the model's data type
                model_dtype = next(self.parameters()).dtype

                # Move tensors to device and ensure consistent dtype
                noised = noised.to(device=device, dtype=model_dtype)
                timesteps = timesteps.to(device=device, dtype=model_dtype)
                eps = eps.to(device=device, dtype=model_dtype)

                if self.condition and x_cond is not None:
                    x_cond = x_cond.to(device=device, dtype=model_dtype)

                    # Create a mask for classifier-free guidance training
                    guidance_mask = torch.rand(len(Xbatch)) >= guidance_dropout

                    if guidance_mask.sum() < len(Xbatch):
                        cond_outputs = self(noised[guidance_mask], timesteps[guidance_mask], x_cond[guidance_mask])
                        uncond_outputs = self(noised[~guidance_mask], timesteps[~guidance_mask], None)
                        predicted_noise = torch.zeros_like(noised)
                        predicted_noise[guidance_mask] = cond_outputs
                        predicted_noise[~guidance_mask] = uncond_outputs
                    else:
                        predicted_noise = self(noised, timesteps, x_cond)
                else:
                    predicted_noise = self(noised, timesteps)

                # Calculate loss
                loss = loss_fn(predicted_noise, eps)
                losses.append(loss.item())

                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                steps += 1



            else:
                # Use provided dataset X
                for i in range(0, len(X), batch_size):
                    Xbatch = X[i:i + batch_size]

                    # Extract conditioning input if needed
                    if self.condition:
                        x_cond = Xbatch[:, :condition_on]
                    else:
                        x_cond = None

                    # Random timesteps
                    timesteps = torch.randint(0, self.diffusion_steps, size=[len(Xbatch), 1])

                    # Apply noise to Xbatch
                    noised, eps = self.noise(Xbatch, timesteps)

                    # Get the model's data type
                    model_dtype = next(self.parameters()).dtype

                    # Move tensors to device and ensure consistent dtype
                    noised = noised.to(device=device, dtype=model_dtype)
                    timesteps = timesteps.to(device=device, dtype=model_dtype)
                    eps = eps.to(device=device, dtype=model_dtype)

                    if self.condition and x_cond is not None:
                        x_cond = x_cond.to(device=device, dtype=model_dtype)

                        # Create a mask for classifier-free guidance training
                        guidance_mask = torch.rand(len(Xbatch)) >= guidance_dropout

                        if guidance_mask.sum() < len(Xbatch):
                            cond_outputs = self(noised[guidance_mask], timesteps[guidance_mask], x_cond[guidance_mask])
                            uncond_outputs = self(noised[~guidance_mask], timesteps[~guidance_mask], None)
                            predicted_noise = torch.zeros_like(noised)
                            predicted_noise[guidance_mask] = cond_outputs
                            predicted_noise[~guidance_mask] = uncond_outputs
                        else:
                            predicted_noise = self(noised, timesteps, x_cond)
                    else:
                        predicted_noise = self(noised, timesteps)

                    # Calculate loss
                    loss = loss_fn(predicted_noise, eps)
                    losses.append(loss.item())

                    # Optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    steps += 1

            # Update learning rate schedule
            scheduler.step()
            pbar.set_description(f"loss: {epoch_loss / steps:.6f}")

        # Return the loss history
        return losses

    # @torch.no_grad()
    def sample(self, nsamples, condition_x=None, device="cpu", eta=0.0, t_start=None, t_end=None,
               init_noise=None):
        """
        Sample from the DDIM model with optional conditioning.

        Args:
            nsamples: Number of samples to generate.
            condition_x: Optional conditioning input.
            device: Device to run the sampling on ("cpu" or "cuda").
            eta: Controls the stochasticity (0 = deterministic DDIM, 1 = DDPM)
            t_start: Starting timestep (default: diffusion_steps - 1)
            t_end: Ending timestep (default: 0)
            init_noise: Optional [nsamples, nfeatures] tensor used as the initial
                        x instead of a fresh torch.randn draw (e.g. antithetic pairs).

        Returns:
            x: Final sampled points.
            xt: List of intermediate sampled points.
        """
        # Move model to device
        self.to(device)
        # self.mean = self.mean.to(device)
        # self.std = self.std.to(device)
        # Get model dtype
        model_dtype = next(self.parameters()).dtype

        # Set default values for t_start and t_end if not provided
        if t_start is None:
            t_start = self.diffusion_steps - 1
        if t_end is None:
            t_end = 0

        # Ensure t_start and t_end are valid
        if t_end > t_start:
            raise ValueError(f"Invalid time range: t_end ({t_end}) cannot be greater than t_start ({t_start})")

        # Initialize random noise (or a caller-provided noise, e.g. antithetic pairs)
        if init_noise is not None:
            x = init_noise.to(device=device, dtype=model_dtype)
        else:
            x = torch.randn(size=(nsamples, self.nfeatures), device=device, dtype=model_dtype)
        xt = [x.clone()]
        pred_x0_l=[]

        # Convert condition_x to the model's dtype if it exists
        if condition_x is not None:
            # mean = self.mean.flatten()[:condition_x.shape[1]].to(device=device, dtype=condition_x.dtype)
            # std = self.std.flatten()[:condition_x.shape[1]].to(device=device, dtype=condition_x.dtype)
            # condition_x = (condition_x - mean) / std
            condition_x = condition_x.to(device=device, dtype=model_dtype)

        # Denoise step by step
        for t in range(t_start, t_end, -1):
            t_batch = torch.full([nsamples, 1], t, device=device, dtype=model_dtype)
            # Predict noise
            # with torch.no_grad():
            predicted_noise = self(x, t_batch, condition_x)

            # DDIM update rule
            alpha_t = self.alphas[t]
            alpha_bar_t = self.baralphas[t]

            # Get alpha_bar_prev (use t_end for the final step if needed)
            if t > t_end:
                alpha_bar_prev = self.baralphas[t - 1]
            else:
                alpha_bar_prev = torch.tensor(1.0, dtype=model_dtype, device=device)

            # Compute sigma (η in the DDIM paper)
            sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * torch.sqrt(
                1 - alpha_bar_t / alpha_bar_prev)

            # DDIM deterministic update
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma_t ** 2) * predicted_noise
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

            # Add noise only if eta > 0
            if eta > 0:
                noise = torch.randn_like(x)
                x = x + sigma_t * noise

            xt.append(x.clone())
            pred_x0_l.append((pred_x0).clone())
        x = x #* self.std + self.mean  # Un-normalize samples

        return x, xt,pred_x0_l

    def sample_ddim_step(self, x_start, t,i=None, condition_x=None, device="cpu", eta=0.0): # TODO integrate into the sample_ddim
        """
        Perform a single DDIM step from x_start.

        Args:
            x_start: Input sample from which the step is taken.
            t: Current timestep.
            condition_x: Optional conditioning input.
            device: Device to run the sampling on ("cpu" or "cuda").
            eta: Controls the stochasticity (0 = deterministic DDIM, 1 = DDPM).

        Returns:
            x_next: The updated sample after one DDIM step.
        """

        # Move model to device
        self.to(device)

        # Get model dtype
        model_dtype = next(self.parameters()).dtype

        # Convert condition_x to the model's dtype if provided
        if condition_x is not None:
            # mean = self.mean.flatten()[:condition_x.shape[1]].to(device=device, dtype=condition_x.dtype)
            # std = self.std.flatten()[:condition_x.shape[1]].to(device=device, dtype=condition_x.dtype)
            # condition_x = (condition_x - mean) / std
            condition_x = condition_x.to(device=device)

        # Ensure x_start is on the correct device and dtype
        x = x_start.to(device=device, dtype=model_dtype)
        # x = ((x - self.mean) / self.std).to(torch.float32)
        # Create timestep tensor
        t_batch = torch.full([x.shape[0], 1], t, device=device)

        # Predict noise
        predicted_noise = self(x, t_batch, condition_x)

        # DDIM update rule
        alpha_t = self.alphas[t]
        alpha_bar_t = self.baralphas[t]

        # Get alpha_bar_prev (use 1.0 for final step)
        alpha_bar_prev = self.baralphas[t - 1] if t > 0 else torch.tensor(1.0, device=device)

        # Compute sigma (η in the DDIM paper)
        sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * torch.sqrt(
            1 - alpha_bar_t / alpha_bar_prev)

        # DDIM deterministic update
        pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
        dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma_t ** 2) * predicted_noise
        x_next = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

        # Add noise only if eta > 0
        if eta > 0:
            noise = torch.randn_like(x)
            x_next = x_next + sigma_t * noise

        # Un-normalize the sample
        x_next = x_next# * self.std + self.mean
        pred_x0 = pred_x0# * self.std + self.mean
        return x_next, pred_x0



