"""
Diffusion Model for Spectrum Data Augmentation.

Generates synthetic but realistic spectrum observation patterns
for replay buffer augmentation, improving sample diversity and
enabling training on underrepresented scenarios.

Uses a conditional denoising diffusion probabilistic model (DDPM)
conditioned on the current spectrum state to generate plausible
future observations.

Reference:
  Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
  "Diffusion Models for Wireless Transceivers", arXiv 2024.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([args.cos(), args.sin()], dim=-1)


class DenoisingBlock(nn.Module):
    """Residual denoising block with time conditioning."""

    def __init__(self, dim: int, time_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_proj = nn.Linear(time_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.net(h) + self.time_proj(t_emb)
        return x + h


class SpectrumDiffusionModel(nn.Module):
    """
    Conditional DDPM for spectrum observation generation.

    Conditions on current spectrum state to generate realistic
    future observations for replay buffer augmentation.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        num_diffusion_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_steps = num_diffusion_steps

        # Noise schedule
        betas = torch.linspace(beta_start, beta_end, num_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

        # Time embedding
        time_dim = hidden_dim
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Condition encoder (current spectrum state)
        self.cond_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Noise prediction network (epsilon-prediction)
        self.input_proj = nn.Linear(obs_dim + hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            DenoisingBlock(hidden_dim, time_dim)
            for _ in range(num_blocks)
        ])
        self.output_proj = nn.Linear(hidden_dim, obs_dim)

    def forward(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict noise in x_noisy given timestep t and condition.

        Args:
            x_noisy:   (B, obs_dim) noised observation
            t:         (B,) int timestep indices
            condition: (B, obs_dim) conditioning (current state)
        Returns:
            eps_pred:  (B, obs_dim) predicted noise
        """
        t_emb = self.time_embed(t)  # (B, time_dim)
        cond = self.cond_encoder(condition)  # (B, hidden_dim)

        h = self.input_proj(torch.cat([x_noisy, cond], dim=-1))

        for block in self.blocks:
            h = block(h, t_emb)

        return self.output_proj(h)

    def training_loss(
        self,
        x0: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DDPM training loss.

        Args:
            x0:        (B, obs_dim) clean target observation
            condition: (B, obs_dim) conditioning state
        Returns:
            loss: scalar MSE loss
        """
        B = x0.shape[0]
        t = torch.randint(0, self.num_steps, (B,), device=x0.device)
        noise = torch.randn_like(x0)

        # Forward diffusion: q(x_t | x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        x_noisy = sqrt_alpha * x0 + sqrt_one_minus * noise

        # Predict noise
        eps_pred = self.forward(x_noisy, t, condition)

        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Generate synthetic observations via reverse diffusion.

        Args:
            condition: (B, obs_dim) conditioning state
            num_samples: number of samples per condition
        Returns:
            samples: (B * num_samples, obs_dim) generated observations
        """
        B = condition.shape[0]
        if num_samples > 1:
            condition = condition.repeat(num_samples, 1)
        total = condition.shape[0]

        # Start from pure noise
        x = torch.randn(total, self.obs_dim, device=condition.device)

        # Reverse diffusion
        for t_idx in reversed(range(self.num_steps)):
            t = torch.full((total,), t_idx, device=x.device, dtype=torch.long)
            eps_pred = self.forward(x, t, condition)

            alpha = self.alphas[t_idx]
            alpha_bar = self.alphas_cumprod[t_idx]
            beta = self.betas[t_idx]

            # DDPM update
            x = (1.0 / alpha.sqrt()) * (
                x - (beta / (1.0 - alpha_bar).sqrt()) * eps_pred
            )

            if t_idx > 0:
                noise = torch.randn_like(x)
                x = x + beta.sqrt() * noise

        return x


class DiffusionAugmenter:
    """
    Replay buffer augmentation using diffusion-generated spectrum data.

    Trains a diffusion model on real transitions and generates
    synthetic ones for underrepresented scenarios.
    """

    def __init__(
        self,
        obs_dim: int,
        device: torch.device,
        hidden_dim: int = 256,
        lr: float = 1e-4,
    ):
        self.model = SpectrumDiffusionModel(obs_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = device

    def train_step(
        self,
        current_obs: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> float:
        """Train diffusion model on a batch of real transitions."""
        self.model.train()
        loss = self.model.training_loss(next_obs, current_obs)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def generate(
        self,
        current_obs: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """Generate synthetic next observations."""
        self.model.eval()
        return self.model.sample(current_obs, num_samples)
