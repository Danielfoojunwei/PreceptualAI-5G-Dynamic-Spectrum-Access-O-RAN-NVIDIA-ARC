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


# ======================================================================
# Diffusion Trajectory Planning (Paradigm Shift 3)
# ======================================================================

class TrajectoryDiffusionModel(nn.Module):
    """
    Diffusion model for planning entire allocation trajectories.

    Instead of single-step augmentation, generates K-step action-state
    trajectories conditioned on the current state, then scores them
    using a value function or world model.

    Inspired by Diffuser (Janner et al., ICML 2022) and Decision
    Diffusion (Ajay et al., NeurIPS 2023), adapted for spectrum.

    Key insight: Handover decisions are multi-step. Switching from LEO
    to FR1 costs 0.3 NOW but may save 5.0 over the next 10 steps by
    avoiding collisions. Trajectory planning captures this.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        horizon: int = 10,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        num_diffusion_steps: int = 50,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.traj_dim = horizon * (obs_dim + action_dim)  # flattened trajectory
        self.num_steps = num_diffusion_steps

        # Noise schedule
        betas = torch.linspace(1e-4, 0.02, num_diffusion_steps)
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

        # State conditioner
        self.state_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Trajectory denoiser
        self.input_proj = nn.Linear(self.traj_dim + hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            DenoisingBlock(hidden_dim, time_dim) for _ in range(num_blocks)
        ])
        self.output_proj = nn.Linear(hidden_dim, self.traj_dim)

    def forward(self, traj_noisy, t, state):
        t_emb = self.time_embed(t)
        s_emb = self.state_encoder(state)
        h = self.input_proj(torch.cat([traj_noisy, s_emb], dim=-1))
        for block in self.blocks:
            h = block(h, t_emb)
        return self.output_proj(h)

    def training_loss(self, trajectories, states):
        """
        Args:
            trajectories: (B, horizon * (obs_dim + action_dim)) flattened
            states: (B, obs_dim) current state conditioning
        """
        B = trajectories.shape[0]
        t = torch.randint(0, self.num_steps, (B,), device=trajectories.device)
        noise = torch.randn_like(trajectories)
        sqrt_a = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_1ma = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        noisy = sqrt_a * trajectories + sqrt_1ma * noise
        eps_pred = self.forward(noisy, t, states)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def plan(self, state, num_candidates=8):
        """
        Generate candidate trajectories and return them for scoring.

        Args:
            state: (B, obs_dim) current state
            num_candidates: number of candidate trajectories per state
        Returns:
            trajectories: (B * num_candidates, horizon, obs_dim + action_dim)
        """
        B = state.shape[0]
        state_expanded = state.repeat_interleave(num_candidates, dim=0)
        total = state_expanded.shape[0]

        x = torch.randn(total, self.traj_dim, device=state.device)

        for t_idx in reversed(range(self.num_steps)):
            t = torch.full((total,), t_idx, device=x.device, dtype=torch.long)
            eps_pred = self.forward(x, t, state_expanded)
            alpha = self.alphas[t_idx]
            alpha_bar = self.alphas_cumprod[t_idx]
            beta = self.betas[t_idx]
            x = (1.0 / alpha.sqrt()) * (x - (beta / (1 - alpha_bar).sqrt()) * eps_pred)
            if t_idx > 0:
                x = x + beta.sqrt() * torch.randn_like(x)

        # Reshape to (B*K, horizon, obs_dim + action_dim)
        trajs = x.view(total, self.horizon, self.obs_dim + self.action_dim)
        return trajs


class DiffusionTrajectoryPlanner:
    """
    Plans multi-step spectrum allocations via diffusion + scoring.

    1. Generate K candidate trajectories via TrajectoryDiffusionModel
    2. Score each trajectory using cumulative predicted reward
    3. Execute the first action of the best trajectory

    This enables look-ahead planning for handover decisions where
    short-term cost leads to long-term gain.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        horizon: int = 10,
        num_candidates: int = 8,
        device: torch.device = torch.device("cpu"),
        hidden_dim: int = 256,
        lr: float = 1e-4,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.device = device

        self.model = TrajectoryDiffusionModel(
            obs_dim, action_dim, horizon, hidden_dim,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Simple reward predictor: (obs, action_onehot) -> scalar reward
        self.reward_predictor = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)
        self.reward_optimizer = torch.optim.Adam(
            self.reward_predictor.parameters(), lr=lr
        )

    def train_step(self, trajectories, states):
        """Train the trajectory diffusion model."""
        self.model.train()
        loss = self.model.training_loss(trajectories, states)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def train_reward(self, obs, actions_onehot, rewards):
        """Train reward predictor on real transitions."""
        self.reward_predictor.train()
        pred = self.reward_predictor(torch.cat([obs, actions_onehot], dim=-1)).squeeze(-1)
        loss = F.mse_loss(pred, rewards)
        self.reward_optimizer.zero_grad()
        loss.backward()
        self.reward_optimizer.step()
        return loss.item()

    @torch.no_grad()
    def plan_action(self, state):
        """
        Plan best first action by generating and scoring trajectories.

        Args:
            state: (1, obs_dim) current state
        Returns:
            best_action: int — action index from best trajectory
        """
        self.model.eval()
        self.reward_predictor.eval()

        # Generate candidates
        trajs = self.model.plan(state, self.num_candidates)  # (K, H, obs+act)
        K = trajs.shape[0]

        # Score each trajectory by cumulative predicted reward
        total_rewards = torch.zeros(K, device=state.device)
        for t in range(self.horizon):
            obs_t = trajs[:, t, :self.obs_dim]
            act_t = trajs[:, t, self.obs_dim:]
            r_pred = self.reward_predictor(
                torch.cat([obs_t, act_t], dim=-1)
            ).squeeze(-1)
            total_rewards += r_pred * (0.99 ** t)  # discounted

        # Pick best trajectory
        best_idx = total_rewards.argmax()
        best_traj = trajs[best_idx]

        # Extract first action (convert continuous to discrete)
        first_action_cont = best_traj[0, self.obs_dim:]
        best_action = first_action_cont.argmax().item()

        return best_action
