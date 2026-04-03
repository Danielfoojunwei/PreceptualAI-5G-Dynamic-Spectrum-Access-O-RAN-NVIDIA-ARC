"""
Hybrid Discrete-Continuous Actor for 5G Spectrum Management.

Extends the discrete SAC actor to support joint:
  - Discrete actions: channel/frequency band selection
  - Continuous actions: power allocation, bandwidth fraction, MCS target

Uses a shared LTC encoder with two output heads:
  - Softmax head for discrete actions
  - Tanh-squashed Gaussian head for continuous actions
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from preceptualai.core.ltc_encoder_gpu import LTCEncoderGPU


class HybridActor(nn.Module):
    """
    Hybrid discrete-continuous actor for spectrum management.

    Shared LTC encoder -> discrete head (softmax) + continuous head (Gaussian).
    """

    def __init__(
        self,
        encoder: LTCEncoderGPU,
        num_discrete_actions: int,
        continuous_action_dim: int,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_discrete = num_discrete_actions
        self.continuous_dim = continuous_action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        latent = encoder.latent_dim

        # Discrete head: channel/frequency selection
        self.discrete_head = nn.Sequential(
            nn.Linear(latent, latent),
            nn.ReLU(),
            nn.Linear(latent, num_discrete_actions),
        )

        # Continuous head: power allocation (mean + log_std)
        self.continuous_mean = nn.Sequential(
            nn.Linear(latent, latent),
            nn.ReLU(),
            nn.Linear(latent, continuous_action_dim),
        )
        self.continuous_log_std = nn.Sequential(
            nn.Linear(latent, latent),
            nn.ReLU(),
            nn.Linear(latent, continuous_action_dim),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (B, T, F) observation sequence

        Returns:
            discrete_probs: (B, num_discrete) action probabilities
            continuous_mean: (B, continuous_dim) mean of Gaussian
            continuous_log_std: (B, continuous_dim) log std of Gaussian
        """
        z = self.encoder(state)

        # Discrete action probabilities
        discrete_logits = self.discrete_head(z)
        discrete_probs = F.softmax(discrete_logits, dim=-1)

        # Continuous action distribution
        mean = self.continuous_mean(z)
        log_std = self.continuous_log_std(z)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)

        return discrete_probs, mean, log_std

    @torch.no_grad()
    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select hybrid action.

        Returns:
            discrete_action: int — selected discrete action
            continuous_action: (continuous_dim,) — continuous values in [-1, 1]
            discrete_probs: (num_discrete,) — for entropy computation
            log_prob: scalar — joint log probability
        """
        probs, mean, log_std = self.forward(state)

        if deterministic:
            discrete_action = probs.argmax(dim=-1).item()
            continuous_action = torch.tanh(mean.squeeze(0))
        else:
            # Sample discrete
            discrete_action = torch.multinomial(probs, 1).squeeze(-1).item()

            # Sample continuous (reparameterization trick)
            std = log_std.exp()
            noise = torch.randn_like(mean)
            raw = mean + std * noise
            continuous_action = torch.tanh(raw.squeeze(0))

        return discrete_action, continuous_action, probs, mean

    @torch.no_grad()
    def get_action_batch(
        self,
        states: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched action selection for vectorized environments.

        Returns:
            discrete_actions: (B,) int tensor
            continuous_actions: (B, continuous_dim) tensor in [-1, 1]
        """
        probs, mean, log_std = self.forward(states)

        if deterministic:
            discrete_actions = probs.argmax(dim=-1)
            continuous_actions = torch.tanh(mean)
        else:
            discrete_actions = torch.multinomial(probs, 1).squeeze(-1)
            std = log_std.exp()
            continuous_actions = torch.tanh(mean + std * torch.randn_like(mean))

        return discrete_actions, continuous_actions

    def evaluate_actions(
        self,
        states: torch.Tensor,
        discrete_actions: torch.Tensor,
        continuous_actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities of given actions (for SAC update).

        Returns:
            discrete_log_probs: (B,) log prob of discrete actions
            continuous_log_probs: (B,) log prob of continuous actions
            discrete_entropy: (B,) entropy of discrete distribution
            continuous_entropy: (B,) entropy of continuous distribution
        """
        probs, mean, log_std = self.forward(states)
        std = log_std.exp()

        # Discrete log probs
        probs_clamped = probs.clamp(min=1e-8)
        log_probs_all = torch.log(probs_clamped)
        discrete_log_probs = log_probs_all.gather(1, discrete_actions.unsqueeze(1)).squeeze(1)
        discrete_entropy = -(probs_clamped * log_probs_all).sum(dim=-1)

        # Continuous log probs (tanh-squashed Gaussian)
        # Inverse tanh to get raw values
        raw_actions = torch.atanh(continuous_actions.clamp(-0.999, 0.999))
        gaussian_log_prob = -0.5 * (
            ((raw_actions - mean) / (std + 1e-8)) ** 2
            + 2 * log_std
            + 1.8378770664093453  # log(2*pi)
        )
        # Tanh squashing correction
        log_det = torch.log(1 - continuous_actions ** 2 + 1e-6)
        continuous_log_probs = (gaussian_log_prob - log_det).sum(dim=-1)
        continuous_entropy = (0.5 + 0.5 * 1.8378770664093453 + log_std).sum(dim=-1)

        return discrete_log_probs, continuous_log_probs, discrete_entropy, continuous_entropy
