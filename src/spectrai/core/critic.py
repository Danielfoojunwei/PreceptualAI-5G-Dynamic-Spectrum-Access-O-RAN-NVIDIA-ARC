"""LTC-based Critic network for Q-value estimation."""

import torch
import torch.nn as nn

from spectrai.core.ltc_encoder import LTCEncoder


class LTCCritic(nn.Module):
    """Q-value network: state sequence → Q-values for all actions via LTC encoder."""

    def __init__(self, encoder: LTCEncoder, num_actions: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.latent_dim, num_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(state))
