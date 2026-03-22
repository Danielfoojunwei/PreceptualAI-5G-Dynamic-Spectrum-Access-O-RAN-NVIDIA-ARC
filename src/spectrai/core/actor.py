"""LTC-based Actor network for discrete action selection."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectrai.core.ltc_encoder import LTCEncoder


class LTCActor(nn.Module):
    """Policy network: state sequence → action probabilities via LTC encoder."""

    def __init__(self, encoder: LTCEncoder, num_actions: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.latent_dim, num_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        z = self.encoder(state)
        return F.softmax(self.head(z), dim=-1)

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        probs = self.forward(state)
        if deterministic:
            return probs.argmax(dim=-1).item(), probs
        return torch.distributions.Categorical(probs).sample().item(), probs
