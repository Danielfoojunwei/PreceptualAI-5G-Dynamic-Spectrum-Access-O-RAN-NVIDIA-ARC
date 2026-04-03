"""
SAC-LTC: Discrete Soft Actor-Critic with Liquid Time-Constant Network Encoder.

Combines the off-policy sample efficiency of SAC with the adaptive temporal
dynamics of Liquid Time-Constant (LTC) networks for Dynamic Spectrum Access.

Architecture:
    Actor:  LTCEncoder → softmax policy head  π(a|s)
    Critic: 2× LTCEncoder → Q-value heads     Q₁(s,·), Q₂(s,·)
    Target: Polyak-averaged critic copies
    Alpha:  Automatic entropy coefficient tuning
"""

import copy
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from preceptualai.core.actor import LTCActor
from preceptualai.core.critic import LTCCritic
from preceptualai.core.ltc_encoder import LTCEncoder
from preceptualai.core.replay_buffer import ReplayBuffer


class SACLTCAgent:
    """
    SAC-LTC: Discrete SAC with Liquid Time-Constant encoder.

    The same discrete SAC algorithm (Christodoulou 2019) with an LTC encoder
    that provides:
      1. Input-dependent time constants τ(x) for adaptive temporal integration
      2. Continuous-time ODE dynamics matching PU Markov process structure
      3. Stable linear attractor (−h + f) ensuring bounded hidden state dynamics
      4. Learnable τ_base floor preventing time-constant collapse
    """

    def __init__(
        self,
        state_shape: Tuple[int, ...],
        num_actions: int,
        input_dim: int,
        device: torch.device,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        num_layers: int = 2,
        dt: float = 1.0,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        learning_starts: int = 1000,
        target_entropy_ratio: float = 0.5,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.num_actions = num_actions

        def _enc():
            return LTCEncoder(input_dim, hidden_dim, latent_dim, num_layers, dt)

        self.actor = LTCActor(_enc(), num_actions).to(device)
        self.critic1 = LTCCritic(_enc(), num_actions).to(device)
        self.critic2 = LTCCritic(_enc(), num_actions).to(device)

        self.target_critic1 = copy.deepcopy(self.critic1).to(device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(device)
        for p in self.target_critic1.parameters():
            p.requires_grad = False
        for p in self.target_critic2.parameters():
            p.requires_grad = False

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=lr)

        self.target_entropy = -target_entropy_ratio * np.log(1.0 / num_actions)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = Adam([self.log_alpha], lr=lr)

        self.replay_buffer = ReplayBuffer(buffer_size, state_shape, device)
        self.train_step_count = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action, _ = self.actor.get_action(state_t, deterministic=deterministic)
        return int(action)

    def update(self) -> Dict[str, float]:
        if len(self.replay_buffer) < self.learning_starts:
            return {}

        batch = self.replay_buffer.sample(self.batch_size)
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # Critic targets
        with torch.no_grad():
            next_probs = self.actor(next_states).clamp(min=1e-8)
            next_log = torch.log(next_probs)
            tq1 = self.target_critic1(next_states)
            tq2 = self.target_critic2(next_states)
            tq = torch.min(tq1, tq2)
            next_v = (next_probs * (tq - self.alpha * next_log)).sum(dim=-1)
            td_target = rewards + self.gamma * (1.0 - dones) * next_v

        # Critic losses
        q1 = self.critic1(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        q2 = self.critic2(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        c1_loss = F.mse_loss(q1, td_target)
        c2_loss = F.mse_loss(q2, td_target)

        self.critic1_optimizer.zero_grad()
        c1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        c2_loss.backward()
        self.critic2_optimizer.step()

        # Actor loss
        probs = self.actor(states).clamp(min=1e-8)
        log_probs = torch.log(probs)
        with torch.no_grad():
            min_q = torch.min(self.critic1(states), self.critic2(states))
        actor_loss = (probs * (self.alpha.detach() * log_probs - min_q)).sum(-1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha
        entropy = -(probs.detach() * log_probs.detach()).sum(-1).mean()
        alpha_loss = self.log_alpha * (entropy - self.target_entropy)

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Polyak
        self._soft_update(self.critic1, self.target_critic1)
        self._soft_update(self.critic2, self.target_critic2)
        self.train_step_count += 1

        return {
            "critic1_loss": c1_loss.item(),
            "critic2_loss": c2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "entropy": entropy.item(),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.mul_(1.0 - self.tau).add_(sp.data, alpha=self.tau)

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "train_step_count": self.train_step_count,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.target_critic1.load_state_dict(ckpt["target_critic1"])
        self.target_critic2.load_state_dict(ckpt["target_critic2"])
        self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))
        self.train_step_count = ckpt["train_step_count"]
