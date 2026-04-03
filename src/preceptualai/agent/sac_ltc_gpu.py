"""
GPU-Maximized SAC-LTC Agent.

Combines all GPU optimizations into one agent:
  1. Fused LTC cells with Heun ODE solver
  2. GPU-resident replay buffer (zero CPU roundtrips at sample time)
  3. Mixed-precision training (BF16 on Blackwell)
  4. Vectorized environment integration (batch push)
  5. Fused optimizer steps
  6. TF32 matmul precision
  7. Gradient clipping for stability
"""

import copy
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from preceptualai.core.ltc_encoder_gpu import LTCEncoderGPU
from preceptualai.core.replay_buffer_gpu import ReplayBufferGPU


class LTCActorGPU(nn.Module):
    """GPU-optimized actor with fused encoder."""

    def __init__(self, encoder: LTCEncoderGPU, num_actions: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.latent_dim, num_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        z = self.encoder(state)
        return F.softmax(self.head(z), dim=-1)

    @torch.no_grad()
    def get_action_batch(self, states: torch.Tensor, deterministic: bool = False):
        """Batched action selection (for vectorized envs)."""
        probs = self.forward(states)
        if deterministic:
            return probs.argmax(dim=-1), probs
        return torch.multinomial(probs, 1).squeeze(-1), probs

    @torch.no_grad()
    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        probs = self.forward(state)
        if deterministic:
            return probs.argmax(dim=-1).item(), probs
        return torch.distributions.Categorical(probs).sample().item(), probs


class LTCCriticGPU(nn.Module):
    """GPU-optimized critic with fused encoder."""

    def __init__(self, encoder: LTCEncoderGPU, num_actions: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.latent_dim, num_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(state))


class SACLTCAgentGPU:
    """
    GPU-maximized SAC-LTC agent.

    All GPU optimizations enabled: fused LTC cells, GPU-resident replay,
    mixed-precision training, gradient clipping, TF32 matmul precision.
    """

    def __init__(
        self,
        state_shape: Tuple[int, ...],
        num_actions: int,
        input_dim: int,
        device: torch.device,
        hidden_dim: int = 256,
        latent_dim: int = 256,
        num_layers: int = 3,
        dt: float = 1.0,
        solver: str = "heun",
        sub_steps: int = 1,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 2_000_000,
        batch_size: int = 1024,
        learning_starts: int = 2000,
        target_entropy_ratio: float = 0.5,
        amp_dtype: Optional[torch.dtype] = torch.bfloat16,
        grad_clip: float = 1.0,
        use_gradient_checkpointing: bool = False,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.num_actions = num_actions
        self.amp_dtype = amp_dtype
        self.grad_clip = grad_clip

        # Enable TF32 for maximum throughput
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        def _enc():
            return LTCEncoderGPU(
                input_dim, hidden_dim, latent_dim, num_layers, dt,
                solver=solver, sub_steps=sub_steps,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )

        # Networks
        self.actor = LTCActorGPU(_enc(), num_actions).to(device)
        self.critic1 = LTCCriticGPU(_enc(), num_actions).to(device)
        self.critic2 = LTCCriticGPU(_enc(), num_actions).to(device)

        self.target_critic1 = copy.deepcopy(self.critic1).to(device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(device)
        for p in self.target_critic1.parameters():
            p.requires_grad = False
        for p in self.target_critic2.parameters():
            p.requires_grad = False

        # Optimizers with fused=True for GPU efficiency
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr, fused=True)
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=lr, fused=True)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=lr, fused=True)

        # Entropy tuning
        self.target_entropy = -target_entropy_ratio * np.log(1.0 / num_actions)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = Adam([self.log_alpha], lr=lr)

        # GPU-resident replay buffer
        self.replay_buffer = ReplayBufferGPU(buffer_size, state_shape, device)

        # AMP gradient scaler (for FP16 only; BF16 doesn't need it)
        self.use_scaler = amp_dtype == torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)

        self.train_step_count = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action, _ = self.actor.get_action(state_t, deterministic=deterministic)
        return int(action)

    @torch.no_grad()
    def select_action_batch(
        self, states: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        """Select actions for a batch of states (vectorized envs)."""
        actions, _ = self.actor.get_action_batch(states, deterministic=deterministic)
        return actions

    def update(self) -> Dict[str, float]:
        """Single SAC update step with AMP and gradient clipping."""
        if len(self.replay_buffer) < self.learning_starts:
            return {}

        batch = self.replay_buffer.sample(self.batch_size)
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        amp_ctx = torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_dtype is not None)

        # ---- Critic update ----
        with torch.no_grad():
            with amp_ctx:
                next_probs = self.actor(next_states).clamp(min=1e-8)
                next_log = torch.log(next_probs)
                tq1 = self.target_critic1(next_states)
                tq2 = self.target_critic2(next_states)
                tq = torch.min(tq1, tq2)
                next_v = (next_probs * (tq - self.alpha * next_log)).sum(dim=-1)
                td_target = rewards + self.gamma * (1.0 - dones) * next_v

        with amp_ctx:
            q1 = self.critic1(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            q2 = self.critic2(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            c1_loss = F.mse_loss(q1, td_target)
            c2_loss = F.mse_loss(q2, td_target)

        self.critic1_optimizer.zero_grad(set_to_none=True)
        if self.use_scaler:
            self.scaler.scale(c1_loss).backward()
            self.scaler.unscale_(self.critic1_optimizer)
            nn.utils.clip_grad_norm_(self.critic1.parameters(), self.grad_clip)
            self.scaler.step(self.critic1_optimizer)
        else:
            c1_loss.backward()
            nn.utils.clip_grad_norm_(self.critic1.parameters(), self.grad_clip)
            self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad(set_to_none=True)
        if self.use_scaler:
            self.scaler.scale(c2_loss).backward()
            self.scaler.unscale_(self.critic2_optimizer)
            nn.utils.clip_grad_norm_(self.critic2.parameters(), self.grad_clip)
            self.scaler.step(self.critic2_optimizer)
        else:
            c2_loss.backward()
            nn.utils.clip_grad_norm_(self.critic2.parameters(), self.grad_clip)
            self.critic2_optimizer.step()

        # ---- Actor update ----
        with amp_ctx:
            probs = self.actor(states).clamp(min=1e-8)
            log_probs = torch.log(probs)
            with torch.no_grad():
                min_q = torch.min(self.critic1(states), self.critic2(states))
            actor_loss = (probs * (self.alpha.detach() * log_probs - min_q)).sum(-1).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        if self.use_scaler:
            self.scaler.scale(actor_loss).backward()
            self.scaler.unscale_(self.actor_optimizer)
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.scaler.step(self.actor_optimizer)
        else:
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.actor_optimizer.step()

        # ---- Alpha update ----
        entropy = -(probs.detach() * log_probs.detach()).sum(-1).mean()
        alpha_loss = self.log_alpha * (entropy - self.target_entropy)

        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ---- Scaler update (FP16 only) ----
        if self.use_scaler:
            self.scaler.update()

        # ---- Polyak target update ----
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

    def update_n(self, n: int = 4) -> Dict[str, float]:
        """Perform n gradient updates per environment step (higher UTD ratio)."""
        metrics = {}
        for _ in range(n):
            m = self.update()
            if m:
                metrics = m  # Keep last
        return metrics

    @torch.no_grad()
    def _soft_update(self, source: nn.Module, target: nn.Module):
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.lerp_(sp.data, self.tau)

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

    def param_count(self) -> Dict[str, int]:
        """Return parameter counts for each network."""
        return {
            "actor": sum(p.numel() for p in self.actor.parameters()),
            "critic1": sum(p.numel() for p in self.critic1.parameters()),
            "critic2": sum(p.numel() for p in self.critic2.parameters()),
            "total": sum(
                p.numel() for net in [self.actor, self.critic1, self.critic2]
                for p in net.parameters()
            ),
        }
