"""Action-conditioned latent dynamics — Mamba-style selective scan.

Predicts `z_{t+1} = f(z_t, a_t)` over the Perceiver latent. Uses a
diagonal selective state-space (Mamba's input-dependent A, B, Δ) without
the CUDA kernel — pure PyTorch recurrent loop, fast enough for the
slot-coarse rollouts the rApp's TD-MPC2 planner queries.

Design choices:
  * Action `a_t` is concatenated to the input projection, giving the
    selective-scan gates conditional access to the action.
  * State propagation is the same diagonal-A ZOH discretisation used by
    `liquid_s4.py`; this module re-implements just the recurrent path
    so latent dynamics can keep its own gating / parametrisation tuned
    for action-conditioning rather than long-context pretraining.
  * `rollout(z0, actions)` produces a (B, T, d_latent) trajectory the
    planner uses to score MPPI candidates.

References:
    Gu, Goel, Ré *S4*, ICLR 2022.
    Gu, Dao *Mamba: Linear-Time Sequence Modeling with Selective State
        Spaces*, 2023 — arXiv:2312.00752.
    Dao, Gu *Transformers are SSMs (Mamba-2)*, ICML 2024 — arXiv:2405.21060.
    Hafner et al. *DreamerV3*, *Nature* 2025 — symlog two-hot heads on top
        of the rolled-out latent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LatentDynamicsConfig:
    d_latent: int = 128
    d_action: int = 16
    d_state: int = 16
    log_dt_min: float = -3.0
    log_dt_max: float = 1.0


class LatentDynamics(nn.Module):
    """Action-conditioned diagonal selective state-space dynamics."""

    def __init__(self, config: LatentDynamicsConfig | None = None):
        super().__init__()
        self.cfg = config or LatentDynamicsConfig()
        H, N, A_dim = (
            self.cfg.d_latent, self.cfg.d_state, self.cfg.d_action,
        )

        # Continuous SSM matrices.
        a = -0.5 * torch.exp(
            torch.linspace(math.log(1.0), math.log(float(N)), N)
        ).repeat(H, 1)
        self.log_neg_a = nn.Parameter(torch.log(-a))

        # Selective B/C made input-dependent — projections from (z, a).
        self.proj_in = nn.Linear(H + A_dim, H)
        self.B_proj = nn.Linear(H + A_dim, H * N)
        self.C_proj = nn.Linear(H + A_dim, H * N)
        self.D = nn.Parameter(torch.zeros(H))

        # Liquid Δ: input-dependent log-step.
        log_dt = torch.empty(H).uniform_(
            self.cfg.log_dt_min, self.cfg.log_dt_max
        )
        self.log_dt_base = nn.Parameter(log_dt)
        self.dt_gate = nn.Linear(H + A_dim, H)

        self.norm = nn.LayerNorm(H)
        self.out_proj = nn.Linear(H, H)

    def step(
        self,
        z: torch.Tensor,                  # (B, d_latent)
        action: torch.Tensor,             # (B, d_action)
        ssm_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Propagate (z, action) → z_next."""
        if z.ndim != 2 or z.shape[1] != self.cfg.d_latent:
            raise ValueError(
                f"z must be (B, {self.cfg.d_latent}), got {tuple(z.shape)}"
            )
        if action.ndim != 2 or action.shape[1] != self.cfg.d_action:
            raise ValueError(
                f"action must be (B, {self.cfg.d_action}), got {tuple(action.shape)}"
            )
        B, H = z.shape
        N = self.cfg.d_state

        u = torch.cat([z, action], dim=-1)
        u_in = self.proj_in(u)                                         # (B, H)
        gate = torch.sigmoid(self.dt_gate(u))                          # (B, H)
        log_dt = self.log_dt_base.unsqueeze(0) + gate                  # (B, H)
        dt = torch.exp(log_dt).unsqueeze(-1)                           # (B, H, 1)

        a = -torch.exp(self.log_neg_a)                                 # (H, N)
        a_d = torch.exp(dt * a.unsqueeze(0))                           # (B, H, N)

        B_t = self.B_proj(u).view(B, H, N)                             # (B, H, N)
        C_t = self.C_proj(u).view(B, H, N)
        b_d = ((a_d - 1.0) / a.unsqueeze(0)) * B_t                     # (B, H, N)

        if ssm_state is None:
            ssm_state = z.new_zeros(B, H, N)
        x_next = a_d * ssm_state + b_d * u_in.unsqueeze(-1)            # (B, H, N)
        y = (C_t * x_next).sum(dim=-1) + self.D * u_in                  # (B, H)
        z_next = self.norm(z + self.out_proj(F.silu(y)))
        return z_next, x_next

    def rollout(
        self, z0: torch.Tensor, actions: torch.Tensor,
    ) -> torch.Tensor:
        """Roll the dynamics forward for T steps.

        z0:      (B, d_latent)
        actions: (B, T, d_action)
        Returns: (B, T, d_latent) — predicted latents at each step.
        """
        if actions.ndim != 3 or actions.shape[2] != self.cfg.d_action:
            raise ValueError(
                f"actions must be (B, T, {self.cfg.d_action}); "
                f"got {tuple(actions.shape)}"
            )
        if z0.shape[0] != actions.shape[0]:
            raise ValueError(
                f"batch mismatch: z0 has {z0.shape[0]} but actions has "
                f"{actions.shape[0]}"
            )
        B, T, _ = actions.shape
        z = z0
        ssm = None
        out: list[torch.Tensor] = []
        for t in range(T):
            z, ssm = self.step(z, actions[:, t, :], ssm)
            out.append(z)
        return torch.stack(out, dim=1)


__all__ = ["LatentDynamics", "LatentDynamicsConfig"]
