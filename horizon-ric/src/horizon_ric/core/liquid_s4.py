"""Liquid-S4 backbone (Hasani et al. 2023, ICLR).

Combines:
  * Structured State-Space Model recurrence with HiPPO-LegS-style
    diagonal state matrix (S4D parametrisation, Gu et al. 2022).
  * Input-dependent time-constants (LTC / CfC, Hasani 2020/2022).

The cell discretises the continuous-time SSM
        dx/dt = A·x + B·u
        y    = C·x
with a Zero-Order-Hold step Δ that is **input-dependent** through a
small gate, giving every channel its own data-driven τ. The recurrence
runs in O(L) per layer; for short slot-coarse rollouts (5–30 s) on the
Orin Nano this comfortably fits inside the A1 budget.

For tighter accuracy on long sequences, swap the diagonal HiPPO-LegS
init for the diagonal-plus-low-rank S4 form. We keep diagonal-only here
because it is enough to capture the multi-rate behaviour the rApp needs
and avoids the FFT-convolution path entirely.

References:
    Gu et al. *Efficiently Modeling Long Sequences with Structured State
        Spaces* (S4), ICLR 2022 — arXiv:2111.00396.
    Gu, Goel, Ré *On the Parameterization and Initialization of
        Diagonal State Space Models* (S4D), 2022 — arXiv:2206.11893.
    Hasani et al. *Liquid Structural State-Space Models*, ICLR 2023
        — arXiv:2209.12951.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LiquidS4Config:
    d_model: int = 64
    d_state: int = 16
    """SSM hidden state size per channel."""
    n_layers: int = 4
    dropout: float = 0.0
    log_dt_min: float = -3.0   # log10 of min Δ
    log_dt_max: float = 1.0    # log10 of max Δ


def _hippo_legs_diagonal(n: int) -> torch.Tensor:
    """HiPPO-LegS diagonal eigenvalue init (S4D-Lin).

    Gives every channel a stable continuous-time pole on the negative
    real axis with logarithmic spacing. This is the simplest of the S4D
    initialisations and the one used by Liquid-S4 ablations.
    """
    return -0.5 * torch.exp(
        torch.linspace(math.log(1.0), math.log(float(n)), n)
    )


class _LiquidS4Layer(nn.Module):
    """Single Liquid-S4 layer with input-dependent Δ.

    State-space matrices stored in continuous form; per-step we compute
    the discretised ZOH transition `A_d = exp(Δ · A)`. Δ is gated on the
    input so different positions can integrate at different time scales.
    """

    def __init__(self, cfg: LiquidS4Config):
        super().__init__()
        self.cfg = cfg
        N = cfg.d_state
        H = cfg.d_model

        # Continuous-time SSM parameters per channel.
        a = _hippo_legs_diagonal(N).repeat(H, 1)         # (H, N)
        self.log_neg_a = nn.Parameter(torch.log(-a))     # learn on log-space
        self.B = nn.Parameter(torch.randn(H, N) * (1.0 / math.sqrt(N)))
        self.C = nn.Parameter(torch.randn(H, N) * (1.0 / math.sqrt(N)))
        self.D = nn.Parameter(torch.zeros(H))            # skip term

        # Liquid input-dependent Δ. Base log Δ initialised log-uniformly.
        log_dt = torch.empty(H).uniform_(cfg.log_dt_min, cfg.log_dt_max)
        self.log_dt_base = nn.Parameter(log_dt)
        self.dt_gate = nn.Linear(H, H)

        self.out_proj = nn.Linear(H, H)
        self.dropout = nn.Dropout(cfg.dropout)
        self.norm = nn.LayerNorm(H)

    def _step(self, u_t: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One recurrent step.

        u_t: (B, H) — input.
        x:   (B, H, N) — SSM hidden state.
        """
        # Liquid Δ: base × sigmoid-gated input contribution.
        gate = torch.sigmoid(self.dt_gate(u_t))                  # (B, H)
        log_dt = self.log_dt_base.unsqueeze(0) + gate            # (B, H)
        dt = torch.exp(log_dt).unsqueeze(-1)                     # (B, H, 1)

        a = -torch.exp(self.log_neg_a)                           # (H, N)
        a_d = torch.exp(dt * a.unsqueeze(0))                     # (B, H, N)
        # Closed-form ZOH for B: (a_d - 1) / a · B
        b_d = ((a_d - 1.0) / a.unsqueeze(0)) * self.B.unsqueeze(0)  # (B, H, N)

        x_next = a_d * x + b_d * u_t.unsqueeze(-1)               # (B, H, N)
        y = (self.C.unsqueeze(0) * x_next).sum(dim=-1) + self.D * u_t  # (B, H)
        return y, x_next

    def forward(
        self, u: torch.Tensor, x0: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """u: (B, L, d_model). Returns (y: (B, L, d_model), x_final)."""
        B, L, H = u.shape
        if H != self.cfg.d_model:
            raise ValueError(
                f"d_model mismatch: layer has {self.cfg.d_model}, got {H}"
            )
        x = (
            x0
            if x0 is not None
            else u.new_zeros(B, H, self.cfg.d_state)
        )
        outputs = []
        for t in range(L):
            y_t, x = self._step(u[:, t, :], x)
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)
        # Standard SSM block: residual + norm + projection + dropout.
        y = self.dropout(self.out_proj(F.silu(y)))
        return self.norm(u + y), x


class LiquidS4(nn.Module):
    """Stack of `n_layers` Liquid-S4 layers.

    Two modes:
      * `forward(u)`        — full-sequence (B, L, d_model) → (B, L, d_model).
      * `step(u_t, states)` — single time-step, suitable for streaming.
    """

    def __init__(self, config: LiquidS4Config | None = None):
        super().__init__()
        self.cfg = config or LiquidS4Config()
        self.layers = nn.ModuleList(
            [_LiquidS4Layer(self.cfg) for _ in range(self.cfg.n_layers)]
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            u, _ = layer(u)
        return u

    def step(
        self, u_t: torch.Tensor, states: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Single-step API for streaming inference."""
        if states is None:
            states = [
                u_t.new_zeros(u_t.shape[0], self.cfg.d_model, self.cfg.d_state)
                for _ in self.layers
            ]
        new_states: list[torch.Tensor] = []
        h = u_t
        for layer, x in zip(self.layers, states, strict=True):
            y, x_next = layer._step(h, x)
            # Match `forward()` block: residual + dropout + projection + norm.
            y = layer.dropout(layer.out_proj(F.silu(y)))
            h = layer.norm(h + y)
            new_states.append(x_next)
        return h, new_states


__all__ = ["LiquidS4", "LiquidS4Config"]
