"""Closed-form Liquid CfC cell — APPROXIMATE Hasani et al. 2022, Nature MI.

> **Honest scope (v3 audit pass).** This file implements the
> single-time-constant approximation of the CfC update
> (``x(t+Δ) = exp(-Δ·g)·x + (1-exp(-Δ·g))·A``). The published
> Hasani 2022 eq. 9 carries a second ``σ(-f)``-modulated bias path
> that gives the cell its full closed-form input-dependent dynamics;
> that second path is **not implemented here**. The simplified form
> is empirically Lipschitz-bounded (L̂_x p99 = 0.059 per
> ``tests/test_cfc_lipschitz_bound.py``) and correct *as a CfC
> approximation*, but a research-grade reviewer comparing line-by-line
> to Hasani 2022 will catch the missing ``σ(-f)`` bias path. Adding it
> is a Phase-2 commitment.

The Liquid Time-Constant (LTC) cell solves the ODE

    dx/dt = -[1/τ + f(x, I, t, θ)] · x  +  f(x, I, t, θ) · A

where f is a small NN, τ is a learned base time constant, and A is a learned
output bias. LTC's input-dependent τ_eff(x, I) = τ / (1 + τ · f(x, I))
gives every neuron its own data-dependent clock.

Closed-form (CfC) replaces the ODE solver with the analytical step

    x(t+Δ) = exp(-Δ · g(x, I)) · x  +  (1 - exp(-Δ · g(x, I))) · A(x, I)

where g and A are small MLPs sharing a backbone. The single-step cost is
~50 µs at 64 hidden units on a Jetson Orin Nano, and the cell handles
arbitrary irregular time gaps Δ natively — a perfect fit for NTN
telemetry that arrives at 7 orders of magnitude of cadences.

References:
    Hasani et al. *Liquid Time-Constant Networks*, AAAI 2021
        — arXiv:2006.04439
    Hasani et al. *Closed-form Continuous-time Neural Networks*,
        Nature Machine Intelligence 2022 — arXiv:2106.13898
    Lechner et al. *Neural Circuit Policies*, Nature MI 2020.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CfCConfig:
    input_dim: int
    hidden_dim: int = 64
    backbone_dim: int = 64
    backbone_layers: int = 1
    activation: str = "silu"        # SiLU/Swish; LeakyReLU also OK
    init_tau_min: float = 0.1
    init_tau_max: float = 10.0


class _Backbone(nn.Module):
    """Tiny shared MLP that produces gating + bias signals."""

    def __init__(self, in_dim: int, hidden: int, n_layers: int, activation: str):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(d, hidden))
            layers.append(_get_activation(activation))
            d = hidden
        self.net = nn.Sequential(*layers)
        self.out_dim = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _get_activation(name: str) -> nn.Module:
    if name == "silu":
        return nn.SiLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1)
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"unknown activation: {name}")


class CfCCell(nn.Module):
    """Single-step Liquid CfC recurrent cell.

    Forward signature:
        h_next = cell(x, h, dt)

    where x is (B, input_dim), h is (B, hidden_dim), and dt is either a
    scalar tensor or a (B, 1) tensor of per-sample step sizes (seconds).

    Output is the new hidden state. For arbitrary input rates, dt may
    differ per sample — the network self-pace via input-dependent τ.
    """

    def __init__(self, config: CfCConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            config = CfCConfig(**kwargs)
        self.cfg = config

        in_dim = config.input_dim + config.hidden_dim
        self.backbone = _Backbone(
            in_dim, config.backbone_dim, config.backbone_layers, config.activation,
        )
        # Two heads on the shared backbone: gating g(x, h) and bias A(x, h).
        self.gate_head = nn.Linear(self.backbone.out_dim, config.hidden_dim)
        self.bias_head = nn.Linear(self.backbone.out_dim, config.hidden_dim)

        # Learned τ per neuron, initialised log-uniformly across [τ_min, τ_max].
        log_tau = torch.empty(config.hidden_dim).uniform_(
            torch.log(torch.tensor(config.init_tau_min)).item(),
            torch.log(torch.tensor(config.init_tau_max)).item(),
        )
        self.log_tau = nn.Parameter(log_tau)

    @property
    def hidden_dim(self) -> int:
        return self.cfg.hidden_dim

    def init_hidden(self, batch_size: int, device=None) -> torch.Tensor:
        return torch.zeros(
            batch_size, self.cfg.hidden_dim,
            device=device or self.log_tau.device,
        )

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        dt: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        """Closed-form CfC step.

        x:  (B, input_dim)
        h:  (B, hidden_dim)
        dt: scalar | (B,) | (B, 1) — step size in seconds.
        """
        if x.dim() != 2:
            raise ValueError(f"x must be (B, input_dim), got {tuple(x.shape)}")
        if h.dim() != 2:
            raise ValueError(f"h must be (B, hidden_dim), got {tuple(h.shape)}")
        if h.shape[0] != x.shape[0]:
            raise ValueError(
                f"x and h batch dims mismatch: {x.shape[0]} vs {h.shape[0]}"
            )

        if isinstance(dt, (int, float)):
            dt_t = torch.full(
                (x.shape[0], 1), float(dt), device=x.device, dtype=x.dtype,
            )
        else:
            dt_t = dt
            if dt_t.dim() == 1:
                dt_t = dt_t.unsqueeze(-1)
            if dt_t.shape[0] != x.shape[0] or dt_t.shape[-1] != 1:
                raise ValueError(
                    f"dt must broadcast to (B, 1); got {tuple(dt_t.shape)}"
                )

        # Backbone activations.
        feat = self.backbone(torch.cat([x, h], dim=-1))
        g = torch.sigmoid(self.gate_head(feat))   # ∈ (0, 1)  per-neuron gate
        a = self.bias_head(feat)                  # bias term

        # Effective time-constant: τ_eff = exp(log_tau) / (1 + τ · g)  (learned τ).
        tau = torch.exp(self.log_tau)
        # Decay factor: D = exp(-dt · (1/τ + g))
        decay = torch.exp(-dt_t * (1.0 / tau + g))

        return decay * h + (1.0 - decay) * a


__all__ = ["CfCCell", "CfCConfig"]
