"""
Liquid Time-Constant (LTC) Cell.

Implements one discretised step of the continuous-time neural ODE:

    τ(x) · dh/dt = -h + f(x, h)

Discretised (Euler):

    h' = h + (Δt / τ(x)) · (-h + f(x, h))

where:
    f(x, h) = tanh(W_h · h + W_x · x + b)
    τ(x)    = τ_base + softplus(W_τ · x + b_τ)

Reference: Hasani et al., "Liquid Time-Constant Networks", AAAI 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LTCCell(nn.Module):
    """
    Liquid Time-Constant cell.

    Implements one discretised step of the continuous-time ODE:
        τ(x) · dh/dt = -h + f(x, h)

    Discretised (Euler):
        h' = h + (Δt / τ(x)) · (-h + f(x, h))

    where:
        f(x, h) = tanh(W_h · h + W_x · x + b)
        τ(x)    = τ_base + softplus(W_τ · x + b_τ)
    """

    def __init__(self, input_dim: int, hidden_dim: int, dt: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dt = dt

        # State transition: f(x, h) = tanh(W_h·h + W_x·x + b)
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.W_x = nn.Linear(input_dim, hidden_dim)

        # Input-dependent time constant: τ(x) = τ_base + softplus(W_τ·x + b_τ)
        self.W_tau = nn.Linear(input_dim, hidden_dim)
        self.tau_base = nn.Parameter(torch.ones(hidden_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_h.weight)
        nn.init.zeros_(self.W_h.bias)
        nn.init.xavier_uniform_(self.W_x.weight)
        nn.init.zeros_(self.W_x.bias)
        nn.init.xavier_uniform_(self.W_tau.weight)
        nn.init.zeros_(self.W_tau.bias)

    def forward(
        self, x_t: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        """
        Single ODE step.

        Args:
            x_t: (B, input_dim) — input at time t
            h:   (B, hidden_dim) — hidden state

        Returns:
            h':  (B, hidden_dim) — updated hidden state
        """
        # Nonlinear activation f(x, h)
        f = torch.tanh(self.W_h(h) + self.W_x(x_t))

        # Input-dependent time constant (always positive)
        tau = self.tau_base + F.softplus(self.W_tau(x_t))

        # Euler step: h' = h + (dt / τ) · (-h + f)
        dh = (self.dt / tau) * (-h + f)
        h_new = h + dh

        return h_new
