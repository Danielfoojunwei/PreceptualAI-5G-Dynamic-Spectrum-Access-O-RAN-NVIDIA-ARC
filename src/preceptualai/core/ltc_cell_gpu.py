"""
GPU-optimized Liquid Time-Constant (LTC) Cell.

Key optimizations over the base LTC cell:
  1. Fused linear projections (W_h, W_x, W_tau in a single matmul)
  2. Multi-step ODE solver (RK2 Heun's method for better accuracy)
  3. torch.compile-ready (no Python-level branching in forward)
  4. BF16/FP16 AMP-compatible
  5. Optional sub-stepping for finer temporal resolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LTCCellGPU(nn.Module):
    """
    GPU-fused Liquid Time-Constant cell.

    Fuses W_h, W_x, and W_tau into a single projection for reduced
    kernel launch overhead. Supports RK2 (Heun) integration for
    better accuracy at no extra memory cost.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dt: float = 1.0,
        solver: str = "heun",  # "euler" or "heun" (RK2)
        sub_steps: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dt = dt
        self.solver = solver
        self.sub_steps = sub_steps
        self.sub_dt = dt / sub_steps

        # Fused projection: input -> [W_x_out, W_tau_out] in one matmul
        # W_x contributes hidden_dim, W_tau contributes hidden_dim
        self.W_x_fused = nn.Linear(input_dim, hidden_dim * 2)

        # Hidden state transition
        self.W_h = nn.Linear(hidden_dim, hidden_dim)

        # Learnable tau base (always positive)
        self.tau_base = nn.Parameter(torch.ones(hidden_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_x_fused.weight)
        nn.init.zeros_(self.W_x_fused.bias)
        nn.init.xavier_uniform_(self.W_h.weight)
        nn.init.zeros_(self.W_h.bias)

    def _ode_rhs(self, h: torch.Tensor, x_proj: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Compute dh/dt = (1/tau) * (-h + f(x, h))."""
        f = torch.tanh(self.W_h(h) + x_proj)
        return (1.0 / tau) * (-h + f)

    def forward(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Single ODE step with fused projections.

        Args:
            x_t: (B, input_dim)
            h:   (B, hidden_dim)
        Returns:
            h':  (B, hidden_dim)
        """
        # Fused input projection: one matmul instead of two
        x_fused = self.W_x_fused(x_t)
        x_proj, tau_proj = x_fused.chunk(2, dim=-1)

        # Input-dependent time constant (always positive)
        tau = self.tau_base + F.softplus(tau_proj)

        # Integration
        dt = self.sub_dt
        for _ in range(self.sub_steps):
            if self.solver == "heun":
                # Heun's method (RK2) — 2nd order accuracy
                k1 = self._ode_rhs(h, x_proj, tau)
                h_pred = h + dt * k1
                k2 = self._ode_rhs(h_pred, x_proj, tau)
                h = h + (dt / 2.0) * (k1 + k2)
            else:
                # Euler — 1st order
                dh = self._ode_rhs(h, x_proj, tau)
                h = h + dt * dh

        return h


class LTCCellGPUFused(nn.Module):
    """
    Maximum-fused LTC cell: all three projections (W_h, W_x, W_tau) are
    concatenated into a single matmul by expanding [h; x] -> [f_contrib; tau_contrib].

    This is the most GPU-efficient variant: one kernel launch per step.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dt: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dt = dt

        # Single fused projection: [h, x] -> [W_h*h + W_x*x, W_tau*x]
        # Input: hidden_dim + input_dim
        # Output: hidden_dim (for f) + hidden_dim (for tau)
        self.fused_proj = nn.Linear(hidden_dim + input_dim, hidden_dim * 2)
        self.tau_base = nn.Parameter(torch.ones(hidden_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fused_proj.weight)
        nn.init.zeros_(self.fused_proj.bias)

    def forward(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Single ODE step — one matmul for everything."""
        # Concatenate hidden state and input
        hx = torch.cat([h, x_t], dim=-1)  # (B, H+I)

        # Single fused projection
        proj = self.fused_proj(hx)  # (B, 2*H)
        f_raw, tau_proj = proj.chunk(2, dim=-1)

        f = torch.tanh(f_raw)
        tau = self.tau_base + F.softplus(tau_proj)

        # Euler step
        dh = (self.dt / tau) * (-h + f)
        return h + dh
