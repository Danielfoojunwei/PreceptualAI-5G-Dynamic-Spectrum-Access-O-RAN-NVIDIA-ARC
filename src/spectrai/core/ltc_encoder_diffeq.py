"""
Multi-layer LTC encoder using torchdiffeq ODE solvers.

Same interface as LTCEncoderGPU but uses LTCCellDiffeq for
adaptive ODE integration and optional adjoint backpropagation.
"""

from typing import Optional

import torch
import torch.nn as nn

from spectrai.core.ltc_cell_diffeq import LTCCellDiffeq


class LTCEncoderDiffeq(nn.Module):
    """
    Multi-layer LTC encoder with torchdiffeq ODE solvers.

    Input:  (B, T, F)
    Output: (B, latent_dim)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        num_layers: int = 2,
        dt: float = 1.0,
        solver: str = "dopri5",
        use_adjoint: bool = True,
        rtol: float = 1e-3,
        atol: float = 1e-4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(
                LTCCellDiffeq(in_dim, hidden_dim, dt, solver, use_adjoint, rtol, atol)
            )

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        self.proj: Optional[nn.Linear] = None
        if hidden_dim != latent_dim:
            self.proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        device = x.device

        h = [torch.zeros(B, self.hidden_dim, device=device, dtype=x.dtype)
             for _ in range(self.num_layers)]

        for t in range(T):
            inp = x[:, t, :]
            for layer_idx in range(self.num_layers):
                h[layer_idx] = self.cells[layer_idx](inp, h[layer_idx])
                h[layer_idx] = self.layer_norms[layer_idx](h[layer_idx])
                inp = h[layer_idx]

        z = h[-1]
        if self.proj is not None:
            z = self.proj(z)
        return z
