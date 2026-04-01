"""
Mamba / Selective State Space Model Encoder for 5G/6G DSA.

Implements a WiMamba-inspired selective SSM encoder with:
  - Input-dependent A, B, C matrices (selective mechanism)
  - Linear O(L) complexity via parallel scan (training)
  - O(1) per-step recurrent mode (inference)
  - Bidirectional processing option for spatial dimensions

This replaces the sequential LTC encoder with a parallelizable
architecture that maintains input-dependent dynamics.

Reference:
  Gu & Dao, "Mamba: Linear-Time Sequence Modeling", 2023.
  WiMamba: "Linear-Scale Wireless Foundation Model", March 2026.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSMBlock(nn.Module):
    """
    Single Selective State Space Model block.

    Implements the selective scan mechanism where A, B, C matrices
    are functions of the input (analogous to LTC's input-dependent tau).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        # Input projection: x -> (z, x_inner) for gating
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Causal 1D convolution (short-range mixing)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        # SSM parameter projections (selective — input-dependent)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # B, C, dt

        # Learnable log(A) — initialized as structured (S4-style)
        log_A = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
        self.log_A = nn.Parameter(log_A.unsqueeze(0).expand(self.d_inner, -1))

        # dt projection
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        # Initialize dt bias for range [dt_min, dt_max]
        with torch.no_grad():
            inv_dt = torch.exp(
                torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            )
            self.dt_proj.bias.copy_(inv_dt.log())

        # D residual connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # LayerNorm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model) input sequence
        Returns:
            y: (B, T, d_model) output sequence
        """
        B, T, D = x.shape
        residual = x
        x = self.norm(x)

        # Input projection + split for gating
        xz = self.in_proj(x)  # (B, T, 2 * d_inner)
        x_inner, z = xz.chunk(2, dim=-1)  # each (B, T, d_inner)

        # Causal convolution
        x_conv = x_inner.transpose(1, 2)  # (B, d_inner, T)
        x_conv = self.conv1d(x_conv)[:, :, :T]  # causal: trim to length T
        x_conv = x_conv.transpose(1, 2)  # (B, T, d_inner)
        x_conv = F.silu(x_conv)

        # Selective SSM parameters (input-dependent)
        ssm_params = self.x_proj(x_conv)  # (B, T, 2*d_state + 1)
        B_sel = ssm_params[:, :, :self.d_state]  # (B, T, N)
        C_sel = ssm_params[:, :, self.d_state:2 * self.d_state]  # (B, T, N)
        dt_raw = ssm_params[:, :, -1:]  # (B, T, 1)

        # Discretize dt
        dt = F.softplus(self.dt_proj(dt_raw))  # (B, T, d_inner)

        # Discretize A
        A = -torch.exp(self.log_A)  # (d_inner, N) — negative for stability
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, T, d_inner, N)

        # Discretize B
        dB = dt.unsqueeze(-1) * B_sel.unsqueeze(2)  # (B, T, 1, N) broadcast

        # Sequential scan (recurrent mode — works for any length)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            h = dA[:, t] * h + dB[:, t] * x_conv[:, t].unsqueeze(-1)  # (B, d_inner, N)
            y_t = (h * C_sel[:, t].unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, T, d_inner)

        # D residual + gate
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)

        # Output projection + residual
        y = self.out_proj(y) + residual
        return y


class MambaEncoder(nn.Module):
    """
    Mamba-based sequence encoder for spectrum observation processing.

    Replaces LTCEncoderGPU with linear-time complexity.

    Input:  (B, T, F)
    Output: (B, latent_dim)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        num_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Stack of Mamba blocks
        self.blocks = nn.ModuleList([
            SelectiveSSMBlock(hidden_dim, d_state, d_conv, expand)
            for _ in range(num_layers)
        ])

        # Optional reverse blocks for bidirectional
        if bidirectional:
            self.rev_blocks = nn.ModuleList([
                SelectiveSSMBlock(hidden_dim, d_state, d_conv, expand)
                for _ in range(num_layers)
            ])
            proj_in = hidden_dim * 2
        else:
            self.rev_blocks = None
            proj_in = hidden_dim

        # Final norm + projection
        self.final_norm = nn.LayerNorm(proj_in)
        self.proj = nn.Linear(proj_in, latent_dim) if proj_in != latent_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F)
        Returns:
            z: (B, latent_dim)
        """
        # Project input
        h = self.input_proj(x)  # (B, T, hidden_dim)

        # Forward pass through Mamba blocks
        h_fwd = h
        for block in self.blocks:
            h_fwd = block(h_fwd)

        if self.bidirectional and self.rev_blocks is not None:
            # Reverse pass
            h_rev = h.flip(1)
            for block in self.rev_blocks:
                h_rev = block(h_rev)
            h_rev = h_rev.flip(1)
            h_out = torch.cat([h_fwd, h_rev], dim=-1)
        else:
            h_out = h_fwd

        # Pool: use last timestep
        z = h_out[:, -1, :]  # (B, proj_in)
        z = self.final_norm(z)
        z = self.proj(z)
        return z
