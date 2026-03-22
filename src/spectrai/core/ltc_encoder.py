"""
Multi-layer Liquid Time-Constant sequence encoder.

Input:  (Batch, Seq_Len, Features)
Output: (Batch, Latent_Dim)

Stacks multiple LTC cells (layers) and uses the final hidden
state of the last layer as the sequence representation.
"""

from typing import Optional

import torch
import torch.nn as nn

from spectrai.core.ltc_cell import LTCCell


class LTCEncoder(nn.Module):
    """
    Multi-layer Liquid Time-Constant sequence encoder.

    Input:  (Batch, Seq_Len, Features)
    Output: (Batch, Latent_Dim)

    Stacks multiple LTC cells (layers) and uses the final hidden
    state of the last layer as the sequence representation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        num_layers: int = 2,
        dt: float = 1.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Stack of LTC layers — first layer takes input_dim, rest take hidden_dim
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(LTCCell(in_dim, hidden_dim, dt))

        # Layer norms between LTC layers for training stability
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        # Optional projection to latent dim
        self.proj: Optional[nn.Linear] = None
        if hidden_dim != latent_dim:
            self.proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F)
        Returns:
            z: (B, latent_dim)
        """
        B, T, F = x.shape

        # Initialise hidden states for each layer
        h = [torch.zeros(B, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]

        # Process sequence step by step
        for t in range(T):
            inp = x[:, t, :]  # (B, F)
            for layer_idx, (cell, norm) in enumerate(zip(self.cells, self.layer_norms)):
                h[layer_idx] = cell(inp, h[layer_idx])
                h[layer_idx] = norm(h[layer_idx])
                inp = h[layer_idx]  # feed to next layer

        # Use final hidden state of last layer
        z = h[-1]  # (B, hidden_dim)
        if self.proj is not None:
            z = self.proj(z)
        return z
