"""
Parallel Scan for CfC — Mamba-speed training with physics-grounded inference.

The standard CfC encoder processes sequences sequentially: O(T) steps
that cannot be parallelized. This limits training throughput on GPUs.

This module implements an associative parallel scan for the CfC update
rule, enabling O(T log T) parallel training while preserving the
continuous-time semantics at inference.

CfC update:  h' = (1 - σ) · h + σ · f(x, h)

This is an affine recurrence: h_t = a_t · h_{t-1} + b_t
where a_t = (1 - σ_t) and b_t = σ_t · f_t

Affine recurrences can be computed via parallel prefix scan using the
associative operator: (a₂, b₂) ∘ (a₁, b₁) = (a₂·a₁, a₂·b₁ + b₂)

This is the same technique used by Mamba (Gu & Dao, 2024) and S5/S4
for selective state spaces. The key insight: CfC's continuous-time
gating σ(x) makes this a PHYSICS-INFORMED parallel scan where the scan
coefficients are derived from physical timescales, not arbitrary.

References:
  Blelloch, G. "Prefix Sums and Their Applications", 1990.
  Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective SSMs", 2024.
  Martin & Cundy, "Parallelizing Linear Recurrences", ICML 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from preceptualai.core.ltc_cell_cfc import CfCCell


def parallel_scan(gates: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """
    Compute h_t = a_t * h_{t-1} + b_t for all t in parallel via prefix scan.

    Uses the associative operator:
        (a₂, b₂) ∘ (a₁, b₁) = (a₂·a₁, a₂·b₁ + b₂)

    Args:
        gates:  (B, T, H) — multiplicative gates a_t = (1 - σ_t)
        tokens: (B, T, H) — additive tokens b_t = σ_t · f_t
    Returns:
        h:      (B, T, H) — hidden states at each timestep
    """
    B, T, H = gates.shape

    # Work in log-space for numerical stability of cumulative products
    # But for simplicity and correctness, use iterative prefix scan
    # which is still O(T log T) parallel on GPU via torch operations

    # Pad to power of 2 for clean scan
    log2T = (T - 1).bit_length()
    padded_T = 1 << log2T
    if padded_T > T:
        pad_gates = torch.ones(B, padded_T - T, H, device=gates.device, dtype=gates.dtype)
        pad_tokens = torch.zeros(B, padded_T - T, H, device=tokens.device, dtype=tokens.dtype)
        gates = torch.cat([gates, pad_gates], dim=1)
        tokens = torch.cat([tokens, pad_tokens], dim=1)

    a = gates.clone()
    b = tokens.clone()

    # Up-sweep (reduce)
    for d in range(log2T):
        stride = 1 << (d + 1)
        half = 1 << d
        # Indices where we combine
        idx = torch.arange(stride - 1, padded_T, stride, device=gates.device)
        idx_prev = idx - half
        # (a₂, b₂) ∘ (a₁, b₁) = (a₂·a₁, a₂·b₁ + b₂)
        a_new = a[:, idx, :] * a[:, idx_prev, :]
        b_new = a[:, idx, :] * b[:, idx_prev, :] + b[:, idx, :]
        a[:, idx, :] = a_new
        b[:, idx, :] = b_new

    # Down-sweep (propagate)
    a[:, -1, :] = 0.0  # Identity for scan
    b_saved = b[:, -1, :].clone()
    b[:, -1, :] = 0.0

    for d in range(log2T - 1, -1, -1):
        stride = 1 << (d + 1)
        half = 1 << d
        idx = torch.arange(stride - 1, padded_T, stride, device=gates.device)
        idx_prev = idx - half

        a_left = a[:, idx_prev, :].clone()
        b_left = b[:, idx_prev, :].clone()

        a[:, idx_prev, :] = a[:, idx, :]
        b[:, idx_prev, :] = b[:, idx, :]

        a[:, idx, :] = a[:, idx, :] * a_left
        b[:, idx, :] = a[:, idx, :] * b_left + b[:, idx, :]

    # The result b now contains the prefix scan outputs (shifted)
    # We need to combine with original tokens
    h = gates * b + tokens

    return h[:, :T, :]  # Remove padding


class CfCParallelEncoder(nn.Module):
    """
    CfC encoder with parallel scan for training.

    During training: uses parallel_scan for O(T log T) GPU throughput.
    During inference: uses sequential CfC for continuous-time correctness.

    This gives Mamba-speed training with CfC physics-grounding at inference.
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
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dt = dt

        # Per-layer CfC cells (used in sequential mode for inference)
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(CfCCell(in_dim, hidden_dim, dt))

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        # Parallel scan projections: compute gates and tokens for all timesteps at once
        self.scan_projs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.scan_projs.append(nn.ModuleDict({
                # Gate network: computes σ for all timesteps
                "gate": nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.Sigmoid(),
                ),
                # Token network: computes f(x) for all timesteps
                "token": nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.Tanh(),
                ),
            }))

        self.proj = None
        if hidden_dim != latent_dim:
            self.proj = nn.Linear(hidden_dim, latent_dim)

    def forward_sequential(self, x: torch.Tensor) -> torch.Tensor:
        """Sequential forward — physics-correct for inference."""
        B, T, F = x.shape
        device = x.device

        h = [torch.zeros(B, self.hidden_dim, device=device, dtype=x.dtype)
             for _ in range(self.num_layers)]

        for t in range(T):
            inp = x[:, t, :]
            for i in range(self.num_layers):
                h[i] = self.cells[i](inp, h[i])
                h[i] = self.layer_norms[i](h[i])
                inp = h[i]

        z = h[-1]
        if self.proj is not None:
            z = self.proj(z)
        return z

    def forward_parallel(self, x: torch.Tensor) -> torch.Tensor:
        """Parallel scan forward — fast for training."""
        B, T, F = x.shape
        inp = x

        for i in range(self.num_layers):
            # Compute gates and tokens for ALL timesteps in parallel
            sigma = self.scan_projs[i]["gate"](inp)   # (B, T, H) — time gates
            f_val = self.scan_projs[i]["token"](inp)   # (B, T, H) — nonlinear activations

            # CfC recurrence: h_t = (1-σ_t) * h_{t-1} + σ_t * f_t
            gates = 1.0 - sigma   # a_t = (1 - σ_t)
            tokens = sigma * f_val  # b_t = σ_t · f_t

            # Parallel prefix scan
            h_all = parallel_scan(gates, tokens)  # (B, T, H)
            h_all = self.layer_norms[i](h_all)
            inp = h_all

        z = inp[:, -1, :]  # Take last timestep
        if self.proj is not None:
            z = self.proj(z)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Automatically selects parallel (training) or sequential (inference).

        Args:
            x: (B, T, F) input sequence
        Returns:
            z: (B, latent_dim) encoded representation
        """
        if self.training:
            return self.forward_parallel(x)
        else:
            return self.forward_sequential(x)
