"""Perceiver IO cross-attention fusion (Jaegle et al., ICLR 2022).

Solves the variable-cardinality problem: per tick we have a varying number
of satellites, UEs, beams, plus fixed rasters (spectrum, weather). All
become tokens; a fixed-size learned latent array `Z [L, d]` cross-
attends to them, then `n_self` self-attention layers refine. The fixed
latent shape gives the world-model rollout a stable state.

Single-head attention is sufficient at our token counts (~10²–10³); for
scale-up swap in `nn.MultiheadAttention`.

Reference:
    Jaegle et al. *Perceiver IO: A General Architecture for Structured
    Inputs & Outputs*, ICLR 2022 — arXiv:2107.14795.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PerceiverConfig:
    n_latents: int = 256
    d_latent: int = 128
    d_input: int = 128
    n_self_layers: int = 2
    n_heads: int = 4
    ff_mult: int = 2
    dropout: float = 0.0


class _CrossAttention(nn.Module):
    def __init__(self, d_q: int, d_kv: int, n_heads: int, dropout: float):
        super().__init__()
        if d_q % n_heads:
            raise ValueError(f"d_q ({d_q}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.d_head = d_q // n_heads
        self.q_proj = nn.Linear(d_q, d_q, bias=False)
        self.k_proj = nn.Linear(d_kv, d_q, bias=False)
        self.v_proj = nn.Linear(d_kv, d_q, bias=False)
        self.out_proj = nn.Linear(d_q, d_q)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, q: torch.Tensor, kv: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """q: (B, Lq, d_q), kv: (B, Lk, d_kv), mask: (B, Lq, Lk) or None."""
        B, Lq, _ = q.shape
        Lk = kv.shape[1]
        Q = self.q_proj(q).view(B, Lq, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(kv).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(kv).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)
        scores = (Q @ K.transpose(-1, -2)) / (self.d_head ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ V).transpose(1, 2).reshape(B, Lq, -1)
        return self.out_proj(out)


class _SelfAttention(_CrossAttention):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__(d_model, d_model, n_heads, dropout)


class _Block(nn.Module):
    def __init__(self, d: int, ff_mult: int, dropout: float):
        super().__init__()
        self.ff = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d * ff_mult),
            nn.GELU(),
            nn.Linear(d * ff_mult, d),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(x)


class PerceiverFusion(nn.Module):
    """Variable-cardinality cross-attention into a fixed latent."""

    def __init__(self, config: PerceiverConfig | None = None):
        super().__init__()
        self.cfg = config or PerceiverConfig()
        self.latents = nn.Parameter(
            torch.randn(self.cfg.n_latents, self.cfg.d_latent) * 0.02
        )
        self.cross_attn = _CrossAttention(
            self.cfg.d_latent, self.cfg.d_input, self.cfg.n_heads, self.cfg.dropout,
        )
        self.cross_norm_q = nn.LayerNorm(self.cfg.d_latent)
        self.cross_norm_kv = nn.LayerNorm(self.cfg.d_input)
        self.cross_ff = _Block(self.cfg.d_latent, self.cfg.ff_mult, self.cfg.dropout)

        self.self_attns = nn.ModuleList(
            [
                _SelfAttention(self.cfg.d_latent, self.cfg.n_heads, self.cfg.dropout)
                for _ in range(self.cfg.n_self_layers)
            ]
        )
        self.self_norms = nn.ModuleList(
            [nn.LayerNorm(self.cfg.d_latent) for _ in range(self.cfg.n_self_layers)]
        )
        self.self_ffs = nn.ModuleList(
            [
                _Block(self.cfg.d_latent, self.cfg.ff_mult, self.cfg.dropout)
                for _ in range(self.cfg.n_self_layers)
            ]
        )

    def forward(
        self, inputs: torch.Tensor, input_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross-attend learned latents to inputs, then self-attend.

        inputs:     (B, N, d_input). N varies per call.
        input_mask: (B, N) — True where the token is valid. Optional.
        Returns:    (B, n_latents, d_latent) — the fused latent.
        """
        if inputs.ndim != 3:
            raise ValueError(
                f"inputs must be (B, N, d_input), got {tuple(inputs.shape)}"
            )
        B = inputs.shape[0]
        z = self.latents.unsqueeze(0).expand(B, -1, -1).contiguous()

        # Cross attention.
        kv = self.cross_norm_kv(inputs)
        # Build (B, n_latents, N) mask if input mask is given
        if input_mask is not None:
            mask_q = input_mask.unsqueeze(1).expand(-1, self.cfg.n_latents, -1)
        else:
            mask_q = None
        z = z + self.cross_attn(self.cross_norm_q(z), kv, mask=mask_q)
        z = self.cross_ff(z)

        # Self attention layers.
        for sa, norm, ff in zip(
            self.self_attns, self.self_norms, self.self_ffs, strict=True,
        ):
            z = z + sa(norm(z), norm(z))
            z = ff(z)
        return z


__all__ = ["PerceiverConfig", "PerceiverFusion"]
