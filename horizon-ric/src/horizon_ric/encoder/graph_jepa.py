"""Graph-JEPA — masked latent prediction over the Perceiver latent.

Action-free pretraining following the I-JEPA / V-JEPA recipe:
    * **Context** encoder (online, learns) sees the visible tokens.
    * **Target** encoder (EMA copy of context) sees the same input but
      *unmasked*; we use its embeddings of the masked positions as the
      prediction target.
    * **Predictor** maps (context_latent, mask_indices) → predicted
      target latent.
    * Loss is smooth-L1 in latent space — never in token space — which
      avoids representation collapse and the wasted-capacity issue of
      pixel/MAE-style reconstruction.

This implementation operates on *Perceiver latents*: the input is the
output of `encoder.PerceiverFusion`, shape (B, n_latents, d_latent). We
mask a subset of latent positions and predict their target embeddings.

References:
    Assran et al. *I-JEPA: Self-Supervised Learning from Images with
        Joint-Embedding Predictive Architecture*, CVPR 2023
        — arXiv:2301.08243.
    Bardes et al. *V-JEPA: Revisiting Feature Prediction for Learning
        Visual Representations from Video*, 2024 — arXiv:2404.08471.
    Meta AI *V-JEPA 2 — Action-Conditioned Predictor*, June 2025
        (Meta AI tech report).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GraphJEPAConfig:
    d_latent: int = 128
    predictor_hidden: int = 256
    predictor_layers: int = 3
    ema_momentum: float = 0.996
    """β for the target-encoder EMA: θ_T ← β·θ_T + (1−β)·θ_C."""
    mask_ratio: float = 0.75
    """Fraction of latent positions masked during pretraining."""


class _Predictor(nn.Module):
    """Lightweight transformer that maps context tokens + mask-position
    queries to predicted target embeddings."""

    def __init__(self, d_latent: int, hidden: int, n_layers: int):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_latent,
            nhead=4,
            dim_feedforward=hidden,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_latent) * 0.02)
        self.pos_embed = nn.Embedding(2048, d_latent)  # plenty of slots

    def forward(
        self,
        context_tokens: torch.Tensor,         # (B, Lc, D)
        context_pos: torch.Tensor,            # (B, Lc) integer positions
        mask_pos: torch.Tensor,               # (B, Lm) integer positions
    ) -> torch.Tensor:
        """Returns predicted target embeddings of shape (B, Lm, D)."""
        B, Lc, D = context_tokens.shape
        Lm = mask_pos.shape[1]
        ctx = context_tokens + self.pos_embed(context_pos)
        masks = self.mask_token.expand(B, Lm, D) + self.pos_embed(mask_pos)
        x = torch.cat([ctx, masks], dim=1)
        out = self.transformer(x)
        # The trailing Lm tokens are the predictor outputs for masked positions.
        return out[:, Lc:, :]


class GraphJEPA(nn.Module):
    """Joint-Embedding Predictive Architecture over Perceiver latents."""

    def __init__(
        self,
        context_encoder: nn.Module,
        config: GraphJEPAConfig | None = None,
    ):
        super().__init__()
        self.cfg = config or GraphJEPAConfig()
        self.context_encoder = context_encoder
        # Target encoder is a frozen EMA copy of the context encoder.
        self.target_encoder = deepcopy(context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = _Predictor(
            self.cfg.d_latent,
            self.cfg.predictor_hidden,
            self.cfg.predictor_layers,
        )

    @torch.no_grad()
    def update_target(self) -> None:
        """EMA update: θ_T ← β·θ_T + (1−β)·θ_C."""
        beta = self.cfg.ema_momentum
        for p_t, p_c in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters(),
            strict=True,
        ):
            p_t.data.mul_(beta).add_(p_c.data, alpha=1.0 - beta)

    @staticmethod
    def random_mask(
        n_tokens: int, mask_ratio: float, batch_size: int, device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample random partition into (context, masked) positions.

        Returns:
            context_pos: (B, Lc) — positions kept by the context encoder.
            mask_pos:    (B, Lm) — positions whose latent is the prediction
                                   target.
        """
        Lm = max(int(round(n_tokens * mask_ratio)), 1)
        Lc = n_tokens - Lm
        if Lc <= 0:
            raise ValueError(
                f"mask_ratio {mask_ratio} leaves zero context tokens"
            )
        # Generate per-sample random permutations.
        perm = torch.argsort(torch.rand(batch_size, n_tokens, device=device), dim=-1)
        return perm[:, :Lc], perm[:, Lc:]

    def loss(
        self,
        inputs: torch.Tensor,                   # (B, N, d_input)
        input_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute JEPA loss for a batch.

        Returns dict with `loss`, `pred_norm`, `target_norm`.
        """
        # Target encoder runs no-grad on the FULL input.
        with torch.no_grad():
            target_full = self.target_encoder(inputs, input_mask=input_mask)
        # Context encoder runs WITH grad on the same input — we still
        # mask-out positions via index selection downstream.
        context_full = self.context_encoder(inputs, input_mask=input_mask)

        B, n_tokens, D = context_full.shape
        ctx_pos, mask_pos = self.random_mask(
            n_tokens, self.cfg.mask_ratio, B, inputs.device,
        )

        ctx_tokens = torch.gather(
            context_full, 1, ctx_pos.unsqueeze(-1).expand(-1, -1, D),
        )
        target_tokens = torch.gather(
            target_full, 1, mask_pos.unsqueeze(-1).expand(-1, -1, D),
        )
        pred_tokens = self.predictor(ctx_tokens, ctx_pos, mask_pos)

        loss = F.smooth_l1_loss(pred_tokens, target_tokens)
        return {
            "loss": loss,
            "pred_norm": pred_tokens.detach().norm(dim=-1).mean(),
            "target_norm": target_tokens.detach().norm(dim=-1).mean(),
        }


__all__ = ["GraphJEPA", "GraphJEPAConfig"]
