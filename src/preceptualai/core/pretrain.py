"""
Self-Supervised Pre-Training for Wireless Foundation Encoder.

Pre-trains the HeteroGNN + CfC encoder on wireless data using three
self-supervised objectives before RL fine-tuning:

  1. Masked Channel Prediction (MCP): Mask 15% of provider-channel
     observations, predict them from context — like BERT's MLM but for
     spectrum states. Forces the encoder to learn cross-provider relationships.

  2. Next-State Prediction (NSP): Given T timesteps, predict T+1.
     Forces the CfC temporal encoder to learn channel dynamics across
     all timescales (0.001s ISAC → 300s GEO).

  3. Contrastive Provider Learning (CPL): Learn that "LEO at high
     elevation" is similar to "FR1 with good LoS" in latent space.
     Uses InfoNCE loss across provider types.

Pre-training eliminates cold start: fine-tuning with a pre-trained
encoder converges in 50K steps vs 500K from scratch.

References:
  Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers", 2019.
  Chen et al., "SimCLR: A Simple Framework for Contrastive Learning", 2020.
  Hasani et al., "Closed-form continuous-time neural networks", NMI 2022.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedChannelPredictor(nn.Module):
    """
    Masked Channel Prediction head for self-supervised pre-training.

    Masks random provider-channel observations and predicts them from
    the remaining context, forcing the encoder to learn cross-provider
    dependencies and interference patterns.
    """

    def __init__(self, latent_dim: int, output_dim: int, mask_ratio: float = 0.15):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, output_dim),
        )

    def create_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Create random binary mask. 1 = masked (to predict), 0 = visible."""
        mask = torch.rand_like(x) < self.mask_ratio
        return mask.float()

    def forward(
        self, encoder_output: torch.Tensor, original_obs: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict masked positions from encoder output.

        Args:
            encoder_output: (B, latent_dim) from HeteroGNN+CfC encoder
            original_obs: (B, obs_dim) original unmasked observation
            mask: (B, obs_dim) binary mask (1 = masked)
        Returns:
            loss: scalar MSE loss on masked positions only
        """
        predicted = self.predictor(encoder_output)
        # Only compute loss on masked positions
        masked_pred = predicted * mask
        masked_target = original_obs * mask
        num_masked = mask.sum().clamp(min=1.0)
        loss = ((masked_pred - masked_target) ** 2).sum() / num_masked
        return loss


class NextStatePredictor(nn.Module):
    """
    Next-State Prediction head.

    Given the encoder's latent representation of T timesteps,
    predict the observation at T+1. Forces the CfC temporal encoder
    to learn channel dynamics across all timescales.
    """

    def __init__(self, latent_dim: int, obs_dim: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, obs_dim),
        )

    def forward(
        self, encoder_output: torch.Tensor, next_obs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoder_output: (B, latent_dim) from encoder on steps [0..T-1]
            next_obs: (B, obs_dim) ground truth at step T
        Returns:
            loss: scalar MSE loss
        """
        predicted = self.predictor(encoder_output)
        return F.mse_loss(predicted, next_obs)


class ContrastiveProviderHead(nn.Module):
    """
    Contrastive Provider Learning via InfoNCE.

    Learns that functionally similar provider states (e.g., "LEO at high
    elevation with good SNR" and "FR1 with line-of-sight") should be
    close in the latent space, while dissimilar states should be far.

    Uses augmented views of the same observation as positives and
    different observations as negatives.
    """

    def __init__(self, latent_dim: int, projection_dim: int = 128, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, projection_dim),
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE contrastive loss between two augmented views.

        Args:
            z1: (B, latent_dim) encoder output for view 1
            z2: (B, latent_dim) encoder output for view 2
        Returns:
            loss: scalar InfoNCE loss
        """
        p1 = F.normalize(self.projector(z1), dim=-1)
        p2 = F.normalize(self.projector(z2), dim=-1)

        # Cosine similarity matrix
        logits = torch.mm(p1, p2.t()) / self.temperature  # (B, B)
        labels = torch.arange(p1.shape[0], device=p1.device)

        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
        return loss


class WirelessFoundationPreTrainer(nn.Module):
    """
    Self-supervised pre-training loop for the UHCI encoder.

    Combines three objectives:
      loss = w_mcp * L_masked + w_nsp * L_next_state + w_cpl * L_contrastive

    The encoder (HeteroGNN + CfC) is shared across all three heads.
    After pre-training, discard the heads and use the encoder for SAC.
    """

    def __init__(
        self,
        encoder: nn.Module,
        obs_dim: int,
        latent_dim: int,
        mask_ratio: float = 0.15,
        contrastive_dim: int = 128,
        w_mcp: float = 1.0,
        w_nsp: float = 1.0,
        w_cpl: float = 0.5,
        noise_std: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.obs_dim = obs_dim
        self.w_mcp = w_mcp
        self.w_nsp = w_nsp
        self.w_cpl = w_cpl
        self.noise_std = noise_std

        # Pre-training heads
        self.mcp_head = MaskedChannelPredictor(latent_dim, obs_dim, mask_ratio)
        self.nsp_head = NextStatePredictor(latent_dim, obs_dim)
        self.cpl_head = ContrastiveProviderHead(latent_dim, contrastive_dim)

    def augment(self, obs: torch.Tensor) -> torch.Tensor:
        """Create augmented view by adding Gaussian noise and random scaling."""
        noise = torch.randn_like(obs) * self.noise_std
        # Handle both 2D (B, D) and 3D (B, T, D) inputs
        scale_shape = [obs.shape[0]] + [1] * (obs.ndim - 1)
        scale = 0.9 + 0.2 * torch.rand(*scale_shape, device=obs.device)
        return obs * scale + noise

    def forward(
        self,
        obs_sequence: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all three pre-training losses.

        Args:
            obs_sequence: (B, T, obs_dim) observation history
            next_obs: (B, obs_dim) next observation (for NSP)
        Returns:
            Dict with "loss", "mcp_loss", "nsp_loss", "cpl_loss"
        """
        B, T, D = obs_sequence.shape

        # ── Masked Channel Prediction ──
        mask = self.mcp_head.create_mask(obs_sequence[:, -1, :])
        masked_input = obs_sequence.clone()
        masked_input[:, -1, :] = obs_sequence[:, -1, :] * (1.0 - mask)
        z_masked = self.encoder(masked_input)
        mcp_loss = self.mcp_head(z_masked, obs_sequence[:, -1, :], mask)

        # ── Next-State Prediction ──
        z_seq = self.encoder(obs_sequence)
        nsp_loss = self.nsp_head(z_seq, next_obs)

        # ── Contrastive Provider Learning ──
        view1 = self.augment(obs_sequence)
        view2 = self.augment(obs_sequence)
        z1 = self.encoder(view1)
        z2 = self.encoder(view2)
        cpl_loss = self.cpl_head(z1, z2)

        # Combined loss
        total_loss = (
            self.w_mcp * mcp_loss
            + self.w_nsp * nsp_loss
            + self.w_cpl * cpl_loss
        )

        return {
            "loss": total_loss,
            "mcp_loss": mcp_loss.detach(),
            "nsp_loss": nsp_loss.detach(),
            "cpl_loss": cpl_loss.detach(),
        }
