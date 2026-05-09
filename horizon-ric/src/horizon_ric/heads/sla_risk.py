"""SLA risk head — multi-horizon SLA breach probability.

Predicts P(SLA_breach) at 30s, 1min, 5min horizons from the shared latent
``z_resource``. Uses two-hot symlog regression over a logit-space grid for
calibrated probability outputs.

References:
    PRD §2.3 (Risk Prediction)
    PARADIGMS.md §1.1 (multi-head shared latent)
    3GPP TS 28.554 §6.x (E2E KPIs we predict against)
    DreamerV3 — two-hot symlog regression
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from horizon_ric.heads._two_hot import (
    make_bins,
    two_hot_decode,
    two_hot_loss,
)


@dataclass
class SLARiskConfig:
    latent_dim: int = 1024
    hidden_dim: int = 256
    num_bins: int = 51
    # Logit-space bin range. P(breach) ∈ [0,1] maps to logits ∈ ~[-7, 7]
    bin_low: float = -7.0
    bin_high: float = 7.0
    horizons_seconds: tuple[int, ...] = field(default_factory=lambda: (30, 60, 300))


class SLARiskHead(nn.Module):
    """Multi-horizon SLA breach probability predictor.

    Output shape per call: dict mapping horizon (str) -> (B,) probability.
    Trained via two-hot CE on logit-space targets.
    """

    def __init__(self, config: SLARiskConfig | None = None):
        super().__init__()
        self.cfg = config or SLARiskConfig()
        self.register_buffer(
            "bins",
            make_bins(self.cfg.num_bins, self.cfg.bin_low, self.cfg.bin_high),
        )
        # One logit head per horizon; small MLP from the shared latent.
        self.heads = nn.ModuleDict(
            {
                f"h_{h}s": nn.Sequential(
                    nn.Linear(self.cfg.latent_dim, self.cfg.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.cfg.hidden_dim, self.cfg.num_bins),
                )
                for h in self.cfg.horizons_seconds
            }
        )

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict SLA breach probability at each horizon.

        Args:
            z: (B, latent_dim) shared latent.

        Returns:
            dict { "h_30s": (B,), "h_60s": (B,), "h_300s": (B,) } of probabilities.
        """
        out: dict[str, torch.Tensor] = {}
        for key, head in self.heads.items():
            logits = head(z)  # (B, num_bins) — bins are in logit space directly
            # apply_symlog=False because bins ARE the logit grid; no warp.
            decoded_logit = two_hot_decode(logits, self.bins, apply_symlog=False)
            out[key] = torch.sigmoid(decoded_logit)
        return out

    def predict_logits(self, z: torch.Tensor, horizon_key: str) -> torch.Tensor:
        """Return raw logits (B, num_bins) for one horizon, for loss computation."""
        return self.heads[horizon_key](z)

    def loss(
        self,
        z: torch.Tensor,
        target_probabilities: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Sum of per-horizon two-hot CE losses.

        Args:
            z: (B, latent_dim) shared latent.
            target_probabilities: dict mapping horizon_key -> (B,) ∈ [0,1].

        Returns:
            Scalar loss.
        """
        total = z.new_zeros(())
        for horizon_key, p_target in target_probabilities.items():
            if horizon_key not in self.heads:
                continue
            # Convert probability → logit (eps for stability)
            p_clamped = p_target.clamp(1e-6, 1.0 - 1e-6)
            logit_target = torch.log(p_clamped / (1.0 - p_clamped))
            logits = self.predict_logits(z, horizon_key)
            # apply_symlog=False — bins live in logit space, target is already logit.
            total = total + two_hot_loss(
                logits, logit_target, self.bins, apply_symlog=False
            )
        return total / max(len(target_probabilities), 1)

    def calibration_summary(
        self,
        predicted: dict[str, torch.Tensor],
        observed: dict[str, torch.Tensor],
        n_bins: int = 10,
    ) -> dict[str, float]:
        """Expected Calibration Error (ECE) per horizon.

        Args:
            predicted: dict { horizon_key: (N,) predicted P(breach) }
            observed: dict { horizon_key: (N,) binary outcomes 0/1 }
            n_bins: number of calibration bins.

        Returns:
            dict { horizon_key: ECE in [0, 1] }, computed per Naeini et al. 2015
            with equal-width bins on [0, 1]. Bin i covers [edges[i], edges[i+1])
            for i < n_bins-1, and [edges[n_bins-1], 1.0] for the last bin so that
            both endpoints (0.0 and 1.0) are included exactly once.
        """
        out: dict[str, float] = {}
        for key in predicted.keys() & observed.keys():
            p = predicted[key].detach().cpu()
            y = observed[key].detach().cpu().float()
            n = max(p.numel(), 1)
            ece = 0.0
            edges = torch.linspace(0.0, 1.0, n_bins + 1)
            for i in range(n_bins):
                if i < n_bins - 1:
                    in_bin = (p >= edges[i]) & (p < edges[i + 1])
                else:
                    in_bin = (p >= edges[i]) & (p <= edges[i + 1])
                count = int(in_bin.sum().item())
                if count == 0:
                    continue
                acc = y[in_bin].mean().item()
                conf = p[in_bin].mean().item()
                ece += (count / n) * abs(acc - conf)
            out[key] = ece
        return out

    def brier_score(
        self,
        predicted: dict[str, torch.Tensor],
        observed: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Per-horizon Brier score (mean squared error vs binary outcomes)."""
        out: dict[str, float] = {}
        for key in predicted.keys() & observed.keys():
            p = predicted[key].detach().cpu().float()
            y = observed[key].detach().cpu().float()
            out[key] = float(((p - y) ** 2).mean().item())
        return out
