"""
Kolmogorov-Arnold Network (KAN) Actor for Interpretable DSA Policy.

Replaces MLP action heads with learnable B-spline activation functions
on edges, providing:
  - Better accuracy with fewer parameters
  - Inherent interpretability (inspect learned allocation functions)
  - Regulatory compliance (verify against optimal solutions)
  - Goal-conditioned allocation (QoS-aware, multi-task)

Reference:
  Liu et al., "KAN: Kolmogorov-Arnold Networks", ICLR 2025.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    """
    KAN layer: replaces nn.Linear with learnable edge functions.

    Each weight is parameterized as a B-spline with learnable
    control points, plus a residual linear+SiLU base function.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        base_activation: str = "silu",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Base linear transformation with activation (residual)
        self.base_weight = nn.Parameter(
            torch.randn(out_features, in_features) * (1.0 / in_features ** 0.5)
        )
        self.base_bias = nn.Parameter(torch.zeros(out_features))

        # B-spline control points (learnable)
        # Grid: grid_size + 2*spline_order + 1 knots
        n_knots = grid_size + spline_order + 1
        self.register_buffer(
            'grid',
            torch.linspace(-1.0, 1.0, n_knots).unsqueeze(0).unsqueeze(0)
        )  # (1, 1, n_knots)

        # Spline coefficients: (out_features, in_features, grid_size + spline_order)
        n_basis = grid_size + spline_order
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, n_basis) * 0.1
        )

        if base_activation == "silu":
            self.base_act = nn.SiLU()
        elif base_activation == "gelu":
            self.base_act = nn.GELU()
        else:
            self.base_act = nn.Identity()

    def _rbf_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate radial basis function (RBF) activations as KAN edge functions.

        Uses Gaussian RBFs centered on grid points — simpler and more
        numerically stable than B-spline recursion.

        Args:
            x: (B, in_features) input values
        Returns:
            bases: (B, in_features, n_basis) basis function values
        """
        n_basis = self.spline_weight.shape[2]
        centers = torch.linspace(-1.0, 1.0, n_basis, device=x.device)  # (n_basis,)
        width = 2.0 / max(n_basis - 1, 1)

        # (B, in_features, 1) - (1, 1, n_basis) -> (B, in_features, n_basis)
        x_expanded = x.unsqueeze(-1)
        centers_expanded = centers.unsqueeze(0).unsqueeze(0)
        bases = torch.exp(-((x_expanded - centers_expanded) ** 2) / (2 * width ** 2))
        return bases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_features)
        Returns:
            y: (B, out_features)
        """
        # Base function (residual linear + activation)
        base_out = F.linear(self.base_act(x), self.base_weight, self.base_bias)

        # Spline function
        spline_basis = self._rbf_basis(x)  # (B, in_features, n_basis)
        # Weighted sum: (out, in, n_basis) @ (B, in, n_basis) -> (B, out)
        spline_out = torch.einsum('oin,bin->bo', self.spline_weight, spline_basis)

        return base_out + spline_out


class KANActor(nn.Module):
    """
    KAN-based actor for interpretable spectrum allocation.

    Uses KAN layers instead of MLP for the policy head,
    providing inherently interpretable action selection.

    Supports optional goal conditioning for multi-task allocation:
    when goal_dim > 0, the actor takes a QoS goal vector and conditions
    the policy on it, enabling one model to serve different application
    profiles (video streaming, V2X, IoT, emergency).
    """

    def __init__(
        self,
        encoder_latent_dim: int,
        num_actions: int,
        kan_hidden_dim: int = 64,
        grid_size: int = 5,
        spline_order: int = 3,
        goal_dim: int = 0,
    ):
        super().__init__()
        self.goal_dim = goal_dim

        # Goal embedding: project QoS goal vector to latent space
        if goal_dim > 0:
            self.goal_encoder = nn.Sequential(
                nn.Linear(goal_dim, kan_hidden_dim),
                nn.LayerNorm(kan_hidden_dim),
                nn.SiLU(),
                nn.Linear(kan_hidden_dim, encoder_latent_dim),
            )
            # FiLM conditioning: goal modulates the encoder output
            # z_conditioned = z * (1 + gamma) + beta
            self.film_gamma = nn.Linear(encoder_latent_dim, encoder_latent_dim)
            self.film_beta = nn.Linear(encoder_latent_dim, encoder_latent_dim)
        else:
            self.goal_encoder = None
            self.film_gamma = None
            self.film_beta = None

        self.kan_layers = nn.Sequential(
            KANLinear(encoder_latent_dim, kan_hidden_dim, grid_size, spline_order),
            nn.LayerNorm(kan_hidden_dim),
            KANLinear(kan_hidden_dim, num_actions, grid_size, spline_order),
        )

    def forward(
        self,
        z: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim) encoder output
            goal: (B, goal_dim) optional QoS goal vector
                  [latency_target, throughput_floor, reliability_class, power_budget]
        Returns:
            probs: (B, num_actions) action probabilities
        """
        if self.goal_encoder is not None and goal is not None:
            # FiLM conditioning: modulate encoder latent with goal
            goal_emb = self.goal_encoder(goal)
            gamma = self.film_gamma(goal_emb)
            beta = self.film_beta(goal_emb)
            z = z * (1.0 + gamma) + beta

        logits = self.kan_layers(z)
        return F.softmax(logits, dim=-1)

    def get_spline_stats(self) -> dict:
        """Return spline coefficient statistics for interpretability analysis."""
        stats = {}
        for i, layer in enumerate(self.kan_layers):
            if isinstance(layer, KANLinear):
                w = layer.spline_weight.detach()
                stats[f"layer_{i}"] = {
                    "mean": w.mean().item(),
                    "std": w.std().item(),
                    "sparsity": (w.abs() < 0.01).float().mean().item(),
                }
        return stats


# ======================================================================
# QoS Goal Profiles
# ======================================================================

# Standard QoS goal vectors for different 5G/6G slice types
# Format: [latency_target_ms, throughput_floor_mbps, reliability_class, power_budget_frac]
# All normalized to [0, 1] range for network input

QOS_PROFILES = {
    "embb":        [0.2, 0.8, 0.5, 0.7],   # Enhanced Mobile Broadband: high throughput
    "urllc":       [0.95, 0.2, 0.95, 0.5],  # Ultra-Reliable Low-Latency: strict latency + reliability
    "mmtc":        [0.1, 0.05, 0.3, 0.1],   # Massive Machine-Type: low power, low throughput
    "v2x":         [0.9, 0.4, 0.9, 0.6],    # Vehicle-to-Everything: low latency + reliability
    "streaming":   [0.3, 0.7, 0.4, 0.6],    # Video streaming: throughput-focused
    "emergency":   [0.8, 0.3, 0.99, 0.9],   # Emergency services: max reliability
    "iot_sensor":  [0.05, 0.01, 0.2, 0.05], # IoT sensors: minimize power
    "xr":          [0.85, 0.6, 0.8, 0.7],   # Extended Reality: balanced latency + throughput
}

QOS_GOAL_DIM = 4  # [latency, throughput, reliability, power]
