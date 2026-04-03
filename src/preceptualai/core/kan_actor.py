"""
Kolmogorov-Arnold Network (KAN) Actor for Interpretable DSA Policy.

Replaces MLP action heads with learnable B-spline activation functions
on edges, providing:
  - Better accuracy with fewer parameters
  - Inherent interpretability (inspect learned allocation functions)
  - Regulatory compliance (verify against optimal solutions)

Reference:
  Liu et al., "KAN: Kolmogorov-Arnold Networks", ICLR 2025.
"""

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
    """

    def __init__(
        self,
        encoder_latent_dim: int,
        num_actions: int,
        kan_hidden_dim: int = 64,
        grid_size: int = 5,
        spline_order: int = 3,
    ):
        super().__init__()
        self.kan_layers = nn.Sequential(
            KANLinear(encoder_latent_dim, kan_hidden_dim, grid_size, spline_order),
            nn.LayerNorm(kan_hidden_dim),
            KANLinear(kan_hidden_dim, num_actions, grid_size, spline_order),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim) encoder output
        Returns:
            probs: (B, num_actions) action probabilities
        """
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
