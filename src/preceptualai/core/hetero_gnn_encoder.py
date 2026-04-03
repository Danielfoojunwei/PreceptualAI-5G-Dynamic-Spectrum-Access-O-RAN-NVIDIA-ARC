"""
Heterogeneous Graph Neural Network Encoder for Multi-Provider Connectivity.

Extends the existing GNN spatial encoder (gnn_encoder.py) to handle
heterogeneous node and edge types — the key requirement for the UHCI paradigm
where nodes are connectivity providers with wildly different physics.

Architecture:
  1. HeteroNodeProjection — projects each provider type's features into a
     shared latent space using type-specific linear layers.
  2. HeteroGraphAttentionLayer — multi-head attention where attention scores
     are conditioned on edge type (coexistence relationship type).
  3. HeteroGNNEncoder — stacks projection + attention layers, pools to a
     single latent vector, and optionally chains into a temporal encoder.

The heterogeneous formulation is critical because:
  - A LEO node has Doppler shift as a key feature; a GEO node has rain fade.
  - Terrestrial-to-satellite interference edges differ fundamentally from
    LEO-to-GEO interference edges.
  - Shared-weight homogeneous GNNs lose this discriminating information.

References:
  Wang et al., "Heterogeneous Graph Attention Network", WWW 2019.
  Hu et al., "HGT: Heterogeneous Graph Transformer", WWW 2020.
  HR-GAT for HetNet spectrum sharing, arXiv March 2026.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from preceptualai.env.provider_registry import (
    PROVIDER_FEATURE_DIM,
    ProviderType,
    get_all_provider_types,
)


class HeteroNodeProjection(nn.Module):
    """
    Project per-provider-type raw features into a shared latent space.

    Each provider type gets its own Linear layer so the model learns type-
    specific transformations, then concatenates all providers into one tensor.
    """

    def __init__(
        self,
        provider_types:   List[ProviderType],
        raw_feature_dim:  int,
        latent_dim:       int,
    ):
        super().__init__()
        self.provider_types = provider_types
        self.latent_dim = latent_dim

        # One projection layer per provider type
        self.projections = nn.ModuleDict({
            pt.name: nn.Linear(raw_feature_dim, latent_dim)
            for pt in provider_types
        })
        self.norms = nn.ModuleDict({
            pt.name: nn.LayerNorm(latent_dim)
            for pt in provider_types
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, P, raw_feature_dim) — raw node features for P providers
        Returns:
            h: (B, P, latent_dim) — type-projected latent node embeddings
        """
        B, P, _ = x.shape
        out = torch.zeros(B, P, self.latent_dim, device=x.device, dtype=x.dtype)
        for i, pt in enumerate(self.provider_types):
            proj = self.projections[pt.name]
            norm = self.norms[pt.name]
            out[:, i, :] = norm(F.silu(proj(x[:, i, :])))
        return out


class HeteroEdgeAttention(nn.Module):
    """
    Edge-conditioned multi-head attention for heterogeneous graphs.

    Attention scores between provider i and j are computed from:
      - Source node embedding h_i
      - Target node embedding h_j
      - Edge type embedding (encodes the coexistence relationship type)

    This allows the model to learn that, e.g., LEO→FR3 interference behaves
    differently from LEO→GEO interference, even if both share the same band.
    """

    def __init__(
        self,
        latent_dim:    int,
        num_heads:     int = 4,
        num_edge_types: int = 3,  # none / co-primary / co-secondary
        dropout:       float = 0.1,
    ):
        super().__init__()
        assert latent_dim % num_heads == 0
        self.latent_dim    = latent_dim
        self.num_heads     = num_heads
        self.head_dim      = latent_dim // num_heads
        self.num_edge_types = num_edge_types

        # Q, K, V projections (shared across edge types)
        self.W_q = nn.Linear(latent_dim, latent_dim, bias=False)
        self.W_k = nn.Linear(latent_dim, latent_dim, bias=False)
        self.W_v = nn.Linear(latent_dim, latent_dim, bias=False)

        # Edge type embeddings
        self.edge_embed = nn.Embedding(num_edge_types, self.head_dim)

        # Output projection
        self.W_o = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        h:       torch.Tensor,    # (B, P, D)
        adj:     torch.Tensor,    # (P, P) coexistence adjacency (0 or 1)
        edge_types: Optional[torch.Tensor] = None,  # (P, P) edge type ids
    ) -> torch.Tensor:
        """
        Args:
            h:          (B, P, D) node embeddings
            adj:        (P, P) binary adjacency
            edge_types: (P, P) integer edge type ids, defaults to ones (all same)
        Returns:
            h_out: (B, P, D) updated node embeddings
        """
        B, P, D = h.shape
        H, Dh = self.num_heads, self.head_dim

        if edge_types is None:
            edge_types = adj.long()  # treat adj == 1 as edge type 1, 0 as type 0

        # Compute Q, K, V — reshape to (B, P, H, Dh)
        Q = self.W_q(h).view(B, P, H, Dh)  # (B, P, H, Dh)
        K = self.W_k(h).view(B, P, H, Dh)
        V = self.W_v(h).view(B, P, H, Dh)

        # Edge type biases: (P, P, Dh)
        e_bias = self.edge_embed(edge_types)   # (P, P, Dh)

        # Attention scores: (B, H, P, P)
        # score[b,h,i,j] = Q[b,i,h] · (K[b,j,h] + edge_bias[i,j]) / sqrt(Dh)
        Q_ = Q.permute(0, 2, 1, 3)  # (B, H, P, Dh)
        K_ = K.permute(0, 2, 1, 3)  # (B, H, P, Dh)
        V_ = V.permute(0, 2, 1, 3)  # (B, H, P, Dh)

        # Add edge bias to keys
        e = e_bias.unsqueeze(0).unsqueeze(0)  # (1, 1, P, P, Dh)
        K_exp = K_.unsqueeze(-2).expand(B, H, P, P, Dh)     # (B, H, P, P, Dh)
        K_biased = K_exp + e.expand(B, H, P, P, Dh)          # (B, H, P, P, Dh)

        Q_exp = Q_.unsqueeze(-1)  # (B, H, P, Dh, 1)
        scores = torch.matmul(K_biased, Q_exp).squeeze(-1) / (Dh ** 0.5)  # (B, H, P, P)

        # Mask non-edges (zero attention on non-adjacent nodes except self)
        self_loop = torch.eye(P, device=h.device).bool()
        mask = (adj.bool() | self_loop).unsqueeze(0).unsqueeze(0)  # (1, 1, P, P)
        scores = scores.masked_fill(~mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)   # (B, H, P, P)
        attn = self.dropout(attn)

        # Aggregate values
        out = torch.matmul(attn, V_)  # (B, H, P, Dh)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, P, D)  # (B, P, D)
        out = self.W_o(out)

        # Residual + norm
        return self.norm(h + out)


class HeteroGNNEncoder(nn.Module):
    """
    Full heterogeneous GNN encoder for multi-provider connectivity graphs.

    Processes a batch of heterogeneous graph snapshots into a fixed-size
    latent vector suitable for the SAC-LTC policy.

    Pipeline:
      raw_node_features (B, P, raw_dim)
        → HeteroNodeProjection → (B, P, latent_dim)
        → N × HeteroEdgeAttention → (B, P, latent_dim)
        → mean pool across providers → (B, latent_dim)
        → output projection → (B, output_dim)
    """

    def __init__(
        self,
        provider_types:  List[ProviderType],
        raw_feature_dim: int,
        latent_dim:      int = 128,
        output_dim:      int = 256,
        num_gnn_layers:  int = 3,
        num_heads:       int = 4,
        dropout:         float = 0.1,
    ):
        super().__init__()
        self.provider_types = provider_types
        self.num_providers  = len(provider_types)
        self.latent_dim     = latent_dim
        self.output_dim     = output_dim

        # Node type projection
        self.node_proj = HeteroNodeProjection(
            provider_types, raw_feature_dim, latent_dim
        )

        # Stack of graph attention layers
        self.gnn_layers = nn.ModuleList([
            HeteroEdgeAttention(latent_dim, num_heads, dropout=dropout)
            for _ in range(num_gnn_layers)
        ])

        # Output MLP
        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        x:          torch.Tensor,     # (B, P, raw_feature_dim)
        adj:        torch.Tensor,     # (P, P) adjacency
        edge_types: Optional[torch.Tensor] = None,  # (P, P) edge type ids
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, P, raw_feature_dim) raw node feature matrix
            adj:  (P, P) binary coexistence adjacency
            edge_types: optional (P, P) edge type ids

        Returns:
            z: (B, output_dim) graph-level embedding
        """
        # Project to shared latent space (type-aware)
        h = self.node_proj(x)    # (B, P, latent_dim)

        # Propagate messages across provider graph
        for layer in self.gnn_layers:
            h = layer(h, adj, edge_types)  # (B, P, latent_dim)

        # Global mean pool (provider-invariant)
        g = h.mean(dim=1)        # (B, latent_dim)

        # Project to output dimension
        z = self.output_proj(g)  # (B, output_dim)
        return z

    def get_node_embeddings(
        self,
        x:   torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return per-node embeddings (B, P, latent_dim) for interpretability.
        """
        h = self.node_proj(x)
        for layer in self.gnn_layers:
            h = layer(h, adj)
        return h


class HeteroGNNTemporalEncoder(nn.Module):
    """
    Heterogeneous GNN encoder chained with a temporal LTC/CfC encoder.

    For each timestep in the history, runs the GNN to get graph embeddings,
    then feeds the embedding sequence to the temporal encoder.

    This gives the agent both:
      - Spatial awareness: who is interfering with whom right now
      - Temporal awareness: how the interference topology is evolving
    """

    def __init__(
        self,
        provider_types:  List[ProviderType],
        raw_feature_dim: int,
        gnn_latent_dim:  int = 128,
        gnn_output_dim:  int = 256,
        temporal_hidden: int = 128,
        output_dim:      int = 256,
        num_gnn_layers:  int = 2,
        temporal_layers: int = 2,
        temporal_backend: str = "cfc",   # "cfc", "ltc", or "mamba"
        use_cfc:         Optional[bool] = None,  # deprecated, use temporal_backend
    ):
        super().__init__()
        self.gnn = HeteroGNNEncoder(
            provider_types  = provider_types,
            raw_feature_dim = raw_feature_dim,
            latent_dim      = gnn_latent_dim,
            output_dim      = gnn_output_dim,
            num_gnn_layers  = num_gnn_layers,
        )

        # Backward compat: use_cfc → temporal_backend
        if use_cfc is not None:
            temporal_backend = "cfc" if use_cfc else "ltc"

        # Temporal encoder: CfC (default, fastest), Mamba (linear O(L)),
        # or LTC-GPU (most accurate ODE integration)
        if temporal_backend == "cfc":
            from preceptualai.core.ltc_cell_cfc import CfCEncoder
            self.temporal = CfCEncoder(
                input_dim  = gnn_output_dim,
                hidden_dim = temporal_hidden,
                latent_dim = output_dim,
                num_layers = temporal_layers,
            )
        elif temporal_backend == "mamba":
            from preceptualai.core.mamba_encoder import MambaEncoder
            self.temporal = MambaEncoder(
                input_dim     = gnn_output_dim,
                hidden_dim    = temporal_hidden,
                latent_dim    = output_dim,
                num_layers    = temporal_layers,
                bidirectional = False,  # causal for real-time inference
            )
        elif temporal_backend == "ltc":
            from preceptualai.core.ltc_encoder_gpu import LTCEncoderGPU
            self.temporal = LTCEncoderGPU(
                input_dim  = gnn_output_dim,
                hidden_dim = temporal_hidden,
                latent_dim = output_dim,
                num_layers = temporal_layers,
            )
        else:
            raise ValueError(f"Unknown temporal_backend: {temporal_backend!r}")

        self.output_dim = output_dim

    def forward(
        self,
        x_seq:      torch.Tensor,       # (B, T, P, raw_feature_dim)
        adj:        torch.Tensor,       # (P, P)
        edge_types: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_seq: (B, T, P, F) sequence of graph snapshots
            adj:   (P, P) adjacency (assumed static across time)
        Returns:
            z: (B, output_dim) spatio-temporal latent embedding
        """
        B, T, P, F = x_seq.shape

        # Apply GNN to each timestep
        x_flat = x_seq.view(B * T, P, F)
        g_flat = self.gnn(x_flat, adj, edge_types)     # (B*T, gnn_output_dim)
        g_seq  = g_flat.view(B, T, -1)                  # (B, T, gnn_output_dim)

        # Temporal encoding
        z = self.temporal(g_seq)   # (B, output_dim)
        return z
