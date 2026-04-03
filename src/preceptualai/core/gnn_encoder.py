"""
Graph Neural Network Encoder for Topology-Aware DSA.

Processes spatial interference relationships between cells/beams
using message-passing graph attention, then feeds per-node embeddings
into the LTC temporal encoder.

Architecture:
  GNN (spatial) -> LTC/CfC/Mamba (temporal) -> Policy head

This factorizes the spatial-temporal DSA problem cleanly:
  - GNN captures multi-hop interference topology
  - Temporal encoder captures channel dynamics per node

Reference:
  HR-GAT: "Spectrum Demand Estimation Using GNNs", March 2026.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Single Graph Attention layer (GAT).

    Computes attention-weighted message passing across graph edges.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.head_dim = out_features // num_heads if concat else out_features

        # Per-head linear projections
        self.W = nn.Linear(in_features, self.head_dim * num_heads, bias=False)

        # Attention mechanism: a^T [Wh_i || Wh_j]
        self.a_src = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        if concat:
            self.out_dim = self.head_dim * num_heads
        else:
            self.out_dim = out_features

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:   (B, N, in_features) node features
            adj: (B, N, N) or (N, N) adjacency matrix (1 = connected)
        Returns:
            h:   (B, N, out_dim) updated node features
        """
        B, N, _ = x.shape

        # Linear projection
        Wh = self.W(x)  # (B, N, head_dim * num_heads)
        Wh = Wh.view(B, N, self.num_heads, self.head_dim)  # (B, N, H, D)

        # Attention scores
        e_src = (Wh * self.a_src.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # (B, N, H)
        e_dst = (Wh * self.a_dst.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # (B, N, H)

        # Pairwise attention: e_ij = LeakyReLU(e_src_i + e_dst_j)
        e = e_src.unsqueeze(2) + e_dst.unsqueeze(1)  # (B, N, N, H)
        e = self.leaky_relu(e)

        # Mask with adjacency
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(B, -1, -1)
        mask = adj.unsqueeze(-1).expand_as(e)  # (B, N, N, H)
        e = e.masked_fill(mask == 0, float('-inf'))

        # Softmax attention
        alpha = F.softmax(e, dim=2)  # (B, N, N, H)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = self.dropout(alpha)

        # Aggregate: h_i = sum_j alpha_ij * Wh_j
        # alpha: (B, N, N, H), Wh: (B, N, H, D)
        h = torch.einsum('bnjh,bjhd->bnhd', alpha, Wh)  # (B, N, H, D)

        if self.concat:
            h = h.reshape(B, N, self.num_heads * self.head_dim)
        else:
            h = h.mean(dim=2)  # (B, N, D)

        return h


class GNNSpatialEncoder(nn.Module):
    """
    Multi-layer Graph Attention Network for spatial interference modeling.

    Takes per-cell observation features and an interference adjacency matrix,
    outputs spatially-aware per-cell embeddings.

    Input:  (B, N, F) node features + (N, N) adjacency
    Output: (B, N, embed_dim) spatially-enriched features
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else embed_dim
            layer = GraphAttentionLayer(in_dim, embed_dim, num_heads, dropout, concat=True)
            self.layers.append(layer)
            self.norms.append(nn.LayerNorm(layer.out_dim))
            if i == 0 and input_dim != layer.out_dim:
                self.input_proj = nn.Linear(input_dim, layer.out_dim)
            elif i == 0:
                self.input_proj = nn.Identity()

        self.output_proj = nn.Linear(self.layers[-1].out_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:   (B, N, input_dim) per-node features
            adj: (N, N) or (B, N, N) adjacency matrix
        Returns:
            h:   (B, N, embed_dim) spatially-enriched embeddings
        """
        h = x
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            h_new = layer(h, adj)
            h_new = norm(h_new)
            h_new = F.elu(h_new)
            # Residual connection (after first layer)
            if i > 0 and h.shape[-1] == h_new.shape[-1]:
                h_new = h_new + h
            h = h_new

        return self.output_proj(h)

    @staticmethod
    def build_interference_adj(
        num_cells: int,
        interference_radius: int = 2,
        self_loops: bool = True,
    ) -> torch.Tensor:
        """
        Build a simple interference adjacency matrix.

        Cells within `interference_radius` hops are connected.
        For real deployments, this should be replaced with actual
        interference measurements or propagation model outputs.

        Returns:
            adj: (num_cells, num_cells) float tensor
        """
        adj = torch.zeros(num_cells, num_cells)
        for i in range(num_cells):
            for j in range(num_cells):
                if abs(i - j) <= interference_radius:
                    adj[i, j] = 1.0
            if self_loops:
                adj[i, i] = 1.0
        return adj


class GNNTemporalEncoder(nn.Module):
    """
    Combined GNN (spatial) + Temporal encoder for topology-aware DSA.

    Architecture:
      1. GNN processes per-cell features at each timestep
      2. Temporal encoder processes GNN output sequence per cell
      3. Pool across cells for global decision

    Input:  (B, T, N, F) observations across T timesteps, N cells, F features
    Output: (B, latent_dim) global spectrum state embedding
    """

    def __init__(
        self,
        input_dim: int,
        num_cells: int,
        gnn_embed_dim: int = 64,
        gnn_layers: int = 2,
        temporal_hidden_dim: int = 128,
        latent_dim: int = 128,
        temporal_layers: int = 2,
    ):
        super().__init__()
        self.num_cells = num_cells
        self.latent_dim = latent_dim

        self.gnn = GNNSpatialEncoder(input_dim, gnn_embed_dim, gnn_layers)

        # Temporal processing per cell (shared weights)
        self.temporal_rnn = nn.GRU(
            gnn_embed_dim, temporal_hidden_dim,
            num_layers=temporal_layers, batch_first=True,
        )

        # Pool + project
        self.pool_proj = nn.Linear(temporal_hidden_dim * num_cells, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:   (B, T, N, F)
            adj: (N, N)
        Returns:
            z:   (B, latent_dim)
        """
        B, T, N, F = x.shape

        # Process each timestep through GNN
        gnn_outs = []
        for t in range(T):
            h_t = self.gnn(x[:, t], adj)  # (B, N, gnn_embed_dim)
            gnn_outs.append(h_t)
        gnn_seq = torch.stack(gnn_outs, dim=1)  # (B, T, N, gnn_embed_dim)

        # Process temporal sequence per cell
        cell_embeds = []
        for n in range(N):
            cell_seq = gnn_seq[:, :, n, :]  # (B, T, gnn_embed_dim)
            _, h_n = self.temporal_rnn(cell_seq)  # h_n: (layers, B, hidden)
            cell_embeds.append(h_n[-1])  # (B, hidden)

        # Concatenate all cells and project
        all_cells = torch.cat(cell_embeds, dim=-1)  # (B, N * hidden)
        z = self.pool_proj(all_cells)
        return z
