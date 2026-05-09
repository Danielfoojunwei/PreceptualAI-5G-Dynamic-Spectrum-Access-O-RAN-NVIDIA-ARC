"""Entity tokeniser — typed per-asset tokens.

For every asset in the resource graph (satellite, gNB cell, WiFi AP, UE,
beam, gateway, edge node), produce ONE token of shape `d_model` via:

    token_i = MLP_type(attr_i) + TypeEmbed(type_i) + PosEnc(xyz_i)

where MLP_type is a per-type small network (so the satellite encoder
sees orbital state, the UE encoder sees SLA + buffer, etc.). Type
embeddings live in a learned table keyed off the `EntityType` enum;
position encoding is a fixed Fourier feature over normalized ECEF.

This module is the input gate for the JEPA / Perceiver IO encoder
specced in SYNTHESIS.md. Output shape is (N_entities_t, d_model).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn

from horizon_ric._scaffold import TrainedMarkerMixin, warn_untrained


class EntityType(str, Enum):
    SAT_NGSO = "sat_ngso"
    SAT_GSO = "sat_gso"
    CELL = "cell"
    AP_WIFI = "ap_wifi"
    UE = "ue"
    BEAM = "beam"
    GATEWAY = "gateway"
    EDGE_NODE = "edge_node"


@dataclass
class EntityTokenizerConfig:
    """Per-type attribute dimensionality + shared tokeniser config."""

    d_model: int = 128
    pos_freqs: int = 8
    """Number of Fourier-feature bands per spatial axis."""
    attr_dims: dict[EntityType, int] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.attr_dims is None:
            # Defaults — must match the upstream feature producers:
            #   SAT_NGSO / SAT_GSO  → encoder/link_state.LINK_STATE_DIM (12)
            #   the rest are reasonable scratch defaults pending real
            #   per-type feature producers landing in Phase-2.
            #
            # Bug fix (v3 honest-audit pass): SAT_NGSO=8 mismatched
            # LINK_STATE_DIM=12 and the documented "wire link_state
            # into the satellite token" path crashed at runtime. Now
            # both satellite types match the LinkState width.
            from horizon_ric.encoder.link_state import LINK_STATE_DIM
            self.attr_dims = {
                EntityType.SAT_NGSO: LINK_STATE_DIM,   # was 8 — broken
                EntityType.SAT_GSO: LINK_STATE_DIM,    # was 6 — broken
                EntityType.CELL: 12,
                EntityType.AP_WIFI: 8,
                EntityType.UE: 16,
                EntityType.BEAM: 10,
                EntityType.GATEWAY: 6,
                EntityType.EDGE_NODE: 8,
            }


@dataclass
class AssetAttributes:
    """One asset's input bundle for the tokeniser."""

    entity_type: EntityType
    attributes: torch.Tensor   # (attr_dim,)
    xyz_m: tuple[float, float, float]
    """ECEF position; the tokeniser normalises internally."""

    def __post_init__(self):
        if self.attributes.ndim != 1:
            raise ValueError(
                f"attributes must be 1-D (attr_dim,), got {tuple(self.attributes.shape)}"
            )


class _PositionEncoder(nn.Module):
    """Fourier-feature encoding over normalized ECEF.

    Each axis gets `n_freqs` log-spaced sine/cosine pairs, giving a
    `6 · n_freqs` output. We normalise ECEF by the GSO orbital radius so
    the input range is bounded ~[−1, 1] for satellites and much smaller
    for ground assets.
    """

    def __init__(self, n_freqs: int):
        super().__init__()
        self.n_freqs = n_freqs
        # Log-spaced frequencies (ICML 2020 NeRF convention).
        freqs = 2.0 ** torch.arange(n_freqs).float()
        self.register_buffer("freqs", freqs)

    @property
    def out_dim(self) -> int:
        return 6 * self.n_freqs

    def forward(self, xyz_m: torch.Tensor) -> torch.Tensor:
        """xyz_m: (B, 3) → (B, 6 · n_freqs)."""
        # Normalise: GSO-radius scale.
        x = xyz_m / 42_164_000.0
        out = []
        for f in self.freqs:
            out.append(torch.sin(x * f))
            out.append(torch.cos(x * f))
        return torch.cat(out, dim=-1)


class EntityTokenizer(nn.Module, TrainedMarkerMixin):
    """Typed tokeniser producing one (d_model,) token per asset.

    .. warning::

        Bare instantiation produces **random projections** (Kaiming-init
        ``nn.Linear`` + NeRF-style positional encoding). Production
        deployments must load a checkpoint produced by
        ``scripts/train_jepa_full.py`` via
        ``EntityTokenizer.from_pretrained(path, config=...)`` —
        otherwise the typed-token embeddings are random and the
        downstream encoder is meaningless. ``UntrainedScaffoldWarning``
        fires at construction.
    """

    def __init__(self, config: EntityTokenizerConfig | None = None):
        super().__init__()
        self.cfg = config or EntityTokenizerConfig()
        self.pos_enc = _PositionEncoder(self.cfg.pos_freqs)
        warn_untrained("EntityTokenizer", "checkpoints/jepa_encoder_v0.1.pt")

        # Per-type attribute MLP.
        self.attr_mlps = nn.ModuleDict(
            {
                t.value: nn.Sequential(
                    nn.Linear(dim, self.cfg.d_model),
                    nn.SiLU(),
                    nn.Linear(self.cfg.d_model, self.cfg.d_model),
                )
                for t, dim in self.cfg.attr_dims.items()
            }
        )

        self.type_embed = nn.Embedding(len(EntityType), self.cfg.d_model)
        self._type_to_idx = {t: i for i, t in enumerate(EntityType)}

        self.pos_proj = nn.Linear(self.pos_enc.out_dim, self.cfg.d_model)

    def forward(self, assets: list[AssetAttributes]) -> torch.Tensor:
        """Tokenize a list of assets — heterogeneous types allowed.

        Returns:
            (N, d_model) tensor in the order assets are passed.
        """
        if not assets:
            return torch.zeros(0, self.cfg.d_model)

        device = next(self.parameters()).device
        tokens: list[torch.Tensor] = []
        for asset in assets:
            t_name = asset.entity_type.value
            mlp = self.attr_mlps[t_name]
            expected_dim = self.cfg.attr_dims[asset.entity_type]
            if asset.attributes.shape[0] != expected_dim:
                raise ValueError(
                    f"asset {asset.entity_type} expects attr_dim={expected_dim}, "
                    f"got {asset.attributes.shape[0]}"
                )
            attr_token = mlp(asset.attributes.to(device).unsqueeze(0)).squeeze(0)
            type_token = self.type_embed.weight[self._type_to_idx[asset.entity_type]]
            xyz = torch.tensor(asset.xyz_m, dtype=torch.float32, device=device).unsqueeze(0)
            pos_token = self.pos_proj(self.pos_enc(xyz)).squeeze(0)
            tokens.append(attr_token + type_token + pos_token)
        return torch.stack(tokens, dim=0)


__all__ = [
    "AssetAttributes",
    "EntityTokenizer",
    "EntityTokenizerConfig",
    "EntityType",
]
