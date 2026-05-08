"""Resource-State encoder (PRD §4.2.2, SYNTHESIS.md Tier C).

What ships today (v0.2 Tier-1):

    spatial_prior        Pure-geometry tensor builder. Zero ML; consumes
                         WGS-84 + ephemeris and emits attention biases
                         (LOS mask, slant range, Doppler) for downstream
                         attention layers. Fei-Fei-Li-style spatial prior
                         that gives geometry to the model rather than
                         making it learn it.
    entity_tokenizer     Typed-asset tokeniser: each asset (sat / cell /
                         UE / beam / gateway) becomes one token via a
                         small per-type MLP over its attribute vector
                         concatenated with a learned type embedding.

Roadmapped next (Tier C in SYNTHESIS.md):

    multi_rate_fusion    SeFT triplets + CRU + Liquid-S4 (Hasani 2023).
    hetero_graph_jepa    HGT + I-JEPA masked latent prediction.
    perceiver_fusion     Perceiver IO cross-attention into a fixed latent.
    kpi_chronos          Time-series foundation encoder for KPI streams.

Until those land, downstream world-model code can already consume the
output of `spatial_prior` (numpy arrays) and `entity_tokenizer`
(torch.Tensor); both are stable interfaces and are tested.
"""

from horizon_ric.encoder.entity_tokenizer import (
    AssetAttributes,
    EntityTokenizer,
    EntityTokenizerConfig,
    EntityType,
)
from horizon_ric.encoder.graph_jepa import GraphJEPA, GraphJEPAConfig
from horizon_ric.encoder.link_state import (
    LINK_STATE_DIM,
    LinkStateInputs,
    compose_link_state,
)
from horizon_ric.encoder.perceiver_fusion import PerceiverConfig, PerceiverFusion
from horizon_ric.encoder.spatial_prior import SpatialPrior, SpatialPriorTensor

__all__ = [
    "AssetAttributes",
    "EntityTokenizer",
    "EntityTokenizerConfig",
    "EntityType",
    "GraphJEPA",
    "GraphJEPAConfig",
    "LINK_STATE_DIM",
    "LinkStateInputs",
    "PerceiverConfig",
    "PerceiverFusion",
    "SpatialPrior",
    "SpatialPriorTensor",
    "compose_link_state",
]
