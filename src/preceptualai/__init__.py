"""
PreceptualAI — Universal Heterogeneous Connectivity Intelligence (UHCI).

A unified AI-native spectrum management system for ALL connectivity providers:
  NTN: LEO · MEO · GEO · HAPS satellites
  TR:  5G FR1 · 5G/6G FR3 · 6G ISAC · Wi-Fi 7

Powered by:
  - Liquid Time-Constant (LTC) neural ODEs for temporal dynamics
  - Closed-form CfC cells for sub-millisecond inference
  - Heterogeneous GNN encoder for multi-provider interference graphs
  - KAN actor for interpretable, regulatory-compliant policies
  - Soft Actor-Critic (SAC) with GPU-resident replay buffer
  - LTC-aware Federated Learning across operator boundaries
"""

__version__ = "0.2.0"

from preceptualai.agent.sac_ltc import SACLTCAgent
from preceptualai.core.actor import LTCActor
from preceptualai.core.critic import LTCCritic
from preceptualai.core.ltc_cell import LTCCell
from preceptualai.core.ltc_encoder import LTCEncoder
from preceptualai.core.replay_buffer import ReplayBuffer

# UHCI — Universal Heterogeneous Connectivity Intelligence
from preceptualai.core.hetero_gnn_encoder import (
    HeteroGNNEncoder,
    HeteroGNNTemporalEncoder,
    HeteroNodeProjection,
)
from preceptualai.core.universal_spectrum_agent import (
    UniversalSpectrumAgent,
    UniversalAgentConfig,
    UniversalCritic,
)
from preceptualai.env.provider_registry import (
    ProviderType,
    ProviderPhysics,
    PROVIDER_REGISTRY,
    coexistence_adjacency,
    provider_node_features,
    NTN_TYPES,
    TR_TYPES,
)
from preceptualai.env.unified_connectivity_env import (
    UnifiedConnectivityEnv,
    UnifiedConnectivityConfig,
)

__all__ = [
    # Legacy / base
    "LTCCell", "LTCEncoder", "LTCActor", "LTCCritic",
    "ReplayBuffer", "SACLTCAgent",
    # UHCI core
    "HeteroGNNEncoder", "HeteroGNNTemporalEncoder", "HeteroNodeProjection",
    "UniversalSpectrumAgent", "UniversalAgentConfig", "UniversalCritic",
    # Provider registry
    "ProviderType", "ProviderPhysics", "PROVIDER_REGISTRY",
    "coexistence_adjacency", "provider_node_features",
    "NTN_TYPES", "TR_TYPES",
    # UHCI environment
    "UnifiedConnectivityEnv", "UnifiedConnectivityConfig",
]
