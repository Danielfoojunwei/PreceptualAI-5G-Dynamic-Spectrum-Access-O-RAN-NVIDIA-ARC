"""
PreceptualAI spectrum environments — simulation, O-RAN, NVIDIA ARC, ISAC,
GPU-vectorized, satellite NTN, and Universal Heterogeneous Connectivity.
"""

from preceptualai.env.base import SpectrumEnv, SpectrumEnvConfig
from preceptualai.env.sim import SimulatedDSAConfig, SimulatedDSAEnv
from preceptualai.env.sim_vectorized import VectorizedDSAEnv
from preceptualai.env.itu_propagation import ITUPropagation
from preceptualai.env.sionna_channel import SionnaChannelGenerator, TDLChannelPyTorch
from preceptualai.env.sim_vectorized_sionna import SionnaVectorizedDSAEnv
from preceptualai.env.data_pipeline import Unified5GDataPipeline
from preceptualai.env.isac_env import ISACVectorizedDSAEnv
from preceptualai.env.fr3_propagation import FR3PropagationModel, NTNInterferenceModel
from preceptualai.env.satellite import SatelliteSpectrumEnv
from preceptualai.env.orbital_dynamics import OrbitalDynamics
from preceptualai.env.provider_registry import (
    ProviderType, ProviderPhysics, PROVIDER_REGISTRY,
    coexistence_adjacency, provider_node_features,
    NTN_TYPES, TR_TYPES, PROVIDER_FEATURE_DIM,
)
from preceptualai.env.unified_connectivity_env import (
    UnifiedConnectivityEnv, UnifiedConnectivityConfig,
)

# Optional: NVIDIA Warp JIT environment
try:
    from preceptualai.env.sim_vectorized_warp import WarpVectorizedDSAEnv
except ImportError:
    pass

__all__ = [
    "SpectrumEnv", "SpectrumEnvConfig",
    "SimulatedDSAEnv", "SimulatedDSAConfig",
    # GPU-vectorized
    "VectorizedDSAEnv", "SionnaVectorizedDSAEnv", "WarpVectorizedDSAEnv",
    # ISAC 6G
    "ISACVectorizedDSAEnv",
    # Physics
    "ITUPropagation", "FR3PropagationModel", "NTNInterferenceModel",
    "SionnaChannelGenerator", "TDLChannelPyTorch",
    # Satellite NTN
    "SatelliteSpectrumEnv", "OrbitalDynamics",
    # Data
    "Unified5GDataPipeline",
    # UHCI — Universal Heterogeneous Connectivity Intelligence
    "ProviderType", "ProviderPhysics", "PROVIDER_REGISTRY",
    "coexistence_adjacency", "provider_node_features",
    "NTN_TYPES", "TR_TYPES", "PROVIDER_FEATURE_DIM",
    "UnifiedConnectivityEnv", "UnifiedConnectivityConfig",
]
