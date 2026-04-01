"""SpectrAI spectrum environments — simulation, O-RAN, NVIDIA ARC, and GPU-vectorized."""

from spectrai.env.base import SpectrumEnv, SpectrumEnvConfig
from spectrai.env.sim import SimulatedDSAConfig, SimulatedDSAEnv
from spectrai.env.sim_vectorized import VectorizedDSAEnv
from spectrai.env.itu_propagation import ITUPropagation
from spectrai.env.sionna_channel import SionnaChannelGenerator, TDLChannelPyTorch
from spectrai.env.sim_vectorized_sionna import SionnaVectorizedDSAEnv
from spectrai.env.data_pipeline import Unified5GDataPipeline

# Optional: NVIDIA Warp JIT environment (requires pip install warp-lang)
try:
    from spectrai.env.sim_vectorized_warp import WarpVectorizedDSAEnv
except ImportError:
    pass

__all__ = [
    "SpectrumEnv",
    "SpectrumEnvConfig",
    "SimulatedDSAEnv",
    "SimulatedDSAConfig",
    # GPU-vectorized
    "VectorizedDSAEnv",
    "SionnaVectorizedDSAEnv",
    "WarpVectorizedDSAEnv",
    # Physics
    "ITUPropagation",
    "SionnaChannelGenerator",
    "TDLChannelPyTorch",
    # Data
    "Unified5GDataPipeline",
]
