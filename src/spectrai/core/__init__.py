"""Core neural network components for SpectrAI."""

from spectrai.core.actor import LTCActor
from spectrai.core.critic import LTCCritic
from spectrai.core.ltc_cell import LTCCell
from spectrai.core.ltc_encoder import LTCEncoder
from spectrai.core.replay_buffer import ReplayBuffer

# GPU-optimized variants
from spectrai.core.ltc_cell_gpu import LTCCellGPU, LTCCellGPUFused
from spectrai.core.ltc_cell_multiscale import MultiScaleLTCCell, MultiScaleLTCEncoder
from spectrai.core.ltc_encoder_gpu import LTCEncoderGPU
from spectrai.core.hybrid_actor import HybridActor
from spectrai.core.replay_buffer_gpu import ReplayBufferGPU

# Closed-Form Continuous-Time (solver-free LTC replacement)
from spectrai.core.ltc_cell_cfc import CfCCell, CfCCellExact, CfCEncoder

# Next-gen architectures
from spectrai.core.mamba_encoder import MambaEncoder, SelectiveSSMBlock
from spectrai.core.gnn_encoder import GNNSpatialEncoder, GNNTemporalEncoder
from spectrai.core.kan_actor import KANActor, KANLinear
from spectrai.core.smooth_actor import SmoothHybridActor, SmODENeuron
from spectrai.core.world_model import LTCWorldModel, DynaWorldModelTrainer
from spectrai.core.fno_surrogate import FNOChannelSurrogate
from spectrai.core.diffusion_augment import SpectrumDiffusionModel, DiffusionAugmenter

# Optional: torchdiffeq-based variants
try:
    from spectrai.core.ltc_cell_diffeq import LTCCellDiffeq
    from spectrai.core.ltc_encoder_diffeq import LTCEncoderDiffeq
except ImportError:
    pass

__all__ = [
    # Base
    "LTCCell", "LTCEncoder", "LTCActor", "LTCCritic", "ReplayBuffer",
    # GPU-optimized
    "LTCCellGPU", "LTCCellGPUFused", "LTCEncoderGPU",
    "MultiScaleLTCCell", "MultiScaleLTCEncoder",
    "HybridActor", "ReplayBufferGPU",
    # CfC (solver-free)
    "CfCCell", "CfCCellExact", "CfCEncoder",
    # Next-gen
    "MambaEncoder", "SelectiveSSMBlock",
    "GNNSpatialEncoder", "GNNTemporalEncoder",
    "KANActor", "KANLinear",
    "SmoothHybridActor", "SmODENeuron",
    "LTCWorldModel", "DynaWorldModelTrainer",
    "FNOChannelSurrogate",
    "SpectrumDiffusionModel", "DiffusionAugmenter",
    # Diffeq (optional)
    "LTCCellDiffeq", "LTCEncoderDiffeq",
]
