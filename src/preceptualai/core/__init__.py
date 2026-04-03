"""Core neural network components for PreceptualAI."""

from preceptualai.core.actor import LTCActor
from preceptualai.core.critic import LTCCritic
from preceptualai.core.ltc_cell import LTCCell
from preceptualai.core.ltc_encoder import LTCEncoder
from preceptualai.core.replay_buffer import ReplayBuffer

# GPU-optimized variants
from preceptualai.core.ltc_cell_gpu import LTCCellGPU, LTCCellGPUFused
from preceptualai.core.ltc_cell_multiscale import MultiScaleLTCCell, MultiScaleLTCEncoder
from preceptualai.core.ltc_encoder_gpu import LTCEncoderGPU
from preceptualai.core.hybrid_actor import HybridActor
from preceptualai.core.replay_buffer_gpu import ReplayBufferGPU

# Closed-Form Continuous-Time (solver-free LTC replacement)
from preceptualai.core.ltc_cell_cfc import CfCCell, CfCCellExact, CfCEncoder

# Next-gen architectures
from preceptualai.core.mamba_encoder import MambaEncoder, SelectiveSSMBlock
from preceptualai.core.gnn_encoder import GNNSpatialEncoder, GNNTemporalEncoder
from preceptualai.core.kan_actor import KANActor, KANLinear
from preceptualai.core.smooth_actor import SmoothHybridActor, SmODENeuron
from preceptualai.core.world_model import LTCWorldModel, DynaWorldModelTrainer
from preceptualai.core.fno_surrogate import FNOChannelSurrogate
from preceptualai.core.diffusion_augment import SpectrumDiffusionModel, DiffusionAugmenter

# Optional: torchdiffeq-based variants
try:
    from preceptualai.core.ltc_cell_diffeq import LTCCellDiffeq
    from preceptualai.core.ltc_encoder_diffeq import LTCEncoderDiffeq
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
