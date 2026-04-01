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

# Optional: torchdiffeq-based variants (require pip install torchdiffeq)
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
    # Diffeq (optional)
    "LTCCellDiffeq", "LTCEncoderDiffeq",
]
