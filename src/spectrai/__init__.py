"""
SpectrAI — AI-Native Dynamic Spectrum Management for O-RAN.

Production O-RAN xApp powered by Liquid Time-Constant (LTC) neural networks
for real-time dynamic spectrum access on NVIDIA ARC hardware.
"""

__version__ = "0.1.0"

from spectrai.core.ltc_cell import LTCCell
from spectrai.core.ltc_encoder import LTCEncoder
from spectrai.core.actor import LTCActor
from spectrai.core.critic import LTCCritic
from spectrai.core.replay_buffer import ReplayBuffer
from spectrai.agent.sac_ltc import SACLTCAgent

__all__ = [
    "LTCCell",
    "LTCEncoder",
    "LTCActor",
    "LTCCritic",
    "ReplayBuffer",
    "SACLTCAgent",
]
