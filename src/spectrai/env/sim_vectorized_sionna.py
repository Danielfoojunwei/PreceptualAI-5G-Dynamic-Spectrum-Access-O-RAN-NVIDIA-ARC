"""
GPU-vectorized DSA environment with NVIDIA Sionna 3GPP channel physics.

Extends VectorizedDSAEnv by replacing the simple Markov + constant-SNR
model with physics-based 3GPP TDL/CDL channel models via the
SionnaChannelGenerator.

Requires: nvidia-sionna for full 3GPP UMa/UMi/RMa models.
Falls back to PyTorch-native TDL model otherwise.
"""

from typing import Optional

import torch

from spectrai.env.sim_vectorized import VectorizedDSAEnv
from spectrai.env.sionna_channel import SionnaChannelGenerator


class SionnaVectorizedDSAEnv(VectorizedDSAEnv):
    """
    VectorizedDSAEnv with 3GPP channel physics for O-RAN training.

    Channel dynamics are driven by a TDL/CDL model instead of
    simple on/off Markov transitions with constant SNR values.
    """

    def __init__(
        self,
        num_envs: int = 256,
        num_channels: int = 10,
        num_features: int = 3,
        sequence_length: int = 16,
        max_steps: int = 200,
        channel_model: str = "TDL-A",
        carrier_freq_hz: float = 3.5e9,
        delay_spread_ns: float = 300.0,
        max_doppler_hz: float = 50.0,
        snr_occupancy_threshold_db: float = 5.0,
        reward_success: float = 1.0,
        reward_collision: float = -1.0,
        reward_switch: float = -0.1,
        device: torch.device = torch.device("cuda"),
    ):
        # Initialize base env (Markov dynamics still used for PU activity)
        super().__init__(
            num_envs=num_envs,
            num_channels=num_channels,
            num_features=num_features,
            sequence_length=sequence_length,
            max_steps=max_steps,
            reward_success=reward_success,
            reward_collision=reward_collision,
            reward_switch=reward_switch,
            device=device,
        )

        self.snr_threshold = snr_occupancy_threshold_db

        # Initialize 3GPP channel model
        self._channel_gen = SionnaChannelGenerator(
            num_envs=num_envs,
            num_channels=num_channels,
            channel_model=channel_model,
            carrier_freq_hz=carrier_freq_hz,
            delay_spread_ns=delay_spread_ns,
            max_doppler_hz=max_doppler_hz,
            device=device,
        )

    def _build_features(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Override: use 3GPP channel physics instead of constant SNR values.

        Features are [SNR_norm, interference, occupancy] per channel,
        where SNR and interference come from the channel model.
        """
        n = mask.sum().item()

        # Get physics-based channel quality
        snr_norm, interference = self._channel_gen.generate_batch()

        # Use channel states from Markov model for PU occupancy
        occupancy = self.channel_states[mask]

        # Modulate SNR by occupancy (occupied channels have degraded SNR)
        snr_masked = snr_norm[mask] * (1.0 - 0.7 * occupancy)
        intf_masked = interference[mask] + 0.5 * occupancy

        features = torch.stack([snr_masked, intf_masked, occupancy], dim=-1)
        return features.reshape(n, -1)
