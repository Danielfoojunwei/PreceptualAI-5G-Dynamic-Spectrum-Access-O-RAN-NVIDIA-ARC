"""
ISAC-Aware Dynamic Spectrum Access Environment.

Extends VectorizedDSAEnv with Integrated Sensing and Communication (ISAC)
for 6G readiness. The agent jointly allocates time/frequency resources
between radar sensing and communication tasks.

The action space becomes: (channel_selection, sensing_fraction)
  - channel_selection: which frequency channel to use
  - sensing_fraction: what fraction of resources to dedicate to sensing

Reward combines communication throughput + sensing quality (detection
probability, range resolution).

Reference:
  3GPP Release 19 ISAC Channel Models Study Item.
  ETSI ISC-001: ISAC Use Cases, 2025.
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from spectrai.env.sim_vectorized import VectorizedDSAEnv


class ISACVectorizedDSAEnv(VectorizedDSAEnv):
    """
    ISAC-aware vectorized DSA environment for 6G.

    Extends VectorizedDSAEnv with:
      - Sensing targets (moving objects with radar cross-section)
      - Dual-function waveform resource allocation
      - Joint communication + sensing reward
    """

    def __init__(
        self,
        num_envs: int = 256,
        num_channels: int = 10,
        num_features: int = 5,  # Extended: [snr, intf, occ, sensing_snr, target_present]
        sequence_length: int = 16,
        max_steps: int = 200,
        num_sensing_targets: int = 3,
        radar_cross_section_dbsm: float = 10.0,
        sensing_bandwidth_hz: float = 100e6,
        comm_weight: float = 0.6,
        sensing_weight: float = 0.4,
        device: torch.device = torch.device("cuda"),
        **kwargs,
    ):
        super().__init__(
            num_envs=num_envs,
            num_channels=num_channels,
            num_features=num_features,
            sequence_length=sequence_length,
            max_steps=max_steps,
            device=device,
            **kwargs,
        )

        self.num_targets = num_sensing_targets
        self.rcs_dbsm = radar_cross_section_dbsm
        self.sensing_bw = sensing_bandwidth_hz
        self.comm_weight = comm_weight
        self.sensing_weight = sensing_weight

        # Sensing target states: (num_envs, num_targets, 3) = [range_m, velocity_mps, rcs_dbsm]
        self.target_states = torch.zeros(num_envs, num_sensing_targets, 3, device=device)

        # Per-channel sensing quality
        self.sensing_snr = torch.zeros(num_envs, num_channels, device=device)

        # Sensing allocation fraction per env
        self.sensing_fraction = torch.full((num_envs,), 0.3, device=device)

    def reset(self, mask: torch.Tensor = None) -> torch.Tensor:
        """Reset with random sensing targets."""
        obs = super().reset(mask)

        if mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        n = mask.sum().item()
        if n > 0:
            # Initialize random targets
            self.target_states[mask, :, 0] = torch.rand(n, self.num_targets, device=self.device) * 1000 + 50  # range: 50-1050m
            self.target_states[mask, :, 1] = torch.randn(n, self.num_targets, device=self.device) * 30  # velocity: ~30 m/s
            self.target_states[mask, :, 2] = self.rcs_dbsm + torch.randn(n, self.num_targets, device=self.device) * 3

        return obs

    def step(self, actions: torch.Tensor, sensing_fractions: torch.Tensor = None):
        """
        ISAC-aware step with joint communication and sensing.

        Args:
            actions: (num_envs,) channel selections
            sensing_fractions: (num_envs,) fraction [0,1] for sensing (optional)
        Returns:
            obs, rewards, dones, infos (with sensing metrics)
        """
        # Update sensing allocation
        if sensing_fractions is not None:
            self.sensing_fraction = sensing_fractions.clamp(0.0, 1.0)

        # Move targets (simple linear motion model)
        dt = 1e-3  # 1ms per step
        self.target_states[:, :, 0] += self.target_states[:, :, 1] * dt
        # Wrap range to [10, 2000]
        self.target_states[:, :, 0] = self.target_states[:, :, 0].clamp(10.0, 2000.0)

        # Compute sensing SNR per channel
        self._compute_sensing_snr()

        # Standard DSA step
        obs, comm_rewards, dones, infos = super().step(actions)

        # Sensing reward: detection probability based on sensing SNR
        sensing_snr_selected = self.sensing_snr.gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        # Detection probability (Swerling-1 model approximation)
        detection_prob = torch.sigmoid(
            (sensing_snr_selected * self.sensing_fraction - 5.0) * 0.5
        )

        # Combined ISAC reward
        sensing_reward = detection_prob * 2.0 - 1.0  # Map to [-1, 1]
        rewards = (
            self.comm_weight * comm_rewards
            + self.sensing_weight * sensing_reward
        )

        # Extended infos
        infos["detection_probability"] = detection_prob
        infos["sensing_snr_db"] = sensing_snr_selected
        infos["sensing_fraction"] = self.sensing_fraction
        infos["comm_reward"] = comm_rewards
        infos["sensing_reward"] = sensing_reward

        return obs, rewards, dones, infos

    def _compute_sensing_snr(self):
        """Compute radar sensing SNR per channel using radar equation."""
        # Simplified radar equation: SNR_radar = Pt * G^2 * lambda^2 * RCS / ((4pi)^3 * R^4 * N0)
        # Use per-channel frequency to compute wavelength
        for ch in range(self.num_channels):
            # Average target contribution
            ranges = self.target_states[:, :, 0]  # (E, T)
            rcs = self.target_states[:, :, 2]  # (E, T) in dBsm

            rcs_linear = 10.0 ** (rcs / 10.0)
            r4 = (ranges ** 4).clamp(min=1.0)
            snr_per_target = rcs_linear / r4 * 1e10  # arbitrary scaling
            snr_total = snr_per_target.sum(dim=1)  # (E,)
            self.sensing_snr[:, ch] = 10.0 * torch.log10(snr_total.clamp(min=1e-10))

    def _build_features(self, mask: torch.Tensor) -> torch.Tensor:
        """Extended features including sensing information."""
        n = mask.sum().item()
        occupancy = self.channel_states[mask]

        # Standard DSA features
        snr = torch.where(
            occupancy < 0.5,
            torch.tensor(self.snr_free, device=self.device),
            torch.tensor(self.snr_occupied, device=self.device),
        )
        snr = snr + torch.randn_like(snr) * self.noise_std

        interference = torch.where(
            occupancy > 0.5,
            torch.tensor(self.interference_occupied, device=self.device),
            torch.tensor(self.interference_free, device=self.device),
        )
        interference = interference + torch.randn_like(interference) * self.noise_std

        # Sensing features
        sensing_snr_norm = torch.clamp(
            (self.sensing_snr[mask] + 10.0) / 40.0, 0.0, 1.0
        )
        target_present = (self.sensing_snr[mask] > 0.0).float()

        # Stack all features
        features = torch.stack(
            [snr, interference, occupancy, sensing_snr_norm, target_present],
            dim=-1,
        )
        return features.reshape(n, -1)
