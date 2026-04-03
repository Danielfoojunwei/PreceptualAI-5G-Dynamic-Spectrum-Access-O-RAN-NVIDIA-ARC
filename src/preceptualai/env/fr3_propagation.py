"""
FR3 Band (7-24 GHz) and NTN Propagation Models for 6G DSA.

Extends ITU-R propagation with:
  - FR3 band-specific path loss models (3GPP TR 38.901)
  - NTN satellite-terrestrial interference models
  - Dynamic satellite pass scheduling
  - Doppler compensation features

FR3 is the critical 6G spectrum band balancing coverage and capacity.

Reference:
  3GPP TR 38.901: Channel model for frequencies from 0.5 to 100 GHz.
  3GPP Release 19: NTN enhancements.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch


class FR3PropagationModel:
    """
    FR3 (7-24 GHz) propagation model for 6G DSA.

    Implements 3GPP TR 38.901 UMa/UMi path loss models adapted
    for the FR3 band with frequency-dependent penetration loss.
    """

    def __init__(self, device: torch.device = torch.device("cuda")):
        self.device = device

    def path_loss_uma_los(
        self,
        freq_ghz: torch.Tensor,
        distance_m: torch.Tensor,
        bs_height_m: float = 25.0,
        ue_height_m: float = 1.5,
    ) -> torch.Tensor:
        """
        3GPP UMa LOS path loss for FR3 band.

        PL = 28.0 + 22*log10(d_3D) + 20*log10(fc)

        Args:
            freq_ghz:    (...) carrier frequency in GHz [7-24]
            distance_m:  (...) 3D distance in meters
        Returns:
            PL: (...) path loss in dB
        """
        d = distance_m.clamp(min=10.0)
        fc = freq_ghz.clamp(min=0.5)

        # UMa LOS (3GPP TR 38.901 Table 7.4.1-1)
        d_bp = 4 * bs_height_m * ue_height_m * fc * 1e9 / 3e8  # breakpoint distance
        pl1 = 28.0 + 22.0 * torch.log10(d) + 20.0 * torch.log10(fc)
        pl2 = 28.0 + 40.0 * torch.log10(d) + 20.0 * torch.log10(fc) - 9.0 * torch.log10(d_bp ** 2 + (bs_height_m - ue_height_m) ** 2)

        pl = torch.where(d <= d_bp, pl1, pl2)
        return pl

    def path_loss_uma_nlos(
        self,
        freq_ghz: torch.Tensor,
        distance_m: torch.Tensor,
    ) -> torch.Tensor:
        """
        3GPP UMa NLOS path loss for FR3 band.

        PL = 13.54 + 39.08*log10(d_3D) + 20*log10(fc) - 0.6*(h_UT - 1.5)
        """
        d = distance_m.clamp(min=10.0)
        fc = freq_ghz.clamp(min=0.5)

        pl_nlos = 13.54 + 39.08 * torch.log10(d) + 20.0 * torch.log10(fc)
        pl_los = self.path_loss_uma_los(freq_ghz, distance_m)

        return torch.max(pl_los, pl_nlos)

    def indoor_penetration_loss(
        self,
        freq_ghz: torch.Tensor,
        building_type: str = "thermally_efficient",
    ) -> torch.Tensor:
        """
        Building penetration loss for FR3 band.

        FR3 experiences significant penetration loss (10-30 dB).
        """
        fc = freq_ghz
        if building_type == "thermally_efficient":
            # Modern energy-efficient buildings
            loss = 5.0 + 14.0 * torch.log10(fc) + 0.3 * fc
        elif building_type == "traditional":
            loss = 5.0 + 10.0 * torch.log10(fc)
        else:
            loss = 5.0 + 12.0 * torch.log10(fc)

        return loss.clamp(min=0.0, max=50.0)

    def fr3_channel_capacity(
        self,
        freq_ghz: torch.Tensor,
        distance_m: torch.Tensor,
        bandwidth_mhz: float = 400.0,
        tx_power_dbm: float = 30.0,
        noise_figure_db: float = 7.0,
        los: bool = True,
    ) -> torch.Tensor:
        """
        Estimate channel capacity for FR3 deployment.

        Returns capacity in Mbps.
        """
        if los:
            pl = self.path_loss_uma_los(freq_ghz, distance_m)
        else:
            pl = self.path_loss_uma_nlos(freq_ghz, distance_m)

        # Thermal noise
        noise_dbm = -174.0 + 10.0 * math.log10(bandwidth_mhz * 1e6) + noise_figure_db

        # SNR
        snr_db = tx_power_dbm - pl - noise_dbm
        snr_linear = 10.0 ** (snr_db / 10.0)

        # Shannon capacity
        capacity_bps = bandwidth_mhz * 1e6 * torch.log2(1.0 + snr_linear)
        return capacity_bps / 1e6  # Mbps


class NTNInterferenceModel:
    """
    Non-Terrestrial Network (NTN) interference model for spectrum sharing.

    Models dynamic satellite-terrestrial interference based on
    satellite pass geometry and frequency coordination.
    """

    def __init__(
        self,
        num_envs: int,
        num_channels: int,
        device: torch.device = torch.device("cuda"),
        satellite_altitude_km: float = 550.0,
        min_elevation_deg: float = 10.0,
    ):
        self.num_envs = num_envs
        self.num_channels = num_channels
        self.device = device
        self.sat_alt = satellite_altitude_km
        self.min_elev = min_elevation_deg

        # Satellite state per environment
        self.sat_elevation = torch.zeros(num_envs, device=device)
        self.sat_visible = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.pass_time_remaining_s = torch.zeros(num_envs, device=device)

        # Which channels are shared with NTN (configurable)
        self.ntn_channels = torch.zeros(num_channels, dtype=torch.bool, device=device)
        self.ntn_channels[:3] = True  # First 3 channels shared

        self._step_count = 0

    def step(self, dt_s: float = 1.0):
        """Advance satellite pass simulation."""
        self._step_count += 1

        # Simple satellite pass model: elevation follows sine curve
        period = 600.0  # 10-minute pass
        phase = (self._step_count * dt_s / period) * 2.0 * math.pi
        max_elev = 45.0 + torch.randn(self.num_envs, device=self.device) * 10.0

        self.sat_elevation = max_elev * torch.sin(torch.tensor(phase, device=self.device))
        self.sat_visible = self.sat_elevation > self.min_elev
        remaining = max(0.0, period / 2 - (self._step_count * dt_s % period))
        self.pass_time_remaining_s = torch.where(
            self.sat_visible,
            torch.tensor(remaining, device=self.device).expand_as(self.sat_elevation),
            torch.zeros_like(self.sat_elevation),
        )

    def get_interference(self) -> torch.Tensor:
        """
        Compute NTN interference level per channel per environment.

        Returns:
            interference: (num_envs, num_channels) in [0, 1]
                          0 = no interference, 1 = maximum satellite interference
        """
        # Interference only on NTN-shared channels when satellite visible
        base_intf = torch.zeros(self.num_envs, self.num_channels, device=self.device)

        if self.sat_visible.any():
            # Higher elevation = higher interference
            elev_norm = (self.sat_elevation / 90.0).clamp(0, 1)  # (E,)
            intf_level = elev_norm * 0.8  # Max 80% interference

            # Apply to NTN channels only
            for ch in range(self.num_channels):
                if self.ntn_channels[ch]:
                    base_intf[:, ch] = intf_level * self.sat_visible.float()

        return base_intf

    def get_features(self) -> Dict[str, torch.Tensor]:
        """Get NTN features for observation augmentation."""
        return {
            "sat_elevation_deg": self.sat_elevation,
            "sat_visible": self.sat_visible.float(),
            "pass_time_remaining_s": self.pass_time_remaining_s,
            "ntn_interference": self.get_interference(),
        }
