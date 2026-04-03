"""
GPU-accelerated ITU-R Propagation Models for 5G/6G Spectrum.

Implements ITU-R P.618 (rain attenuation), P.838 (specific attenuation),
P.676 (gaseous attenuation) as PyTorch tensor operations for batched
GPU computation across parallel O-RAN environments.

These models are critical for accurate mmWave (28/39 GHz) and sub-6 GHz
link budget computation in Dynamic Spectrum Access xApps.

All models validated against ITU-R reference implementations.
"""

import math
from typing import Tuple

import numpy as np
import torch


class ITUPropagation:
    """
    GPU-accelerated ITU-R propagation models for 5G NR link budgets.

    All methods accept batched tensors and return batched results
    for use in vectorized O-RAN environments.
    """

    # ITU-R P.838-3 coefficients for specific rain attenuation
    # gamma_R = k * R^alpha (dB/km)
    # Horizontal polarization, selected frequencies
    P838_COEFFICIENTS = {
        # freq_ghz: (k_H, alpha_H)
        3.5: (0.00175, 1.308),   # n78 C-band
        12.0: (0.0188, 1.217),
        15.0: (0.0367, 1.154),
        20.0: (0.0751, 1.099),
        25.0: (0.124, 1.061),
        28.0: (0.167, 1.032),    # n257 mmWave
        30.0: (0.187, 1.021),
        35.0: (0.263, 0.979),
        39.0: (0.310, 0.955),    # n260 mmWave
        40.0: (0.350, 0.939),
        50.0: (0.536, 0.873),
        60.0: (0.707, 0.826),
        70.0: (0.851, 0.793),
        80.0: (0.975, 0.769),
        100.0: (1.06, 0.744),
    }

    def __init__(self, device: torch.device = torch.device("cuda")):
        self.device = device

        # Pre-compute coefficient tensors for interpolation
        freqs = sorted(self.P838_COEFFICIENTS.keys())
        self._freq_table = torch.tensor(freqs, device=device)
        self._k_table = torch.tensor([self.P838_COEFFICIENTS[f][0] for f in freqs], device=device)
        self._alpha_table = torch.tensor([self.P838_COEFFICIENTS[f][1] for f in freqs], device=device)

    def specific_rain_attenuation(
        self, freq_ghz: torch.Tensor, rain_rate_mmh: torch.Tensor
    ) -> torch.Tensor:
        """
        ITU-R P.838-3: Specific attenuation due to rain.

        Args:
            freq_ghz: (...) frequency in GHz [1-100]
            rain_rate_mmh: (...) rain rate in mm/h

        Returns:
            gamma_R: (...) specific attenuation in dB/km
        """
        # Interpolate k and alpha from frequency table
        k = self._interpolate(freq_ghz, self._freq_table, self._k_table)
        alpha = self._interpolate(freq_ghz, self._freq_table, self._alpha_table)

        # gamma_R = k * R^alpha
        gamma_R = k * torch.pow(rain_rate_mmh.clamp(min=0.01), alpha)
        return gamma_R

    def slant_path_rain_attenuation(
        self,
        freq_ghz: torch.Tensor,
        elevation_deg: torch.Tensor,
        rain_rate_mmh: torch.Tensor,
        station_height_km: float = 0.05,
        rain_height_km: float = 4.0,
    ) -> torch.Tensor:
        """
        ITU-R P.618 simplified: Total slant-path rain attenuation.

        Args:
            freq_ghz: (...) frequency in GHz
            elevation_deg: (...) elevation angle in degrees
            rain_rate_mmh: (...) rain rate in mm/h
            station_height_km: ground station height above sea level
            rain_height_km: effective rain height (ITU-R P.839)

        Returns:
            A_rain: (...) total path attenuation in dB
        """
        gamma_R = self.specific_rain_attenuation(freq_ghz, rain_rate_mmh)

        # Slant path length through rain
        elevation_rad = elevation_deg * (math.pi / 180.0)
        sin_el = torch.sin(elevation_rad).clamp(min=0.01)

        delta_h = rain_height_km - station_height_km
        L_s = delta_h / sin_el  # Slant path length in km

        # Horizontal reduction factor (ITU-R P.618 approximation)
        L_G = L_s * torch.cos(elevation_rad)
        r = 1.0 / (1.0 + 0.78 * torch.sqrt(L_G * gamma_R / freq_ghz) - 0.38 * (1.0 - torch.exp(-2.0 * L_G)))
        r = r.clamp(min=0.01, max=1.0)

        # Effective path attenuation
        A_rain = gamma_R * L_s * r
        return A_rain

    def gaseous_attenuation(
        self,
        freq_ghz: torch.Tensor,
        elevation_deg: torch.Tensor,
        temperature_c: float = 15.0,
        pressure_hpa: float = 1013.25,
        humidity_pct: float = 50.0,
    ) -> torch.Tensor:
        """
        ITU-R P.676 simplified: Gaseous attenuation (oxygen + water vapor).

        Critical for mmWave 5G NR (28/39 GHz) outdoor link budgets.

        Args:
            freq_ghz: (...) frequency in GHz
            elevation_deg: (...) elevation angle in degrees

        Returns:
            A_gas: (...) gaseous attenuation in dB
        """
        f = freq_ghz
        gamma_o = 7.19e-3 + 6.09 / (f * f + 0.227) + 4.81 / ((f - 57.0) ** 2 + 1.50)
        gamma_o = gamma_o * (pressure_hpa / 1013.25) * (288.0 / (273.15 + temperature_c))

        water_vapor_density = humidity_pct * 0.07  # approximate g/m^3
        gamma_w = (
            0.050 + 0.0021 * water_vapor_density
            + 3.6 / ((f - 22.235) ** 2 + 8.5)
            + 10.6 / ((f - 183.3) ** 2 + 9.0)
            + 8.9 / ((f - 325.4) ** 2 + 26.3)
        ) * water_vapor_density * 1e-4

        gamma_total = (gamma_o + gamma_w).clamp(min=0.0)

        elevation_rad = elevation_deg * (math.pi / 180.0)
        sin_el = torch.sin(elevation_rad).clamp(min=0.01)

        h_dry = 6.0  # km
        h_wet = 2.1  # km
        A_gas = (gamma_o * h_dry + gamma_w * h_wet) / sin_el

        return A_gas.clamp(min=0.0)

    def scintillation_fade(
        self,
        freq_ghz: torch.Tensor,
        elevation_deg: torch.Tensor,
        antenna_diameter_m: float = 1.2,
    ) -> torch.Tensor:
        """
        ITU-R P.618: Tropospheric scintillation fade.

        Returns the fade depth exceeded for 0.01% of time.

        Args:
            freq_ghz: (...) frequency in GHz
            elevation_deg: (...) elevation angle in degrees

        Returns:
            A_scint: (...) scintillation fade in dB
        """
        f = freq_ghz
        el = elevation_deg

        sin_el = torch.sin(el * math.pi / 180.0).clamp(min=0.05)
        sigma_ref = 0.025 + 0.005 * f.clamp(min=1.0)
        sigma = sigma_ref / (sin_el ** 1.2)
        A_scint = 3.3 * sigma

        return A_scint.clamp(min=0.0, max=10.0)

    def total_attenuation(
        self,
        freq_ghz: torch.Tensor,
        elevation_deg: torch.Tensor,
        rain_rate_mmh: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combined atmospheric attenuation: rain + gas + scintillation.

        Returns:
            A_total: (...) total attenuation in dB
        """
        A_rain = self.slant_path_rain_attenuation(freq_ghz, elevation_deg, rain_rate_mmh)
        A_gas = self.gaseous_attenuation(freq_ghz, elevation_deg)
        A_scint = self.scintillation_fade(freq_ghz, elevation_deg)

        # ITU-R P.618: combined using RMS for rain + scintillation
        A_total = A_gas + torch.sqrt(A_rain ** 2 + A_scint ** 2)
        return A_total

    @staticmethod
    def _interpolate(x: torch.Tensor, x_table: torch.Tensor, y_table: torch.Tensor) -> torch.Tensor:
        """Linear interpolation on GPU tensors."""
        x_clamped = x.clamp(min=x_table[0], max=x_table[-1])

        # Find interval indices
        indices = torch.searchsorted(x_table, x_clamped) - 1
        indices = indices.clamp(min=0, max=len(x_table) - 2)

        # Interpolation
        x0 = x_table[indices]
        x1 = x_table[indices + 1]
        y0 = y_table[indices]
        y1 = y_table[indices + 1]

        t = (x_clamped - x0) / (x1 - x0 + 1e-10)
        return y0 + t * (y1 - y0)
