"""
Satellite Spectrum Environment for PreceptualAI.

GPU-vectorized environment simulating satellite-to-ground dynamic spectrum
access across Ka/Ku/V bands with ITU-R propagation physics, orbital
dynamics, and multi-beam interference.

Designed for training SAC-LTC agents on space data center connectivity.
"""

from typing import Dict, Tuple

import torch

from preceptualai.env.itu_propagation import ITUPropagation
from preceptualai.env.orbital_dynamics import OrbitalDynamics, GroundStation, REFERENCE_STATIONS


class SatelliteSpectrumEnv:
    """
    GPU-vectorized satellite spectrum environment.

    State: per-beam channel quality (SNR after attenuation), weather,
           elevation, Doppler, handover countdown, interference.
    Action: frequency band selection per ground station (discrete).
    Reward: throughput - interference penalty - handover penalty.

    All computation runs on GPU tensors for vectorized parallel simulation.
    """

    # Frequency bands available (GHz)
    BANDS = {
        "Ku_low": 12.0,
        "Ku_high": 18.0,
        "Ka_down": 20.0,
        "Ka_up": 30.0,
        "V_low": 40.0,
        "V_high": 50.0,
    }

    def __init__(
        self,
        num_envs: int = 256,
        num_stations: int = 5,
        num_bands: int = 6,
        num_satellites: int = 10,
        sequence_length: int = 16,
        max_steps: int = 200,
        altitude_km: float = 550.0,
        dt_seconds: float = 1.0,
        rain_rate_range: Tuple[float, float] = (0.0, 50.0),
        reward_throughput_weight: float = 1.0,
        reward_interference_weight: float = 0.5,
        reward_handover_weight: float = 0.2,
        reward_power_weight: float = 0.1,
        device: torch.device = torch.device("cuda"),
    ):
        self.num_envs = num_envs
        self.num_stations = num_stations
        self.num_bands = num_bands
        self.num_satellites = num_satellites
        self.sequence_length = sequence_length
        self.max_steps = max_steps
        self.dt_seconds = dt_seconds
        self.device = device

        # Reward weights
        self.w_throughput = reward_throughput_weight
        self.w_interference = reward_interference_weight
        self.w_handover = reward_handover_weight
        self.w_power = reward_power_weight

        # Frequency band values (GHz)
        band_freqs = list(self.BANDS.values())[:num_bands]
        self.band_freqs = torch.tensor(band_freqs, device=device)

        # ITU-R propagation model
        self.propagation = ITUPropagation(device=device)

        # Orbital dynamics (shared across envs — deterministic)
        stations = REFERENCE_STATIONS[:num_stations]
        self.orbital = OrbitalDynamics(
            altitude_km=altitude_km,
            num_satellites=num_satellites,
            ground_stations=stations,
            device=device,
        )

        # State tensors
        num_features = 5  # [snr_norm, rain_norm, elevation_norm, doppler_norm, handover_norm]
        obs_dim = num_stations * num_features
        self.num_features = num_features

        self.history = torch.zeros(num_envs, sequence_length, obs_dim, device=device)
        self.rain_rates = torch.zeros(num_envs, num_stations, device=device)
        self.prev_actions = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.step_counts = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.current_bands = torch.zeros(num_envs, num_stations, dtype=torch.long, device=device)

        # Rain dynamics (Markov model for rain rate evolution)
        self.rain_range = rain_rate_range

    @property
    def action_space_size(self) -> int:
        """Number of discrete actions = num_bands (select band per station)."""
        return self.num_bands

    @property
    def obs_dim(self) -> int:
        return self.num_stations * self.num_features

    def reset(self, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        n = mask.sum().item()
        if n == 0:
            return self._get_observation()

        # Reset rain rates (random initialization)
        self.rain_rates[mask] = torch.rand(n, self.num_stations, device=self.device) * 10.0

        # Reset state
        self.step_counts[mask] = 0
        self.prev_actions[mask] = -1
        self.current_bands[mask] = 0

        # Reset orbital dynamics
        self.orbital._time_s = 0.0

        # Fill history
        self.history[mask] = 0.0
        for t in range(self.sequence_length):
            self._evolve_weather(mask)
            self.orbital.step(self.dt_seconds)
            features = self._build_features(mask)
            self.history[mask, t] = features

        return self._get_observation()

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            actions: (num_envs,) int — frequency band index [0, num_bands)
        """
        self.step_counts += 1

        # Evolve weather (rain rates)
        all_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._evolve_weather(all_mask)

        # Advance orbital dynamics
        self.orbital.step(self.dt_seconds)

        # Get channel quality for selected bands
        elevation = self.orbital.get_elevation_angles()  # (S, G)
        best_sat = self.orbital.get_best_satellite()  # (G,)
        best_elevation = elevation[best_sat, torch.arange(self.num_stations, device=self.device)]  # (G,)

        # Expand for all envs: (E, G)
        elev_expanded = best_elevation.unsqueeze(0).expand(self.num_envs, -1)

        # Get frequency for selected band per env
        selected_freq = self.band_freqs[actions.clamp(0, self.num_bands - 1)]  # (E,)

        # Compute attenuation for each env's selected band at each station
        # Use first station's rain rate as representative
        freq_expanded = selected_freq.unsqueeze(1).expand(-1, self.num_stations)  # (E, G)
        rain_expanded = self.rain_rates  # (E, G)

        # Total atmospheric attenuation
        attenuation = self.propagation.total_attenuation(
            freq_expanded, elev_expanded.clamp(min=1.0), rain_expanded
        )  # (E, G)

        # SNR = base_snr - attenuation (dB scale)
        base_snr = 30.0  # dB, clear-sky SNR
        effective_snr = (base_snr - attenuation).clamp(min=-10.0, max=40.0)  # (E, G)

        # Throughput (Shannon capacity proxy): C = log2(1 + SNR_linear)
        snr_linear = 10 ** (effective_snr / 10.0)
        throughput = torch.log2(1.0 + snr_linear).mean(dim=1)  # (E,) avg across stations

        # Interference: higher frequency bands have less congestion (reward V-band usage)
        interference = (1.0 - selected_freq / 75.0).clamp(min=0) * 0.5  # (E,)

        # Handover penalty
        switched = (self.prev_actions >= 0) & (actions != self.prev_actions)
        handover_penalty = switched.float() * 0.3  # (E,)

        # Power cost (higher freq needs more power)
        power_cost = (selected_freq / 50.0).clamp(max=1.0) * 0.2  # (E,)

        # Reward
        rewards = (
            self.w_throughput * throughput
            - self.w_interference * interference
            - self.w_handover * handover_penalty
            - self.w_power * power_cost
        )

        self.prev_actions = actions
        self.current_bands[:, :] = actions.unsqueeze(1)

        # Build features and update history
        features = self._build_features(all_mask)
        self.history = torch.roll(self.history, -1, dims=1)
        self.history[:, -1] = features

        # Done detection
        dones = self.step_counts >= self.max_steps

        obs = self._get_observation()

        # Success: effective SNR > 10 dB at majority of stations
        success = (effective_snr > 10.0).float().mean(dim=1) > 0.5

        infos = {
            "success": success,
            "collision": ~success,
            "throughput": throughput,
            "attenuation_mean": attenuation.mean(dim=1),
            "effective_snr_mean": effective_snr.mean(dim=1),
        }

        return obs, rewards, dones, infos

    def auto_reset(self, dones: torch.Tensor) -> torch.Tensor:
        if dones.any():
            self.reset(mask=dones)
        return self._get_observation()

    def _evolve_weather(self, mask: torch.Tensor):
        """Evolve rain rates using an Ornstein-Uhlenbeck process."""
        n = mask.sum().item()
        if n == 0:
            return

        # OU process: dR = theta * (mu - R) * dt + sigma * dW
        theta = 0.01  # mean reversion rate
        mu = 5.0  # long-term mean rain rate (mm/h)
        sigma = 2.0  # volatility

        current = self.rain_rates[mask]
        noise = torch.randn_like(current) * sigma * (self.dt_seconds ** 0.5)
        new_rain = current + theta * (mu - current) * self.dt_seconds + noise
        self.rain_rates[mask] = new_rain.clamp(min=0.0, max=self.rain_range[1])

    def _build_features(self, mask: torch.Tensor) -> torch.Tensor:
        """Build observation features for masked environments."""
        n = mask.sum().item()

        # Get orbital state
        elevation = self.orbital.get_elevation_angles()  # (S, G)
        best_sat = self.orbital.get_best_satellite()  # (G,)
        best_elev = elevation[best_sat, torch.arange(self.num_stations, device=self.device)]
        doppler = self.orbital.get_doppler_shift()  # (S, G)
        best_doppler = doppler[best_sat, torch.arange(self.num_stations, device=self.device)]
        time_to_set = self.orbital.get_time_to_set()  # (S, G)
        best_tts = time_to_set[best_sat, torch.arange(self.num_stations, device=self.device)]

        # Expand to env batch
        elev_exp = best_elev.unsqueeze(0).expand(n, -1)  # (n, G)
        doppler_exp = best_doppler.unsqueeze(0).expand(n, -1)
        tts_exp = best_tts.unsqueeze(0).expand(n, -1)

        # Rain attenuation at Ka 30 GHz (representative)
        rain = self.rain_rates[mask]
        freq_30 = torch.full_like(rain, 30.0)
        attenuation = self.propagation.slant_path_rain_attenuation(
            freq_30, elev_exp.clamp(min=1.0), rain
        )

        # Normalize features to [0, 1]
        snr_norm = (30.0 - attenuation).clamp(min=-10, max=40) / 50.0  # map [-10,40] to [0,1]
        rain_norm = rain / self.rain_range[1]
        elev_norm = elev_exp.clamp(min=0, max=90) / 90.0
        doppler_norm = (doppler_exp / 600e3).clamp(-1, 1) * 0.5 + 0.5  # map to [0,1]
        handover_norm = (tts_exp / 600.0).clamp(0, 1)  # 600s max
        handover_norm[tts_exp == float("inf")] = 0.0  # no satellite = 0

        # Stack features: (n, G, 5) -> (n, G*5)
        features = torch.stack([
            snr_norm, rain_norm, elev_norm, doppler_norm, handover_norm
        ], dim=-1)

        return features.reshape(n, -1)

    def _get_observation(self) -> torch.Tensor:
        noise = torch.randn_like(self.history) * 0.05
        return self.history + noise
