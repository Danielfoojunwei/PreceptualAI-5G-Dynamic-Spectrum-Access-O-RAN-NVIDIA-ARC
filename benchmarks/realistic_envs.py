"""
Realistic and Heterogeneous DSA Environments for Fair Benchmarking.

Two environment variants that address the weaknesses of the original DSAEnv:

1. RealisticDSAEnv: Removes the occupancy leak, adds realistic noise,
   and makes the environment harder so heuristics can't trivially solve it.

2. HeterogeneousDSAEnv: Multi-provider environment where channels have
   different physics (latency, Doppler, fading timescales, SNR ranges).
   Models mixed LEO + 5G FR1 + WiFi 7 scenarios where heuristics must
   reason across provider types.

These environments expose where RL agents outperform traditional methods
on problems that actually resemble real wireless challenges.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ======================================================================
# 1. Realistic DSA Environment (no occupancy leak)
# ======================================================================

class RealisticDSAEnv(gym.Env):
    """
    Realistic Dynamic Spectrum Access without the occupancy leak.

    Key differences from DSAEnv:
    - NO direct occupancy feature in the observation (only SNR + interference)
    - Higher observation noise (real sensing uncertainty)
    - Correlated PU activity across channels (bursty traffic)
    - Time-varying PU transition probabilities (non-stationary)
    - Channel-specific SNR baselines (heterogeneous quality)

    The agent must INFER occupancy from noisy SNR/interference patterns
    over the history window — this is the real challenge in cognitive radio.

    State: (sequence_length, num_channels * 2)  — only SNR + interference
    Action: Discrete channel selection
    Reward: +1 success, -1 collision, -0.1 switching cost
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_channels: int = 10,
        sequence_length: int = 16,
        num_features: int = 2,        # SNR + interference only (NO occupancy)
        pu_on_prob: float = 0.3,
        pu_off_prob: float = 0.5,
        noise_std: float = 0.25,      # Much higher noise than original (0.1)
        max_steps: int = 200,
        nonstationarity: float = 0.05, # PU transition prob drift rate
        correlation: float = 0.3,      # Cross-channel PU correlation
    ):
        super().__init__()

        self.num_channels = num_channels
        self.sequence_length = sequence_length
        self.num_features = num_features  # 2: SNR + interference (no occupancy)
        self.base_pu_on_prob = pu_on_prob
        self.base_pu_off_prob = pu_off_prob
        self.noise_std = noise_std
        self.max_steps = max_steps
        self.nonstationarity = nonstationarity
        self.correlation = correlation

        self.action_space = spaces.Discrete(num_channels)
        obs_dim = num_channels * num_features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(sequence_length, obs_dim),
            dtype=np.float32,
        )

        # Per-channel baseline SNR when free (heterogeneous quality)
        self._base_snr = None
        self._channel_states = None
        self._history = None
        self._prev_action = None
        self._step_count = 0
        # Time-varying PU probabilities
        self._pu_on_prob = None
        self._pu_off_prob = None

    def _init_channels(self) -> np.ndarray:
        return (np.random.rand(self.num_channels) < 0.5).astype(np.float32)

    def _step_channels(self) -> None:
        """Advance PU with correlation and nonstationarity."""
        # Drift transition probabilities (non-stationary)
        self._pu_on_prob += np.random.randn(self.num_channels) * self.nonstationarity
        self._pu_off_prob += np.random.randn(self.num_channels) * self.nonstationarity
        self._pu_on_prob = np.clip(self._pu_on_prob, 0.05, 0.7)
        self._pu_off_prob = np.clip(self._pu_off_prob, 0.1, 0.8)

        # Correlated PU activity: shared random drive
        shared_drive = np.random.rand() < self.correlation

        for ch in range(self.num_channels):
            if self._channel_states[ch] == 0:
                prob = self._pu_on_prob[ch]
                if shared_drive:
                    prob = min(prob * 1.5, 0.9)  # Burst: all channels more likely to turn on
                if np.random.rand() < prob:
                    self._channel_states[ch] = 1.0
            else:
                prob = self._pu_off_prob[ch]
                if shared_drive:
                    prob = max(prob * 0.5, 0.05)  # Burst: channels less likely to turn off
                if np.random.rand() < prob:
                    self._channel_states[ch] = 0.0

    def _build_features(self) -> np.ndarray:
        """
        Build features WITHOUT occupancy — only SNR and interference.

        The key difference: agents must infer channel state from noisy
        continuous signals rather than reading a binary flag.
        """
        occupancy = self._channel_states.copy()

        # SNR: varies per channel, noisy, with realistic dynamics
        # Free channels: high SNR with variation
        # Occupied channels: lower SNR but NOT a clean binary signal
        snr_free = self._base_snr + np.random.randn(self.num_channels) * 0.15
        snr_occ = self._base_snr * 0.3 + np.random.randn(self.num_channels) * 0.2
        snr = np.where(occupancy == 0, snr_free, snr_occ)

        # Interference: continuous, overlapping distributions
        # Free channels still have background interference
        # Occupied channels have higher interference but distributions overlap
        interf_free = 0.15 + np.abs(np.random.randn(self.num_channels) * 0.15)
        interf_occ = 0.55 + np.abs(np.random.randn(self.num_channels) * 0.25)
        interference = np.where(occupancy == 0, interf_free, interf_occ)

        # Stack: [snr_0, interf_0, snr_1, interf_1, ...] — NO occupancy
        features = np.stack([snr, interference], axis=-1).reshape(-1)
        return features.astype(np.float32)

    def _get_observation(self) -> np.ndarray:
        obs = self._history.copy()
        obs += np.random.randn(*obs.shape).astype(np.float32) * self.noise_std
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._channel_states = self._init_channels()
        self._prev_action = None
        self._step_count = 0

        # Heterogeneous baseline SNR per channel
        self._base_snr = 0.6 + np.random.rand(self.num_channels) * 0.6  # [0.6, 1.2]

        # Initialize per-channel PU probabilities with variation
        self._pu_on_prob = np.full(self.num_channels, self.base_pu_on_prob)
        self._pu_on_prob += np.random.randn(self.num_channels) * 0.05
        self._pu_on_prob = np.clip(self._pu_on_prob, 0.05, 0.7)

        self._pu_off_prob = np.full(self.num_channels, self.base_pu_off_prob)
        self._pu_off_prob += np.random.randn(self.num_channels) * 0.05
        self._pu_off_prob = np.clip(self._pu_off_prob, 0.1, 0.8)

        obs_dim = self.num_channels * self.num_features
        self._history = np.zeros((self.sequence_length, obs_dim), dtype=np.float32)
        for t in range(self.sequence_length):
            self._step_channels()
            self._history[t] = self._build_features()

        obs = self._get_observation()
        info = {"channel_states": self._channel_states.copy()}
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action)
        self._step_count += 1

        self._step_channels()

        channel_occupied = self._channel_states[action] == 1.0
        reward = -1.0 if channel_occupied else 1.0
        if self._prev_action is not None and action != self._prev_action:
            reward -= 0.1
        self._prev_action = action

        new_features = self._build_features()
        self._history = np.roll(self._history, shift=-1, axis=0)
        self._history[-1] = new_features

        obs = self._get_observation()
        terminated = False
        truncated = self._step_count >= self.max_steps

        info = {
            "channel_states": self._channel_states.copy(),
            "collision": channel_occupied,
            "success": not channel_occupied,
        }
        return obs, reward, terminated, truncated, info


# ======================================================================
# 2. Heterogeneous Multi-Provider DSA Environment
# ======================================================================

class HeterogeneousDSAEnv(gym.Env):
    """
    Multi-provider heterogeneous spectrum access environment.

    Simulates 3 provider types simultaneously:
    - LEO Satellite (4 channels): High latency, Doppler, intermittent visibility
    - 5G FR1 (4 channels): Medium quality, urban fading
    - WiFi 7 (4 channels): High quality but crowded, fast fading

    Each provider has different:
    - SNR characteristics (baseline, variance)
    - PU dynamics (different transition rates)
    - Interference patterns
    - Observation noise levels
    - Switching costs (cross-provider handover is expensive)

    The observation encodes per-channel SNR + interference + a provider-type
    indicator, but NO direct occupancy. Agent must learn which provider
    type to use AND which channel within that provider.

    This is where heuristics fail: argmax(SNR) doesn't account for
    provider-specific dynamics, handover costs, or visibility windows.

    State: (sequence_length, total_channels * 3)  — SNR + interference + provider_id
    Action: Discrete(total_channels)
    """

    metadata = {"render_modes": ["human"]}

    # Provider configurations
    PROVIDERS = {
        "leo": {
            "num_channels": 4,
            "base_snr": 0.5,       # Lower SNR (long path)
            "snr_variance": 0.3,   # High variance (scintillation)
            "pu_on_prob": 0.15,    # Lower PU activity (dedicated spectrum)
            "pu_off_prob": 0.7,    # PUs leave quickly
            "interf_base": 0.1,
            "interf_occ": 0.4,
            "noise_std": 0.2,
            "visibility_prob": 0.85,  # LEO pass intermittency
            "provider_id": 0.0,
            "handover_cost": 0.0,     # intra-provider switching
            "cross_handover_cost": 0.4,  # cross-provider switching
        },
        "fr1": {
            "num_channels": 4,
            "base_snr": 0.8,
            "snr_variance": 0.15,
            "pu_on_prob": 0.35,    # Busy urban spectrum
            "pu_off_prob": 0.45,
            "interf_base": 0.2,
            "interf_occ": 0.7,
            "noise_std": 0.1,
            "visibility_prob": 1.0,  # Always available
            "provider_id": 0.5,
            "handover_cost": 0.0,
            "cross_handover_cost": 0.3,
        },
        "wifi7": {
            "num_channels": 4,
            "base_snr": 1.0,       # Highest SNR (short range)
            "snr_variance": 0.25,  # Variable (indoor fading)
            "pu_on_prob": 0.5,     # Very crowded unlicensed band
            "pu_off_prob": 0.6,    # Fast turnover
            "interf_base": 0.3,    # High background interference
            "interf_occ": 0.9,     # Very congested when occupied
            "noise_std": 0.15,
            "visibility_prob": 1.0,
            "provider_id": 1.0,
            "handover_cost": 0.0,
            "cross_handover_cost": 0.25,
        },
    }

    def __init__(
        self,
        sequence_length: int = 16,
        max_steps: int = 200,
        observation_noise_std: float = 0.15,
    ):
        super().__init__()

        self.sequence_length = sequence_length
        self.max_steps = max_steps
        self.observation_noise_std = observation_noise_std
        self.num_features = 3  # SNR + interference + provider_id

        # Build channel map: [(provider_name, local_channel_idx), ...]
        self._channel_map = []
        self._provider_names = []
        self._provider_ranges = {}  # provider_name -> (start_idx, end_idx)
        idx = 0
        for pname, pcfg in self.PROVIDERS.items():
            start = idx
            for ch in range(pcfg["num_channels"]):
                self._channel_map.append((pname, ch))
                idx += 1
            self._provider_ranges[pname] = (start, idx)
            self._provider_names.append(pname)

        self.num_channels = len(self._channel_map)  # total: 12
        self.action_space = spaces.Discrete(self.num_channels)

        obs_dim = self.num_channels * self.num_features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(sequence_length, obs_dim),
            dtype=np.float32,
        )

        self._channel_states = None
        self._visibility = None
        self._history = None
        self._prev_action = None
        self._prev_provider = None
        self._step_count = 0

    def _init_channels(self):
        self._channel_states = np.zeros(self.num_channels, dtype=np.float32)
        self._visibility = np.ones(self.num_channels, dtype=np.float32)
        for i, (pname, _) in enumerate(self._channel_map):
            self._channel_states[i] = float(np.random.rand() < 0.5)
            pcfg = self.PROVIDERS[pname]
            self._visibility[i] = float(np.random.rand() < pcfg["visibility_prob"])

    def _step_channels(self):
        """Step each channel with provider-specific dynamics."""
        for i, (pname, _) in enumerate(self._channel_map):
            pcfg = self.PROVIDERS[pname]

            # Update visibility (LEO pass intermittency)
            if np.random.rand() < 0.02:  # 2% chance visibility changes per step
                self._visibility[i] = float(np.random.rand() < pcfg["visibility_prob"])

            # PU dynamics per provider
            if self._channel_states[i] == 0:
                if np.random.rand() < pcfg["pu_on_prob"]:
                    self._channel_states[i] = 1.0
            else:
                if np.random.rand() < pcfg["pu_off_prob"]:
                    self._channel_states[i] = 0.0

    def _build_features(self) -> np.ndarray:
        """Build features: SNR + interference + provider_id (NO occupancy)."""
        features = np.zeros((self.num_channels, self.num_features), dtype=np.float32)

        for i, (pname, _) in enumerate(self._channel_map):
            pcfg = self.PROVIDERS[pname]
            occ = self._channel_states[i]
            vis = self._visibility[i]

            # SNR depends on occupancy, provider physics, and visibility
            if occ == 0 and vis > 0:
                snr = pcfg["base_snr"] + np.random.randn() * pcfg["snr_variance"] * 0.5
            elif occ == 1 and vis > 0:
                snr = pcfg["base_snr"] * 0.3 + np.random.randn() * pcfg["snr_variance"]
            else:
                # Not visible (LEO below horizon)
                snr = np.random.randn() * 0.05  # Noise floor

            # Interference
            if occ == 0:
                interf = pcfg["interf_base"] + abs(np.random.randn() * 0.1)
            else:
                interf = pcfg["interf_occ"] + abs(np.random.randn() * 0.15)

            # Apply per-provider noise
            snr += np.random.randn() * pcfg["noise_std"]
            interf += abs(np.random.randn() * pcfg["noise_std"] * 0.5)

            features[i, 0] = snr
            features[i, 1] = interf
            features[i, 2] = pcfg["provider_id"]  # Provider type indicator

        return features.reshape(-1)

    def _get_observation(self) -> np.ndarray:
        obs = self._history.copy()
        obs += np.random.randn(*obs.shape).astype(np.float32) * self.observation_noise_std
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._init_channels()
        self._prev_action = None
        self._prev_provider = None
        self._step_count = 0

        obs_dim = self.num_channels * self.num_features
        self._history = np.zeros((self.sequence_length, obs_dim), dtype=np.float32)
        for t in range(self.sequence_length):
            self._step_channels()
            self._history[t] = self._build_features()

        obs = self._get_observation()
        info = {"channel_states": self._channel_states.copy()}
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action)
        self._step_count += 1

        self._step_channels()

        pname, local_ch = self._channel_map[action]
        pcfg = self.PROVIDERS[pname]

        # Check visibility (LEO intermittency)
        if self._visibility[action] == 0:
            reward = -1.5  # Tried to use invisible satellite — worse than collision
            success = False
            collision = True
        elif self._channel_states[action] == 1.0:
            reward = -1.0  # Collision
            success = False
            collision = True
        else:
            reward = 1.0   # Success
            success = True
            collision = False

        # Switching cost: intra-provider vs cross-provider handover
        if self._prev_action is not None:
            prev_pname = self._channel_map[self._prev_action][0]
            if prev_pname != pname:
                # Cross-provider handover (expensive)
                reward -= pcfg["cross_handover_cost"]
            elif action != self._prev_action:
                # Same provider, different channel (cheap)
                reward -= 0.05

        self._prev_action = action
        self._prev_provider = pname

        new_features = self._build_features()
        self._history = np.roll(self._history, shift=-1, axis=0)
        self._history[-1] = new_features

        obs = self._get_observation()
        terminated = False
        truncated = self._step_count >= self.max_steps

        info = {
            "channel_states": self._channel_states.copy(),
            "collision": collision,
            "success": success,
            "provider": pname,
            "visibility": self._visibility.copy(),
        }
        return obs, reward, terminated, truncated, info
