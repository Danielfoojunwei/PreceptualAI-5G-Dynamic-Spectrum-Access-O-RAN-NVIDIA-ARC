"""
Real 5G Trace Environment for PreceptualAI.

Replays real 5G network measurements from the UCC MISL 5G Dataset
(https://github.com/uccmisl/5Gdataset) as a spectrum management environment.

Features extracted from real data:
  - RSRP (Reference Signal Received Power)
  - RSRQ (Reference Signal Received Quality)
  - SNR (Signal-to-Noise Ratio)
  - CQI (Channel Quality Indicator)
  - RSSI (Received Signal Strength Indicator)
"""

import glob
import os
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class Real5GEnv(gym.Env):
    """
    Spectrum environment driven by real 5G network traces.

    Pre-loads all traces as numpy arrays for fast step execution.
    Channels are synthesized from real signal metrics with per-channel variation.
    """

    metadata = {"render_modes": ["human"]}

    FEATURE_COLS = ["RSRP", "RSRQ", "SNR", "CQI", "RSSI"]

    def __init__(
        self,
        data_dir: str,
        num_channels: int = 10,
        sequence_length: int = 16,
        max_steps: int = 200,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.num_features = len(self.FEATURE_COLS)
        self.sequence_length = sequence_length
        self.max_steps = max_steps

        # Pre-load and convert ALL traces to numpy (fast)
        self._traces, self._snr_traces = self._load_and_preprocess(data_dir)
        self._current_trace_idx = 0
        self._trace_pos = 0

        # Gymnasium spaces
        self.action_space = spaces.Discrete(num_channels)
        obs_dim = num_channels * self.num_features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(sequence_length, obs_dim),
            dtype=np.float32,
        )

        self._history: np.ndarray = np.zeros(
            (sequence_length, num_channels * self.num_features), dtype=np.float32
        )
        self._channel_quality: np.ndarray = np.zeros(num_channels, dtype=np.float32)
        self._prev_action: Optional[int] = None
        self._step_count: int = 0

        # Pre-generate per-channel noise seeds for reproducibility
        self._rng = np.random.RandomState(42)

    def _load_and_preprocess(self, data_dir: str):
        """Load all CSVs, extract features, normalize, convert to numpy arrays."""
        csv_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
        csv_files = [f for f in csv_files if "MACOSX" not in f]

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")

        # First pass: collect all values for normalization
        all_values: Dict[str, list] = {col: [] for col in self.FEATURE_COLS}
        raw_traces = []

        for f in csv_files:
            try:
                df = pd.read_csv(f)
                available = [c for c in self.FEATURE_COLS if c in df.columns]
                if len(available) >= 3:
                    raw_traces.append(df)
                    for col in self.FEATURE_COLS:
                        if col in df.columns:
                            vals = pd.to_numeric(df[col], errors="coerce").dropna()
                            all_values[col].extend(vals.tolist())
            except Exception:
                continue

        if not raw_traces:
            raise ValueError(f"No valid trace files in {data_dir}")

        # Compute normalization stats
        means = {}
        stds = {}
        for col in self.FEATURE_COLS:
            vals = np.array(all_values[col]) if all_values[col] else np.array([0.0])
            means[col] = float(vals.mean())
            stds[col] = float(vals.std()) if vals.std() > 0 else 1.0

        # Second pass: convert to normalized numpy arrays (FAST at runtime)
        numpy_traces = []
        snr_traces = []

        for df in raw_traces:
            n_rows = len(df)
            features = np.zeros((n_rows, self.num_features), dtype=np.float32)

            for i, col in enumerate(self.FEATURE_COLS):
                if col in df.columns:
                    vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values
                    features[:, i] = (vals - means[col]) / stds[col]

            numpy_traces.append(features)

            # Extract raw SNR for channel quality (before normalization)
            if "SNR" in df.columns:
                raw_snr = pd.to_numeric(df["SNR"], errors="coerce").fillna(0.0).values
                snr_traces.append(raw_snr.astype(np.float32))
            else:
                snr_traces.append(np.zeros(n_rows, dtype=np.float32))

        total_rows = sum(len(t) for t in numpy_traces)
        print(f"[Real5GEnv] Loaded {len(numpy_traces)} traces, "
              f"{total_rows} measurements, pre-cached as numpy")

        return numpy_traces, snr_traces

    def _get_features_at(self, trace_idx: int, pos: int) -> np.ndarray:
        """Get pre-computed features and expand to multi-channel (FAST)."""
        trace = self._traces[trace_idx]
        row = trace[min(pos, len(trace) - 1)]  # (num_features,) numpy — instant

        # Expand single measurement to num_channels with per-channel variation
        features = np.tile(row, (self.num_channels, 1))  # (C, F)
        noise = self._rng.randn(self.num_channels, self.num_features).astype(np.float32) * 0.3
        features = features + noise
        return np.asarray(features.reshape(-1))  # (C*F,)

    def _get_channel_quality(self) -> np.ndarray:
        """Derive channel occupancy from real SNR values."""
        snr_trace = self._snr_traces[self._current_trace_idx]
        pos = min(self._trace_pos, len(snr_trace) - 1)
        base_snr = snr_trace[pos]

        # Per-channel SNR variation
        channel_snr = base_snr + self._rng.randn(self.num_channels).astype(np.float32) * 3.0
        # Channels with SNR < 5 dB are "occupied" (poor quality)
        occupied = (channel_snr < 5.0).astype(np.float32)
        return np.asarray(occupied)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.RandomState(seed)

        self._current_trace_idx = self._rng.randint(0, len(self._traces))
        trace = self._traces[self._current_trace_idx]
        max_start = max(0, len(trace) - self.max_steps - self.sequence_length)
        self._trace_pos = self._rng.randint(0, max(1, max_start))

        self._prev_action = None
        self._step_count = 0

        # Fill history buffer from pre-cached numpy
        obs_dim = self.num_channels * self.num_features
        self._history = np.zeros((self.sequence_length, obs_dim), dtype=np.float32)
        for t in range(self.sequence_length):
            self._history[t] = self._get_features_at(
                self._current_trace_idx, self._trace_pos + t
            )
        self._trace_pos += self.sequence_length
        self._channel_quality = self._get_channel_quality()

        info = {
            "channel_quality": self._channel_quality.copy(),
            "data_source": "5G-production-dataset (UCC MISL)",
        }
        return self._history.copy(), info

    def step(self, action: int):  # type: ignore[override]
        assert self.action_space.contains(action)
        self._step_count += 1

        trace = self._traces[self._current_trace_idx]
        self._trace_pos = min(self._trace_pos + 1, len(trace) - 1)

        self._channel_quality = self._get_channel_quality()

        # Reward from real channel quality
        channel_occupied = self._channel_quality[action] == 1.0
        reward = -1.0 if channel_occupied else 1.0

        if self._prev_action is not None and action != self._prev_action:
            reward -= 0.1
        self._prev_action = action

        # Update history from pre-cached numpy (FAST)
        new_features = self._get_features_at(self._current_trace_idx, self._trace_pos)
        self._history = np.roll(self._history, shift=-1, axis=0)
        self._history[-1] = new_features

        terminated = False
        truncated = (self._step_count >= self.max_steps) or (self._trace_pos >= len(trace) - 1)

        info = {
            "channel_quality": self._channel_quality.copy(),
            "collision": bool(channel_occupied),
            "success": bool(not channel_occupied),
            "data_source": "5G-production-dataset (UCC MISL)",
        }
        return self._history.copy(), reward, terminated, truncated, info

    def render(self):
        quality = self._channel_quality.astype(int)
        bar = " ".join(f"{'X' if q else '.'}" for q in quality)
        print(f"Step {self._step_count:>4d} | Channels: [{bar}] | "
              f"Trace {self._current_trace_idx} pos {self._trace_pos}")
