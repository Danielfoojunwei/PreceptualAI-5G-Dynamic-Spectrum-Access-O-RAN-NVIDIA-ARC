"""
Abstract base class for SpectrAI spectrum environments.

Provides the shared Gymnasium interface, history-buffer management,
and reward logic used by all concrete environment implementations
(simulated, O-RAN live, NVIDIA AODT).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pydantic import BaseModel, Field, field_validator


class SpectrumEnvConfig(BaseModel):
    """Validated configuration for any SpectrumEnv derivative."""

    num_channels: int = Field(10, ge=1, description="Number of wireless channels (N).")
    num_features: int = Field(3, ge=1, description="Features per channel.")
    sequence_length: int = Field(16, ge=1, description="Observation history window (T).")
    feature_names: List[str] = Field(
        default_factory=lambda: ["snr", "interference", "occupancy"],
        description="Human-readable feature names.",
    )
    max_steps: int = Field(200, ge=1, description="Maximum steps per episode.")
    reward_success: float = Field(1.0, description="Reward for successful transmission.")
    reward_collision: float = Field(-1.0, description="Penalty for PU collision.")
    reward_switch: float = Field(-0.1, description="Cost for switching channels.")

    @field_validator("feature_names")
    @classmethod
    def _check_feature_names_length(cls, v: List[str], info: Any) -> List[str]:
        num_features = info.data.get("num_features")
        if num_features is not None and len(v) != num_features:
            raise ValueError(
                f"feature_names length ({len(v)}) must match "
                f"num_features ({num_features})"
            )
        return v

    model_config = {"frozen": False, "extra": "allow"}


class SpectrumEnv(gym.Env, ABC):
    """
    Abstract Gymnasium environment for dynamic spectrum access.

    Subclasses must implement:
        _get_channel_states() -> np.ndarray of shape (num_channels,)
        _build_features()     -> np.ndarray of shape (num_channels * num_features,)

    The base class provides:
        reset(), step(), _get_observation(), history buffer management.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Optional[SpectrumEnvConfig] = None, **kwargs: Any):
        super().__init__()

        if config is None:
            config = SpectrumEnvConfig(**kwargs)

        self.config = config
        self.num_channels: int = config.num_channels
        self.num_features: int = config.num_features
        self.sequence_length: int = config.sequence_length
        self.max_steps: int = config.max_steps

        self.action_space = spaces.Discrete(config.num_channels)

        obs_dim = config.num_channels * config.num_features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(config.sequence_length, obs_dim),
            dtype=np.float32,
        )

        # Internal state
        self._channel_states: Optional[np.ndarray] = None
        self._history: Optional[np.ndarray] = None
        self._prev_action: Optional[int] = None
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _get_channel_states(self) -> np.ndarray:
        """
        Return the current binary occupancy vector.

        Returns:
            np.ndarray of shape (num_channels,) with 1.0 = occupied, 0.0 = free.
        """

    @abstractmethod
    def _build_features(self) -> np.ndarray:
        """
        Build a per-channel feature vector for the current timestep.

        Returns:
            np.ndarray of shape (num_channels * num_features,).
        """

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        """
        Return a copy of the history buffer.

        Shape: (sequence_length, num_channels * num_features).
        """
        assert self._history is not None
        return np.array(self._history, copy=True)

    def _init_history(self) -> np.ndarray:
        """Allocate a zeroed history buffer."""
        obs_dim = self.num_channels * self.num_features
        return np.zeros((self.sequence_length, obs_dim), dtype=np.float32)

    def _append_to_history(self, features: np.ndarray) -> None:
        """Shift history left by one step and append new features."""
        assert self._history is not None
        self._history = np.roll(self._history, shift=-1, axis=0)
        self._history[-1] = features

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self._prev_action = None
        self._step_count = 0
        self._history = self._init_history()

        # Fill the history buffer with initial observations
        for t in range(self.sequence_length):
            self._channel_states = self._get_channel_states()
            self._history[t] = self._build_features()

        obs = self._get_observation()
        assert self._channel_states is not None
        info: Dict[str, Any] = {"channel_states": self._channel_states.copy()}
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        self._step_count += 1

        # Advance channel dynamics
        self._channel_states = self._get_channel_states()

        # Compute reward
        channel_occupied = bool(self._channel_states[action] == 1.0)
        if channel_occupied:
            reward = self.config.reward_collision
        else:
            reward = self.config.reward_success

        # Switching cost
        if self._prev_action is not None and action != self._prev_action:
            reward += self.config.reward_switch

        self._prev_action = action

        # Update history buffer
        new_features = self._build_features()
        self._append_to_history(new_features)

        obs = self._get_observation()
        terminated = False
        truncated = self._step_count >= self.max_steps

        info: Dict[str, Any] = {
            "channel_states": self._channel_states.copy(),
            "collision": channel_occupied,
            "success": not channel_occupied,
            "step": self._step_count,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        if self._channel_states is None:
            return
        occ = self._channel_states.astype(int)
        bar = " ".join("X" if o else "." for o in occ)
        print(f"Step {self._step_count:>4d} | Channels: [{bar}]")
