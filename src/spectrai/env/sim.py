"""
Simulated Dynamic Spectrum Access environment.

Refactored from the original ``benchmarks/dsa_env.py``.  Extends the abstract
:class:`SpectrumEnv` base class and adds configurable SNR, interference,
noise, and per-channel heterogeneous Markov transition rates.

When instantiated with default parameters the behaviour is identical to the
original ``DSAEnv``.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import Field

from spectrai.env.base import SpectrumEnv, SpectrumEnvConfig


class SimulatedDSAConfig(SpectrumEnvConfig):
    """Extended configuration for the simulated DSA environment."""

    # Markov dynamics -- scalar or per-channel arrays
    pu_on_probs: Union[float, List[float]] = Field(
        0.3,
        description=(
            "Probability of PU transitioning OFF -> ON. "
            "Scalar applies to all channels; list must have length num_channels."
        ),
    )
    pu_off_probs: Union[float, List[float]] = Field(
        0.5,
        description=(
            "Probability of PU transitioning ON -> OFF. "
            "Scalar applies to all channels; list must have length num_channels."
        ),
    )

    # Feature generation
    snr_free: float = Field(1.0, description="Mean SNR when a channel is free.")
    snr_occupied: float = Field(0.2, description="Mean SNR when a channel is occupied.")
    interference_occupied: float = Field(
        0.8, description="Mean interference when channel is occupied."
    )
    interference_free: float = Field(
        0.1, description="Mean interference when channel is free."
    )
    feature_noise_std: float = Field(
        0.05, description="Std-dev of Gaussian noise on per-feature values."
    )

    # Observation noise
    observation_noise_std: float = Field(
        0.1,
        description="Std-dev of Gaussian noise applied to the full observation buffer.",
    )


class SimulatedDSAEnv(SpectrumEnv):
    """
    Simulated multi-channel DSA environment with Markov PU dynamics.

    Drop-in replacement for the original ``benchmarks/dsa_env.py:DSAEnv``
    when constructed with default parameters.  All tuning knobs are
    exposed via :class:`SimulatedDSAConfig`.

    Examples::

        # Quick — same behaviour as the original DSAEnv
        env = SimulatedDSAEnv()

        # Customised
        env = SimulatedDSAEnv(
            num_channels=20,
            pu_on_probs=[0.2] * 10 + [0.5] * 10,
            pu_off_probs=[0.6] * 10 + [0.3] * 10,
            snr_free=1.5,
        )

        # Via config object
        cfg = SimulatedDSAConfig(num_channels=5, max_steps=500)
        env = SimulatedDSAEnv(config=cfg)
    """

    def __init__(
        self,
        config: Optional[SimulatedDSAConfig] = None,
        **kwargs: Any,
    ):
        if config is None:
            config = SimulatedDSAConfig(**kwargs)
        elif not isinstance(config, SimulatedDSAConfig):
            # Allow passing a base SpectrumEnvConfig — upcast silently
            config = SimulatedDSAConfig(**config.model_dump())

        super().__init__(config=config)
        self._sim_cfg: SimulatedDSAConfig = config

        # Expand scalar Markov rates to per-channel arrays
        self._pu_on_probs = self._expand_rates(
            config.pu_on_probs, config.num_channels, "pu_on_probs"
        )
        self._pu_off_probs = self._expand_rates(
            config.pu_off_probs, config.num_channels, "pu_off_probs"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_rates(
        value: Union[float, Sequence[float]], n: int, name: str
    ) -> np.ndarray:
        """Convert a scalar or list of rates to an (n,) float64 array."""
        if isinstance(value, (int, float)):
            return np.full(n, float(value), dtype=np.float64)
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape != (n,):
            raise ValueError(
                f"{name} must be a scalar or a sequence of length {n}, "
                f"got shape {arr.shape}"
            )
        return arr

    # ------------------------------------------------------------------
    # Markov channel dynamics
    # ------------------------------------------------------------------

    def _init_channels(self) -> np.ndarray:
        """Initialise PU occupancy uniformly at random."""
        return (self.np_random.random(self.num_channels) < 0.5).astype(np.float32)

    def _step_channels(self) -> None:
        """Advance each channel's ON/OFF Markov process by one step."""
        rand = self.np_random.random(self.num_channels)
        for ch in range(self.num_channels):
            if self._channel_states[ch] == 0.0:  # type: ignore[index]
                if rand[ch] < self._pu_on_probs[ch]:
                    self._channel_states[ch] = 1.0  # type: ignore[index]
            else:
                if rand[ch] < self._pu_off_probs[ch]:
                    self._channel_states[ch] = 0.0  # type: ignore[index]

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def _get_channel_states(self) -> np.ndarray:
        """Advance Markov dynamics and return updated occupancy vector."""
        if self._channel_states is None:
            self._channel_states = self._init_channels()
        else:
            self._step_channels()
        return self._channel_states

    def _build_features(self) -> np.ndarray:
        """
        Build per-channel feature vector ``[SNR, interference, occupancy]``.

        Returns:
            np.ndarray of shape ``(num_channels * num_features,)``.
        """
        cfg = self._sim_cfg
        assert self._channel_states is not None
        occupancy = self._channel_states.copy()

        noise_snr = (
            self.np_random.standard_normal(self.num_channels).astype(np.float32)
            * cfg.feature_noise_std
        )
        noise_intf = (
            self.np_random.standard_normal(self.num_channels).astype(np.float32)
            * cfg.feature_noise_std
        )

        snr = np.where(occupancy == 0.0, cfg.snr_free, cfg.snr_occupied) + noise_snr
        interference = (
            np.where(occupancy == 1.0, cfg.interference_occupied, cfg.interference_free)
            + noise_intf
        )

        features = np.stack([snr, interference, occupancy], axis=-1).reshape(-1)
        return features.astype(np.float32)

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        """Return observation with configurable Gaussian noise."""
        assert self._history is not None
        obs = self._history.copy()
        if self._sim_cfg.observation_noise_std > 0.0:
            obs += (
                self.np_random.standard_normal(obs.shape).astype(np.float32)
                * self._sim_cfg.observation_noise_std
            )
        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment — reinitialise channels before filling history."""
        # Use gym.Env.reset directly to initialise np_random
        super(SpectrumEnv, self).reset(seed=seed)

        self._channel_states = None
        self._prev_action = None
        self._step_count = 0
        self._history = self._init_history()

        # Fill history buffer with initial observations
        for t in range(self.sequence_length):
            self._channel_states = self._get_channel_states()
            self._history[t] = self._build_features()

        obs = self._get_observation()
        assert self._channel_states is not None
        info: Dict[str, Any] = {"channel_states": self._channel_states.copy()}
        return obs, info
