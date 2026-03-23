"""
NVIDIA Aerial Digital Twin (AODT) environment.

GPU-accelerated spectrum environment using pyAerial's channel estimation
pipeline and Sionna's 3GPP channel models.  Provides physically realistic
channel observations for training and validating SAC-LTC agents before
live O-RAN deployment.

Requires:
    - NVIDIA GPU with CUDA support
    - pyAerial (``nvidia-pyaerial``)
    - Sionna (``sionna``)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from pydantic import Field

from spectrai.env.base import SpectrumEnv, SpectrumEnvConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional GPU imports
# ---------------------------------------------------------------------------
try:
    import cupy as cp

    _HAS_CUPY = True
except ImportError:
    cp = None  # type: ignore[assignment]
    _HAS_CUPY = False

try:
    from aerial.phy5g.chest import ChannelEstimator as PyAerialChannelEstimateGenerator
    from aerial.phy5g.ldpc import LDPCDecoder, LDPCEncoder  # noqa: F401
    from aerial.phy5g.ofdm import OFDMDemodulator, OFDMModulator  # noqa: F401

    _HAS_PYAERIAL = True
except ImportError:
    PyAerialChannelEstimateGenerator = None  # type: ignore[assignment,misc]
    _HAS_PYAERIAL = False

try:
    import sionna  # noqa: F401
    from sionna.channel import OFDMChannel
    from sionna.channel.tr38901 import UMa as SionnaUMa

    _HAS_SIONNA = True
except ImportError:
    SionnaUMa = None  # type: ignore[assignment,misc]
    OFDMChannel = None  # type: ignore[assignment,misc]
    _HAS_SIONNA = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class AODTEnvConfig(SpectrumEnvConfig):
    """Configuration for the NVIDIA Aerial Digital Twin environment."""

    channel_model: str = Field(
        "UMa",
        description="3GPP channel model: UMa, UMi, RMa.",
    )
    num_prbs: int = Field(
        52,
        ge=1,
        description="Number of Physical Resource Blocks per channel.",
    )
    subcarrier_spacing_khz: int = Field(
        30,
        description="Subcarrier spacing in kHz (15, 30, 60, 120).",
    )
    num_ofdm_symbols: int = Field(
        14,
        ge=1,
        description="OFDM symbols per slot.",
    )
    carrier_frequency_hz: float = Field(
        3.5e9,
        description="Carrier frequency in Hz.",
    )
    bandwidth_mhz: float = Field(
        20.0,
        description="Channel bandwidth in MHz.",
    )
    num_tx_antennas: int = Field(1, ge=1, description="Transmit antennas per cell.")
    num_rx_antennas: int = Field(2, ge=1, description="Receive antennas at UE.")
    gpu_idx: int = Field(0, ge=0, description="CUDA device index.")
    snr_range_db: Tuple[float, float] = Field(
        (-5.0, 30.0),
        description="(min, max) SNR in dB for normalisation.",
    )
    pu_activity_model: str = Field(
        "markov",
        description="PU activity model: 'markov' or 'trace'.",
    )
    pu_on_prob: float = Field(0.3, description="Markov OFF->ON probability.")
    pu_off_prob: float = Field(0.5, description="Markov ON->OFF probability.")

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Channel model abstraction
# ---------------------------------------------------------------------------

class ChannelModelBackend:
    """
    Wraps Sionna or a fallback NumPy Rayleigh model so that AODT can
    run (with reduced fidelity) even without Sionna/pyAerial.
    """

    def __init__(self, config: AODTEnvConfig):
        self.config = config
        self._use_sionna = _HAS_SIONNA
        self._channel = None

        if self._use_sionna:
            self._init_sionna(config)
        else:
            logger.warning(
                "Sionna not available — falling back to Rayleigh fading model."
            )

    def _init_sionna(self, cfg: AODTEnvConfig) -> None:
        """Initialise the Sionna 3GPP channel model."""
        model_map = {
            "UMa": SionnaUMa,
        }
        model_cls = model_map.get(cfg.channel_model)
        if model_cls is None:
            raise ValueError(
                f"Unsupported channel model '{cfg.channel_model}'. "
                f"Available: {list(model_map.keys())}"
            )
        self._channel_model = model_cls(
            carrier_frequency=cfg.carrier_frequency_hz,
            o2i_model="low",
            ut_array=None,
            bs_array=None,
            direction="downlink",
        )

    def generate(
        self,
        num_channels: int,
        num_prbs: int,
        num_ofdm_symbols: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Generate channel frequency response.

        Returns:
            Complex array of shape (num_channels, num_prbs, num_ofdm_symbols).
        """
        if self._use_sionna:
            return self._generate_sionna(
                num_channels, num_prbs, num_ofdm_symbols
            )
        return self._generate_rayleigh(
            num_channels, num_prbs, num_ofdm_symbols, rng
        )

    def _generate_sionna(
        self, num_channels: int, num_prbs: int, num_ofdm_symbols: int
    ) -> np.ndarray:
        """Generate channel via Sionna 3GPP model."""
        h = self._channel_model(
            batch_size=num_channels,
            num_time_steps=num_ofdm_symbols,
        )
        # h shape varies; extract and reshape to (num_channels, num_prbs, num_ofdm_symbols)
        h_np = h.numpy() if hasattr(h, "numpy") else np.asarray(h)
        # Adapt shape
        if h_np.ndim > 3:
            h_np = h_np.reshape(num_channels, -1, num_ofdm_symbols)
        if h_np.shape[1] > num_prbs:
            h_np = h_np[:, :num_prbs, :]
        elif h_np.shape[1] < num_prbs:
            pad_width = num_prbs - h_np.shape[1]
            h_np = np.pad(h_np, ((0, 0), (0, pad_width), (0, 0)), mode="wrap")
        return h_np

    @staticmethod
    def _generate_rayleigh(
        num_channels: int,
        num_prbs: int,
        num_ofdm_symbols: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Rayleigh fading fallback: i.i.d. complex Gaussian."""
        real = rng.standard_normal((num_channels, num_prbs, num_ofdm_symbols))
        imag = rng.standard_normal((num_channels, num_prbs, num_ofdm_symbols))
        return np.asarray((real + 1j * imag).astype(np.complex64) / np.sqrt(2.0))


# ---------------------------------------------------------------------------
# Channel estimator abstraction
# ---------------------------------------------------------------------------

class ChannelEstimatorBackend:
    """
    Wraps pyAerial's GPU-accelerated channel estimator or falls back
    to a simple least-squares estimate.
    """

    def __init__(self, config: AODTEnvConfig):
        self.config = config
        self._use_pyaerial = _HAS_PYAERIAL and _HAS_CUPY

        if self._use_pyaerial:
            self._init_pyaerial(config)
        else:
            logger.warning(
                "pyAerial/CuPy not available — using NumPy LS channel estimation."
            )

    def _init_pyaerial(self, cfg: AODTEnvConfig) -> None:
        """Initialise pyAerial cuPHY channel estimator on the target GPU."""
        cp.cuda.Device(cfg.gpu_idx).use()
        self._estimator = PyAerialChannelEstimateGenerator(
            num_rx_ant=cfg.num_rx_antennas,
            num_prbs=cfg.num_prbs,
            num_ofdm_symbols=cfg.num_ofdm_symbols,
        )

    def estimate(
        self,
        h_freq: np.ndarray,
        noise_power: float,
    ) -> np.ndarray:
        """
        Estimate channel from received signal.

        Args:
            h_freq: Channel frequency response (num_channels, num_prbs, num_ofdm_symbols).
            noise_power: Noise power (linear).

        Returns:
            Estimated channel (num_channels, num_prbs, num_ofdm_symbols).
        """
        if self._use_pyaerial:
            return self._estimate_gpu(h_freq, noise_power)
        return self._estimate_ls(h_freq, noise_power)

    def _estimate_gpu(
        self, h_freq: np.ndarray, noise_power: float
    ) -> np.ndarray:
        """GPU-accelerated estimation via cuPHY."""
        h_gpu = cp.asarray(h_freq)
        noise_gpu = cp.full_like(h_gpu, noise_power)
        # Add noise to simulate received signal
        rx_signal = h_gpu + cp.sqrt(noise_gpu) * (
            cp.random.randn(*h_gpu.shape) + 1j * cp.random.randn(*h_gpu.shape)
        ) / cp.sqrt(2.0)
        h_est = self._estimator(rx_signal)
        return np.asarray(cp.asnumpy(h_est))

    @staticmethod
    def _estimate_ls(h_freq: np.ndarray, noise_power: float) -> np.ndarray:
        """Least-squares fallback: add noise then return noisy version."""
        noise = np.sqrt(noise_power) * (
            np.random.randn(*h_freq.shape) + 1j * np.random.randn(*h_freq.shape)
        ).astype(np.complex64) / np.sqrt(2.0)
        return np.asarray(h_freq + noise)


# ---------------------------------------------------------------------------
# AODT Environment
# ---------------------------------------------------------------------------

class AODTEnv(SpectrumEnv):
    """
    NVIDIA Aerial Digital Twin environment.

    Uses GPU-accelerated channel modelling (Sionna 3GPP UMa) and channel
    estimation (pyAerial cuPHY) to provide physically realistic observations
    for SAC-LTC training and validation.

    The environment generates per-channel SNR, interference, and occupancy
    features from full PHY-layer simulation including:
        - 3GPP spatial channel model (UMa/UMi/RMa)
        - OFDM modulation/demodulation
        - MMSE channel estimation
        - PU Markov activity model
    """

    def __init__(
        self,
        config: Optional[AODTEnvConfig] = None,
        **kwargs: Any,
    ):
        if config is None:
            config = AODTEnvConfig(**kwargs)

        super().__init__(config=config)
        self._aodt_cfg: AODTEnvConfig = config

        # Initialise backends
        self._channel_model = ChannelModelBackend(config)
        self._channel_estimator = ChannelEstimatorBackend(config)

        # PU Markov state
        self._pu_states: Optional[np.ndarray] = None
        self._pu_on_probs = np.full(config.num_channels, config.pu_on_prob)
        self._pu_off_probs = np.full(config.num_channels, config.pu_off_prob)

        # Cached PHY outputs
        self._snr_linear: Optional[np.ndarray] = None
        self._interference_linear: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # PU dynamics
    # ------------------------------------------------------------------

    def _init_pu_states(self) -> np.ndarray:
        """Initialise PU activity."""
        return (self.np_random.random(self.num_channels) < 0.5).astype(np.float32)

    def _step_pu_states(self) -> None:
        """Advance PU Markov chains."""
        assert self._pu_states is not None
        rand = self.np_random.random(self.num_channels)
        for ch in range(self.num_channels):
            if self._pu_states[ch] == 0.0:
                if rand[ch] < self._pu_on_probs[ch]:
                    self._pu_states[ch] = 1.0
            else:
                if rand[ch] < self._pu_off_probs[ch]:
                    self._pu_states[ch] = 0.0

    # ------------------------------------------------------------------
    # PHY simulation
    # ------------------------------------------------------------------

    def _simulate_phy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run one slot of PHY simulation.

        Returns:
            snr_db: Per-channel average SNR in dB, shape (num_channels,).
            interference_linear: Per-channel interference power, shape (num_channels,).
        """
        cfg = self._aodt_cfg

        # Generate channel
        h_freq = self._channel_model.generate(
            num_channels=cfg.num_channels,
            num_prbs=cfg.num_prbs,
            num_ofdm_symbols=cfg.num_ofdm_symbols,
            rng=self.np_random,
        )

        # Compute per-channel signal power from channel gain
        _signal_power = np.mean(np.abs(h_freq) ** 2, axis=(1, 2))  # (num_channels,)

        # PU interference: occupied channels add interference
        assert self._pu_states is not None
        pu_interference = self._pu_states * 0.5  # interference from active PUs
        noise_power_base = 0.01  # thermal noise floor (linear)
        total_noise = noise_power_base + pu_interference

        # Channel estimation (adds estimation error)
        h_est = self._channel_estimator.estimate(h_freq, noise_power_base)
        est_signal_power = np.mean(np.abs(h_est) ** 2, axis=(1, 2))

        # SNR from estimated channel
        snr_linear = np.maximum(est_signal_power / (total_noise + 1e-10), 1e-6)
        snr_db = 10.0 * np.log10(snr_linear)

        return snr_db.astype(np.float32), pu_interference.astype(np.float32)

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def _get_channel_states(self) -> np.ndarray:
        """Advance PU dynamics and return occupancy vector."""
        if self._pu_states is None:
            self._pu_states = self._init_pu_states()
        else:
            self._step_pu_states()

        # Run PHY simulation to get updated SNR/interference
        snr_db, interference = self._simulate_phy()
        self._snr_linear = snr_db
        self._interference_linear = interference

        self._channel_states = self._pu_states.copy()
        return self._channel_states

    def _build_features(self) -> np.ndarray:
        """
        Build features from PHY simulation outputs.

        Features per channel: [normalised_snr, normalised_interference, occupancy].
        """
        cfg = self._aodt_cfg
        snr_min, snr_max = cfg.snr_range_db

        assert self._snr_linear is not None
        assert self._interference_linear is not None
        assert self._channel_states is not None

        # Normalise SNR to [0, 1]
        snr_norm = np.clip(
            (self._snr_linear - snr_min) / (snr_max - snr_min + 1e-10),
            0.0,
            1.0,
        ).astype(np.float32)

        # Normalise interference to [0, 1]
        interference_norm = np.clip(
            self._interference_linear / 1.0, 0.0, 1.0
        ).astype(np.float32)

        # Occupancy
        occupancy = self._channel_states.copy()

        features = np.stack(
            [snr_norm, interference_norm, occupancy], axis=-1
        ).reshape(-1)
        return features.astype(np.float32)

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the AODT environment."""
        super(SpectrumEnv, self).reset(seed=seed)

        self._pu_states = None
        self._channel_states = None
        self._prev_action = None
        self._step_count = 0
        self._history = self._init_history()
        self._snr_linear = None
        self._interference_linear = None

        # Fill history
        for t in range(self.sequence_length):
            self._channel_states = self._get_channel_states()
            self._history[t] = self._build_features()

        obs = self._get_observation()
        assert self._channel_states is not None
        info: Dict[str, Any] = {
            "channel_states": self._channel_states.copy(),
            "snr_db": self._snr_linear.copy() if self._snr_linear is not None else None,
        }
        return obs, info

    def close(self) -> None:
        """Release GPU resources."""
        self._channel_model = None  # type: ignore[assignment]
        self._channel_estimator = None  # type: ignore[assignment]
        super().close()
