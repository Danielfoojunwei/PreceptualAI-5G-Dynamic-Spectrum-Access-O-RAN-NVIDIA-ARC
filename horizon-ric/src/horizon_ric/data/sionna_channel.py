"""Sionna channel-model integration for NVIDIA-native NTN+AI-RAN training.

This module wraps NVIDIA's Sionna link-level simulator
(https://github.com/nvlabs/sionna) to generate 3GPP TR 38.901 / TR 38.811
TDL channel realisations for the world model and the constraint layer.

REAL DEPENDENCY ONLY. Sionna is a hard requirement of this module: if it
cannot be imported the constructor raises ImportError with install
instructions. There is no numpy fallback or stand-in path.

Sionna 2.0 ships TDL-A..E from TR 38.901 in
``sionna.phy.channel.tr38901.TDL``. The TR 38.811 NTN-TDL-* profiles are
spec extensions of the same TDL framework; the SionnaChannelGenerator
maps the NTN-TDL-A/B/C/D requests to the real TR 38.901 TDL-A/B/C/D
backbones at the spec'd RMS delay spread for the band/elevation, which
captures the relevant fading statistics with the real Sionna PyTorch
implementation. If a future Sionna release ships a dedicated
``sionna.channel.tr38811.TDL``, ``_resolve_tdl_class`` picks it up
automatically.

Surface:

    generate_channel(scenario, ue_speed_kmh, frequency_hz, n_samples)
        -> np.ndarray (n_samples, n_rx, n_tx, n_paths) complex64

    to_telemetry_events(channels, t0_utc, source_id)
        -> Iterator[TelemetryEvent]   (modality="kpm_ntn")

`to_telemetry_events` deliberately serialises **statistics** (mean tap
power, RMS delay spread, K-factor) rather than the raw complex tensor —
the bus would otherwise be flooded.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import Any, Literal

import numpy as np

from horizon_ric.io.schemas import TelemetryEvent
from horizon_ric.planner.physics.tr38811 import channel_state

logger = logging.getLogger(__name__)


_SIONNA_INSTALL_HINT = (
    "Sionna is not installed. Install it with:\n"
    "    pip install sionna\n"
    "Sionna 2.0+ uses PyTorch as its backend (no TensorFlow). On aarch64 "
    "GB10 / Jetson hosts the install pulls in mitsuba + drjit and is "
    "~600 MB. NO STAND-IN BACKEND IS PROVIDED — this module requires real "
    "Sionna."
)

try:  # pragma: no cover - exercised at import time
    import sionna  # noqa: F401  type: ignore[import-not-found]

    _SIONNA_IMPORT_ERROR: Exception | None = None
except Exception as _exc:  # pragma: no cover - depends on host
    _SIONNA_IMPORT_ERROR = _exc


# Scenarios from 3GPP TR 38.811 Annex C (NTN-TDL profiles).
NTNScenario = Literal[
    "NTN-TDL-A",
    "NTN-TDL-B",
    "NTN-TDL-C",
    "NTN-TDL-D",
]

# Map NTN-TDL-* requests onto the real TR 38.901 TDL-* models that
# Sionna ships. The TR 38.811 NTN-TDL profiles are TR 38.901 TDL
# extensions: A/B are NLOS, C/D are LOS Rician.
_NTN_TO_TDL_MODEL: dict[str, str] = {
    "NTN-TDL-A": "A",
    "NTN-TDL-B": "B",
    "NTN-TDL-C": "C",
    "NTN-TDL-D": "D",
}


def _resolve_tdl_class() -> Any:
    """Return the real Sionna TDL class.

    Prefers ``sionna.channel.tr38811.TDL`` if the installed version
    ships TR 38.811 NTN profiles; otherwise uses the canonical
    ``sionna.phy.channel.tr38901.TDL`` (Sionna 2.x layout) or the
    legacy ``sionna.channel.tr38901.TDL`` (Sionna 0.x layout).
    """
    if _SIONNA_IMPORT_ERROR is not None:
        raise ImportError(_SIONNA_INSTALL_HINT) from _SIONNA_IMPORT_ERROR

    # Try TR 38.811 first (newest Sionna releases may ship it).
    try:
        from sionna.channel.tr38811 import TDL as _NTN_TDL  # type: ignore[import-not-found]

        return _NTN_TDL, "tr38811"
    except Exception:
        pass

    # Sionna 2.x canonical PyTorch path.
    try:
        from sionna.phy.channel.tr38901 import TDL as _TDL2  # type: ignore[import-not-found]

        return _TDL2, "tr38901_phy"
    except Exception:
        pass

    # Sionna 0.x TensorFlow legacy path.
    from sionna.channel.tr38901 import TDL as _TDL0  # type: ignore[import-not-found]

    return _TDL0, "tr38901"


def _channel_stats(channels: np.ndarray) -> dict[str, float]:
    """Compute serialisable summary stats from a (S, Rx, Tx, P) tensor."""
    pwr = np.mean(np.abs(channels) ** 2, axis=(0, 1, 2))  # (P,)
    total = float(np.sum(pwr))
    if total <= 0.0:
        return {
            "mean_power": 0.0,
            "rms_delay_spread_taps": 0.0,
            "k_factor_db": float("-inf"),
            "n_paths": int(channels.shape[-1]),
        }
    norm = pwr / total
    taps = np.arange(channels.shape[-1])
    mean_tap = float(np.sum(taps * norm))
    rms = float(np.sqrt(np.sum(((taps - mean_tap) ** 2) * norm)))
    peak = float(np.max(pwr))
    rest = total - peak
    k_db = 10.0 * np.log10(peak / rest) if rest > 0 else float("inf")
    return {
        "mean_power": float(np.mean(np.abs(channels) ** 2)),
        "rms_delay_spread_taps": rms,
        "k_factor_db": float(k_db),
        "n_paths": int(channels.shape[-1]),
    }


class SionnaChannelGenerator:
    """Public façade for Sionna NTN-TDL channel generation.

    Always uses real Sionna. If Sionna is not importable, the constructor
    raises ImportError with install instructions. There is no fallback.
    """

    SUPPORTED_SCENARIOS: tuple[str, ...] = tuple(_NTN_TO_TDL_MODEL)

    def __init__(
        self,
        seed: int = 0,
        n_rx: int = 1,
        n_tx: int = 1,
        # Kept for backward-compatible call sites; both are now ignored
        # because the only backend is real Sionna.
        use_standin: bool = False,  # noqa: ARG002 — kept for API compat
        require_sionna: bool = True,  # noqa: ARG002 — kept for API compat
    ) -> None:
        if _SIONNA_IMPORT_ERROR is not None:
            raise ImportError(_SIONNA_INSTALL_HINT) from _SIONNA_IMPORT_ERROR

        self._tdl_cls, self._tdl_layout = _resolve_tdl_class()
        self.seed = int(seed)
        self.n_rx = int(n_rx)
        self.n_tx = int(n_tx)
        self.backend_name = "sionna"

        # Seed Sionna's RNG (PyTorch backend in Sionna 2.x, TF in 0.x).
        try:
            import torch

            torch.manual_seed(self.seed)
        except Exception:  # pragma: no cover
            pass
        try:
            import tensorflow as tf  # type: ignore[import-not-found]

            tf.random.set_seed(self.seed)
        except Exception:
            pass

        self._tdl_cache: dict[tuple[str, float, float], Any] = {}

    # ── Backend lookup ──────────────────────────────────────────────────
    def _get_tdl(
        self, scenario: str, frequency_hz: float, ds_ns: float, v_mps: float
    ) -> Any:
        model = _NTN_TO_TDL_MODEL[scenario]
        # Sionna's TDL is parameterised; we must rebuild when speed changes
        # because min/max_speed are constructor args in the PyTorch layout.
        key = (model, float(frequency_hz), float(ds_ns), float(v_mps))
        if key in self._tdl_cache:
            return self._tdl_cache[key]

        kwargs: dict[str, Any] = dict(
            model=model,
            delay_spread=float(ds_ns) * 1e-9,
            carrier_frequency=float(frequency_hz),
            num_rx_ant=self.n_rx,
            num_tx_ant=self.n_tx,
            min_speed=float(v_mps),
            max_speed=float(v_mps),
        )
        tdl = self._tdl_cls(**kwargs)
        self._tdl_cache[key] = tdl
        return tdl

    # ── Public API ──────────────────────────────────────────────────────

    def generate_channel(
        self,
        scenario: str,
        ue_speed_kmh: float,
        frequency_hz: float,
        n_samples: int,
        elevation_deg: float = 30.0,
        environment: str = "rural",
        **_: Any,
    ) -> np.ndarray:
        if scenario not in self.SUPPORTED_SCENARIOS:
            raise ValueError(
                f"Unknown scenario {scenario!r}; expected one of "
                f"{list(self.SUPPORTED_SCENARIOS)}."
            )

        # Pull the spec'd RMS delay spread for the band/elev when possible;
        # otherwise use a sensible default per profile.
        try:
            cs = channel_state(environment, frequency_hz, elevation_deg)
            ds_ns = float(cs.delay_spread_ns)
        except ValueError:
            ds_ns = 100.0  # spec default for S-band

        v_mps = float(ue_speed_kmh) / 3.6
        tdl = self._get_tdl(scenario, frequency_hz, ds_ns, v_mps)

        # Sionna's TDL call:
        #   h, tau = tdl(batch_size, num_time_steps, sampling_frequency)
        # h has shape (batch, n_rx, n_rx_ant, n_tx, n_tx_ant, n_paths, n_steps).
        # We want (n_samples, n_rx, n_tx, n_paths). We treat n_samples as
        # the time dimension (1 ms slot, 30 kHz SCS) and use a single TX/RX
        # link.
        sampling_frequency = 1e3  # 1 ms inter-sample period (30 kHz SCS slot)
        h, _tau = tdl(
            batch_size=1,
            num_time_steps=int(n_samples),
            sampling_frequency=sampling_frequency,
        )

        # Sionna 2.x returns torch.Tensor; 0.x returns tf.Tensor.
        try:
            arr = h.detach().cpu().numpy()
        except AttributeError:
            arr = h.numpy()

        # Sionna 2.x TDL returns a 7-D tensor:
        # (batch, n_rx_link, n_rx_ant, n_tx_link, n_tx_ant, n_paths, n_steps).
        # We always call with batch=1 and let n_rx_link / n_tx_link default to
        # 1, so collapse those three singleton dims and surface
        # (n_steps, n_rx_ant, n_tx_ant, n_paths).
        if arr.ndim == 7:
            # batch=axis0=1, rx_link=axis1=1, tx_link=axis3=1.
            arr = arr[0, 0, :, 0, :, :, :]  # (n_rx_ant, n_tx_ant, n_paths, n_steps)
            arr = np.transpose(arr, (3, 0, 1, 2))  # (n_steps, n_rx_ant, n_tx_ant, n_paths)
        elif arr.ndim == 6:
            # Older TF layout: (batch, n_rx_ant, n_tx_link, n_tx_ant, n_paths, n_steps).
            arr = arr[0]
            if arr.shape[1] == 1:
                arr = arr[:, 0, :, :, :]
            arr = np.transpose(arr, (3, 0, 1, 2))
        else:  # pragma: no cover - defensive
            # Generic fallback: assume the last axis is time.
            arr = np.moveaxis(arr, -1, 0)

        return arr.astype(np.complex64, copy=False)

    def to_telemetry_events(
        self,
        channels: np.ndarray,
        t0_utc: datetime,
        source_id: str,
        scenario: str = "NTN-TDL-A",
        frequency_hz: float = 2.0e9,
    ) -> Iterator[TelemetryEvent]:
        if t0_utc.tzinfo is None:
            raise ValueError("t0_utc must be timezone-aware UTC")
        n = int(channels.shape[0])
        for i in range(n):
            stats = _channel_stats(channels[i : i + 1])
            ts = t0_utc + timedelta(milliseconds=i)
            yield TelemetryEvent(
                event_id=str(uuid.uuid4()),
                modality="kpm_ntn",
                source_id=source_id,
                ts_utc=ts,
                sequence=i,
                payload={
                    "scenario": scenario,
                    "frequency_hz": float(frequency_hz),
                    "backend": self.backend_name,
                    **stats,
                },
                tags={"channel_model": f"sionna_{self._tdl_layout}"},
            )


__all__ = [
    "NTNScenario",
    "SionnaChannelGenerator",
]
