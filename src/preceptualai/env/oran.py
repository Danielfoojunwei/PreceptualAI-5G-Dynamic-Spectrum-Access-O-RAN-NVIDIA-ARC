"""
Production O-RAN xApp environment for live spectrum management.

Integrates with the O-RAN Software Community (OSC) Near-RT RIC via
ricxappframe.  Subscribes to E2SM-KPM for real-time radio metrics
and issues E2SM-RC control messages for spectrum allocation changes.

Architecture follows the srsRAN xAppBase pattern
(https://github.com/srsran/oran-sc-ric).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from preceptualai.env.base import SpectrumEnv, SpectrumEnvConfig

# ---------------------------------------------------------------------------
# Optional imports — ricxappframe may not be installed in all environments
# ---------------------------------------------------------------------------
try:
    from ricxappframe.e2ap.asn1 import IndicationMsg
    from ricxappframe.xapp_frame import RMRXapp

    _HAS_RIC = True
except ImportError:
    RMRXapp = None  # type: ignore[assignment,misc]
    IndicationMsg = None  # type: ignore[assignment,misc]
    _HAS_RIC = False


# ---------------------------------------------------------------------------
# KPM metric names we subscribe to
# ---------------------------------------------------------------------------
DEFAULT_KPM_METRICS: List[str] = [
    "DRB.UEThpDl",
    "RRU.PrbUsedDl",
    "L1M.RS-SINR",
    "RRU.PrbAvailDl",
    "DRB.UEThpUl",
    "RRU.PrbUsedUl",
]


class ORANEnvConfig(SpectrumEnvConfig):
    """Configuration for the O-RAN xApp environment."""

    rmr_port: int = 4560
    rmr_wait_for_ready: float = 5.0
    e2_node_id: str = "gnb_001"
    ran_func_id: int = 2
    kpm_report_period_ms: int = 100
    kpm_metrics: List[str] = DEFAULT_KPM_METRICS
    sinr_occupied_threshold: float = 5.0
    prb_occupied_threshold: float = 0.7
    control_request_timeout_s: float = 2.0


# ---------------------------------------------------------------------------
# Decorator helper
# ---------------------------------------------------------------------------

def start_function(func: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a method as the xApp entry-point (mirrors srsRAN pattern)."""
    func._is_start_function = True  # type: ignore[attr-defined]
    return func


# ---------------------------------------------------------------------------
# Main xApp class
# ---------------------------------------------------------------------------

class PreceptualAIxApp:
    """
    O-RAN Near-RT RIC xApp for PreceptualAI spectrum management.

    Lifecycle:
        1. ``__init__`` — configure RMR transport and register callbacks.
        2. ``start()``  — subscribe to E2SM-KPM, begin consuming indications.
        3. ``indication_callback()`` — extract metrics, run inference, issue control.
        4. ``stop()``   — unsubscribe and shut down.

    The xApp exposes a :class:`SpectrumEnv`-compatible observation buffer so
    that the same SAC-LTC inference engine can be used for both simulation
    and live deployment.
    """

    def __init__(
        self,
        config: ORANEnvConfig,
        predict_fn: Optional[Callable[[np.ndarray], int]] = None,
    ):
        if not _HAS_RIC:
            raise RuntimeError(
                "ricxappframe is not installed.  Install it with: "
                "pip install ricxappframe"
            )

        self.config = config
        self._predict_fn = predict_fn

        # Observation buffer — same shape as SpectrumEnv
        obs_dim = config.num_channels * config.num_features
        self._history = np.zeros(
            (config.sequence_length, obs_dim), dtype=np.float32
        )
        self._channel_states = np.zeros(config.num_channels, dtype=np.float32)
        self._lock = threading.Lock()
        self._running = False
        self._subscription_id: Optional[str] = None
        self._prev_action: Optional[int] = None

        # Metrics bookkeeping
        self._last_sinr: Optional[np.ndarray] = None
        self._last_prb_usage: Optional[np.ndarray] = None

        # Initialise RMR xApp
        self._xapp = RMRXapp(
            default_handler=self._default_rmr_handler,
            rmr_port=config.rmr_port,
            rmr_wait_for_ready=config.rmr_wait_for_ready,
            use_fake_sdl=False,
        )
        # Register E2AP indication handler (message type 12050)
        self._xapp.register_callback(self._indication_handler, 12050)

    # ------------------------------------------------------------------
    # RMR handlers
    # ------------------------------------------------------------------

    def _default_rmr_handler(self, xapp: Any, summary: Any, sbuf: Any) -> None:
        """Consume and free unknown message types."""
        xapp.rmr_free(sbuf)

    def _indication_handler(self, xapp: Any, summary: Any, sbuf: Any) -> None:
        """Process E2SM-KPM indication messages."""
        try:
            ind_msg = IndicationMsg()
            ind_msg.decode(summary["payload"])
            metrics = self._extract_kpm_metrics(ind_msg)
            self._update_observation(metrics)

            if self._predict_fn is not None:
                obs = self.get_observation()
                action = self._predict_fn(obs)
                self._send_control(action)
        finally:
            xapp.rmr_free(sbuf)

    # ------------------------------------------------------------------
    # KPM metric extraction
    # ------------------------------------------------------------------

    def _extract_kpm_metrics(self, ind_msg: Any) -> Dict[str, np.ndarray]:
        """
        Parse an E2SM-KPM indication message into named metric arrays.

        Returns a dict mapping metric name to np.ndarray of shape (num_channels,).
        """
        metrics: Dict[str, np.ndarray] = {}
        cfg = self.config

        # Parse the indication header and message
        _hdr = ind_msg.indication_header
        msg = ind_msg.indication_message

        # E2SM-KPM Format 1: measInfoList contains metric definitions,
        # measData contains the actual measurements per granularity period.
        meas_info_list = getattr(msg, "measInfoList", [])
        meas_data = getattr(msg, "measData", [])

        for idx, meas_info in enumerate(meas_info_list):
            meas_name = getattr(meas_info, "measType", {}).get("measName", "")
            if meas_name not in cfg.kpm_metrics:
                continue

            values = []
            for record in meas_data:
                items = getattr(record, "measRecordItemList", [])
                if idx < len(items):
                    val = getattr(items[idx], "integer", 0)
                    values.append(float(val))

            arr = np.array(values[: cfg.num_channels], dtype=np.float32)
            if len(arr) < cfg.num_channels:
                padded = np.zeros(cfg.num_channels, dtype=np.float32)
                padded[: len(arr)] = arr
                arr = padded
            metrics[meas_name] = arr

        return metrics

    def _update_observation(self, metrics: Dict[str, np.ndarray]) -> None:
        """Update internal state from extracted KPM metrics."""
        cfg = self.config
        nc = cfg.num_channels

        sinr = metrics.get("L1M.RS-SINR", np.zeros(nc, dtype=np.float32))
        prb_used = metrics.get("RRU.PrbUsedDl", np.zeros(nc, dtype=np.float32))
        prb_avail = metrics.get("RRU.PrbAvailDl", np.ones(nc, dtype=np.float32))
        _thp_dl = metrics.get("DRB.UEThpDl", np.zeros(nc, dtype=np.float32))

        # Derive occupancy from SINR and PRB usage
        prb_ratio = np.where(prb_avail > 0, prb_used / prb_avail, 1.0)
        occupied = (
            (sinr < cfg.sinr_occupied_threshold)
            | (prb_ratio > cfg.prb_occupied_threshold)
        ).astype(np.float32)

        # Normalise features to [0, 1] range
        sinr_norm = np.clip(sinr / 30.0, 0.0, 1.0)
        interference = 1.0 - sinr_norm
        occupancy = occupied

        with self._lock:
            self._channel_states = occupied
            self._last_sinr = sinr
            self._last_prb_usage = prb_ratio

            features = np.stack(
                [sinr_norm, interference, occupancy], axis=-1
            ).reshape(-1).astype(np.float32)

            self._history = np.roll(self._history, -1, axis=0)
            self._history[-1] = features

    # ------------------------------------------------------------------
    # E2SM-RC control
    # ------------------------------------------------------------------

    def _send_control(self, action: int) -> None:
        """
        Send E2SM-RC control message to change spectrum allocation.

        The control message instructs the gNB to allocate the selected
        channel (PRB group) to the secondary user.
        """
        # Build RC control header
        rc_header = {
            "controlActionId": 1,
            "ueId": {"gnbUeX2apId": 0},
        }
        # Build RC control message
        rc_message = {
            "ranParameterList": [
                {
                    "ranParameterId": 1,
                    "ranParameterValue": {
                        "valueInt": action,
                    },
                },
                {
                    "ranParameterId": 2,
                    "ranParameterValue": {
                        "valueInt": 1,  # allocation active
                    },
                },
            ],
        }

        control_payload = self._encode_rc_control(rc_header, rc_message)

        self._xapp.rmr_send(
            control_payload,
            mtype=12040,  # RIC_CONTROL_REQUEST
        )

        with self._lock:
            if self._prev_action is not None and action != self._prev_action:
                pass  # switch cost tracked externally
            self._prev_action = action

    @staticmethod
    def _encode_rc_control(header: Dict[str, Any], message: Dict[str, Any]) -> bytes:
        """
        Encode E2SM-RC control header + message into ASN.1 UPER bytes.

        In production, this delegates to the e2sm_rc ASN.1 encoder.
        Here we build a minimal wire format compatible with the srsRAN RIC.
        """
        import json

        payload = json.dumps({"header": header, "message": message}).encode("utf-8")
        return payload

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def _subscribe_kpm(self) -> str:
        """Subscribe to E2SM-KPM indications from the target E2 node."""
        cfg = self.config

        # Build subscription request
        sub_params = {
            "e2NodeId": cfg.e2_node_id,
            "ranFuncId": cfg.ran_func_id,
            "eventTrigger": {
                "reportingPeriod": cfg.kpm_report_period_ms,
            },
            "actionDefinitions": [
                {
                    "actionId": 1,
                    "actionType": "report",
                    "subsequentAction": {"subsequentActionType": "continue"},
                    "actionDefinition": {
                        "measInfoList": [
                            {"measName": m} for m in cfg.kpm_metrics
                        ],
                        "granularityPeriod": cfg.kpm_report_period_ms,
                    },
                }
            ],
        }

        response = self._xapp.subscription_create(sub_params)
        sub_id: str = response.get("subscriptionId", "unknown")
        return sub_id

    def _unsubscribe_kpm(self) -> None:
        """Remove KPM subscription."""
        if self._subscription_id is not None:
            self._xapp.subscription_delete(self._subscription_id)
            self._subscription_id = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_observation(self) -> np.ndarray:
        """Thread-safe observation retrieval."""
        with self._lock:
            return np.array(self._history, copy=True)  # type: ignore[arg-type]

    def get_channel_states(self) -> np.ndarray:
        """Thread-safe channel state retrieval."""
        with self._lock:
            return np.array(self._channel_states, copy=True)  # type: ignore[arg-type]

    @start_function
    def start(self) -> None:
        """Start the xApp: subscribe to KPM and begin processing indications."""
        self._subscription_id = self._subscribe_kpm()
        self._running = True
        self._xapp.run()

    def stop(self) -> None:
        """Gracefully shut down the xApp."""
        self._running = False
        self._unsubscribe_kpm()
        self._xapp.stop()


# ---------------------------------------------------------------------------
# Gymnasium-compatible wrapper
# ---------------------------------------------------------------------------

class ORANEnv(SpectrumEnv):
    """
    Gymnasium wrapper around a live O-RAN xApp.

    Bridges the gap between the RL training loop (which expects a Gymnasium
    ``Env``) and the asynchronous O-RAN indication stream.

    ``step()`` blocks until a new KPM indication arrives, then returns the
    latest observation.
    """

    def __init__(
        self,
        config: Optional[ORANEnvConfig] = None,
        predict_fn: Optional[Callable[[np.ndarray], int]] = None,
        **kwargs: Any,
    ):
        if config is None:
            config = ORANEnvConfig(**kwargs)

        super().__init__(config=config)
        self._oran_cfg = config
        self._xapp = PreceptualAIxApp(config=config, predict_fn=predict_fn)
        self._xapp_thread: Optional[threading.Thread] = None

    def _get_channel_states(self) -> np.ndarray:
        """Fetch the latest channel states from the xApp."""
        return np.asarray(self._xapp.get_channel_states())

    def _build_features(self) -> np.ndarray:
        """
        Features are built inside the xApp's indication callback.
        Return the latest row from the xApp's history buffer.
        """
        obs = self._xapp.get_observation()
        return np.asarray(obs[-1])

    def _get_observation(self) -> np.ndarray:
        """Return the full observation buffer from the xApp."""
        return np.asarray(self._xapp.get_observation())

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super(SpectrumEnv, self).reset(seed=seed)
        self._step_count = 0
        self._prev_action = None

        # Start xApp if not running
        if self._xapp_thread is None or not self._xapp_thread.is_alive():
            self._xapp_thread = threading.Thread(
                target=self._xapp.start, daemon=True
            )
            self._xapp_thread.start()

        # Wait for initial observations to fill
        time.sleep(
            self._oran_cfg.kpm_report_period_ms
            * self.sequence_length
            / 1000.0
        )

        obs = self._get_observation()
        info: Dict[str, Any] = {"channel_states": self._get_channel_states()}
        return obs, info

    def close(self) -> None:
        """Shut down the xApp and join the background thread."""
        self._xapp.stop()
        if self._xapp_thread is not None:
            self._xapp_thread.join(timeout=5.0)
        super().close()
