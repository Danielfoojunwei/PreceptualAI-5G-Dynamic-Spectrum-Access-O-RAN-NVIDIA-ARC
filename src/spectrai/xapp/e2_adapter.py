"""
Production E2 interface adapter for the O-RAN Near-RT RIC.

Wraps E2SM-KPM (monitoring) and E2SM-RC (control) service model
interactions following the srsRAN e2sm module patterns.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional RIC imports
# ---------------------------------------------------------------------------
try:
    from ricxappframe.xapp_frame import RMRXapp

    _HAS_RIC = True
except ImportError:
    RMRXapp = None  # type: ignore[assignment,misc]
    _HAS_RIC = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class KPMMetrics:
    """Container for a single KPM indication report."""

    timestamp_ms: int = 0
    e2_node_id: str = ""
    metrics: Dict[str, np.ndarray] = field(default_factory=dict)
    raw_payload: Optional[bytes] = None


@dataclass
class RCControlRequest:
    """Container for an E2SM-RC control action."""

    action_id: int = 1
    target_channel: int = 0
    allocation_active: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# E2 Adapter
# ---------------------------------------------------------------------------

class E2Adapter:
    """
    Adapter for E2SM-KPM and E2SM-RC interactions with the Near-RT RIC.

    Provides three main operations:
        1. ``subscribe_kpm()``   — establish a KPM subscription for periodic
           radio metric reports.
        2. ``extract_metrics()`` — parse the latest KPM indication into
           structured metric arrays.
        3. ``send_rc_control()`` — issue an E2SM-RC control request to change
           spectrum allocation at the gNB.

    Thread-safe: a background thread can process indications while the
    main thread reads metrics and sends controls.
    """

    # RIC message types
    RIC_INDICATION = 12050
    RIC_CONTROL_REQUEST = 12040
    RIC_CONTROL_ACK = 12041
    RIC_CONTROL_FAILURE = 12042

    def __init__(
        self,
        xapp: Any,
        e2_node_id: str = "gnb_001",
        ran_func_id: int = 2,
        kpm_metrics: Optional[List[str]] = None,
        report_period_ms: int = 100,
        num_channels: int = 10,
    ):
        if not _HAS_RIC:
            raise RuntimeError(
                "ricxappframe is not installed. "
                "Install with: pip install ricxappframe"
            )

        self._xapp = xapp
        self._e2_node_id = e2_node_id
        self._ran_func_id = ran_func_id
        self._kpm_metric_names = kpm_metrics or [
            "DRB.UEThpDl",
            "RRU.PrbUsedDl",
            "L1M.RS-SINR",
            "RRU.PrbAvailDl",
            "DRB.UEThpUl",
            "RRU.PrbUsedUl",
        ]
        self._report_period_ms = report_period_ms
        self._num_channels = num_channels

        self._subscription_id: Optional[str] = None
        self._latest_metrics: Optional[KPMMetrics] = None
        self._lock = threading.Lock()
        self._indication_count = 0
        self._control_ack_count = 0
        self._control_fail_count = 0

        # Callbacks
        self._on_metrics: Optional[Callable[[KPMMetrics], None]] = None

    # ------------------------------------------------------------------
    # KPM Subscription
    # ------------------------------------------------------------------

    def subscribe_kpm(
        self,
        on_metrics: Optional[Callable[[KPMMetrics], None]] = None,
    ) -> str:
        """
        Subscribe to E2SM-KPM indications from the target E2 node.

        Args:
            on_metrics: Optional callback invoked for each parsed indication.

        Returns:
            Subscription ID string.
        """
        self._on_metrics = on_metrics

        # Register the indication callback with the RMR xApp
        self._xapp.register_callback(self._handle_indication, self.RIC_INDICATION)
        self._xapp.register_callback(self._handle_control_ack, self.RIC_CONTROL_ACK)
        self._xapp.register_callback(
            self._handle_control_failure, self.RIC_CONTROL_FAILURE
        )

        # Build E2SM-KPM subscription request
        sub_params = {
            "e2NodeId": self._e2_node_id,
            "ranFuncId": self._ran_func_id,
            "eventTrigger": {
                "reportingPeriod": self._report_period_ms,
            },
            "actionDefinitions": [
                {
                    "actionId": 1,
                    "actionType": "report",
                    "subsequentAction": {
                        "subsequentActionType": "continue",
                        "timeToWait": "w1ms",
                    },
                    "actionDefinition": {
                        "style": 1,
                        "measInfoList": [
                            {"measName": m} for m in self._kpm_metric_names
                        ],
                        "granularityPeriod": self._report_period_ms,
                    },
                }
            ],
        }

        response = self._xapp.subscription_create(sub_params)
        self._subscription_id = response.get("subscriptionId", "unknown")
        logger.info(
            "KPM subscription created: id=%s, node=%s, period=%dms, metrics=%s",
            self._subscription_id,
            self._e2_node_id,
            self._report_period_ms,
            self._kpm_metric_names,
        )
        return self._subscription_id

    def unsubscribe_kpm(self) -> None:
        """Remove the active KPM subscription."""
        if self._subscription_id is not None:
            self._xapp.subscription_delete(self._subscription_id)
            logger.info("KPM subscription deleted: id=%s", self._subscription_id)
            self._subscription_id = None

    # ------------------------------------------------------------------
    # Indication handling
    # ------------------------------------------------------------------

    def _handle_indication(self, xapp: Any, summary: Any, sbuf: Any) -> None:
        """Process incoming E2SM-KPM indication messages."""
        try:
            payload = summary.get("payload", b"")
            metrics = self._parse_kpm_indication(payload)

            with self._lock:
                self._latest_metrics = metrics
                self._indication_count += 1

            if self._on_metrics is not None:
                self._on_metrics(metrics)

        except Exception:
            logger.exception("Failed to process KPM indication")
        finally:
            xapp.rmr_free(sbuf)

    def _handle_control_ack(self, xapp: Any, summary: Any, sbuf: Any) -> None:
        """Process E2SM-RC control acknowledgement."""
        with self._lock:
            self._control_ack_count += 1
        logger.debug("RC control ACK received (total=%d)", self._control_ack_count)
        xapp.rmr_free(sbuf)

    def _handle_control_failure(self, xapp: Any, summary: Any, sbuf: Any) -> None:
        """Process E2SM-RC control failure."""
        with self._lock:
            self._control_fail_count += 1
        logger.warning(
            "RC control FAILURE received (total=%d)", self._control_fail_count
        )
        xapp.rmr_free(sbuf)

    def _parse_kpm_indication(self, payload: bytes) -> KPMMetrics:
        """
        Parse E2SM-KPM indication payload into structured metrics.

        The payload follows E2SM-KPM v2 Indication Message Format 1:
        measData contains measRecordList entries, each with measRecordItems
        corresponding to the measInfoList from the subscription.
        """
        metrics = KPMMetrics(
            timestamp_ms=int(time.time() * 1000),
            e2_node_id=self._e2_node_id,
            raw_payload=payload,
        )

        try:
            # ASN.1 UPER decoding of the indication message
            from ricxappframe.e2ap.asn1 import IndicationMsg

            ind_msg = IndicationMsg()
            ind_msg.decode(payload)

            msg = ind_msg.indication_message
            meas_info_list = getattr(msg, "measInfoList", [])
            meas_data = getattr(msg, "measData", [])

            for idx, meas_info in enumerate(meas_info_list):
                meas_name = getattr(meas_info, "measType", {}).get("measName", "")
                if not meas_name:
                    continue

                values: List[float] = []
                for record in meas_data:
                    items = getattr(record, "measRecordItemList", [])
                    if idx < len(items):
                        val = getattr(items[idx], "integer", 0)
                        values.append(float(val))

                arr = np.zeros(self._num_channels, dtype=np.float32)
                arr[: min(len(values), self._num_channels)] = values[
                    : self._num_channels
                ]
                metrics.metrics[meas_name] = arr

        except ImportError:
            logger.warning(
                "ricxappframe.e2ap.asn1 not available — storing raw payload only."
            )
        except Exception:
            logger.exception("KPM indication parsing failed")

        return metrics

    # ------------------------------------------------------------------
    # Metric extraction
    # ------------------------------------------------------------------

    def extract_metrics(self) -> Optional[KPMMetrics]:
        """
        Return the most recently received KPM metrics.

        Thread-safe. Returns ``None`` if no indication has been received yet.
        """
        with self._lock:
            return self._latest_metrics

    def get_metric_array(self, metric_name: str) -> np.ndarray:
        """
        Extract a single named metric as a NumPy array.

        Returns zeros of shape ``(num_channels,)`` if the metric is not available.
        """
        with self._lock:
            if self._latest_metrics is None:
                return np.zeros(self._num_channels, dtype=np.float32)
            return self._latest_metrics.metrics.get(
                metric_name,
                np.zeros(self._num_channels, dtype=np.float32),
            ).copy()

    # ------------------------------------------------------------------
    # RC Control
    # ------------------------------------------------------------------

    def send_rc_control(self, request: RCControlRequest) -> bool:
        """
        Send an E2SM-RC control request to the gNB.

        Args:
            request: Control action specification.

        Returns:
            True if the RMR send succeeded, False otherwise.
        """
        # Encode E2SM-RC control header
        rc_header = {
            "controlActionId": request.action_id,
            "ueId": {"gnbUeX2apId": 0},
        }

        # Encode E2SM-RC control message
        ran_params = [
            {
                "ranParameterId": 1,
                "ranParameterValue": {"valueInt": request.target_channel},
            },
            {
                "ranParameterId": 2,
                "ranParameterValue": {
                    "valueInt": 1 if request.allocation_active else 0,
                },
            },
        ]
        # Append any extra parameters
        for key, value in request.parameters.items():
            ran_params.append(
                {
                    "ranParameterId": hash(key) % 1000 + 100,
                    "ranParameterValue": {"valueInt": int(value)},
                }
            )

        rc_message = {"ranParameterList": ran_params}

        # Serialise to wire format
        import json

        payload = json.dumps(
            {"header": rc_header, "message": rc_message}
        ).encode("utf-8")

        success = self._xapp.rmr_send(payload, mtype=self.RIC_CONTROL_REQUEST)
        if success:
            logger.debug(
                "RC control sent: channel=%d, active=%s",
                request.target_channel,
                request.allocation_active,
            )
        else:
            logger.warning("RC control send failed for channel %d", request.target_channel)

        return bool(success)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def indication_count(self) -> int:
        """Total KPM indications received."""
        with self._lock:
            return self._indication_count

    @property
    def control_ack_count(self) -> int:
        """Total RC control ACKs received."""
        with self._lock:
            return self._control_ack_count

    @property
    def control_failure_count(self) -> int:
        """Total RC control failures received."""
        with self._lock:
            return self._control_fail_count

    @property
    def is_subscribed(self) -> bool:
        """Whether a KPM subscription is currently active."""
        return self._subscription_id is not None
