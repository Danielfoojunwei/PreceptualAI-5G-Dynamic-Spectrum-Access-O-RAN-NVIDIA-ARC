"""NVIDIA ARC-OTA real-world I/Q feed consumer.

ARC-OTA (Aerial RAN CoLab — Over The Air) is NVIDIA's reference
deployment for streaming real-world physical-layer telemetry — uplink
I/Q, FAPI L1/L2 messages, KPM counters — out of a live cuBB-driven
small cell. The native producer surface is gRPC; for PreceptualAI's
pilot deployments we run an HTTP bridge in front of it so the rApp side
of the wire only needs ``httpx`` (already in deps) and not ``grpcio``.

The bridge contract used here is intentionally narrow:

  * ``GET  {endpoint_url}/arc-ota/poll``    — long-poll for the next
    batch of capture messages. Returns JSON ``{"messages": [...]}``.
    Each message is a dict with at minimum ``capture_type``
    (e.g. ``uplink_iq``, ``fapi_l1``) plus a ``payload`` blob and an
    ``ingest_ts_ns`` ARC-OTA timestamp.
  * On disconnect / non-2xx, we reconnect with exponential backoff
    starting at 50 ms, doubling each attempt up to
    ``reconnect_backoff_max_s``.

For the gRPC-native path, install ``grpcio`` and switch to
``ARCOTAGrpcConsumer`` (Phase-2). The HTTP bridge is the path we use
in CI and in the integration tests because it can be driven by an
``httpx.MockTransport`` end-to-end.

Translation: each ARC-OTA message becomes one
``horizon_ric.io.schemas.TelemetryEvent``. The mapping is:

  * ``capture_type == "uplink_iq"``    → modality ``spectrum_iq``
  * ``capture_type == "fapi_l1"``      → modality ``kpm_5g``
  * ``capture_type == "kpm"``          → modality ``kpm_5g``
  * anything else                      → modality ``kpm_5g`` (with a
    ``tags["arc_ota_capture_type"]`` annotation so the consumer can
    still distinguish on the bus).

Two Prometheus metrics are exposed module-level:

  * ``arc_ota_connected``   — gauge, 1 when the consumer believes the
    upstream bridge is healthy, 0 otherwise.
  * ``arc_ota_msgs_total``  — counter, total ARC-OTA messages
    translated and yielded downstream.

Both are unlabeled because in pilot deployments there is exactly one
ARC-OTA bridge per rApp instance.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import httpx
import structlog
from prometheus_client import Counter, Gauge
from pydantic import BaseModel, Field

from horizon_ric.io.schemas import Modality, TelemetryEvent

logger = structlog.get_logger(__name__)

# ─── Prometheus metrics (module-scoped, unlabeled) ────────────────────────

arc_ota_connected = Gauge(
    "arc_ota_connected",
    "1 when the ARC-OTA HTTP bridge consumer is connected, 0 otherwise.",
)
arc_ota_msgs_total = Counter(
    "arc_ota_msgs_total",
    "Total ARC-OTA messages translated to TelemetryEvent and yielded.",
)
# Initialize the gauge to 0 at import so scrapes before first connect
# show the disconnected state rather than absent series.
arc_ota_connected.set(0)


# ─── Capture-type → Modality mapping ─────────────────────────────────────

_CAPTURE_TO_MODALITY: dict[str, Modality] = {
    "uplink_iq": "spectrum_iq",
    "downlink_iq": "spectrum_iq",
    "fapi_l1": "kpm_5g",
    "fapi_l2": "kpm_5g",
    "kpm": "kpm_5g",
}


def _ingest_ts_to_utc(ts_ns: int | None) -> datetime:
    """Map ARC-OTA's nanoseconds-since-epoch to a tz-aware UTC datetime.

    Falls back to ``datetime.now(UTC)`` when the bridge omits the field
    so a malformed message still yields a schema-valid event rather
    than crashing the consumer loop.
    """

    if ts_ns is None:
        return datetime.now(tz=timezone.utc)
    secs, rem_ns = divmod(int(ts_ns), 1_000_000_000)
    base = datetime.fromtimestamp(secs, tz=timezone.utc)
    # Python datetime resolves to microseconds; drop sub-microsecond ns.
    return base.replace(microsecond=rem_ns // 1000)


# ─── Configuration ───────────────────────────────────────────────────────


class ARCOTAConfig(BaseModel):
    """Configuration for the ARC-OTA HTTP-bridge consumer."""

    endpoint_url: str
    """Base URL of the ARC-OTA HTTP bridge, e.g. ``https://arc-ota.local``."""
    grpc_timeout_s: float = 5.0
    """Per-poll HTTP timeout. Named after the upstream gRPC deadline so
    the same value carries through when the gRPC-native consumer lands."""
    reconnect_backoff_max_s: float = 5.0
    """Upper bound on the exponential backoff between reconnect attempts."""
    capture_filter: list[str] = Field(default_factory=list)
    """If non-empty, only ARC-OTA messages whose ``capture_type`` is in
    this list are translated and yielded. Empty list (default) lets all
    capture types through."""
    source_id: str = "nvidia_arc_ota"
    """Stable producer identity stamped onto each ``TelemetryEvent``."""


# ─── Consumer ────────────────────────────────────────────────────────────


class ARCOTAConsumer:
    """Async consumer for the ARC-OTA HTTP-bridged real-time feed.

    Lifecycle:

        cfg = ARCOTAConfig(endpoint_url="https://arc-ota.local")
        consumer = ARCOTAConsumer(cfg)
        await consumer.connect()
        async for event in consumer.stream():
            ...
        await consumer.close()

    For tests, pass an ``httpx.BaseTransport`` (e.g. ``MockTransport``)
    to the constructor to avoid touching the network.
    """

    def __init__(
        self,
        config: ARCOTAConfig,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.cfg = config
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._connected = False
        self._closed = False
        self._stream_task: asyncio.Task[None] | None = None

    # -- lifecycle ---------------------------------------------------------

    async def connect(self) -> None:
        """Open the underlying ``httpx.AsyncClient`` and mark connected."""

        if self._client is not None:
            return
        self._client = httpx.AsyncClient(
            base_url=self.cfg.endpoint_url,
            timeout=self.cfg.grpc_timeout_s,
            transport=self._transport,
        )
        self._connected = True
        self._closed = False
        arc_ota_connected.set(1)
        logger.info(
            "arc_ota.connect",
            endpoint=self.cfg.endpoint_url,
            timeout_s=self.cfg.grpc_timeout_s,
        )

    async def close(self) -> None:
        """Cancel the streaming task (if any) and close the HTTP client."""

        self._closed = True
        self._connected = False
        arc_ota_connected.set(0)
        task = self._stream_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        logger.info("arc_ota.close")

    def is_connected(self) -> bool:
        return self._connected and not self._closed

    # -- streaming ---------------------------------------------------------

    async def stream(self) -> AsyncIterator[TelemetryEvent]:
        """Yield ``TelemetryEvent``s translated from ARC-OTA messages.

        Reconnects with exponential backoff on transport errors and
        non-2xx responses; the ``arc_ota_connected`` gauge tracks the
        believed connection state across the retry cycle.
        """

        if self._client is None:
            raise RuntimeError(
                "ARCOTAConsumer.stream() called before connect()"
            )
        seq = 0
        attempt = 0
        while not self._closed:
            try:
                resp = await self._client.get("/arc-ota/poll")
                resp.raise_for_status()
                body = resp.json()
            except (httpx.HTTPError, ValueError) as e:
                # Either the transport blew up or the body wasn't JSON.
                self._connected = False
                arc_ota_connected.set(0)
                backoff = min(
                    0.05 * (2 ** attempt),
                    self.cfg.reconnect_backoff_max_s,
                )
                logger.warning(
                    "arc_ota.reconnect",
                    attempt=attempt,
                    backoff_s=backoff,
                    error=str(e),
                )
                attempt += 1
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    return
                if self._closed:
                    return
                # Mark connected again ahead of the next poll so the
                # gauge reflects intent during the retry cycle.
                self._connected = True
                arc_ota_connected.set(1)
                continue

            # Successful poll → reset backoff.
            attempt = 0
            self._connected = True
            arc_ota_connected.set(1)

            messages = body.get("messages", []) if isinstance(body, dict) else []
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                capture_type = str(msg.get("capture_type", "kpm"))
                if (
                    self.cfg.capture_filter
                    and capture_type not in self.cfg.capture_filter
                ):
                    continue
                event = self._translate(msg, capture_type, seq)
                seq += 1
                arc_ota_msgs_total.inc()
                yield event

            # If the bridge returned an explicit "done" sentinel, exit
            # cleanly — this is what the test fixtures use to end the
            # stream without forcing a real disconnect.
            if isinstance(body, dict) and body.get("done"):
                return

    # -- translation -------------------------------------------------------

    def _translate(
        self, msg: dict[str, Any], capture_type: str, seq: int
    ) -> TelemetryEvent:
        """Translate one ARC-OTA wire message into a ``TelemetryEvent``."""

        modality: Modality = _CAPTURE_TO_MODALITY.get(capture_type, "kpm_5g")
        ts_ns_raw = msg.get("ingest_ts_ns")
        try:
            ts_ns: int | None = int(ts_ns_raw) if ts_ns_raw is not None else None
        except (TypeError, ValueError):
            ts_ns = None
        ts_utc = _ingest_ts_to_utc(ts_ns)
        event_id = str(
            msg.get("event_id")
            or f"arc-ota-{capture_type}-{seq}"
        )
        payload_blob = msg.get("payload")
        payload: dict[str, Any] = (
            dict(payload_blob)
            if isinstance(payload_blob, dict)
            else {"raw": payload_blob}
        )
        # Preserve fields that aren't in the canonical TelemetryEvent
        # but might be useful downstream.
        for k in ("cell_id", "ue_id", "frame", "slot"):
            if k in msg and k not in payload:
                payload[k] = msg[k]

        tags = {
            "arc_ota_capture_type": capture_type,
            "format": "arc_ota_http_bridge",
        }
        return TelemetryEvent(
            event_id=event_id,
            modality=modality,
            source_id=self.cfg.source_id,
            ts_utc=ts_utc,
            monotonic_ns=ts_ns,
            sequence=seq,
            payload=payload,
            tags=tags,
        )


__all__ = ["ARCOTAConfig", "ARCOTAConsumer", "arc_ota_connected", "arc_ota_msgs_total"]
