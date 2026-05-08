"""O1 NETCONF/YANG adapter (O-RAN.WG10 OAM, §6).

The O1 interface is the SMO ↔ Managed Element (E2 nodes, RU/DU/CU)
fault/configuration/performance management channel. It speaks NETCONF
1.1 over SSH per RFC 6242, with payloads governed by 3GPP TS 28.541
NRM YANG and O-RAN.WG10 SC YANG augmentations.

This module wraps `ncclient` (Apache 2.0) and provides:

  * `connect()`           — open and authenticate the NETCONF session.
  * `get_config()`        — retrieve a YANG subtree (XPath/subtree filter).
  * `edit_config()`       — push a `<config>` into running (default) or
                            candidate datastore.
  * `subscribe_streams()` — RFC 5277 NETCONF notifications stream
                            ('NETCONF', 'oran-fm-alarm', 'oran-pm-pm') —
                            we yield each notification as a TelemetryEvent.

ncclient performs synchronous I/O internally; this adapter offers an
async surface by running the calls in a thread executor — sufficient
for O1 cadence (seconds-class), and avoids re-implementing the
NETCONF/SSH transport.

References:
    O-RAN.WG10.O1-Interface-v06.00 §6.
    3GPP TS 28.541 NRM YANG modules.
    RFC 6241 (NETCONF Protocol), RFC 6242 (NETCONF over SSH),
    RFC 5277 (NETCONF Event Notifications).
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)

try:  # pragma: no cover — optional dep
    from ncclient import manager as nc_manager
    from ncclient.transport.errors import TransportError
    from ncclient.xml_ import to_ele
    _NCCLIENT_AVAILABLE = True
except ImportError:  # pragma: no cover
    nc_manager = None
    TransportError = Exception
    to_ele = None
    _NCCLIENT_AVAILABLE = False


_NETCONF_NOTIF_NS = "urn:ietf:params:xml:ns:netconf:notification:1.0"


@dataclass
class O1AdapterConfig:
    """NETCONF connection parameters."""

    host: str
    port: int = 830
    username: str = "horizon-rapp"
    password: str | None = None
    key_filename: str | None = None
    """Path to an SSH private key — preferred over password in production."""
    hostkey_verify: bool = True
    timeout_seconds: float = 30.0
    look_for_keys: bool = True
    allow_agent: bool = True
    device_params: dict[str, object] = field(
        default_factory=lambda: {"name": "default"}
    )


class O1Adapter:
    """Thin async wrapper over `ncclient.manager`."""

    def __init__(self, config: O1AdapterConfig):
        if not _NCCLIENT_AVAILABLE:  # pragma: no cover
            raise RuntimeError(
                "O1Adapter requires `ncclient`. Install with `pip install ncclient`."
            )
        self.cfg = config
        self._mgr: object | None = None
        self._lock = asyncio.Lock()
        from horizon_ric.runtime.circuit_breaker import (
            AsyncCircuitBreaker,
            BreakerConfig,
        )

        self._cb = AsyncCircuitBreaker(
            BreakerConfig(name="horizon.o1", fail_max=5, reset_timeout=30.0)
        )

    @property
    def circuit_breaker(self):
        return self._cb

    @property
    def is_connected(self) -> bool:
        return self._mgr is not None and getattr(self._mgr, "connected", False)

    async def connect(self) -> None:
        """Open NETCONF/SSH session."""
        async with self._lock:
            if self.is_connected:
                return
            loop = asyncio.get_running_loop()
            try:
                self._mgr = await loop.run_in_executor(
                    None, self._connect_blocking
                )
            except TransportError as exc:
                raise RuntimeError(f"O1 NETCONF connect failed: {exc}") from exc
            logger.info(
                "o1.connect.ok host=%s port=%s",
                self.cfg.host, self.cfg.port,
            )

    def _connect_blocking(self) -> object:  # pragma: no cover — IO
        return nc_manager.connect(
            host=self.cfg.host,
            port=self.cfg.port,
            username=self.cfg.username,
            password=self.cfg.password,
            key_filename=self.cfg.key_filename,
            hostkey_verify=self.cfg.hostkey_verify,
            timeout=self.cfg.timeout_seconds,
            look_for_keys=self.cfg.look_for_keys,
            allow_agent=self.cfg.allow_agent,
            device_params=self.cfg.device_params,
        )

    async def close(self) -> None:
        async with self._lock:
            if self._mgr is None:
                return
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(None, self._mgr.close_session)
            finally:
                self._mgr = None
                logger.info("o1.close.ok")

    async def get_config(
        self, source: str = "running", filter_subtree: str | None = None,
    ) -> str:
        """Fetch a YANG datastore subtree as NETCONF XML reply."""
        if not self.is_connected:  # pragma: no cover
            raise RuntimeError("O1Adapter not connected")
        loop = asyncio.get_running_loop()

        def _run() -> str:  # pragma: no cover — IO
            kwargs: dict[str, object] = {"source": source}
            if filter_subtree:
                kwargs["filter"] = ("subtree", filter_subtree)
            reply = self._mgr.get_config(**kwargs)
            return reply.xml

        async def _do():
            return await loop.run_in_executor(None, _run)

        return await self._cb.call(_do)

    async def edit_config(
        self,
        config_xml: str,
        target: str = "running",
        default_operation: str = "merge",
    ) -> str:
        """Push a `<config>` subtree into the target datastore."""
        if not self.is_connected:  # pragma: no cover
            raise RuntimeError("O1Adapter not connected")
        loop = asyncio.get_running_loop()

        def _run() -> str:  # pragma: no cover — IO
            reply = self._mgr.edit_config(
                target=target,
                config=config_xml,
                default_operation=default_operation,
            )
            return reply.xml

        return await loop.run_in_executor(None, _run)

    async def lock(self, target: str = "candidate") -> str:
        """Acquire a NETCONF lock on the target datastore."""
        if not self.is_connected:  # pragma: no cover
            raise RuntimeError("O1Adapter not connected")
        loop = asyncio.get_running_loop()

        def _run() -> str:  # pragma: no cover — IO
            return self._mgr.lock(target=target).xml

        return await loop.run_in_executor(None, _run)

    async def unlock(self, target: str = "candidate") -> str:
        """Release a NETCONF lock on the target datastore."""
        if not self.is_connected:  # pragma: no cover
            raise RuntimeError("O1Adapter not connected")
        loop = asyncio.get_running_loop()

        def _run() -> str:  # pragma: no cover — IO
            return self._mgr.unlock(target=target).xml

        return await loop.run_in_executor(None, _run)

    async def discard_changes(self) -> str:
        """Discard pending edits in the candidate datastore."""
        if not self.is_connected:  # pragma: no cover
            raise RuntimeError("O1Adapter not connected")
        loop = asyncio.get_running_loop()

        def _run() -> str:  # pragma: no cover — IO
            return self._mgr.discard_changes().xml

        return await loop.run_in_executor(None, _run)

    async def commit(self) -> str:
        """Commit the candidate datastore to running."""
        if not self.is_connected:  # pragma: no cover
            raise RuntimeError("O1Adapter not connected")
        loop = asyncio.get_running_loop()

        def _run() -> str:  # pragma: no cover — IO
            return self._mgr.commit().xml

        return await loop.run_in_executor(None, _run)

    async def create_subscription(
        self,
        stream: str | None = None,
        start_time: str | None = None,
        stop_time: str | None = None,
    ) -> str:
        """Issue an RFC 5277 `<create-subscription>` RPC.

        Returns the raw `<rpc-reply>` XML (`<ok/>` on success). After the
        server accepts the subscription, every event the server publishes
        on the chosen stream will be delivered as a NETCONF
        `<notification>` and can be drained via :py:meth:`iter_notifications`.
        """
        if not self.is_connected:  # pragma: no cover
            raise RuntimeError("O1Adapter not connected")
        loop = asyncio.get_running_loop()

        def _run() -> str:  # pragma: no cover — IO
            attrs: list[str] = []
            inner = ""
            if stream:
                inner += f"<stream>{stream}</stream>"
            if start_time:
                inner += f"<startTime>{start_time}</startTime>"
            if stop_time:
                inner += f"<stopTime>{stop_time}</stopTime>"
            rpc_xml = (
                f'<create-subscription xmlns="{_NETCONF_NOTIF_NS}">'
                f"{inner}</create-subscription>"
            )
            reply = self._mgr.dispatch(to_ele(rpc_xml))
            return reply.xml

        return await loop.run_in_executor(None, _run)

    async def iter_notifications(
        self,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float = 0.2,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield NETCONF notifications (post-`create_subscription`).

        Each yielded dict carries:
          * ``modality``: ``"o1_netconf"`` (TelemetryEvent-compatible tag)
          * ``stream``: best-effort stream name (None if unscoped)
          * ``raw_xml``: the full `<notification>` payload
          * ``received_ts``: monotonic receive timestamp (seconds)

        Stops yielding when ``timeout_seconds`` elapses with no new
        notification, or when the underlying session closes.
        """
        if not self.is_connected:  # pragma: no cover
            raise RuntimeError("O1Adapter not connected")
        loop = asyncio.get_running_loop()
        deadline = (
            time.monotonic() + timeout_seconds
            if timeout_seconds is not None
            else None
        )

        def _take() -> object:  # pragma: no cover — IO
            return self._mgr.take_notification(block=False)

        while True:
            if deadline is not None and time.monotonic() > deadline:
                return
            notif = await loop.run_in_executor(None, _take)
            if notif is None:
                await asyncio.sleep(poll_interval_seconds)
                continue
            raw = getattr(notif, "notification_xml", None) or str(notif)
            yield {
                "modality": "o1_netconf",
                "stream": getattr(notif, "stream", None),
                "raw_xml": raw,
                "received_ts": time.monotonic(),
            }

    @asynccontextmanager
    async def session(self) -> AsyncIterator["O1Adapter"]:
        """Async context manager — connect/close around a using-block."""
        await self.connect()
        try:
            yield self
        finally:
            await self.close()


__all__ = ["O1Adapter", "O1AdapterConfig"]
