"""Connector ABCs — the stable extension surface for adding data flow.

A connector is anything that moves typed messages in or out of PreceptualAI.
Three roles:

    Source         async-iterates TelemetryEvents
    Sink           consumes any schema (Telemetry / Action / Audit)
    BiConnector    bidirectional (gRPC streaming, Kafka req/reply)

All connectors share a `ConnectorConfig` Pydantic model: callers either
pass a config dict (loaded from YAML/JSON) or instantiate the model
directly. Config is validated up-front; runtime errors are explicit and
typed (`ConnectorError` subclasses).

Lifecycle (mirrors O-RAN.WG2 R1AP states):

    INIT          → instance created, no I/O yet
    STARTED       → connect() succeeded, ready to read/write
    DEGRADED      → transient failure, automatic retry pending
    STOPPED       → close() called, no further I/O

Implementations override at least: `connect`, `close`, plus `read` (Source)
or `write` (Sink) or both (BiConnector). Async APIs throughout — sync
glue is provided by the registry.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, AsyncIterator

from pydantic import BaseModel, ConfigDict

from horizon_ric.io.schemas import (
    AuditRecord,
    FeatureFrame,
    PolicyAction,
    TelemetryEvent,
)

# ─── State machine + errors ──────────────────────────────────────────────


class ConnectorState(str, Enum):
    INIT = "init"
    STARTED = "started"
    DEGRADED = "degraded"
    STOPPED = "stopped"


class ConnectorError(Exception):
    """Base for all connector errors."""


class ConnectorConfigError(ConnectorError):
    """Raised when configuration is invalid."""


class ConnectorIOError(ConnectorError):
    """Raised on transient I/O failure (caller may retry)."""


# ─── Config base ─────────────────────────────────────────────────────────


class ConnectorConfig(BaseModel):
    """Shared connector configuration.

    Subclass per-connector to add fields; parsed from dict at register
    time. Fields here are the universal ones every connector exposes.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    """Stable identifier for this instance — used as `source_id` on events."""
    enabled: bool = True
    retry_max_attempts: int = 5
    retry_backoff_seconds: float = 1.0
    deadline_seconds: float | None = None
    """Optional per-call deadline; None = no deadline."""
    tags: dict[str, str] = {}
    """Operator-supplied labels (cluster, region, owner)."""


# ─── ABCs ────────────────────────────────────────────────────────────────


class Connector(ABC):
    """Base for any connector. Carries config, state, and logger."""

    def __init__(self, config: ConnectorConfig):
        self.cfg = config
        self._state = ConnectorState.INIT

    @property
    def name(self) -> str:
        return self.cfg.name

    @property
    def state(self) -> ConnectorState:
        return self._state

    @abstractmethod
    async def connect(self) -> None:
        """Establish the underlying transport. Idempotent."""

    @abstractmethod
    async def close(self) -> None:
        """Tear down the underlying transport."""

    async def __aenter__(self) -> "Connector":
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()


class Source(Connector):
    """Async iterator of TelemetryEvents."""

    @abstractmethod
    def stream(self) -> AsyncIterator[TelemetryEvent]:
        """Return an async iterator over events. Must respect cfg.deadline."""

    async def read(self, n: int = 1) -> list[TelemetryEvent]:
        """Read up to n events from the stream (default 1)."""
        out: list[TelemetryEvent] = []
        async for event in self.stream():
            out.append(event)
            if len(out) >= n:
                break
        return out


class Sink(Connector):
    """Consumer of any I/O schema. Subclasses override the type they accept."""

    @abstractmethod
    async def write(self, message: TelemetryEvent | FeatureFrame
                                 | PolicyAction | AuditRecord) -> None:
        """Push one message. Implementations must be idempotent on retry."""

    async def write_many(
        self,
        messages: list[TelemetryEvent | FeatureFrame
                       | PolicyAction | AuditRecord],
    ) -> None:
        """Default impl: serial writes. Override for batched transports."""
        for m in messages:
            await self.write(m)


class BiConnector(Source, Sink):
    """Bidirectional (e.g. gRPC streaming, Kafka request/reply)."""


# ─── Retry helper ────────────────────────────────────────────────────────


async def with_retry(
    fn,
    *,
    attempts: int,
    backoff_seconds: float,
    on_retry=None,
):
    """Async retry helper used inside Source/Sink implementations."""
    last_exc: Exception | None = None
    for i in range(max(attempts, 1)):
        try:
            return await fn()
        except ConnectorError as exc:
            last_exc = exc
            if i + 1 >= attempts:
                raise
            if on_retry is not None:
                on_retry(i, exc)
            await asyncio.sleep(backoff_seconds * (2 ** i))
    raise last_exc  # pragma: no cover — unreachable


__all__ = [
    "BiConnector",
    "Connector",
    "ConnectorConfig",
    "ConnectorConfigError",
    "ConnectorError",
    "ConnectorIOError",
    "ConnectorState",
    "Sink",
    "Source",
    "with_retry",
]
