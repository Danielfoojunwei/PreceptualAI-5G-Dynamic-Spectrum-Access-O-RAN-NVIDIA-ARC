"""Example external connector — synthetic TelemetryEvent generator.

Demonstrates the entry-point plug-in path documented in
``horizon_ric/io/registry.py``. Installing this package via
``pip install ./examples/external_connector`` makes
``RandomTelemetrySource`` discoverable as ``random_telemetry`` without
touching the core codebase.
"""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timezone
from typing import AsyncIterator

from pydantic import Field

from horizon_ric.io.connector import (
    ConnectorConfig,
    ConnectorState,
    Source,
)
from horizon_ric.io.schemas import TelemetryEvent


class RandomTelemetryConfig(ConnectorConfig):
    """Configuration model for :class:`RandomTelemetrySource`."""

    rate_hz: float = Field(default=1.0, gt=0.0)
    """Events per second to emit."""

    n_events: int = Field(default=10, ge=1)
    """How many events to emit before the stream completes."""

    seed: int | None = None
    """Optional RNG seed for reproducibility."""


class RandomTelemetrySource(Source):
    """Emits ``n_events`` synthetic ``ue_qos`` telemetry events at
    ``rate_hz`` and then completes."""

    Config = RandomTelemetryConfig

    def __init__(self, config: RandomTelemetryConfig):
        super().__init__(config)
        self._rng = random.Random(config.seed)

    async def connect(self) -> None:
        self._state = ConnectorState.STARTED

    async def close(self) -> None:
        self._state = ConnectorState.STOPPED

    async def stream(self) -> AsyncIterator[TelemetryEvent]:
        cfg: RandomTelemetryConfig = self.cfg  # type: ignore[assignment]
        if self._state != ConnectorState.STARTED:
            await self.connect()
        period_s = 1.0 / cfg.rate_hz
        for i in range(cfg.n_events):
            ev = TelemetryEvent(
                event_id=f"random-{cfg.name}-{i}",
                source_id=cfg.name,
                modality="ue_qos",
                ts_utc=datetime.now(timezone.utc),
                sequence=i,
                payload={
                    "ue_id": f"ue-{i}",
                    "rsrp_dbm": self._rng.uniform(-110.0, -70.0),
                    "throughput_mbps": self._rng.uniform(1.0, 200.0),
                },
            )
            yield ev
            if i + 1 < cfg.n_events:
                await asyncio.sleep(period_s)


__all__ = ["RandomTelemetryConfig", "RandomTelemetrySource"]
