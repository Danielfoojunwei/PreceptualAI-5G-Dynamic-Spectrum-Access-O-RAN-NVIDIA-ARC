"""JSONL-on-disk Source + Sink.

Use cases:
  * Replaying recorded telemetry into the encoder.
  * Tee-ing a live stream to disk for offline replay.
  * Trivial integration tests without spinning up Kafka.

One JSON object per line; pydantic round-trip via `.model_dump_json()` and
`model_validate_json()`. The format is `{"schema": "...", ...rest}` so a
single file can carry mixed schemas (common in audit logs).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator

from pydantic import Field

from horizon_ric.io.connector import (
    ConnectorConfig,
    ConnectorIOError,
    ConnectorState,
    Sink,
    Source,
)
from horizon_ric.io.schemas import (
    AuditRecord,
    FeatureFrame,
    PolicyAction,
    TelemetryEvent,
)

_SCHEMA_FOR = {
    "telemetry": TelemetryEvent,
    "feature_frame": FeatureFrame,
    "policy_action": PolicyAction,
    "audit": AuditRecord,
}


class _FileConfig(ConnectorConfig):
    path: str = Field(..., description="path to the JSONL file")
    follow: bool = False
    """If True (Source), tail the file for appended lines."""


class FileSource(Source):
    """Read TelemetryEvents from a JSONL file (one event per line)."""

    Config = _FileConfig
    cfg: _FileConfig

    async def connect(self) -> None:
        path = Path(self.cfg.path)
        if not path.exists():
            raise ConnectorIOError(f"file not found: {path}")
        self._path = path
        self._state = ConnectorState.STARTED

    async def close(self) -> None:
        self._state = ConnectorState.STOPPED

    async def stream(self) -> AsyncIterator[TelemetryEvent]:
        if self._state != ConnectorState.STARTED:
            raise ConnectorIOError("FileSource not connected")
        loop = asyncio.get_running_loop()

        def _read_all() -> list[str]:
            with self._path.open("r", encoding="utf-8") as f:
                return [line for line in f if line.strip()]

        lines = await loop.run_in_executor(None, _read_all)
        for line in lines:
            obj = json.loads(line)
            # Match the envelope FileSink writes: {"schema": tag, "data": payload}.
            # Tolerate raw payloads too (older files without an envelope).
            if isinstance(obj, dict) and "schema" in obj and "data" in obj:
                cls = _SCHEMA_FOR.get(obj["schema"])
                if cls is None:
                    raise ConnectorIOError(
                        f"unknown schema tag in JSONL: {obj['schema']!r}"
                    )
                payload = obj["data"]
            else:
                cls = TelemetryEvent
                payload = obj
            # The Source contract returns only TelemetryEvent; skip non-telemetry
            # rows (audit records, feature frames) so a heterogeneous file
            # stream still parses without crashing.
            if cls is not TelemetryEvent:
                continue
            yield cls.model_validate(payload)


class FileSink(Sink):
    """Append messages to a JSONL file. Tags each line with its schema name
    so a heterogeneous stream replays cleanly."""

    Config = _FileConfig
    cfg: _FileConfig

    async def connect(self) -> None:
        path = Path(self.cfg.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Touch / truncate handled by caller; we always append.
        path.touch(exist_ok=True)
        self._path = path
        self._lock = asyncio.Lock()
        self._state = ConnectorState.STARTED

    async def close(self) -> None:
        self._state = ConnectorState.STOPPED

    async def write(
        self,
        message: TelemetryEvent | FeatureFrame | PolicyAction | AuditRecord,
    ) -> None:
        if self._state != ConnectorState.STARTED:
            raise ConnectorIOError("FileSink not connected")

        for tag, cls in _SCHEMA_FOR.items():
            if isinstance(message, cls):
                schema_tag = tag
                break
        else:  # pragma: no cover — schemas in __init__ are exhaustive
            raise ConnectorIOError(
                f"unsupported message type {type(message).__name__}"
            )

        line = (
            json.dumps({"schema": schema_tag, "data": message.model_dump(mode="json")})
            + "\n"
        )
        async with self._lock:
            loop = asyncio.get_running_loop()

            def _append() -> None:
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(line)

            await loop.run_in_executor(None, _append)


__all__ = ["FileSink", "FileSource"]
