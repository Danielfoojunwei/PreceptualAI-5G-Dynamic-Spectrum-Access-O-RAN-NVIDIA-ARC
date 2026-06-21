"""HTTP/REST Source + Sink.

Source: polls a JSON endpoint at a configured interval; each response item
becomes a TelemetryEvent.

Sink: POSTs each message to a webhook URL.

Both reuse `httpx.AsyncClient` (already declared in pyproject.toml deps).
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import httpx
from pydantic import Field, field_validator

from horizon_ric.io.connector import (
    ConnectorConfig,
    ConnectorConfigError,
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


class _HttpSourceConfig(ConnectorConfig):
    url: str
    method: str = "GET"
    poll_interval_seconds: float = 5.0
    request_headers: dict[str, str] = Field(default_factory=dict)
    bearer_token: str | None = None
    """Optional OAuth2 / static bearer."""
    response_array_path: str | None = None
    """JSON path to the events array; None = response is the array."""

    @field_validator("method")
    @classmethod
    def _method_uppercase(cls, v: str) -> str:
        v = v.upper()
        if v not in {"GET", "POST"}:
            raise ValueError("method must be GET or POST")
        return v


class _HttpSinkConfig(ConnectorConfig):
    url: str
    method: str = "POST"
    request_headers: dict[str, str] = Field(default_factory=dict)
    bearer_token: str | None = None


def _client(headers: dict[str, str], bearer: str | None) -> httpx.AsyncClient:
    h = dict(headers)
    if bearer:
        h["Authorization"] = f"Bearer {bearer}"
    return httpx.AsyncClient(headers=h, timeout=30.0)


class HttpRestSource(Source):
    """Poll a REST endpoint, emit each response item as a TelemetryEvent."""

    Config = _HttpSourceConfig
    cfg: _HttpSourceConfig

    async def connect(self) -> None:
        self._client = _client(self.cfg.request_headers, self.cfg.bearer_token)
        self._stop = asyncio.Event()
        self._state = ConnectorState.STARTED

    async def close(self) -> None:
        self._stop.set()
        await self._client.aclose()
        self._state = ConnectorState.STOPPED

    async def stream(self) -> AsyncIterator[TelemetryEvent]:
        if self._state != ConnectorState.STARTED:
            raise ConnectorIOError("HttpRestSource not connected")
        while not self._stop.is_set():
            try:
                resp = await self._client.request(self.cfg.method, self.cfg.url)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise ConnectorIOError(f"GET {self.cfg.url}: {exc}") from exc
            data = resp.json()
            items = data
            if self.cfg.response_array_path:
                for key in self.cfg.response_array_path.split("."):
                    items = items[key]
            if not isinstance(items, list):
                raise ConnectorConfigError(
                    "expected array; got " + type(items).__name__
                )
            for item in items:
                yield TelemetryEvent.model_validate(item)
            await asyncio.sleep(self.cfg.poll_interval_seconds)


class HttpRestSink(Sink):
    """POST each message to a webhook URL."""

    Config = _HttpSinkConfig
    cfg: _HttpSinkConfig

    async def connect(self) -> None:
        self._client = _client(self.cfg.request_headers, self.cfg.bearer_token)
        self._state = ConnectorState.STARTED

    async def close(self) -> None:
        await self._client.aclose()
        self._state = ConnectorState.STOPPED

    async def write(
        self,
        message: TelemetryEvent | FeatureFrame | PolicyAction | AuditRecord,
    ) -> None:
        if self._state != ConnectorState.STARTED:
            raise ConnectorIOError("HttpRestSink not connected")
        try:
            resp = await self._client.request(
                self.cfg.method, self.cfg.url,
                json=message.model_dump(mode="json"),
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ConnectorIOError(f"{self.cfg.method} {self.cfg.url}: {exc}") from exc


__all__ = ["HttpRestSink", "HttpRestSource"]
