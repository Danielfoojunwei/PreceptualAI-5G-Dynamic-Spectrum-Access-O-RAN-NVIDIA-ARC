"""VIAVI D4AI sandbox integration adapters.

The VIAVI Data-Driven AI (D4AI) sandbox publishes four products that
PreceptualAI rApps care about: TM500 UE Emulator, an RU Power Meter,
XhaulAdvisor (eCPRI / 1588 PTP capture), and TeraVM Core Emulator
(control-plane signaling). Each product exposes its own wire format and
operational gating; this module provides a *uniform* adapter shape so
the rApp side code does not need to special-case any of them.

HONEST status as of M3:

* :class:`TM500Adapter` — ``mode = "functional"``. We ship a working
  ``replay_fixture()`` path that reads JSON-lines telemetry off disk
  and yields :class:`~horizon_ric.io.schemas.TelemetryEvent` instances,
  plus an ``httpx.AsyncClient``-based ``stream()`` against the
  documented TM500 HTTP API. Both paths are covered by tests.
* :class:`PowerMeterAdapter` — ``mode = "documented-only"``. The class
  surface (constructor + lifecycle methods) is fixed so calling code
  can be written today, but ``connect()`` raises
  ``NotImplementedError("contact VIAVI ops")`` until a real Power Meter
  bench is provisioned.
* :class:`XhaulAdvisorAdapter` — ``mode = "documented-only"``. Same
  shape, same ``NotImplementedError("contact VIAVI ops")`` gating.
* :class:`TeraVMCoreAdapter` — ``mode = "documented-only"``. Same.

Sales engineers can reflect on the ``.mode`` class attribute to decide
whether a given adapter can be live-demoed (``"functional"``) or only
shown as a code-level integration point (``"documented-only"``).

Why the explicit "contact VIAVI ops" string: it lets ops greppably
identify exactly which VIAVI integrations are blocked on bench
provisioning, rather than an opaque ``NotImplementedError``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

from horizon_ric.io.schemas import TelemetryEvent

_VIAVI_OPS_HINT = "contact VIAVI ops"


# ─── TM500 UE Emulator (functional) ─────────────────────────────────────


class TM500Adapter:
    """Adapter for the VIAVI TM500 UE Emulator HTTP API.

    The TM500 publishes UE-side KPIs (RSRP, RSRQ, BLER, throughput) over
    a documented HTTP/JSON streaming endpoint. This adapter consumes that
    endpoint via :class:`httpx.AsyncClient` and emits canonical
    :class:`TelemetryEvent` objects on the rApp bus.

    A ``replay_fixture()`` path is also provided so demos and CI can
    exercise the full event-shape pipeline without a TM500 box —
    fixtures are JSON-lines files where each line is a serialised
    ``TelemetryEvent``.
    """

    mode: str = "functional"

    def __init__(self, base_url: str, *, timeout_s: float = 5.0) -> None:
        self.base_url: str = base_url.rstrip("/")
        self.timeout_s: float = timeout_s
        self._client: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        """Open an :class:`httpx.AsyncClient` against the TM500 API.

        Idempotent: a second call after a successful connect is a no-op.
        Tests can pre-populate ``self._client`` with a mock transport
        client before calling :meth:`stream`.
        """

        if self._client is not None:
            return
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_s,
        )

    async def stream(self) -> AsyncIterator[TelemetryEvent]:
        """Stream UE KPI events from the TM500 ``/telemetry`` endpoint.

        The TM500 returns a newline-delimited JSON stream; each line is
        a TelemetryEvent-shaped dict. Yields one TelemetryEvent per line.
        """

        if self._client is None:
            raise RuntimeError(
                "TM500Adapter.stream() called before connect()"
            )
        async with self._client.stream("GET", "/telemetry") as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                obj: dict[str, Any] = json.loads(line)
                yield TelemetryEvent(**obj)

    async def replay_fixture(
        self, fixture_path: Path
    ) -> AsyncIterator[TelemetryEvent]:
        """Yield TelemetryEvents from a JSON-lines fixture file.

        Each line is parsed as a ``TelemetryEvent``; this lets demos run
        the full rApp without a live TM500. Returns an async iterator so
        the call site looks identical to :meth:`stream`.
        """

        path = Path(fixture_path)
        if not path.exists():
            raise FileNotFoundError(f"TM500 fixture not found: {path}")

        async def _gen() -> AsyncIterator[TelemetryEvent]:
            with path.open("r", encoding="utf-8") as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line:
                        continue
                    obj: dict[str, Any] = json.loads(line)
                    yield TelemetryEvent(**obj)

        return _gen()

    async def close(self) -> None:
        """Close the underlying httpx client. Idempotent."""

        if self._client is not None:
            await self._client.aclose()
            self._client = None


# ─── Documented-only adapters ───────────────────────────────────────────


class _DocumentedOnlyAdapter:
    """Shared implementation for the three documented-only adapters.

    Subclasses override :attr:`mode` and the docstring; lifecycle
    methods all raise :class:`NotImplementedError` with the canonical
    ``"contact VIAVI ops"`` hint string.
    """

    mode: str = "documented-only"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Stash construction args so subclasses can introspect them
        # (and so the constructor signature stays permissive — every
        # VIAVI bench is configured slightly differently).
        self._args: tuple[Any, ...] = args
        self._kwargs: dict[str, Any] = dict(kwargs)

    async def connect(self) -> None:
        raise NotImplementedError(_VIAVI_OPS_HINT)

    async def stream(self) -> AsyncIterator[TelemetryEvent]:
        raise NotImplementedError(_VIAVI_OPS_HINT)
        # Make the function a real async generator for static analysers.
        if False:  # pragma: no cover
            yield  # type: ignore[unreachable]

    async def close(self) -> None:
        # Closing a never-opened adapter is a no-op; this matches the
        # TM500 semantics so callers can use ``finally: await x.close()``.
        return None


class PowerMeterAdapter(_DocumentedOnlyAdapter):
    """RU power-readings adapter — documented-only.

    Exposes the same lifecycle shape as :class:`TM500Adapter` so the
    rApp side can be coded against the interface today. ``connect()``
    raises ``NotImplementedError("contact VIAVI ops")`` until a real
    Power Meter bench is provisioned.
    """

    mode: str = "documented-only"


class XhaulAdvisorAdapter(_DocumentedOnlyAdapter):
    """eCPRI / transport / 1588 PTP capture adapter — documented-only.

    Production XhaulAdvisor exposes per-flow eCPRI capture, transport
    KPIs, and 1588 PTP synchronisation data. This stub holds the
    constructor surface but raises ``NotImplementedError("contact VIAVI
    ops")`` from ``connect()`` until a real bench is available.
    """

    mode: str = "documented-only"


class TeraVMCoreAdapter(_DocumentedOnlyAdapter):
    """TeraVM Core Emulator adapter — documented-only.

    Production TeraVM Core emits 5G control-plane signalling and
    session/bearer context. This stub holds the constructor surface but
    raises ``NotImplementedError("contact VIAVI ops")`` from
    ``connect()`` until a real bench is available.
    """

    mode: str = "documented-only"


__all__ = [
    "PowerMeterAdapter",
    "TM500Adapter",
    "TeraVMCoreAdapter",
    "XhaulAdvisorAdapter",
]
