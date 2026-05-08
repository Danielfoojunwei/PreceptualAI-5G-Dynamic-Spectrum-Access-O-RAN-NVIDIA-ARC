"""space-track.org REST API client.

Authoritative LEO/satellite catalog (US Space Force / 18 SDS). Public
documentation: https://www.space-track.org/documentation

Authentication is a session-cookie flow:

    POST /ajaxauth/login   identity=<email>&password=<pass>
    GET  /basicspacedata/query/...                             (Cookie set)

Credentials are taken from the constructor or, by default, from the
environment (`SPACE_TRACK_IDENTITY`, `SPACE_TRACK_PASSWORD`). They are
NEVER printed, logged, or written to disk by this module — `__repr__`
returns `SpaceTrackClient(identity='[redacted]')` regardless of state.

The free tier is rate-limited to ~30 requests/minute. We enforce a
client-side minimum gap of 2.0 s between any two requests (a single
`asyncio.Lock` guards the timestamp) so a tight loop of catalog queries
won't get the account banned.

Also exposed: `SpaceTrackSource`, an `io.Source` adapter that streams
the full GP catalog as `TelemetryEvent(modality="tle")` records via the
connector registry under the name `space_track`.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import quote

import httpx
import structlog

from horizon_ric.io.connector import (
    ConnectorConfig,
    ConnectorIOError,
    ConnectorState,
    Source,
)
from horizon_ric.io.schemas import TelemetryEvent

logger = structlog.get_logger(__name__)


_DEFAULT_BASE_URL = "https://www.space-track.org"
_DEFAULT_USER_AGENT = (
    "horizon-ric/0.2 (+research; danielfoojunwei@gmail.com)"
)
# Free tier ~30 req/min => 2 s minimum gap, leaves headroom for jitter.
_MIN_REQUEST_GAP_SECONDS = 2.0


class SpaceTrackClient:
    """Async REST client for space-track.org.

    Construct with explicit credentials or rely on the env fallback.
    Use as an async context manager so the session is torn down cleanly
    even on error::

        async with SpaceTrackClient() as st:
            tles = await st.get_tles_by_norad_ids([25544])
    """

    def __init__(
        self,
        identity: str | None = None,
        password: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        timeout_seconds: float = 60.0,
        user_agent: str = _DEFAULT_USER_AGENT,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        # Env fallback. Empty strings are treated as unset.
        self._identity = identity or os.environ.get("SPACE_TRACK_IDENTITY") or None
        self._password = password or os.environ.get("SPACE_TRACK_PASSWORD") or None
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._user_agent = user_agent

        self._client: httpx.AsyncClient = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout_seconds,
            headers={"User-Agent": user_agent},
            transport=transport,
        )
        self._authenticated: bool = False
        self._rate_lock = asyncio.Lock()
        self._last_request_at: float = 0.0

    # ── lifecycle ────────────────────────────────────────────────────────

    def __repr__(self) -> str:  # never expose the real identity
        return "SpaceTrackClient(identity='[redacted]')"

    async def __aenter__(self) -> "SpaceTrackClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        try:
            if self._authenticated:
                await self.logout()
        finally:
            await self.close()

    async def close(self) -> None:
        if not self._client.is_closed:
            await self._client.aclose()

    # ── auth ─────────────────────────────────────────────────────────────

    def _require_credentials(self) -> tuple[str, str]:
        if not self._identity or not self._password:
            raise RuntimeError(
                "space-track credentials are not set. Provide them via the "
                "SpaceTrackClient(identity=..., password=...) constructor or "
                "the SPACE_TRACK_IDENTITY and SPACE_TRACK_PASSWORD environment "
                "variables. Sign up for a free account at "
                "https://www.space-track.org/auth/createAccount"
            )
        return self._identity, self._password

    async def login(self) -> None:
        """POST credentials to /ajaxauth/login. Idempotent."""
        identity, password = self._require_credentials()
        await self._respect_rate_limit()
        try:
            resp = await self._client.post(
                "/ajaxauth/login",
                data={"identity": identity, "password": password},
            )
        except httpx.HTTPError as exc:
            logger.error("space_track.auth_transport_failed", error=type(exc).__name__)
            raise

        # Auth failure paths:
        #   * 401/403 directly
        #   * 200 OK but no session cookie set (space-track sometimes returns
        #     a "Failed" body with 200)
        cookie_set = any(
            c.name in {"chocolatechip", "spacetrack_csrf_cookie"}
            for c in self._client.cookies.jar
        )
        if resp.status_code in (401, 403) or not cookie_set:
            logger.error(
                "space_track.auth_failed",
                status=resp.status_code,
                cookie_set=cookie_set,
            )
            self._authenticated = False
            raise RuntimeError(
                f"space-track authentication failed (status={resp.status_code}). "
                "Verify SPACE_TRACK_IDENTITY and SPACE_TRACK_PASSWORD."
            )
        self._authenticated = True
        logger.info("space_track.auth_ok")

    async def logout(self) -> None:
        """Best-effort GET /ajaxauth/logout. Never raises."""
        try:
            await self._client.get("/ajaxauth/logout")
        except httpx.HTTPError:  # pragma: no cover - cleanup path
            pass
        self._authenticated = False

    async def _ensure_session(self) -> None:
        if not self._authenticated:
            await self.login()

    # ── rate limiter ─────────────────────────────────────────────────────

    async def _respect_rate_limit(self) -> None:
        """Sleep so consecutive calls keep below ~30 req/min."""
        async with self._rate_lock:
            now = time.monotonic()
            wait = _MIN_REQUEST_GAP_SECONDS - (now - self._last_request_at)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request_at = time.monotonic()

    # ── core query ───────────────────────────────────────────────────────

    async def _get(self, path: str) -> httpx.Response:
        await self._ensure_session()
        await self._respect_rate_limit()
        resp = await self._client.get(path)
        if resp.status_code in (401, 403):
            # Session may have expired; mark and surface.
            self._authenticated = False
            logger.error("space_track.session_expired", status=resp.status_code)
            resp.raise_for_status()
        resp.raise_for_status()
        return resp

    # ── TLE queries ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_tle_block(text: str) -> list[tuple[str, str]]:
        """Pair up consecutive line-1/line-2 records from a TLE blob."""
        lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
        out: list[tuple[str, str]] = []
        i = 0
        while i + 1 < len(lines):
            l1, l2 = lines[i], lines[i + 1]
            if l1.startswith("1 ") and l2.startswith("2 "):
                out.append((l1, l2))
                i += 2
            else:
                i += 1  # skip stray name lines etc.
        return out

    async def get_tles_by_norad_ids(
        self, norad_ids: list[int]
    ) -> list[tuple[str, str]]:
        if not norad_ids:
            return []
        ids_csv = ",".join(str(int(n)) for n in norad_ids)
        path = (
            f"/basicspacedata/query/class/gp/NORAD_CAT_ID/{ids_csv}"
            "/orderby/EPOCH desc/format/tle"
        )
        resp = await self._get(path)
        return self._parse_tle_block(resp.text)

    @staticmethod
    def _parse_3le_block(text: str) -> list[tuple[str, str, str]]:
        """Parse a 3LE blob into (name, line1, line2) triples.

        TLE names commonly contain spaces (e.g. "STARLINK-1008 ") so we
        cannot split on whitespace; instead, anchor on the structural
        markers — line1 starts with "1 " and line2 with "2 " at column 0,
        and the name is whatever non-empty line precedes them.
        """
        lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
        out: list[tuple[str, str, str]] = []
        i = 0
        while i + 2 < len(lines) + 1:
            if i + 2 >= len(lines):
                break
            name, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
            if l1.startswith("1 ") and l2.startswith("2 "):
                # Strip the optional "0 " prefix that some 3LE formatters emit.
                if name.startswith("0 "):
                    name = name[2:]
                out.append((name, l1, l2))
                i += 3
            else:
                i += 1
        return out

    async def get_recent_gp_xml(self, window_days: int = 30) -> str:
        """GP catalog snapshot from the last `window_days` days, as XML.

        Mirrors the URL the user supplied:
        ``/basicspacedata/query/class/gp/EPOCH/>now-{N}/orderby/NORAD_CAT_ID,EPOCH/format/xml``
        """
        n = max(int(window_days), 1)
        path = (
            f"/basicspacedata/query/class/gp/EPOCH/%3Enow-{n}"
            "/orderby/NORAD_CAT_ID,EPOCH/format/xml"
        )
        resp = await self._get(path)
        return resp.text

    async def get_recent_gp_3le(
        self, window_days: int = 30
    ) -> list[tuple[str, str, str]]:
        """GP catalog snapshot from the last `window_days` days, as 3LE.

        Returns a list of ``(name_line, line1, line2)`` triples. Names may
        contain spaces (e.g. ``"STARLINK-1008  "``) and are stripped of the
        optional ``"0 "`` prefix some 3LE formatters emit.
        """
        n = max(int(window_days), 1)
        path = (
            f"/basicspacedata/query/class/gp/EPOCH/%3Enow-{n}"
            "/orderby/NORAD_CAT_ID,EPOCH/format/3le"
        )
        resp = await self._get(path)
        return self._parse_3le_block(resp.text)

    async def get_gp_history_tle(
        self,
        start_date: str,
        end_date: str,
        show_empty: bool = True,
    ) -> list[tuple[str, str]]:
        """gp_history slice in TLE format for a CREATION_DATE window.

        ``start_date`` / ``end_date`` are ISO-8601 ``YYYY-MM-DD`` strings
        and are passed through verbatim to space-track. When
        ``show_empty=True`` the URL ends with ``/emptyresult/show`` so
        intervals with zero records still come back as a 200 (empty body)
        rather than the default 404.
        """
        path = (
            f"/basicspacedata/query/class/gp_history/CREATION_DATE/"
            f"{start_date}--{end_date}/orderby/NORAD_CAT_ID,EPOCH/format/tle"
        )
        if show_empty:
            path += "/emptyresult/show"
        resp = await self._get(path)
        return self._parse_tle_block(resp.text)

    async def get_tles_by_constellation(
        self, constellation_name: str
    ) -> list[tuple[str, str]]:
        # ~~ is space-track's wildcard "contains" syntax for string columns.
        wildcard = f"~~{constellation_name}~~"
        path = (
            f"/basicspacedata/query/class/gp/OBJECT_NAME/{quote(wildcard, safe='~')}"
            "/orderby/EPOCH desc/format/tle"
        )
        resp = await self._get(path)
        return self._parse_tle_block(resp.text)

    # ── decay / conjunction ──────────────────────────────────────────────

    async def get_decay_history(self, window_days: int = 30) -> list[dict]:
        n = max(int(window_days), 1)
        path = (
            f"/basicspacedata/query/class/decay/DECAY_EPOCH/%3Enow-{n}/format/json"
        )
        resp = await self._get(path)
        data = resp.json()
        return list(data) if isinstance(data, list) else []

    async def get_conjunction_events(self, window_days: int = 7) -> list[dict]:
        n = max(int(window_days), 1)
        # CDM uses TCA (Time of Closest Approach) as its temporal column.
        path = (
            f"/basicspacedata/query/class/cdm_public/TCA/%3Enow-{n}/format/json"
        )
        resp = await self._get(path)
        data = resp.json()
        return list(data) if isinstance(data, list) else []

    # ── bulk catalog ─────────────────────────────────────────────────────

    async def download_full_catalog(self, out_path: Path) -> int:
        """Stream the full GP catalog to `out_path`. Returns satellite count."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        path = "/basicspacedata/query/class/gp/orderby/NORAD_CAT_ID asc/format/tle"
        resp = await self._get(path)
        out_path.write_text(resp.text)
        pairs = self._parse_tle_block(resp.text)
        logger.info(
            "space_track.catalog_downloaded",
            sats=len(pairs),
            path=str(out_path),
        )
        return len(pairs)


# ─── Source registry adapter ─────────────────────────────────────────────


class SpaceTrackSourceConfig(ConnectorConfig):
    """Config for the `space_track` connector.

    Credentials come from the environment by default; passing them in the
    config is supported but discouraged because configs are routinely
    persisted to disk.
    """

    # Path where the bulk-downloaded TLE blob is cached. Re-runs read
    # straight from disk if the file already exists and `refresh=False`.
    catalog_path: str = "data/space_track/full_catalog.tle"
    refresh: bool = False
    # Optional credential overrides; default is to use env vars.
    identity: str | None = None
    password: str | None = None
    base_url: str = _DEFAULT_BASE_URL


class SpaceTrackSource(Source):
    """`io.Source` wrapper around `SpaceTrackClient`.

    On `connect()` the full GP catalog is materialised under
    `cfg.catalog_path` (unless cached). `stream()` then yields one
    `TelemetryEvent(modality="tle")` per (line1, line2) pair.
    """

    Config = SpaceTrackSourceConfig
    cfg: SpaceTrackSourceConfig  # type: ignore[assignment]

    def __init__(self, config: SpaceTrackSourceConfig) -> None:
        super().__init__(config)
        self._client: SpaceTrackClient | None = None
        self._tles: list[tuple[str, str]] = []

    async def connect(self) -> None:
        path = Path(self.cfg.catalog_path)
        if path.exists() and not self.cfg.refresh:
            self._tles = SpaceTrackClient._parse_tle_block(path.read_text())
        else:
            self._client = SpaceTrackClient(
                identity=self.cfg.identity,
                password=self.cfg.password,
                base_url=self.cfg.base_url,
            )
            await self._client.download_full_catalog(path)
            self._tles = SpaceTrackClient._parse_tle_block(path.read_text())
        self._state = ConnectorState.STARTED

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
        self._state = ConnectorState.STOPPED

    async def stream(self) -> AsyncIterator[TelemetryEvent]:  # type: ignore[override]
        if self._state != ConnectorState.STARTED:
            raise ConnectorIOError("SpaceTrackSource not connected")
        ts = datetime.now(timezone.utc)
        for line1, line2 in self._tles:
            # NORAD ID lives at columns 3-7 of TLE line 1.
            norad = line1[2:7].strip() if len(line1) >= 7 else ""
            yield TelemetryEvent(
                event_id=str(uuid.uuid4()),
                modality="tle",
                source_id=self.cfg.name,
                ts_utc=ts,
                payload={
                    "line1": line1,
                    "line2": line2,
                    "norad_cat_id": norad,
                },
            )


__all__ = [
    "SpaceTrackClient",
    "SpaceTrackSource",
    "SpaceTrackSourceConfig",
]
