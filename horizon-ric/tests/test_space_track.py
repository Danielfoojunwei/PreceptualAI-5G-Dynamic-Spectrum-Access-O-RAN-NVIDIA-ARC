"""Unit tests for `horizon_ric.data.space_track.SpaceTrackClient`.

Every test uses `httpx.MockTransport` — no real network traffic is ever
issued against space-track.org from this suite.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

import httpx
import pytest

from horizon_ric.data.space_track import SpaceTrackClient

_FAKE_TLE = (
    "ISS (ZARYA)\n"
    "1 25544U 98067A   24123.45678901  .00012345  00000-0  22334-4 0  9990\n"
    "2 25544  51.6400 123.4567 0001234  12.3456 347.6543 15.50000000123450\n"
)


def _make_client(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    identity: str = "user@example.com",
    password: str = "hunter2-very-secret",
) -> SpaceTrackClient:
    """SpaceTrackClient wired to a MockTransport `handler`."""
    transport = httpx.MockTransport(handler)
    return SpaceTrackClient(
        identity=identity,
        password=password,
        transport=transport,
    )


# ─── tests ──────────────────────────────────────────────────────────────


async def test_login_success():
    """200 + Set-Cookie should mark the client authenticated."""

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/ajaxauth/logout":
            return httpx.Response(200)
        assert req.url.path == "/ajaxauth/login"
        body = req.content.decode()
        assert "identity=" in body and "password=" in body
        return httpx.Response(
            200,
            headers={"set-cookie": "chocolatechip=abc123; Path=/"},
            text="",
        )

    async with _make_client(handler) as st:
        await st.login()
        assert st._authenticated is True


async def test_login_failure_raises():
    """401 should raise RuntimeError, not leak the body."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="Login failed")

    async with _make_client(handler) as st:
        with pytest.raises(RuntimeError, match="authentication failed"):
            await st.login()
        assert st._authenticated is False


async def test_missing_credentials_raises(monkeypatch):
    """No identity + no env → clear, helpful RuntimeError."""
    monkeypatch.delenv("SPACE_TRACK_IDENTITY", raising=False)
    monkeypatch.delenv("SPACE_TRACK_PASSWORD", raising=False)

    st = SpaceTrackClient(identity=None, password=None)
    try:
        with pytest.raises(RuntimeError, match="credentials are not set"):
            await st.login()
    finally:
        await st.close()


async def test_get_tles_by_norad_ids():
    """A valid TLE block must parse into (line1, line2) tuples."""

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/ajaxauth/login":
            return httpx.Response(
                200, headers={"set-cookie": "chocolatechip=abc; Path=/"}
            )
        if req.url.path == "/ajaxauth/logout":
            return httpx.Response(200)
        assert "NORAD_CAT_ID/25544" in req.url.path
        assert req.url.path.endswith("/format/tle")
        return httpx.Response(200, text=_FAKE_TLE)

    async with _make_client(handler) as st:
        # Skip rate-limit waits in the test (we test that separately).
        st._last_request_at = 0.0
        tles = await st.get_tles_by_norad_ids([25544])
        assert len(tles) == 1
        l1, l2 = tles[0]
        assert l1.startswith("1 25544")
        assert l2.startswith("2 25544")


async def test_constellation_query_uses_wildcard():
    """The OBJECT_NAME query path must contain the `~~` wildcard."""
    captured: dict[str, str] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/ajaxauth/login":
            return httpx.Response(
                200, headers={"set-cookie": "chocolatechip=abc; Path=/"}
            )
        if req.url.path == "/ajaxauth/logout":
            return httpx.Response(200)
        captured["path"] = req.url.path
        return httpx.Response(200, text=_FAKE_TLE)

    async with _make_client(handler) as st:
        await st.get_tles_by_constellation("STARLINK")

    assert "~~" in captured["path"]
    assert "STARLINK" in captured["path"]
    assert "OBJECT_NAME" in captured["path"]


async def test_rate_limiter_enforces_min_gap(monkeypatch):
    """Two consecutive query calls should be spaced ≥1.5 s apart."""

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/ajaxauth/login":
            return httpx.Response(
                200, headers={"set-cookie": "chocolatechip=abc; Path=/"}
            )
        if req.url.path == "/ajaxauth/logout":
            return httpx.Response(200)
        return httpx.Response(200, text=_FAKE_TLE)

    # Tighten the gap a touch so the test stays under 5 s of wall-time
    # while still proving the limiter actually *waits*.
    import horizon_ric.data.space_track as st_mod

    monkeypatch.setattr(st_mod, "_MIN_REQUEST_GAP_SECONDS", 1.7)

    async with _make_client(handler) as st:
        await st.login()  # consumes one slot
        t0 = time.monotonic()
        await st.get_tles_by_norad_ids([25544])
        await st.get_tles_by_norad_ids([25545])
        elapsed = time.monotonic() - t0

    # Two query calls + the post-login slot must collectively take at
    # least one full gap; assert ≥1.5 s so the test isn't fragile.
    assert elapsed >= 1.5, f"rate limiter did not enforce a gap (elapsed={elapsed:.2f}s)"


async def test_get_recent_gp_xml_returns_body():
    """`get_recent_gp_xml` returns the XML body verbatim and uses the
    documented URL pattern."""
    captured: dict[str, str] = {}
    xml_body = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<gp>\n"
        "  <object>\n"
        "    <NORAD_CAT_ID>25544</NORAD_CAT_ID>\n"
        "    <OBJECT_NAME>ISS (ZARYA)</OBJECT_NAME>\n"
        "    <EPOCH>2026-05-04T12:34:56.789</EPOCH>\n"
        "  </object>\n"
        "</gp>\n"
    )

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/ajaxauth/login":
            return httpx.Response(
                200, headers={"set-cookie": "chocolatechip=abc; Path=/"}
            )
        if req.url.path == "/ajaxauth/logout":
            return httpx.Response(200)
        captured["path"] = req.url.path
        return httpx.Response(200, text=xml_body)

    async with _make_client(handler) as st:
        st._last_request_at = 0.0
        body = await st.get_recent_gp_xml(window_days=30)

    assert body == xml_body
    # URL anatomy: gp / EPOCH / >now-30 / orderby / NORAD_CAT_ID,EPOCH / format / xml
    assert "/basicspacedata/query/class/gp/EPOCH/" in captured["path"]
    assert "%3Enow-30" in captured["path"] or ">now-30" in captured["path"]
    assert "/orderby/NORAD_CAT_ID,EPOCH/" in captured["path"]
    assert captured["path"].endswith("/format/xml")


async def test_get_recent_gp_3le_parses_names_with_spaces():
    """`get_recent_gp_3le` must preserve TLE names that contain spaces.

    Common real-world names: ``STARLINK-1008``, ``ISS (ZARYA)``,
    ``ONEWEB-0123  ``. The parser must NOT splite them.
    """
    captured: dict[str, str] = {}
    body = (
        "STARLINK-1008  \n"
        "1 44714U 19074B   24123.45678901  .00012345  00000-0  22334-4 0  9990\n"
        "2 44714  53.0000 100.0000 0001234  12.3456 347.6543 15.06000000123456\n"
        "ISS (ZARYA)\n"
        "1 25544U 98067A   24123.45678901  .00012345  00000-0  22334-4 0  9990\n"
        "2 25544  51.6400 123.4567 0001234  12.3456 347.6543 15.50000000123450\n"
    )

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/ajaxauth/login":
            return httpx.Response(
                200, headers={"set-cookie": "chocolatechip=abc; Path=/"}
            )
        if req.url.path == "/ajaxauth/logout":
            return httpx.Response(200)
        captured["path"] = req.url.path
        return httpx.Response(200, text=body)

    async with _make_client(handler) as st:
        st._last_request_at = 0.0
        triples = await st.get_recent_gp_3le(window_days=30)

    assert captured["path"].endswith("/format/3le")
    assert len(triples) == 2
    name0, l1_0, l2_0 = triples[0]
    # Name must be preserved including the trailing space we stripped via rstrip,
    # and the embedded internal spaces must be retained for ISS (ZARYA).
    assert name0 == "STARLINK-1008"
    assert l1_0.startswith("1 44714")
    assert l2_0.startswith("2 44714")
    name1, l1_1, l2_1 = triples[1]
    assert name1 == "ISS (ZARYA)"  # space inside name preserved
    assert l1_1.startswith("1 25544")
    assert l2_1.startswith("2 25544")


async def test_get_gp_history_tle_url_includes_date_range():
    """The exact date range string `2026-05-04--2026-05-05` must appear
    verbatim in the request path."""
    captured: dict[str, str] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/ajaxauth/login":
            return httpx.Response(
                200, headers={"set-cookie": "chocolatechip=abc; Path=/"}
            )
        if req.url.path == "/ajaxauth/logout":
            return httpx.Response(200)
        captured["path"] = req.url.path
        return httpx.Response(200, text=_FAKE_TLE)

    async with _make_client(handler) as st:
        st._last_request_at = 0.0
        pairs = await st.get_gp_history_tle(
            start_date="2026-05-04",
            end_date="2026-05-05",
            show_empty=True,
        )

    assert "/basicspacedata/query/class/gp_history/CREATION_DATE/" in captured["path"]
    assert "2026-05-04--2026-05-05" in captured["path"]
    assert "/orderby/NORAD_CAT_ID,EPOCH/" in captured["path"]
    assert "/format/tle" in captured["path"]
    assert captured["path"].endswith("/emptyresult/show")
    # And the parser still returned the standard (l1, l2) pairs.
    assert len(pairs) == 1
    assert pairs[0][0].startswith("1 25544")

    # Without show_empty the suffix should be absent.
    captured.clear()

    async with _make_client(handler) as st:
        st._last_request_at = 0.0
        await st.get_gp_history_tle(
            start_date="2026-05-04",
            end_date="2026-05-05",
            show_empty=False,
        )
    assert captured["path"].endswith("/format/tle")
    assert "emptyresult" not in captured["path"]


async def test_no_credentials_in_logs(caplog):
    """structlog/log records must never contain the password string."""
    secret = "top-SECRET-do-not-leak-12345"

    def handler(req: httpx.Request) -> httpx.Response:
        # Force the auth-failed code path; that's where logging is densest.
        return httpx.Response(401)

    caplog.set_level(logging.DEBUG)
    async with _make_client(handler, password=secret) as st:
        with pytest.raises(RuntimeError):
            await st.login()

    blob = "\n".join(rec.getMessage() for rec in caplog.records)
    assert secret not in blob
    # The repr must not leak the identity either.
    assert repr(st) == "SpaceTrackClient(identity='[redacted]')"
