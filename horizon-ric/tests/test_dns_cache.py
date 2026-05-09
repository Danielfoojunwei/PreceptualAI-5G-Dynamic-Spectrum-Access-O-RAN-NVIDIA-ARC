"""Tests for `horizon_ric.runtime.dns_cache`.

We don't want these tests to depend on the real DNS resolver, so we
patch ``socket.getaddrinfo`` at the call site to return a deterministic
record set. The TTL clock is stubbed via the cache's `_now` method so
expiry is exercised without `time.sleep` in the test path.
"""

from __future__ import annotations

import socket
from typing import Any

import pytest

from horizon_ric.runtime.dns_cache import (
    CachingDNSTransport,
    TTLDNSCache,
    build_caching_async_client,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def _fake_getaddrinfo(addrs: list[str]):
    """Return a `getaddrinfo` stand-in that yields the supplied addrs."""

    def _impl(host: str, port: int = 0, family: int = 0, *args: Any, **kwargs: Any):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port)) for a in addrs]

    return _impl


# ── (a) cache hit returns same address ──────────────────────────────────────


def test_cache_hit_returns_same_address(monkeypatch):
    """Two consecutive resolves within the TTL must hit the cache and
    return the *same* address tuple (object equality suffices because
    the cache stores a frozen tuple)."""
    cache = TTLDNSCache(ttl_seconds=30.0)
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo(["10.0.0.1"]))

    first = cache.resolve("smo.example.com")
    second = cache.resolve("smo.example.com")

    assert first == ("10.0.0.1",)
    assert first == second
    # One miss (first call), one hit (second call).
    assert cache.misses == 1
    assert cache.hits == 1


# ── (b) cache expires after TTL ─────────────────────────────────────────────


def test_cache_expires_after_ttl(monkeypatch):
    """After the TTL elapses the cache must re-resolve. We drive the
    clock via the `_now` hook on the cache instance — no real sleep."""
    cache = TTLDNSCache(ttl_seconds=5.0)

    seq = iter([100.0, 100.0, 106.0, 106.0])
    monkeypatch.setattr(cache, "_now", lambda: next(seq))

    # Resolver returns a different RR set each call.
    resolved = iter([["10.0.0.1"], ["10.0.0.2"]])
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port=0, family=0, *a, **kw: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (next(resolved)[0], port))
        ],
    )

    first = cache.resolve("smo.example.com")
    second = cache.resolve("smo.example.com")  # past TTL → re-resolves

    assert first == ("10.0.0.1",)
    assert second == ("10.0.0.2",)
    assert cache.misses == 2  # both went to the live resolver
    assert cache.hits == 0


# ── (c) failure mode falls back to live resolution ──────────────────────────


def test_failure_mode_falls_back(monkeypatch):
    """When the live resolver throws, we surface the error rather than
    serving a stale entry — and crucially we DO NOT poison the cache
    with the failure (next attempt must hit the resolver again).

    A subsequent successful resolve must succeed and populate the cache.
    """
    cache = TTLDNSCache(ttl_seconds=30.0)

    calls = {"n": 0}

    def flaky(host, port=0, family=0, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise socket.gaierror("name or service not known")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.9", port))]

    monkeypatch.setattr(socket, "getaddrinfo", flaky)

    with pytest.raises(socket.gaierror):
        cache.resolve("smo.example.com")
    # Cache must be empty — a negative result must not be cached.
    assert len(cache) == 0

    # Resolver recovers; live lookup runs again and result is cached.
    addrs = cache.resolve("smo.example.com")
    assert addrs == ("10.0.0.9",)
    assert len(cache) == 1
    assert calls["n"] == 2


# ── extra coverage: builder smoke + IP-literal short-circuit ────────────────


def test_build_caching_async_client_smoke():
    client = build_caching_async_client(
        base_url="http://nonrtric.local:8080", ttl_seconds=30.0
    )
    assert client.base_url.host == "nonrtric.local"
    # Don't make a real request — just confirm the transport plumbed in.
    assert isinstance(client._transport, CachingDNSTransport)


def test_invalidate_clears_entries(monkeypatch):
    cache = TTLDNSCache(ttl_seconds=30.0)
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo(["10.0.0.1"]))
    cache.resolve("a.example.com")
    cache.resolve("b.example.com")
    assert len(cache) == 2
    n = cache.invalidate(host="a.example.com")
    assert n == 1
    assert len(cache) == 1
    n = cache.invalidate()
    assert n == 1
    assert len(cache) == 0
