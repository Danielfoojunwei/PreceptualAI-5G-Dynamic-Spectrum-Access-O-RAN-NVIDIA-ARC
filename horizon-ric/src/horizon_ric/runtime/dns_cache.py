"""TTL-based DNS resolver cache for the SMO/A1/R1 endpoints.

Why this exists
---------------
Every ``httpx`` request to the SMO opens a fresh socket; on Linux that
triggers a `getaddrinfo(3)` call into glibc + `nss-resolve` + the
local stub resolver. Under steady soak (1 emit / 5 s) the lookups are
fine, but during a burst (e.g. a fault-injection storm) we hammer the
resolver hundreds of times per second for the *same* host. That is
both wasteful and a real source of tail latency on busy edge nodes
where the resolver is also serving NTP/Kubelet/etc.

This module provides a small, dependency-free TTL cache around
``socket.getaddrinfo`` and an ``httpx`` transport hook that consults
it before each connect. The default TTL is 30 s — short enough that a
genuine SMO failover (CNAME flip) is observed inside the breaker's
``reset_timeout=30 s`` window, long enough to absorb a burst.

If the optional ``aiodns`` package is installed we use its async
resolver; otherwise we fall back to the stdlib (run inside a thread
pool via ``loop.run_in_executor`` to keep the event loop unblocked).

Failure mode: every miss falls back to live resolution. We never serve
stale records past their TTL, and we never block the call on a cache
fault (negative results bypass the cache entirely).

Wire-up
-------
The A1Adapter / R1Adapter accept an optional `dns_cache_ttl_s`
field on their config. When > 0 the adapter installs this transport.
Default is 0 (OFF) for backward compat with the existing soak/CI
tests, which assume vanilla httpx behaviour.

References:
    RFC 1035 §3.2.1 (TTL semantics).
    glibc getaddrinfo(3) — no client-side caching by default.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from dataclasses import dataclass
from typing import Optional

import httpx
import structlog

logger = structlog.get_logger(__name__)

# ── try optional aiodns -------------------------------------------------------
try:  # pragma: no cover — exercised only when installed
    import aiodns  # type: ignore

    _HAVE_AIODNS = True
except Exception:  # noqa: BLE001 — any import-side failure is fine
    _HAVE_AIODNS = False


# ── core TTL cache ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Entry:
    addrs: tuple[str, ...]
    expires_at: float


class TTLDNSCache:
    """Thread-safe TTL cache keyed by ``(host, port, family)``.

    We cache the *resolved* address list, not the full ``getaddrinfo``
    tuple, because httpx only needs a host string. The cache is
    intentionally tiny and synchronous — DNS RR sets are small and the
    lookup is dominated by the I/O it replaces.
    """

    def __init__(self, ttl_seconds: float = 30.0, max_entries: int = 1024):
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be > 0, got {ttl_seconds}")
        self._ttl = float(ttl_seconds)
        self._max = int(max_entries)
        self._table: dict[tuple[str, int, int], _Entry] = {}
        self._lock = threading.Lock()
        # Stats — handy for /metrics later, no functional purpose now.
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    # -- internal helpers ----------------------------------------------------

    def _now(self) -> float:
        return time.monotonic()

    def _evict_if_full(self) -> None:
        if len(self._table) < self._max:
            return
        # Drop the entry with the soonest expiry — cheap, no LRU bookkeeping.
        oldest_key = min(self._table, key=lambda k: self._table[k].expires_at)
        del self._table[oldest_key]
        self.evictions += 1

    # -- sync resolution -----------------------------------------------------

    def resolve(
        self, host: str, port: int = 0, family: int = socket.AF_UNSPEC
    ) -> tuple[str, ...]:
        """Return a cached address list, refreshing on miss/expiry.

        On any resolver failure we surface the exception — callers
        treat that as a connect-time error, identical to the no-cache
        path.
        """
        key = (host, port, family)
        now = self._now()
        with self._lock:
            entry = self._table.get(key)
            if entry is not None and entry.expires_at > now:
                self.hits += 1
                return entry.addrs

        # Miss / stale → live lookup outside the lock so a slow resolver
        # does not stall other callers.
        try:
            infos = socket.getaddrinfo(host, port, family)
        except socket.gaierror:
            # Negative result: do NOT cache; fall through to caller.
            with self._lock:
                self.misses += 1
            raise
        addrs = tuple({info[4][0] for info in infos})
        if not addrs:
            raise socket.gaierror(f"no addresses for {host!r}")

        with self._lock:
            self.misses += 1
            self._evict_if_full()
            self._table[key] = _Entry(addrs=addrs, expires_at=self._now() + self._ttl)
        return addrs

    # -- async resolution ----------------------------------------------------

    async def aresolve(
        self,
        host: str,
        port: int = 0,
        family: int = socket.AF_UNSPEC,
    ) -> tuple[str, ...]:
        """Same as :meth:`resolve` but never blocks the event loop.

        Uses ``aiodns`` if available; otherwise dispatches the stdlib
        call to the default thread pool.
        """
        key = (host, port, family)
        now = self._now()
        with self._lock:
            entry = self._table.get(key)
            if entry is not None and entry.expires_at > now:
                self.hits += 1
                return entry.addrs

        if _HAVE_AIODNS:  # pragma: no cover — optional path
            try:
                resolver = aiodns.DNSResolver()
                ans = await resolver.gethostbyname(host, family or socket.AF_INET)
                addrs = tuple(ans.addresses)
            except Exception:
                # Fall back to live stdlib lookup before erroring out;
                # this matches the failure-mode contract.
                addrs = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: tuple({i[4][0] for i in socket.getaddrinfo(host, port, family)})
                )
        else:
            addrs = await asyncio.get_running_loop().run_in_executor(
                None, lambda: tuple({i[4][0] for i in socket.getaddrinfo(host, port, family)})
            )
        if not addrs:
            raise socket.gaierror(f"no addresses for {host!r}")
        with self._lock:
            self.misses += 1
            self._evict_if_full()
            self._table[key] = _Entry(addrs=addrs, expires_at=self._now() + self._ttl)
        return addrs

    # -- maintenance helpers -------------------------------------------------

    def invalidate(self, host: Optional[str] = None) -> int:
        """Drop a single host (or the whole cache when ``host is None``).

        Returns the number of entries removed — useful for soak telemetry.
        """
        with self._lock:
            if host is None:
                n = len(self._table)
                self._table.clear()
                return n
            keys = [k for k in self._table if k[0] == host]
            for k in keys:
                del self._table[k]
            return len(keys)

    def __len__(self) -> int:
        with self._lock:
            return len(self._table)


# ── httpx transport wrapper ─────────────────────────────────────────────────


class CachingDNSTransport(httpx.AsyncBaseTransport):
    """``httpx`` async transport that pre-resolves the host through the
    TTL cache, then delegates to a real ``httpx.AsyncHTTPTransport``.

    httpx itself uses anyio + the stdlib for connect; what we rewrite is
    the *URL host* on the request to a cached IP, leaving the original
    ``Host:`` header intact so TLS SNI and HTTP routing are unchanged.
    """

    def __init__(
        self,
        cache: TTLDNSCache,
        inner: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self._cache = cache
        self._inner = inner or httpx.AsyncHTTPTransport()

    async def handle_async_request(
        self, request: httpx.Request
    ) -> httpx.Response:
        url = request.url
        host = url.host
        # Skip already-numeric hosts — IP literals don't go through DNS.
        if host and not _looks_like_ip(host):
            try:
                addrs = await self._cache.aresolve(host)
                if addrs:
                    # Preserve the original Host header so the server still
                    # routes by name; only swap the connect target.
                    if "host" not in (k.lower() for k in request.headers):
                        request.headers["Host"] = host
                    new_url = url.copy_with(host=addrs[0])
                    request = httpx.Request(
                        method=request.method,
                        url=new_url,
                        headers=request.headers,
                        content=request.content,
                        extensions=request.extensions,
                    )
            except Exception as exc:  # noqa: BLE001 — fall back to inner
                logger.warning("dns_cache.fallback", host=host, error=str(exc))
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()


def _looks_like_ip(host: str) -> bool:
    """Conservative check for ``host`` being an IP literal."""
    # IPv6 in URLs comes wrapped in [].
    if host.startswith("[") and host.endswith("]"):
        return True
    # Crude IPv4 sniff — full validation is the kernel's job.
    parts = host.split(".")
    if len(parts) == 4 and all(p.isdigit() for p in parts):
        return True
    return False


def build_caching_async_client(
    base_url: str,
    *,
    ttl_seconds: float = 30.0,
    timeout: float = 10.0,
    cache: Optional[TTLDNSCache] = None,
) -> httpx.AsyncClient:
    """Convenience builder for adapter wiring.

    Adapters call this when ``dns_cache_ttl_s > 0``; the returned
    ``AsyncClient`` is otherwise a normal httpx client and remains
    backward-compatible with all existing call sites.
    """
    cache = cache or TTLDNSCache(ttl_seconds=ttl_seconds)
    transport = CachingDNSTransport(cache=cache)
    return httpx.AsyncClient(base_url=base_url, timeout=timeout, transport=transport)


__all__ = [
    "CachingDNSTransport",
    "TTLDNSCache",
    "build_caching_async_client",
]
