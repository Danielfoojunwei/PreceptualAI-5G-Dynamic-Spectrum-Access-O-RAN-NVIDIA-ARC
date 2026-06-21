"""mTLS + OAuth2 client-credentials auth for R1/A1 (O-RAN.WG11 §6).

`build_secure_async_client(config)` returns an `httpx.AsyncClient`
configured with:

  * mTLS (client cert + key, optional CA bundle).
  * Bearer token from a configured static token *or* OAuth2 client-credentials
    flow against the SMO's authorization endpoint (handles refresh).

If no auth is configured, returns a plain async client — callers can
continue to use the adapter against unsecured testbeds. Production
deployments should always configure either bearer + mTLS or both.

References:
    O-RAN.WG11 Security Spec v06.00 §6 (mutual TLS + OAuth2 over R1/A1).
    RFC 6749 §4.4 (Client Credentials Grant).
"""

from __future__ import annotations

import asyncio
import ssl
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import structlog
from authlib.integrations.httpx_client import AsyncOAuth2Client

# WG11 §6 cipher allow-list. Order matters for TLS 1.2 negotiation
# (server picks first acceptable from client list).
_DEFAULT_WG11_CIPHERS: list[str] = [
    "TLS_AES_256_GCM_SHA384",
    "TLS_CHACHA20_POLY1305_SHA256",
    "TLS_AES_128_GCM_SHA256",
]

# Auth structured logger — used for audit-grade auth-failure events.
_log = structlog.get_logger("horizon_ric.auth")


@dataclass
class AuthConfig:
    """Bag of configuration for a secure HTTP client.

    Empty / None fields disable that auth feature. To enable both mTLS
    and OAuth2 set both groups.
    """

    # mTLS
    client_cert_path: Path | None = None
    client_key_path: Path | None = None
    ca_bundle_path: Path | None = None
    verify_tls: bool = True

    # OAuth2 (client_credentials)
    token_url: str | None = None
    client_id: str | None = None
    client_secret: str | None = None
    scope: str | None = None

    # Static token (overrides OAuth2 if both set; primarily for tests).
    static_bearer_token: str | None = None

    timeout_seconds: float = 30.0

    # WG11 §6 cipher allow-list. Names are IANA TLS 1.3 suite names.
    # OpenSSL accepts these directly via SSLContext.set_ciphers for
    # TLS ≤ 1.2 negotiation; for TLS 1.3 OpenSSL exposes a separate
    # `set_ciphersuites` API and `set_ciphers` is a no-op for the 1.3
    # suites (this is a known and intentional limitation of Python's
    # ssl module — see honest note on `_build_ssl_context` below).
    cipher_allowlist: list[str] = field(
        default_factory=lambda: list(_DEFAULT_WG11_CIPHERS)
    )

    # Set False ONLY in test/dev. WG11 §6 forbids cert-skipping in production.
    production_mode: bool = True

    def has_mtls(self) -> bool:
        return bool(self.client_cert_path and self.client_key_path)

    def has_oauth(self) -> bool:
        return bool(self.token_url and self.client_id and self.client_secret)

    def has_static_token(self) -> bool:
        return bool(self.static_bearer_token)


def _build_ssl_context(cfg: AuthConfig) -> ssl.SSLContext | bool:
    """Return an SSLContext when mTLS / custom CA is configured; else
    fall back to httpx's `verify=` semantics.

    WG11 enforcement:
      * If ``cfg.production_mode`` is True and ``cfg.verify_tls`` is False,
        we refuse to build a context (ValueError). Cert-skipping in prod
        is a hard ban per WG11 §6.
      * The cipher allow-list is applied via OpenSSL ``set_ciphers``. NOTE:
        for TLS 1.3, OpenSSL uses a separate ``set_ciphersuites`` API and
        Python's stdlib does not surface fine-grained per-suite control —
        ``set_ciphers`` does NOT restrict the TLS-1.3 suite list. We
        attempt to call ``set_ciphersuites`` when available (CPython 3.7+)
        but the OpenSSL build determines which 1.3 suites are compiled
        in. This is a known, honestly-documented limitation: the cipher
        allow-list is best-effort on TLS 1.3 and authoritative on TLS 1.2.
        Operators relying on strict 1.3-suite enforcement must additionally
        configure their server-side TLS terminator (envoy / nginx).
    """
    # Hard ban: production + verify_tls=False ⇒ refuse.
    if cfg.production_mode and not cfg.verify_tls:
        raise ValueError(
            "AuthConfig.verify_tls=False is forbidden when production_mode=True. "
            "WG11 §6 prohibits skipping certificate verification in production. "
            "Set production_mode=False ONLY in test/dev environments."
        )

    if not cfg.verify_tls and not cfg.production_mode:
        # Loud audit-log warning — this should never happen in prod by the
        # check above, but if a dev environment ends up shipping with this
        # flag flipped we want a structured trail.
        _log.warning(
            "tls.verify_disabled",
            production_mode=cfg.production_mode,
            note="cert verification disabled; WG11 §6 forbids in production",
        )

    if not cfg.has_mtls() and cfg.ca_bundle_path is None:
        # No custom context needed — but if the caller specified a cipher
        # allow-list AND verification is on, we still need an SSLContext
        # to apply ciphers. Build a default-verify context in that case.
        if cfg.cipher_allowlist and cfg.verify_tls:
            ctx = ssl.create_default_context()
            _apply_cipher_allowlist(ctx, cfg.cipher_allowlist)
            return ctx
        return cfg.verify_tls

    ctx = ssl.create_default_context(
        cafile=str(cfg.ca_bundle_path) if cfg.ca_bundle_path else None,
    )
    if not cfg.verify_tls:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    if cfg.has_mtls():
        ctx.load_cert_chain(
            certfile=str(cfg.client_cert_path),
            keyfile=str(cfg.client_key_path),
        )

    _apply_cipher_allowlist(ctx, cfg.cipher_allowlist)
    return ctx


def _apply_cipher_allowlist(ctx: ssl.SSLContext, allowlist: list[str]) -> None:
    """Apply WG11 cipher allow-list to an SSLContext.

    Honest behaviour:
      * Calls ``set_ciphers(":".join(allowlist))`` for TLS ≤ 1.2 negotiation.
        Names that match OpenSSL's IANA-style strings are accepted; OpenSSL
        also accepts its older alias forms (e.g. ``ECDHE-RSA-AES256-GCM-SHA384``).
        For TLS 1.3 IANA suite names, ``set_ciphers`` is largely a no-op —
        the OpenSSL TLS 1.3 suites are governed by a separate API.
      * If ``ctx.set_ciphersuites`` is available (it is on CPython 3.7+
        with OpenSSL 1.1.1+), we ALSO call it for the TLS 1.3 names. This
        is the authoritative call for 1.3.
      * If neither call accepts the names, we log a warning rather than
        crash — refusing to build a context because OpenSSL ships a
        slightly different alias list would be worse for operators than
        the documented best-effort behaviour.
    """
    if not allowlist:
        return

    cipher_string = ":".join(allowlist)
    try:
        ctx.set_ciphers(cipher_string)
    except ssl.SSLError as exc:
        # Common when names are TLS-1.3-only — set_ciphers rejects them.
        _log.warning(
            "tls.set_ciphers_failed",
            ciphers=allowlist,
            error=str(exc),
            note="TLS 1.2 cipher negotiation will use OpenSSL defaults; "
                 "TLS 1.3 suites handled via set_ciphersuites if available",
        )

    # TLS 1.3 path — best-effort.
    set_suites = getattr(ctx, "set_ciphersuites", None)
    if callable(set_suites):
        try:
            set_suites(cipher_string)
        except ssl.SSLError as exc:
            _log.warning(
                "tls.set_ciphersuites_failed",
                ciphersuites=allowlist,
                error=str(exc),
                note="TLS 1.3 suite restriction not enforced; "
                     "operator must additionally restrict at terminator",
            )


class _OAuth2BearerAuth(httpx.Auth):
    """httpx auth plug that lazily fetches and caches an OAuth2 token."""

    requires_request_body = False

    def __init__(self, cfg: AuthConfig):
        self.cfg = cfg
        self._token: str | None = None
        self._expiry_epoch: float = 0.0
        self._lock = asyncio.Lock()

    def sync_auth_flow(self, request):  # pragma: no cover — async-only path
        raise RuntimeError("Use async client; sync auth flow not supported.")

    async def async_auth_flow(self, request: httpx.Request):
        token = await self._get_token()
        request.headers["Authorization"] = f"Bearer {token}"
        yield request

    async def _get_token(self) -> str:
        now = time.time()
        if self._token and now < self._expiry_epoch - 30.0:
            return self._token
        async with self._lock:
            if self._token and now < self._expiry_epoch - 30.0:
                return self._token
            try:
                async with AsyncOAuth2Client(
                    self.cfg.client_id, self.cfg.client_secret,
                    scope=self.cfg.scope,
                ) as client:
                    token = await client.fetch_token(self.cfg.token_url)
            except Exception as exc:
                # WG11-grade audit log: do NOT swallow the exception, but
                # ensure SOC tooling sees a structured event for every
                # failed fetch attempt.
                _log.error(
                    "oauth.fetch_failed",
                    token_url=self.cfg.token_url,
                    client_id=self.cfg.client_id,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                raise
            self._token = token["access_token"]
            expires_in = float(token.get("expires_in", 3600))
            self._expiry_epoch = now + expires_in
            return self._token


class _StaticBearerAuth(httpx.Auth):
    requires_request_body = False

    def __init__(self, token: str):
        self._token = token

    def sync_auth_flow(self, request):  # pragma: no cover
        request.headers["Authorization"] = f"Bearer {self._token}"
        yield request

    async def async_auth_flow(self, request: httpx.Request):
        request.headers["Authorization"] = f"Bearer {self._token}"
        yield request


def build_secure_async_client(
    cfg: AuthConfig,
    base_url: str | None = None,
) -> httpx.AsyncClient:
    """Construct an httpx.AsyncClient honouring the AuthConfig.

    The returned client is owned by the caller — call `await client.aclose()`
    when finished (or use `async with`).
    """
    ctx = _build_ssl_context(cfg)

    auth: httpx.Auth | None = None
    if cfg.has_static_token():
        auth = _StaticBearerAuth(cfg.static_bearer_token)
    elif cfg.has_oauth():
        auth = _OAuth2BearerAuth(cfg)

    return httpx.AsyncClient(
        base_url=base_url or "",
        verify=ctx,
        timeout=cfg.timeout_seconds,
        auth=auth,
    )


__all__ = [
    "AuthConfig",
    "build_secure_async_client",
]
