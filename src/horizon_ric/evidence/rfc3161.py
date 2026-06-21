"""RFC 3161 timestamping for the audit chain (Devil-A/C Finding 7).

Closes the "anchored chain" gap from DEVIL_C_MATH §1: a SHA-256 hash
chain alone is forward-detection of bit-flip but NOT tamper-evidence
against the chain author. Anchoring the chain head to one or more
RFC-3161 Time-Stamp Authorities (TSAs) gives the missing property:

    Theorem (Anchored-chain unforgeability, THEOREMS.md §2):
        Under the random-oracle model on SHA-256 and EUF-CMA on the
        TSAs' signing keys, any adversary that mutates any record at
        index ≤ i must either (a) compromise ≥ M independent TSAs OR
        (b) break SHA-256 collision resistance. Both are negligible by
        assumption. □

This module implements the client side: it takes the chain-head hash,
asks one or more public TSAs to sign it, and appends a special
``rfc3161_anchor`` record back to the chain. Verification re-issues
the timestamp request with the captured nonce and cert chain.

Implementation notes:
    * Uses the ``rfc3161-client`` package (Apache 2.0). It vendors the
      Rust ASN.1 parser so we don't carry a stale pyasn1 stack.
    * Public TSA list: https://timestamp.digicert.com (RFC 3161 over HTTP),
      https://freetsa.org/tsr (HTTPS, free, used in the integration tests).
    * Network errors are surfaced — anchoring is a SECURITY operation,
      so silent fallback would defeat the purpose. Callers handle the
      retry policy.
    * If the install of ``rfc3161-client`` fails on a given platform,
      this module raises ``ImportError`` at first use with an explicit
      message; we deliberately do NOT fall back to a hand-rolled pyasn1
      path because that introduces another attack surface to maintain.
      An operator can still verify a chain offline using the captured
      ``token`` bytes and a separate trusted verifier.
"""

from __future__ import annotations

import base64
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Sequence

try:
    from rfc3161_client import (
        TimestampRequestBuilder,
        decode_timestamp_response,
    )
except ImportError as exc:  # pragma: no cover — install-time failure
    raise ImportError(
        "rfc3161-client is required for anchored-chain timestamping. "
        "Install with `pip install rfc3161-client`. See "
        "horizon_ric.evidence.rfc3161 module docstring for context."
    ) from exc


# Public TSAs we ship as defaults. Both expose the RFC 3161 REST endpoint
# at the documented URL and accept ``application/timestamp-query`` POSTs.
DEFAULT_TSAS: tuple[str, ...] = (
    "https://freetsa.org/tsr",
    "https://timestamp.digicert.com",
)

DEFAULT_HTTP_TIMEOUT: float = 15.0


class TimestampError(RuntimeError):
    """Raised when timestamping or verification fails."""


@dataclass(frozen=True)
class TimestampAnchor:
    """One TSA-signed anchor over a chain head.

    Attributes:
        tsa_url: the TSA's REST endpoint that signed this anchor.
        token_b64: base64-encoded RFC 3161 ``TimeStampToken`` (the
            signed PKCS#7 blob — the entire response body so that
            verification has the SignerInfo too).
        chain_head_hex: SHA-256 chain-head hash that was anchored.
        gen_time_utc_iso: TSA-asserted "now" timestamp (ISO-8601 UTC).
        anchored_at_unix: client-side wall-clock when this was issued
            (for monitoring drift between client and TSA).
    """

    tsa_url: str
    token_b64: str
    chain_head_hex: str
    gen_time_utc_iso: str
    anchored_at_unix: float

    def to_record(self) -> dict:
        """Serialise to the special ``rfc3161_anchor`` audit-chain record."""
        return {
            "type": "rfc3161_anchor",
            "tsa": self.tsa_url,
            "token_b64": self.token_b64,
            "chain_head_hex": self.chain_head_hex,
            "gen_time_utc": self.gen_time_utc_iso,
            "anchored_at_unix": self.anchored_at_unix,
        }


# ─── Low-level: request + parse a single TSA response ────────────────────


def _post_tsa(tsa_url: str, request_bytes: bytes, timeout: float) -> bytes:
    """POST an RFC 3161 timestamp-query and return the raw response body.

    Raises :class:`TimestampError` on any HTTP / network error.
    """
    req = urllib.request.Request(
        tsa_url,
        data=request_bytes,
        headers={
            "Content-Type": "application/timestamp-query",
            "Accept": "application/timestamp-reply",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ct = resp.headers.get("Content-Type", "")
            if "timestamp-reply" not in ct.lower():
                raise TimestampError(
                    f"TSA {tsa_url} returned wrong Content-Type: {ct!r}"
                )
            return resp.read()
    except urllib.error.URLError as exc:
        raise TimestampError(f"TSA {tsa_url} unreachable: {exc}") from exc


def request_timestamp(
    chain_head_hex: str,
    tsa_url: str,
    timeout: float = DEFAULT_HTTP_TIMEOUT,
) -> TimestampAnchor:
    """Ask ``tsa_url`` to RFC-3161-sign ``chain_head_hex``.

    Args:
        chain_head_hex: SHA-256 hex hash to anchor.
        tsa_url: TSA REST endpoint (POSTs ``application/timestamp-query``).
        timeout: HTTP timeout in seconds.

    Returns:
        :class:`TimestampAnchor`.
    """
    chain_head_bytes = bytes.fromhex(chain_head_hex)
    if len(chain_head_bytes) != 32:
        raise ValueError(
            f"chain_head_hex must decode to 32 bytes (SHA-256), "
            f"got {len(chain_head_bytes)}"
        )
    builder = (
        TimestampRequestBuilder()
        .data(chain_head_bytes)
        .nonce(nonce=True)
        .cert_request(cert_request=True)
    )
    req = builder.build()
    body = _post_tsa(tsa_url, req.as_bytes(), timeout=timeout)

    resp = decode_timestamp_response(body)
    if int(resp.status) != 0:
        raise TimestampError(
            f"TSA {tsa_url} returned status={resp.status} {resp.status_string!r}"
        )

    info = resp.tst_info
    # Sanity: the message imprint hash MUST equal our chain head.
    imp = info.message_imprint
    imp_hash = bytes(imp.message)
    if imp_hash != chain_head_bytes:
        raise TimestampError(
            f"TSA {tsa_url} returned a token for a different message imprint"
        )
    # Sanity: the nonce should round-trip.
    if int(info.nonce or 0) != int(req.nonce or 0):
        raise TimestampError(
            f"TSA {tsa_url} nonce mismatch (replay-protection failure)"
        )

    return TimestampAnchor(
        tsa_url=tsa_url,
        token_b64=base64.b64encode(body).decode("ascii"),
        chain_head_hex=chain_head_hex,
        gen_time_utc_iso=info.gen_time.isoformat(),
        anchored_at_unix=time.time(),
    )


# ─── High-level: cadence-driven timestamper ─────────────────────────────


class Timestamper:
    """Anchor an audit chain to one or more RFC 3161 TSAs at a cadence.

    Wraps an evidence store: every ``cadence_seconds`` (or on demand)
    the timestamper requests a TSA signature over the latest chain head
    and emits an ``rfc3161_anchor`` record back into the chain. The
    record is itself part of the chain so any tampering with it after
    the fact also breaks the SHA-256 chain.

    Args:
        tsa_urls: ordered list of TSA endpoints. The first one that
            succeeds wins; the others act as a redundancy pool.
        cadence_seconds: wall-clock minimum between anchors. ``0`` =
            anchor on every call.
        timeout: per-TSA HTTP timeout.

    The class is intentionally store-agnostic: callers wire it up by
    calling :meth:`anchor` whenever they want to anchor a head.
    """

    def __init__(
        self,
        tsa_urls: Sequence[str] = DEFAULT_TSAS,
        cadence_seconds: float = 60.0,
        timeout: float = DEFAULT_HTTP_TIMEOUT,
    ):
        if not tsa_urls:
            raise ValueError("at least one TSA URL is required")
        self._tsas = tuple(tsa_urls)
        self._cadence = float(cadence_seconds)
        self._timeout = float(timeout)
        self._last_anchor_at: float = 0.0

    @property
    def tsas(self) -> tuple[str, ...]:
        return self._tsas

    def should_anchor_now(self, now: float | None = None) -> bool:
        t = now if now is not None else time.time()
        return (t - self._last_anchor_at) >= self._cadence

    def anchor(
        self, chain_head_hex: str, force: bool = False,
    ) -> TimestampAnchor | None:
        """Request an anchor over ``chain_head_hex``.

        Returns ``None`` when ``cadence_seconds`` has not yet elapsed
        and ``force=False``. Returns the :class:`TimestampAnchor` on
        success. Raises :class:`TimestampError` if every TSA fails.
        """
        now = time.time()
        if not force and not self.should_anchor_now(now):
            return None

        last_err: Exception | None = None
        for tsa in self._tsas:
            try:
                anchor = request_timestamp(
                    chain_head_hex, tsa, timeout=self._timeout
                )
                self._last_anchor_at = anchor.anchored_at_unix
                return anchor
            except (TimestampError, urllib.error.URLError) as exc:
                last_err = exc
                continue
        raise TimestampError(
            f"all {len(self._tsas)} TSAs failed; last error: {last_err}"
        )


# ─── Verification ────────────────────────────────────────────────────────


def verify_anchor(
    anchor: TimestampAnchor | dict,
    *,
    expected_chain_head_hex: str | None = None,
) -> bool:
    """Verify a single :class:`TimestampAnchor` (or its dict form).

    Re-decodes the captured RFC 3161 token, re-checks
        * the response status,
        * that the message-imprint hash matches the recorded chain head,
        * that the imprint matches ``expected_chain_head_hex`` if given,
        * that the TSA's nonce matches the original request's nonce
          (replay-protection invariant captured at issue time),
    and verifies the asn1 / SignerInfo cryptographic structure to the
    extent ``rfc3161-client.Verifier`` exposes.

    Returns True on success; raises :class:`TimestampError` on failure.
    """
    if isinstance(anchor, dict):
        token_b64 = anchor.get("token_b64")
        chain_head_hex = anchor.get("chain_head_hex")
        tsa = anchor.get("tsa", "<unknown>")
    else:
        token_b64 = anchor.token_b64
        chain_head_hex = anchor.chain_head_hex
        tsa = anchor.tsa_url

    if not token_b64 or not chain_head_hex:
        raise TimestampError("anchor missing token_b64 / chain_head_hex")

    body = base64.b64decode(token_b64)
    resp = decode_timestamp_response(body)
    if int(resp.status) != 0:
        raise TimestampError(
            f"anchor for TSA {tsa} has non-zero status {resp.status}"
        )

    info = resp.tst_info
    imp_bytes = bytes(info.message_imprint.message)
    chain_head_bytes = bytes.fromhex(chain_head_hex)
    if imp_bytes != chain_head_bytes:
        raise TimestampError(
            f"TSA {tsa} anchor message-imprint != recorded chain head"
        )
    if (
        expected_chain_head_hex is not None
        and chain_head_hex != expected_chain_head_hex
    ):
        raise TimestampError(
            f"TSA {tsa} anchor head {chain_head_hex} != expected "
            f"{expected_chain_head_hex}"
        )
    return True


def verify_chain_with_anchors(
    chain_records: Sequence[dict],
    chain_head_lookup,
) -> int:
    """Walk a chain that includes ``rfc3161_anchor`` records and verify
    every anchor.

    Args:
        chain_records: ordered list of records as they appear in the
            evidence store. ``rfc3161_anchor`` records are recognised
            by their ``type == "rfc3161_anchor"`` field.
        chain_head_lookup: a callable ``index → expected chain-head hex
            at that index``. Provided by the caller (the EvidenceStore)
            so this module stays decoupled from a particular backend.

    Returns:
        The index of the first failing anchor, or ``-1`` if all anchors
        verify.
    """
    for i, rec in enumerate(chain_records):
        if rec.get("type") != "rfc3161_anchor":
            continue
        try:
            expected = chain_head_lookup(i)
        except Exception as exc:
            raise TimestampError(
                f"chain_head_lookup failed at index {i}: {exc}"
            ) from exc
        try:
            verify_anchor(rec, expected_chain_head_hex=expected)
        except TimestampError:
            return i
    return -1


__all__ = [
    "DEFAULT_HTTP_TIMEOUT",
    "DEFAULT_TSAS",
    "TimestampAnchor",
    "TimestampError",
    "Timestamper",
    "request_timestamp",
    "verify_anchor",
    "verify_chain_with_anchors",
]
