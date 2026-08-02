"""A fail-closed NVCF invocation client.

Speaks the protocol in :mod:`horizon_nvcf.protocol`: POST to the pass-through
endpoint, and if the response is 202, poll the status endpoint with the
``NVCF-REQID`` the service returned until it resolves or the budget runs out.

Three properties matter more than feature coverage, because the thing on the
other end of this socket is an *untrusted inference service* whose output
becomes a proposal to change a radio network:

**No response ever becomes a default action.** Every non-fulfilled
disposition returns ``payload=None``. There is no "empty result" a caller
could iterate over and treat as "the agent proposed nothing, carry on".

**The version is part of the identity.** An NVCF ``functionId`` is stable
across redeployments; only ``versionId`` is immutable. :class:`NvcfClient`
therefore invokes the versioned path whenever a version is configured, and
:mod:`horizon_nvcf.agent` refuses to build an envelope from a response whose
version is not the authorised one.

**Unknown is not success.** A status code the contract does not declare for
the endpoint is :data:`~horizon_nvcf.protocol.Disposition.UNKNOWN` and
carries no payload, rather than being optimistically parsed.

The transport is injected. ``NvcfClient`` takes any callable with the shape
``(method, url, headers, body) -> (status, headers, body)``; the default
built on :mod:`urllib.request` is used against a real endpoint, and tests and
gates drive a recorded one. No third-party HTTP dependency is added.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from horizon_nvcf.protocol import (
    Disposition,
    InvocationOutcome,
    header,
    invoke_path,
    poll_path,
)
from horizon_nvcf.protocol import classify as _classify

__all__ = ["NvcfClient", "NvcfClientConfig", "Transport", "urllib_transport"]

# (method, url, headers, body) -> (status, headers, body)
Transport = Callable[
    [str, str, Mapping[str, str], bytes | None], "tuple[int, Mapping[str, str], bytes]"
]

DEFAULT_BASE_URL = "https://api.nvcf.nvidia.com"

# NVCF documents a 5 MB inline request-payload limit, above which the caller
# must upload an Asset and reference it via NVCF-INPUT-ASSET-REFERENCES. We
# refuse rather than truncate: a silently-clipped telemetry batch is a wrong
# proposal built from a partial picture, which is worse than no proposal.
MAX_INLINE_PAYLOAD_BYTES = 5 * 1024 * 1024


@dataclass(frozen=True)
class NvcfClientConfig:
    """Everything needed to reach one NVCF-hosted function."""

    function_id: str
    version_id: str | None = None
    base_url: str = DEFAULT_BASE_URL
    api_key: str | None = None
    # NVCF-POLL-SECONDS: how long the service holds the connection open before
    # replying 202. Kept short so control-loop latency stays bounded.
    poll_seconds: int = 5
    # Client-side budget for the whole invoke+poll sequence.
    deadline_s: float = 30.0
    poll_interval_s: float = 0.5
    max_polls: int = 60

    def __post_init__(self) -> None:
        if not self.function_id:
            raise ValueError("function_id is required")
        if self.deadline_s <= 0:
            raise ValueError("deadline_s must be positive")
        if self.max_polls < 1:
            raise ValueError("max_polls must be at least 1")


def urllib_transport(
    method: str, url: str, headers: Mapping[str, str], body: bytes | None
) -> tuple[int, Mapping[str, str], bytes]:
    """Default transport. Never follows redirects — 302 is meaningful here."""

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *_a: Any, **_kw: Any) -> None:
            return None

    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
    try:
        with opener.open(req) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:  # 3xx/4xx/5xx all arrive here
        return exc.code, dict(exc.headers or {}), exc.read()


class NvcfClient:
    """Invoke an NVCF function and resolve its result, fail-closed throughout."""

    def __init__(
        self,
        config: NvcfClientConfig,
        transport: Transport | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = config
        self._transport = transport or urllib_transport
        self._sleep = sleep
        self._now = monotonic

    # ── headers ──────────────────────────────────────────────────────────
    def _headers(self, asset_refs: list[str] | None = None) -> dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            # Header names come from the contract, so a rename upstream raises
            # here instead of producing a request that is quietly ignored.
            header("NVCF-POLL-SECONDS"): str(self.cfg.poll_seconds),
        }
        if self.cfg.api_key:
            h["Authorization"] = f"Bearer {self.cfg.api_key}"
        if asset_refs:
            h[header("NVCF-INPUT-ASSET-REFERENCES")] = ",".join(asset_refs)
        return h

    @staticmethod
    def _read(headers: Mapping[str, str], name: str) -> str | None:
        """Case-insensitive header read — HTTP header names are not case-sensitive."""
        target = name.upper()
        for k, v in headers.items():
            if k.upper() == target:
                return v
        return None

    def _meta(self, headers: Mapping[str, str]) -> dict[str, Any]:
        pct = self._read(headers, header("NVCF-PERCENT-COMPLETE"))
        try:
            pct_i = int(pct) if pct is not None else None
        except ValueError:
            pct_i = None
        return {
            "request_id": self._read(headers, header("NVCF-REQID")),
            "nvcf_status": self._read(headers, header("NVCF-STATUS")),
            "percent_complete": pct_i,
            "location": self._read(headers, "Location"),
        }

    @staticmethod
    def _parse(body: bytes) -> dict[str, Any] | None:
        """Parse a fulfilled body. Anything that is not a JSON object is None."""
        try:
            obj = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return obj if isinstance(obj, dict) else None

    # ── invocation ───────────────────────────────────────────────────────
    def invoke(
        self, payload: Mapping[str, Any], *, asset_refs: list[str] | None = None
    ) -> InvocationOutcome:
        """Invoke the function and resolve the result, or explain why not."""
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        if len(body) > MAX_INLINE_PAYLOAD_BYTES:
            return InvocationOutcome(
                disposition=Disposition.REFUSED,
                status=0,
                detail=(
                    f"request payload is {len(body)} bytes, above NVCF's "
                    f"{MAX_INLINE_PAYLOAD_BYTES}-byte inline limit; upload it "
                    "as an Asset and pass asset_refs instead. Refusing rather "
                    "than truncating."
                ),
            )

        url = self.cfg.base_url.rstrip("/") + invoke_path(
            self.cfg.function_id, self.cfg.version_id
        )
        started = self._now()
        try:
            status, headers, raw = self._transport(
                "POST", url, self._headers(asset_refs), body
            )
        except OSError as exc:
            return InvocationOutcome(
                disposition=Disposition.REFUSED,
                status=0,
                detail=f"transport failure invoking NVCF: {exc}",
            )

        meta = self._meta(headers)
        disp = _classify(status, "invoke")

        if disp == Disposition.FULFILLED:
            return self._fulfilled(status, meta, raw, polls=0)
        if disp != Disposition.PENDING:
            return InvocationOutcome(
                disposition=disp,
                status=status,
                detail=self._terminal_detail(disp, status, meta),
                **meta,
            )

        request_id = meta["request_id"]
        if not request_id:
            return InvocationOutcome(
                disposition=Disposition.UNKNOWN,
                status=status,
                detail=(
                    "NVCF returned 202 without an "
                    f"{header('NVCF-REQID')} header; the result is "
                    "unreachable, so there is nothing to wait for"
                ),
                **{k: v for k, v in meta.items() if k != "request_id"},
            )
        return self._poll(request_id, started)

    # ── polling ──────────────────────────────────────────────────────────
    def _poll(self, request_id: str, started: float) -> InvocationOutcome:
        url = self.cfg.base_url.rstrip("/") + poll_path(request_id)
        headers = {"Accept": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"

        last: dict[str, Any] = {"request_id": request_id}
        for attempt in range(1, self.cfg.max_polls + 1):
            if self._now() - started >= self.cfg.deadline_s:
                return InvocationOutcome(
                    disposition=Disposition.PENDING,
                    status=202,
                    polls=attempt - 1,
                    detail=(
                        f"deadline of {self.cfg.deadline_s}s elapsed with the "
                        "invocation still pending; no proposal was produced"
                    ),
                    **last,
                )
            self._sleep(self.cfg.poll_interval_s)
            try:
                status, hdrs, raw = self._transport("GET", url, headers, None)
            except OSError as exc:
                return InvocationOutcome(
                    disposition=Disposition.REFUSED,
                    status=0,
                    polls=attempt,
                    detail=f"transport failure polling NVCF: {exc}",
                    **last,
                )
            meta = self._meta(hdrs)
            meta["request_id"] = meta["request_id"] or request_id
            last = meta
            disp = _classify(status, "poll")
            if disp == Disposition.FULFILLED:
                return self._fulfilled(status, meta, raw, polls=attempt)
            if disp == Disposition.PENDING:
                continue
            return InvocationOutcome(
                disposition=disp,
                status=status,
                polls=attempt,
                detail=self._terminal_detail(disp, status, meta),
                **meta,
            )

        return InvocationOutcome(
            disposition=Disposition.PENDING,
            status=202,
            polls=self.cfg.max_polls,
            detail=(
                f"exhausted {self.cfg.max_polls} polls with the invocation "
                "still pending; no proposal was produced"
            ),
            **last,
        )

    # ── outcomes ─────────────────────────────────────────────────────────
    def _fulfilled(
        self, status: int, meta: dict[str, Any], raw: bytes, *, polls: int
    ) -> InvocationOutcome:
        payload = self._parse(raw)
        if payload is None:
            return InvocationOutcome(
                disposition=Disposition.UNKNOWN,
                status=status,
                polls=polls,
                detail=(
                    "NVCF reported the invocation fulfilled but the body is "
                    "not a JSON object; refusing to treat it as a proposal"
                ),
                **meta,
            )
        return InvocationOutcome(
            disposition=Disposition.FULFILLED,
            status=status,
            polls=polls,
            payload=payload,
            detail="invocation fulfilled",
            **meta,
        )

    @staticmethod
    def _terminal_detail(disp: str, status: int, meta: Mapping[str, Any]) -> str:
        if disp == Disposition.REDIRECT:
            return (
                f"NVCF returned {status} with Location={meta.get('location')!r}; "
                "the result exceeds the inline response limit. Fetching it is "
                "the caller's decision — this client will not follow a "
                "redirect to an unvalidated host on its own."
            )
        if disp == Disposition.THROTTLED:
            return (
                f"NVCF returned {status}: rate limited. The invocation did not "
                "run, so there is no proposal — back off and retry."
            )
        if disp == Disposition.REFUSED:
            return (
                f"NVCF returned {status}: the invocation was refused "
                "(credentials, scope or credits). No proposal was produced."
            )
        return f"NVCF returned an undeclared status {status} for this endpoint"
