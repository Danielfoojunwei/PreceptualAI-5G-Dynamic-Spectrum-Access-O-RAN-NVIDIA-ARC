"""Horizon's decision plane, presented over NVCF's own invocation contract.

The other direction
-------------------
:mod:`horizon_nvcf.agent` treats NVCF as the place *agents* live. This module
is the mirror: it makes Horizon-RIC itself deployable **as an NVCF function**,
so an operator running NVCF self-managed on their own GPU cluster can deploy
the Shield next to the agents it governs, addressed by the same control plane,
the same health contract and the same invocation protocol as everything else
on that cluster.

Why the fit is exact rather than adapted
----------------------------------------
NVCF's pass-through invocation is deliberately asynchronous: the endpoint
holds the connection for ``NVCF-POLL-SECONDS``, and if the work has not
finished it replies **202** with an ``NVCF-REQID`` the caller polls. That is
usually described as a concession to slow GPUs.

For a *safety* plane it is the required shape, for a different reason.
:class:`~horizon_agentic.cycle.TransactionCycle` cannot answer a submission
when it arrives, because answering would decide that agent's request in
isolation — and deciding requests in isolation is the exact failure the whole
transaction machinery exists to remove. Its ``submit()`` already returns a
*receipt*, not a decision. NVCF's 202-and-poll is the wire form of that
receipt, and 200-on-poll is the wire form of the epoch's committed
certificate.

So the mapping is not a wrapper. It is the same idea twice:

    NVCF                          Horizon
    ────────────────────────      ─────────────────────────────────
    POST /pexec/functions/{id} →  TransactionCycle.submit(envelope)
    202 + NVCF-REQID           →  SubmissionReceipt(epoch, position)
    GET  /pexec/status/{rid}   →  look up that epoch's result
    202, NVCF-PERCENT-COMPLETE →  epoch still open
    200 + body                 →  TransactionCertificate
    429                        →  epoch full (max_members)
    403                        →  wrong functionId / unauthorised caller

Scope. This is a stdlib :mod:`http.server` handler — no framework, no new
dependency. It is the protocol surface, exercised by gate G10 against a real
socket. Production deployment (TLS, the NVCA cluster agent, autoscaling)
belongs to NVCF itself; the function manifest in ``nvcf/deploy/`` is what
declares this service to it.
"""

from __future__ import annotations

import json
import re
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from horizon_agentic.bundle import TransactionResult
from horizon_agentic.cycle import TransactionCycle
from horizon_agentic.envelope import AgentActionEnvelope

from horizon_nvcf.protocol import contract, header

__all__ = [
    "HEALTH_EXPECTED_STATUS",
    "HEALTH_URI",
    "HorizonFunctionService",
    "make_handler",
    "serve",
]

# Must match nvcf/deploy/horizon-shield-function.json's `health` block. G10
# asserts the two agree rather than trusting this comment.
HEALTH_URI = "/v2/health"
HEALTH_EXPECTED_STATUS = 200

_INVOKE_RE = re.compile(
    r"^/v2/nvcf/pexec/functions/(?P<fn>[^/]+)(?:/versions/(?P<ver>[^/]+))?/?$"
)
_STATUS_RE = re.compile(r"^/v2/nvcf/pexec/status/(?P<rid>[^/]+)/?$")

MAX_BODY_BYTES = 1 * 1024 * 1024


@dataclass(frozen=True)
class _Pending:
    epoch: int
    agent_id: str
    position: int


class HorizonFunctionService:
    """The stateful core: request ids, epoch lookup, decision retrieval.

    Separated from the HTTP handler so gates and tests can drive the protocol
    logic without a socket, and so the socket layer holds no state.
    """

    def __init__(
        self,
        cycle: TransactionCycle,
        *,
        function_id: str,
        version_id: str,
        max_tracked: int = 4096,
    ) -> None:
        self.cycle = cycle
        self.function_id = function_id
        self.version_id = version_id
        self._lock = threading.Lock()
        self._pending: dict[str, _Pending] = {}
        self._results: dict[int, TransactionResult] = {}
        self._max_tracked = max_tracked
        self._seq = 0

    # ── identity ─────────────────────────────────────────────────────────
    def addresses_us(self, fn: str, ver: str | None) -> bool:
        """True iff a request path names this function (and version, if given)."""
        if fn != self.function_id:
            return False
        return ver is None or ver == self.version_id

    # ── submission ───────────────────────────────────────────────────────
    def submit(self, envelope: AgentActionEnvelope) -> tuple[int, dict[str, Any]]:
        """Queue ``envelope``; return ``(http_status, body)``.

        202 with a request id is the success case. A structurally refused
        submission (epoch full, duplicate principal) is 429 — the request did
        not run and retrying later is the correct response, which is exactly
        what NVCF's 429 means.
        """
        receipt = self.cycle.submit(envelope)
        if not receipt.accepted:
            return 429, {
                "accepted": False,
                "epoch": receipt.epoch,
                "reason": receipt.reason,
            }
        with self._lock:
            self._seq += 1
            rid = f"h-{receipt.epoch}-{self._seq:06d}"
            self._pending[rid] = _Pending(
                receipt.epoch, receipt.agent_id, receipt.position
            )
            self._evict_locked()
        return 202, {
            "requestId": rid,
            "epoch": receipt.epoch,
            "position": receipt.position,
        }

    def _evict_locked(self) -> None:
        """Bound the tracking tables. Oldest request ids go first."""
        while len(self._pending) > self._max_tracked:
            self._pending.pop(next(iter(self._pending)))
        while len(self._results) > self._max_tracked:
            self._results.pop(next(iter(self._results)))

    # ── epoch closure ────────────────────────────────────────────────────
    def record(self, result: TransactionResult) -> None:
        """Register a decided epoch so its submitters can poll for it."""
        with self._lock:
            self._results[result.certificate.epoch] = result
            self._evict_locked()

    def close_epoch(self, **kwargs: Any) -> TransactionResult:
        """Close the open epoch and make its result pollable."""
        result = self.cycle.close(**kwargs)
        self.record(result)
        return result

    # ── polling ──────────────────────────────────────────────────────────
    def poll(self, request_id: str) -> tuple[int, dict[str, Any]]:
        """Return ``(http_status, body)`` for a status poll.

        403 for an unknown request id, deliberately: an id we never issued is
        indistinguishable from one issued to somebody else, and answering 404
        would let a caller enumerate which ids exist.
        """
        with self._lock:
            pending = self._pending.get(request_id)
            result = self._results.get(pending.epoch) if pending else None

        if pending is None:
            return 403, {"error": "unknown request id"}
        if result is None:
            return 202, {
                "requestId": request_id,
                "epoch": pending.epoch,
                "status": "pending",
            }

        cert = result.certificate
        body: dict[str, Any] = {
            "requestId": request_id,
            "epoch": cert.epoch,
            "committed": result.committed,
            "certificate": cert.to_dict(),
        }
        if result.committed:
            body["actions"] = [
                {"agent_id": m.agent_id, "action": dict(m.action)}
                for m in result.members
                if m.agent_id == pending.agent_id
            ]
        else:
            body["refusals"] = list(result.refusals)
        return 200, body

    # ── NVCF response headers ────────────────────────────────────────────
    def headers_for(self, request_id: str | None, status: int) -> dict[str, str]:
        """The NVCF-* headers this response must carry, named from the contract."""
        h: dict[str, str] = {}
        if request_id:
            h[header("NVCF-REQID")] = request_id
        h[header("NVCF-STATUS")] = "fulfilled" if status == 200 else "pending-evaluation"
        h[header("NVCF-PERCENT-COMPLETE")] = "100" if status == 200 else "0"
        return h


def make_handler(service: HorizonFunctionService) -> type[BaseHTTPRequestHandler]:
    """Build a handler class bound to ``service``."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "horizon-shield-nvcf"

        # Quiet by default: a control plane's stderr is not a request log.
        def log_message(self, *_a: Any) -> None:  # noqa: A003
            return

        # ── helpers ──────────────────────────────────────────────────────
        def _send(
            self, status: int, body: Mapping[str, Any], extra: Mapping[str, str] = {}
        ) -> None:
            raw = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            for k, v in extra.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(raw)

        def _internal_error(self, exc: BaseException) -> None:
            """An internal fault is 500 — never a 403, never a dropped socket.

            403 would read as "refused", which a caller may legitimately stop
            retrying on. Dropping the connection (the BaseHTTPRequestHandler
            default when a handler raises) is fail-closed but undiagnosable.
            500 is undeclared for these endpoints in NVCF's contract, so
            :func:`horizon_nvcf.protocol.classify` maps it to UNKNOWN and the
            client produces no proposal — a fault in the plane composes to
            silence at the agent, which is the correct direction.
            """
            print(
                f"horizon-nvcf: internal error handling {self.path}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            try:
                self._send(500, {"error": "internal error"})
            except OSError:  # pragma: no cover — peer already gone
                pass

        # ── GET: health and status ───────────────────────────────────────
        def do_GET(self) -> None:  # noqa: N802
            try:
                self._do_get()
            except Exception as exc:  # noqa: BLE001
                self._internal_error(exc)

        def do_POST(self) -> None:  # noqa: N802
            try:
                self._do_post()
            except Exception as exc:  # noqa: BLE001
                self._internal_error(exc)

        def _do_get(self) -> None:
            if self.path.rstrip("/") == HEALTH_URI.rstrip("/"):
                self._send(HEALTH_EXPECTED_STATUS, {"status": "ok"})
                return
            m = _STATUS_RE.match(self.path)
            if not m:
                self._send(403, {"error": "unknown path"})
                return
            rid = m.group("rid")
            status, body = service.poll(rid)
            self._send(status, body, service.headers_for(rid, status))

        # ── POST: invocation ─────────────────────────────────────────────
        def _do_post(self) -> None:
            m = _INVOKE_RE.match(self.path)
            if not m:
                self._send(403, {"error": "unknown path"})
                return
            if not service.addresses_us(m.group("fn"), m.group("ver")):
                self._send(403, {"error": "function/version not served here"})
                return

            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                self._send(403, {"error": "bad Content-Length"})
                return
            # Size is checked BEFORE reading, so an oversized body is refused
            # rather than buffered.
            if length <= 0 or length > MAX_BODY_BYTES:
                self._send(
                    403, {"error": f"body must be 1..{MAX_BODY_BYTES} bytes"}
                )
                return

            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                envelope = AgentActionEnvelope.from_dict(payload)
            except Exception:  # noqa: BLE001 — an oracle-free refusal
                # Deliberately one message for every parse/validation failure:
                # a caller must not learn which field it got wrong.
                self._send(403, {"error": "envelope rejected"})
                return

            status, body = service.submit(envelope)
            self._send(
                status, body, service.headers_for(body.get("requestId"), status)
            )

    return Handler


def serve(
    service: HorizonFunctionService, host: str = "127.0.0.1", port: int = 8000
) -> ThreadingHTTPServer:
    """Start the service on ``host:port``. Returns the (already serving) server."""
    httpd = ThreadingHTTPServer((host, port), make_handler(service))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def declared_contract_paths() -> dict[str, str]:
    """The contract paths this service implements, read from the contract."""
    c = contract()
    return {
        "invoke": c["invoke"]["invokeFunction"]["path"],
        "invoke_versioned": c["invoke"]["invokeFunction_1"]["path"],
        "poll": c["poll"]["path"],
    }
