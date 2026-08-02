"""Receiving an envelope from outside this process.

`to_dict`/`from_dict` made an envelope expressible as JSON; nothing consumed
one. This is the consumer: bytes in, a submission receipt out, with the
transport left as a separate and thinner layer.

The split is deliberate. :class:`EnvelopeIngress` is transport-agnostic and is
where every decision that matters lives — size limits, parse refusals, what a
caller is told when it is refused. :func:`serve_line_delimited` binds it to a
socket using nothing but `asyncio`, so a deployment gets something runnable
without this package acquiring a web framework, and can replace it with one
without touching the decisions.

Three rules the boundary enforces, each because the obvious alternative is
worse:

**A size limit before parsing.** `json.loads` on attacker-controlled bytes with
no bound is a memory exhaustion, and it happens before any authentication runs
— refusing on length is the only check available at that point.

**Refusals say what is wrong, not what would work.** An ingress that reported
"unknown agent" versus "bad signature" versus "nonce reused" hands an attacker
an oracle for probing the registry. Everything below authentication returns the
same shape.

**The receipt is not a decision.** Submission puts an envelope in an epoch;
:class:`~horizon_agentic.cycle.TransactionCycle` decides the epoch as a unit.
Returning anything decision-shaped here would reintroduce exactly the
per-agent evaluation the cycle exists to prevent.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Callable

from horizon_agentic.cycle import SubmissionReceipt, TransactionCycle
from horizon_agentic.envelope import AgentActionEnvelope

__all__ = [
    "IngressResult",
    "EnvelopeIngress",
    "serve_line_delimited",
    "MAX_ENVELOPE_BYTES",
]

# Generous for an action with a PRB allocation, small enough that a flood costs
# the sender something. A legitimate envelope measured under 2 KiB.
MAX_ENVELOPE_BYTES = 64 * 1024


@dataclass(frozen=True)
class IngressResult:
    """What the sender is told. Deliberately uninformative when refused."""

    accepted: bool
    epoch: int = -1
    detail: str = ""
    receipt: SubmissionReceipt | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"accepted": self.accepted, "epoch": self.epoch, "detail": self.detail}


class EnvelopeIngress:
    """Turns received bytes into a queued submission, or a refusal.

    ``on_reject`` receives the *real* reason. The sender does not: an ingress
    that distinguishes "unknown agent" from "bad signature" is a registry
    oracle, so the operator gets the detail through this callback and the wire
    gets a flat refusal.
    """

    def __init__(
        self,
        cycle: TransactionCycle,
        *,
        max_bytes: int = MAX_ENVELOPE_BYTES,
        on_reject: Callable[[str], None] | None = None,
    ) -> None:
        self._cycle = cycle
        self._max_bytes = max_bytes
        self._on_reject = on_reject

    def _reject(self, internal: str) -> IngressResult:
        if self._on_reject is not None:
            self._on_reject(internal)
        return IngressResult(False, detail="envelope refused")

    def submit_bytes(self, payload: bytes) -> IngressResult:
        if len(payload) > self._max_bytes:
            # Checked before parsing: this is the only bound available before
            # attacker-controlled bytes reach a parser.
            return self._reject(
                f"payload of {len(payload)} bytes exceeds the {self._max_bytes} limit"
            )
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return self._reject(f"payload is not UTF-8 JSON: {exc}")
        if not isinstance(decoded, dict):
            return self._reject(f"payload is a {type(decoded).__name__}, not an object")
        try:
            envelope = AgentActionEnvelope.from_dict(decoded)
        except (ValueError, TypeError) as exc:
            return self._reject(f"envelope did not parse: {exc}")
        return self.submit(envelope)

    def submit(self, envelope: AgentActionEnvelope) -> IngressResult:
        """Queue a parsed envelope into the open epoch.

        Nothing is authenticated or authorised here. That happens when the
        epoch closes, over the whole bundle — doing any of it now would decide
        one agent in isolation, and the cycle exists precisely so that does not
        happen.
        """
        receipt = self._cycle.submit(envelope)
        if not receipt.accepted:
            # A structural refusal — full epoch, duplicate principal — is not
            # security-sensitive and is safe to relay verbatim. It also tells a
            # well-behaved agent to retry, which a flat refusal would not.
            return IngressResult(
                False, epoch=receipt.epoch, detail=receipt.reason, receipt=receipt
            )
        return IngressResult(True, epoch=receipt.epoch, detail="queued", receipt=receipt)


async def serve_line_delimited(
    ingress: EnvelopeIngress,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
) -> asyncio.base_events.Server:
    """One JSON envelope per line in, one JSON result per line out.

    Newline-delimited JSON over TCP because it needs no dependency and no
    framing negotiation. It is a reference binding, not a production ingress:
    there is no TLS, no backpressure beyond the reader's own limit, and no
    authentication of the *connection* — the envelope's signature authenticates
    the request, which is the property that matters here, but a deployment on
    an untrusted network wants a transport that authenticates the peer too.
    """

    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while True:
                try:
                    line = await reader.readline()
                except (asyncio.LimitOverrunError, ValueError):
                    # The reader's buffer limit was exceeded before a newline
                    # arrived: a sender streaming without framing. Drop the
                    # connection rather than trying to resynchronise.
                    break
                if not line:
                    break
                result = ingress.submit_bytes(line.strip())
                writer.write(
                    (json.dumps(result.to_dict(), sort_keys=True) + "\n").encode()
                )
                await writer.drain()
        finally:
            writer.close()

    return await asyncio.start_server(
        handle, host, port, limit=ingress._max_bytes + 1024
    )
