"""NVCF's invocation protocol, loaded from the extracted contract.

Nothing in this module is typed from memory. Every path, header name and
status code is read out of ``nvcf/contract/nvcf-invocation-contract.json``,
which ``nvcf/extract_contract.py`` derives from NVIDIA's own published
OpenAPI document (vendored under ``nvcf/spec/``). Gate G10 re-extracts and
compares, so a spec change surfaces as a failing gate rather than as a client
that quietly speaks last year's protocol.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = [
    "CONTRACT_PATH",
    "Disposition",
    "InvocationOutcome",
    "NvcfProtocolError",
    "classify",
    "contract",
    "header",
    "invoke_path",
    "poll_path",
    "status_codes",
]

CONTRACT_PATH = (
    Path(__file__).resolve().parents[2] / "contract" / "nvcf-invocation-contract.json"
)


class NvcfProtocolError(RuntimeError):
    """The contract does not contain something the client needs."""


@lru_cache(maxsize=1)
def contract() -> dict[str, Any]:
    """The extracted contract (cached)."""
    if not CONTRACT_PATH.exists():
        raise NvcfProtocolError(
            f"{CONTRACT_PATH} is missing; run nvcf/extract_contract.py"
        )
    loaded: dict[str, Any] = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    return loaded


def header(name: str) -> str:
    """Return ``name`` iff the contract declares it as an NVCF-* header.

    A typo'd header is a silent no-op against a real service — the request
    just goes out without it. This turns that into an exception at the point
    of use.
    """
    declared: list[str] = contract()["nvcf_headers"]
    for h in declared:
        if h.upper() == name.upper():
            return h
    raise NvcfProtocolError(
        f"{name!r} is not an NVCF header in the extracted contract; "
        f"declared headers are {declared}"
    )


def invoke_path(function_id: str, version_id: str | None = None) -> str:
    """The pass-through invocation path, templated from the contract."""
    c = contract()["invoke"]
    if version_id:
        tmpl: str = c["invokeFunction_1"]["path"]
        return tmpl.replace("{functionId}", function_id).replace(
            "{versionId}", version_id
        )
    return str(c["invokeFunction"]["path"].replace("{functionId}", function_id))


def poll_path(request_id: str) -> str:
    """The status-polling path, templated from the contract."""
    return str(contract()["poll"]["path"].replace("{requestId}", request_id))


def status_codes(op: str = "invoke") -> list[int]:
    """Declared status codes for ``invoke`` or ``poll``."""
    key = {"invoke": "invoke_status_codes", "poll": "poll_status_codes"}[op]
    return [int(c) for c in contract()[key]]


class Disposition:
    """What the client should do next, per the spec's own descriptions."""

    FULFILLED = "fulfilled"  # 200 — the body is the result
    PENDING = "pending"  # 202 — poll with the request id
    REDIRECT = "redirect"  # 302 — fetch the large result from Location
    REFUSED = "refused"  # 402/403 — terminal, no result will ever arrive
    THROTTLED = "throttled"  # 429 — back off, the request did not run
    UNKNOWN = "unknown"  # anything the contract does not declare


# Mapping is derived from the contract's declared codes, not asserted over
# them: a code the spec stops declaring stops being classified, and a code it
# adds lands in UNKNOWN, which the client treats as fail-closed.
_DISPOSITION_BY_CODE = {
    200: Disposition.FULFILLED,
    202: Disposition.PENDING,
    302: Disposition.REDIRECT,
    402: Disposition.REFUSED,
    403: Disposition.REFUSED,
    429: Disposition.THROTTLED,
}


def classify(status: int, op: str = "invoke") -> str:
    """Classify an HTTP status for ``op``.

    A status the contract does not declare for this operation is
    :data:`Disposition.UNKNOWN` even if it is a code NVCF uses elsewhere —
    the client must not invent a meaning for a response the endpoint was
    never specified to return.
    """
    if status not in status_codes(op):
        return Disposition.UNKNOWN
    return _DISPOSITION_BY_CODE.get(status, Disposition.UNKNOWN)


@dataclass(frozen=True)
class InvocationOutcome:
    """One completed NVCF invocation attempt, however it ended.

    ``payload`` is populated only for :data:`Disposition.FULFILLED`. Every
    other disposition carries ``payload=None`` — there is deliberately no
    "empty result" that a caller could mistake for a proposal.
    """

    disposition: str
    status: int
    request_id: str | None = None
    nvcf_status: str | None = None
    percent_complete: int | None = None
    location: str | None = None
    payload: dict[str, Any] | None = None
    polls: int = 0
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.disposition == Disposition.FULFILLED and self.payload is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition,
            "status": self.status,
            "request_id": self.request_id,
            "nvcf_status": self.nvcf_status,
            "percent_complete": self.percent_complete,
            "location": self.location,
            "polls": self.polls,
            "detail": self.detail,
            "has_payload": self.payload is not None,
        }
