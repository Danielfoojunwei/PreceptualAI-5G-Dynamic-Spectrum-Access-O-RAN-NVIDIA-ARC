"""Evidence for the transaction, including — especially — the refused ones.

The system's headline promise is signed evidence of every correction, refusal
and execution. The multi-agent layer as first written did not keep it. The
Shield produced a per-action certificate for each member, but the *transaction*
produced nothing: no record of which aggregate limit was breached, no record of
whose request was dropped and under what priority, and for a refused
transaction no artefact at all. An auditor asking "why was agent X's request
not applied on Tuesday?" had nothing to read.

Refusals matter more than commits here, not less. A commit leaves a trace in
the network; a refusal leaves nothing behind unless something writes it down,
and a refusal is exactly the event an operator will later be asked to justify —
to a regulator, to the agent's vendor, or to whoever noticed the capacity did
not arrive.

So :class:`TransactionCertificate` is issued for **every** evaluated
transaction, committed or refused, and carries:

* the telemetry verdict, per check
* every authority verdict, with the problems that caused a refusal
* every aggregate check, satisfied or not, with its margin
* the resolution: who was dropped, at what operator priority, against which
  invariant
* digests of the admitted actions and their per-action certificates

Two properties are load-bearing.

**The chain localises tampering.** Each entry commits to its predecessor's
hash, so altering an old record invalidates every later one and
:func:`verify_chain` reports the exact index where the chain first breaks —
matching the behaviour the rest of the repository already relies on.

**Signing covers the decision, not a summary of it.** Canonical bytes are
produced the same way as ``canonical_certificate_bytes`` in
``horizon_ric.shield.signing``: ``json.dumps`` with sorted keys and compact
separators over the record minus its own signature fields. Same discipline, so
a verifier written for one works for the other.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from horizon_ric.shield.signing import key_fingerprint

__all__ = [
    "TransactionCertificate",
    "ChainEntry",
    "TransactionEvidenceChain",
    "canonical_transaction_bytes",
    "sign_transaction",
    "signed_transaction",
    "verify_transaction",
    "verify_chain",
    "ChainBreak",
    "utc_now_iso",
    "check_dict",
    "digest_of",
]

_SIGNATURE_FIELDS = ("signature", "signing_key_fingerprint")

GENESIS = "0" * 64


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class TransactionCertificate:
    """What the transaction decided, and everything it decided it from.

    Deliberately verbose. A record that omits the checks that *passed* cannot
    distinguish "this limit was evaluated and held" from "this limit was never
    evaluated", and those have very different meanings to an auditor.
    """

    transaction_id: str
    issued_at: str
    committed: bool
    refusals: tuple[str, ...] = ()
    telemetry_checks: tuple[dict[str, Any], ...] = ()
    authority_verdicts: tuple[dict[str, Any], ...] = ()
    aggregate_checks: tuple[dict[str, Any], ...] = ()
    dropped: tuple[dict[str, Any], ...] = ()
    admitted: tuple[dict[str, Any], ...] = ()
    resolution_rounds: int = 0
    signature: str | None = None
    signing_key_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def canonical_transaction_bytes(cert: TransactionCertificate) -> bytes:
    """Stable bytes for signing and verification.

    Identical construction to ``canonical_certificate_bytes``: sorted keys,
    compact separators, signature fields removed so signing and verification
    agree whether or not the record has already been stamped. No ``default=``
    handler, which makes JSON-primitive values normative here too — a
    non-serialisable value raises at signing time rather than producing bytes
    that cannot be reproduced.
    """
    payload = cert.to_dict()
    for name in _SIGNATURE_FIELDS:
        payload.pop(name, None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_transaction(cert: TransactionCertificate, private_key: Ed25519PrivateKey) -> str:
    return private_key.sign(canonical_transaction_bytes(cert)).hex()


def signed_transaction(
    cert: TransactionCertificate, private_key: Ed25519PrivateKey
) -> TransactionCertificate:
    return dataclasses.replace(
        cert,
        signature=sign_transaction(cert, private_key),
        signing_key_fingerprint=key_fingerprint(private_key.public_key()),
    )


def verify_transaction(
    cert: TransactionCertificate, public_key: Ed25519PublicKey
) -> bool:
    """False for a missing, malformed, or invalid signature — never raises."""
    if not cert.signature:
        return False
    try:
        signature = bytes.fromhex(cert.signature)
    except ValueError:
        return False
    try:
        public_key.verify(signature, canonical_transaction_bytes(cert))
    except Exception:
        return False
    return True


@dataclass(frozen=True)
class ChainEntry:
    """One certificate, bound to its predecessor."""

    index: int
    prev_hash: str
    entry_hash: str
    certificate: TransactionCertificate

    def recompute_hash(self) -> str:
        return chain_hash(self.prev_hash, self.certificate)


def chain_hash(prev_hash: str, cert: TransactionCertificate) -> str:
    """SHA-256 over the predecessor hash followed by the certificate's bytes.

    The predecessor is included in the hashed material rather than merely
    stored alongside it. Storing it without hashing it would let an attacker
    rewrite history and re-link the chain, which is the whole failure mode this
    construction exists to prevent.

    The *signed* bytes are hashed, not the signature, so a chain remains
    verifiable for structure even where a signing key is unavailable.
    """
    return hashlib.sha256(
        prev_hash.encode("ascii") + canonical_transaction_bytes(cert)
    ).hexdigest()


@dataclass
class ChainBreak:
    """Where a chain first stops verifying, and why."""

    index: int
    reason: str


@dataclass
class TransactionEvidenceChain:
    """An append-only hash chain of transaction certificates.

    In-memory and deliberately simple: this is the evidence *structure*, not a
    durable store. Persisting it is the deployment's problem, and
    ``horizon_ric.evidence.store`` already exists for that. What matters here
    is that the structure makes tampering detectable and localisable.
    """

    entries: list[ChainEntry] = field(default_factory=list)

    @property
    def head(self) -> str:
        return self.entries[-1].entry_hash if self.entries else GENESIS

    def append(
        self,
        cert: TransactionCertificate,
        *,
        private_key: Ed25519PrivateKey | None = None,
    ) -> ChainEntry:
        """Append ``cert``, signing it first when a key is supplied.

        Signing before hashing matters: the hash covers the canonical bytes,
        which exclude the signature, so the two operations commute — but
        stamping the signature after computing the entry hash would leave a
        record whose stored signature was never part of any chained material.
        """
        stamped = signed_transaction(cert, private_key) if private_key else cert
        prev = self.head
        entry = ChainEntry(
            index=len(self.entries),
            prev_hash=prev,
            entry_hash=chain_hash(prev, stamped),
            certificate=stamped,
        )
        self.entries.append(entry)
        return entry

    def verify(self, *, public_key: Ed25519PublicKey | None = None) -> ChainBreak | None:
        """Return the first break, or ``None`` if the chain is intact."""
        return verify_chain(self.entries, public_key=public_key)


def verify_chain(
    entries: Sequence[ChainEntry], *, public_key: Ed25519PublicKey | None = None
) -> ChainBreak | None:
    """Walk the chain and report the first index that does not verify.

    Reporting the index rather than a bare boolean is the point: "the evidence
    was altered" is much less useful to an operator than "record 41 was
    altered", and the chain construction is what makes the stronger statement
    possible.
    """
    expected_prev = GENESIS
    for position, entry in enumerate(entries):
        if entry.index != position:
            return ChainBreak(position, f"entry index {entry.index} out of order")
        if entry.prev_hash != expected_prev:
            return ChainBreak(
                position,
                f"prev_hash {entry.prev_hash[:12]}... does not match the "
                f"preceding entry hash {expected_prev[:12]}...",
            )
        recomputed = entry.recompute_hash()
        if recomputed != entry.entry_hash:
            return ChainBreak(
                position,
                "certificate content does not match its recorded hash "
                "(record altered after it was chained)",
            )
        if public_key is not None and not verify_transaction(
            entry.certificate, public_key
        ):
            return ChainBreak(position, "signature does not verify")
        expected_prev = entry.entry_hash
    return None


# ── building a certificate from a transaction result ─────────────────────
def check_dict(check: Any) -> dict[str, Any]:
    """Flatten an ``InvariantCheck`` (or a trust check) to JSON primitives.

    ``canonical_transaction_bytes`` runs ``json.dumps`` with no ``default=``,
    so anything that is not already a JSON primitive has to be flattened here
    or signing raises. Doing it explicitly beats a ``default=str`` that would
    silently serialise an object's repr into signed evidence.
    """
    out: dict[str, Any] = {}
    for name in ("invariant_id", "check_id"):
        value = getattr(check, name, None)
        if value is not None:
            out["id"] = str(value)
            break
    for source, target in (("satisfied", "passed"), ("passed", "passed")):
        value = getattr(check, source, None)
        if value is not None:
            out[target] = bool(value)
            break
    margin = getattr(check, "margin", None)
    if margin is not None:
        # inf is not representable in JSON and json.dumps emits a bare
        # `Infinity`, which is not valid JSON and would make the signed bytes
        # unparseable by a conforming verifier.
        out["margin"] = None if margin in (float("inf"), float("-inf")) else float(margin)
    unit = getattr(check, "unit", None)
    if unit is not None:
        out["unit"] = str(unit)
    detail = getattr(check, "detail", None)
    if detail is not None:
        out["detail"] = str(detail)
    return out


def digest_of(payload: Mapping[str, Any]) -> str:
    """SHA-256 over a canonicalised mapping, for referencing without inlining."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
