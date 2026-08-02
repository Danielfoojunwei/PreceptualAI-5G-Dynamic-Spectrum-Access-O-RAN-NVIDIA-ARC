"""Binding an action to the transaction that admitted it.

The main package's proudest structural property is that a Shield-refused action
is *unable* to become an E2 control message: ``control_from_disposition``
raises on a blocked disposition, and the A1 path requires a certificate. Both
gates inspect the **per-action** ``SafetyCertificate``.

That is exactly why the transaction layer needs its own binding, and why adding
an emission path naively would be worse than having none. Every member of a
refused bundle carries a clean, valid, signed per-action certificate — that is
the *premise* of the whole design: each action is individually impeccable and
only the combination is not. So a downstream gate checking the per-action
certificate would admit an action from a bundle the aggregate layer refused,
and would be right to by its own rules. ``TransactionRefused`` protects an
in-process attribute read; it protects nothing on a wire.

:func:`emittable` is the only supported way to get actions out of a result, and
:class:`TransactionBinding` travels with each one so a receiver can check the
transaction rather than just the action. The binding names the transaction, its
certificate digest, the epoch, and — where the transaction was signed — the
signature over that certificate.

What this does *not* do is claim an emission path exists. Nothing here speaks
A1 or E2. It defines the artefact a future binding would have to carry and
refuses to produce one for a refused transaction, so that the naive integration
is closed off before somebody writes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from horizon_agentic.bundle import TransactionResult
from horizon_agentic.evidence import (
    TransactionCertificate,
    digest_of,
    verify_transaction,
)
from horizon_ric.shield.certificate import SafetyCertificate

__all__ = [
    "TransactionBinding",
    "EmittableAction",
    "emittable",
    "verify_binding",
]


@dataclass(frozen=True)
class TransactionBinding:
    """Proof that an action was admitted by a transaction, not merely by a Shield."""

    transaction_id: str
    epoch: int
    certificate_digest: str
    signature: str
    signing_key_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "epoch": self.epoch,
            "certificate_digest": self.certificate_digest,
            "signature": self.signature,
            "signing_key_fingerprint": self.signing_key_fingerprint,
        }


@dataclass(frozen=True)
class EmittableAction:
    """An action cleared for emission, with both certificates that clear it."""

    agent_id: str
    action: Mapping[str, Any]
    certificate: SafetyCertificate
    binding: TransactionBinding


def emittable(result: TransactionResult) -> tuple[EmittableAction, ...]:
    """The actions a transaction cleared, or nothing at all.

    Raises :class:`~horizon_agentic.bundle.TransactionRefused` for a refused
    transaction rather than returning an empty tuple. An empty tuple would be
    indistinguishable from "this transaction committed and had nothing to do",
    and a caller looping over it would emit nothing and log nothing — a silent
    failure exactly where a loud one is wanted.
    """
    members = result.members  # raises TransactionRefused when refused
    cert: TransactionCertificate = result.certificate
    binding = TransactionBinding(
        transaction_id=cert.transaction_id,
        epoch=cert.epoch,
        certificate_digest=digest_of(cert.to_dict()),
        signature=cert.signature or "",
        signing_key_fingerprint=cert.signing_key_fingerprint or "",
    )
    return tuple(
        EmittableAction(
            agent_id=m.agent_id,
            action=m.action,
            certificate=m.certificate,
            binding=binding,
        )
        for m in members
    )


def verify_binding(
    binding: TransactionBinding,
    certificate: TransactionCertificate,
    *,
    public_key: Ed25519PublicKey | None = None,
) -> bool:
    """Check that a binding really refers to a committed, unaltered transaction.

    This is what a receiving RIC would run. Three things have to hold, and the
    third is the one that matters: the transaction must have *committed*. A
    binding that referenced a refused transaction, or a certificate altered
    after the fact, must not verify.
    """
    if binding.certificate_digest != digest_of(certificate.to_dict()):
        return False
    if binding.transaction_id != certificate.transaction_id:
        return False
    if not certificate.committed:
        return False
    if public_key is not None:
        return verify_transaction(certificate, public_key)
    return True
