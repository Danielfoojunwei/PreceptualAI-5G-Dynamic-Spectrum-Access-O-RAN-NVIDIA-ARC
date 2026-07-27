"""Ed25519 signing of SafetyCertificates.

The Shield's :class:`~horizon_ric.shield.certificate.SafetyCertificate` is the
audit-grade trust object for one decision. This module adds an optional
cryptographic layer on top: the certificate's canonical JSON (a stable,
``sort_keys`` dump of its dict *minus* the signature fields) is signed with an
Ed25519 private key, and the hex signature plus a SHA-256 fingerprint of the
public key are stamped onto the certificate's additive optional fields.

Key management is deliberately explicit — keys are NEVER generated implicitly
at runtime. Generate one with::

    python -m horizon_ric.shield.signing --generate /etc/horizon/cert-signing.pem

and point ``HORIZON_CERT_SIGNING_KEY_PATH`` at it. A configured-but-unreadable
key path is a hard startup error (fail loudly, never sign with a surprise key).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from horizon_ric.shield.certificate import SafetyCertificate

# Fields excluded from the signed payload — the signature cannot cover itself.
_SIGNATURE_FIELDS = ("signature", "signing_key_fingerprint")


def canonical_certificate_bytes(cert: SafetyCertificate) -> bytes:
    """Stable byte serialisation of the certificate for signing/verification.

    ``sort_keys`` + compact separators over ``cert.to_dict()`` with the
    signature fields removed, so signing and verification agree regardless
    of whether the certificate has already been stamped.
    """
    payload: dict[str, Any] = cert.to_dict()
    for name in _SIGNATURE_FIELDS:
        payload.pop(name, None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def generate_signing_key(path: str | Path) -> Ed25519PrivateKey:
    """Generate a new Ed25519 private key and write it to ``path`` (PEM, 0600).

    Refuses to overwrite an existing file — key rotation must be explicit.
    """
    p = Path(path)
    if p.exists():
        raise FileExistsError(
            f"refusing to overwrite existing signing key at {p}; "
            "remove it first if rotation is intended"
        )
    key = Ed25519PrivateKey.generate()
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pem)
    p.chmod(0o600)
    return key


def load_signing_key(path: str | Path) -> Ed25519PrivateKey:
    """Load an Ed25519 private key from a PEM file.

    Raises a clear error when the file is missing/unreadable or is not an
    Ed25519 key — callers treat this as a hard startup failure. Keys are
    never generated implicitly here.
    """
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise RuntimeError(
            f"cannot read Ed25519 certificate-signing key at {p}: {exc}. "
            "Generate one explicitly with "
            f"`python -m horizon_ric.shield.signing --generate {p}`."
        ) from exc
    key = serialization.load_pem_private_key(raw, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError(
            f"signing key at {p} is a {type(key).__name__}, not an Ed25519 "
            "private key — refusing to sign certificates with it"
        )
    return key


def key_fingerprint(public_key: Ed25519PublicKey) -> str:
    """SHA-256 hex fingerprint of the raw 32-byte Ed25519 public key."""
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()


def sign_certificate(cert: SafetyCertificate, private_key: Ed25519PrivateKey) -> str:
    """Return the hex Ed25519 signature over the certificate's canonical JSON."""
    return private_key.sign(canonical_certificate_bytes(cert)).hex()


def signed_certificate(
    cert: SafetyCertificate, private_key: Ed25519PrivateKey
) -> SafetyCertificate:
    """Return a copy of ``cert`` stamped with signature + key fingerprint."""
    return dataclasses.replace(
        cert,
        signature=sign_certificate(cert, private_key),
        signing_key_fingerprint=key_fingerprint(private_key.public_key()),
    )


def verify_certificate(cert: SafetyCertificate, public_key: Ed25519PublicKey) -> bool:
    """Verify ``cert.signature`` against the certificate's canonical JSON.

    Returns False for a missing, malformed, or invalid signature — a tampered
    certificate never verifies.
    """
    if not cert.signature:
        return False
    try:
        signature = bytes.fromhex(cert.signature)
    except ValueError:
        return False
    try:
        public_key.verify(signature, canonical_certificate_bytes(cert))
    except InvalidSignature:
        return False
    return True


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m horizon_ric.shield.signing",
        description="Manage Ed25519 certificate-signing keys for the Shield.",
    )
    parser.add_argument(
        "--generate",
        metavar="PATH",
        help="Generate a new Ed25519 private key (PEM, mode 0600) at PATH.",
    )
    args = parser.parse_args(argv)
    if args.generate:
        key = generate_signing_key(args.generate)
        print(
            f"generated Ed25519 signing key at {args.generate} "
            f"(fingerprint {key_fingerprint(key.public_key())})"
        )
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover — exercised via subprocess in tests
    sys.exit(_main())


__all__ = [
    "canonical_certificate_bytes",
    "generate_signing_key",
    "key_fingerprint",
    "load_signing_key",
    "sign_certificate",
    "signed_certificate",
    "verify_certificate",
]
