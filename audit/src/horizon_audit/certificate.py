"""Offline Ed25519 verification of Horizon SafetyCertificates.

Reproduces ``horizon_ric.shield.signing`` from first principles.

A signed :class:`SafetyCertificate` (see ``horizon_ric.shield.certificate``)
is a JSON object carrying, among its fields, ``signature`` (hex Ed25519
signature) and ``signing_key_fingerprint`` (SHA-256 hex of the raw 32-byte
public key). The signed bytes are the certificate's canonical JSON with those
two fields removed::

    canonical_certificate_bytes =
        json.dumps({k: v for k, v in cert.items()
                    if k not in ("signature", "signing_key_fingerprint")},
                   sort_keys=True, separators=(",", ":")).encode("utf-8")

Verification loads the operator's published Ed25519 *public* key (PEM or raw
hex) and calls ``Ed25519PublicKey.verify``. A missing, malformed or invalid
signature verifies as ``False`` — a tampered certificate never passes.

Certificates may also appear *embedded* inside a ``DecisionRecord`` (e.g. at
``chosen_action.certificate``). :func:`find_embedded_certificates` walks a
record and returns every object that carries certificate signature fields,
with the JSON path where it was found, so the CLI can verify any it finds or
honestly report "no embedded certificate".
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

# Fields excluded from the signed payload — the signature cannot cover itself.
_SIGNATURE_FIELDS = ("signature", "signing_key_fingerprint")


def canonical_certificate_bytes(cert: dict[str, Any]) -> bytes:
    """Stable byte serialisation of a certificate dict for verification.

    ``sort_keys`` + compact separators over the certificate with the two
    signature fields removed. Matches
    ``horizon_ric.shield.signing.canonical_certificate_bytes``.
    """
    payload = {k: v for k, v in cert.items() if k not in _SIGNATURE_FIELDS}
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def key_fingerprint(public_key: Ed25519PublicKey) -> str:
    """SHA-256 hex fingerprint of the raw 32-byte Ed25519 public key."""
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()


def load_public_key(source: str | Path) -> Ed25519PublicKey:
    """Load an Ed25519 public key.

    Accepts either a path to a PEM/DER public-key file, a PEM string, or a
    64-char hex encoding of the raw 32-byte public key.
    """
    text: str | None = None
    raw: bytes | None = None
    p = Path(source) if not _looks_like_pem(str(source)) else None
    if p is not None and p.exists():
        data = p.read_bytes()
        if data.lstrip().startswith(b"-----BEGIN"):
            return _load_pem_public(data)
        # Try DER, then hex text.
        try:
            key = serialization.load_der_public_key(data)
            if isinstance(key, Ed25519PublicKey):
                return key
        except Exception:
            pass
        text = data.decode("ascii", errors="ignore").strip()
    else:
        text = str(source).strip()

    if text is not None and "-----BEGIN" in text:
        return _load_pem_public(text.encode("utf-8"))

    if text is not None:
        hexcandidate = text.strip()
        try:
            raw = bytes.fromhex(hexcandidate)
        except ValueError:
            raw = None
    if raw is not None and len(raw) == 32:
        return Ed25519PublicKey.from_public_bytes(raw)

    raise ValueError(
        f"could not load an Ed25519 public key from {source!r}: expected a "
        "PEM/DER public-key file, a PEM string, or 64 hex chars (raw 32-byte key)"
    )


def _looks_like_pem(s: str) -> bool:
    return "-----BEGIN" in s


def _load_pem_public(data: bytes) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(data)
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError(
            f"public key is a {type(key).__name__}, not an Ed25519 public key"
        )
    return key


def verify_certificate(cert: dict[str, Any], public_key: Ed25519PublicKey) -> bool:
    """Verify ``cert['signature']`` against the certificate's canonical JSON.

    Returns ``False`` for a missing, malformed or invalid signature. Matches
    ``horizon_ric.shield.signing.verify_certificate`` byte-for-byte.
    """
    sig_hex = cert.get("signature")
    if not sig_hex:
        return False
    try:
        signature = bytes.fromhex(sig_hex)
    except (ValueError, TypeError):
        return False
    try:
        public_key.verify(signature, canonical_certificate_bytes(cert))
    except InvalidSignature:
        return False
    return True


def fingerprint_matches(cert: dict[str, Any], public_key: Ed25519PublicKey) -> bool:
    """Whether the certificate's recorded key fingerprint names ``public_key``."""
    fp = cert.get("signing_key_fingerprint")
    return bool(fp) and fp == key_fingerprint(public_key)


def _is_certificate_like(obj: Any) -> bool:
    """A dict is certificate-like if it carries a hex signature field."""
    return (
        isinstance(obj, dict)
        and isinstance(obj.get("signature"), str)
        and bool(obj.get("signature"))
        and "signing_key_fingerprint" in obj
    )


def _is_full_certificate(obj: dict[str, Any]) -> bool:
    """A *full*, self-verifiable SafetyCertificate carries the fields the
    signature was actually computed over (not just the record-embedded
    digest subset ``{safe, projected, violated_ids, signature, ...}``)."""
    return _is_certificate_like(obj) and "action_safe" in obj and "decision_id" in obj


@dataclass(frozen=True)
class EmbeddedCertificate:
    path: str
    cert: dict[str, Any]
    full: bool  # True if self-verifiable; False if only a signature digest ref


def find_embedded_certificates(record: Any, _prefix: str = "record") -> list[EmbeddedCertificate]:
    """Recursively find certificate-like objects inside a DecisionRecord.

    Returns every object carrying a ``signature`` field, tagged with the JSON
    path where it was found and whether it is a *full* (self-verifiable)
    certificate or merely the embedded signature-digest reference that a
    DecisionRecord stores at ``chosen_action.certificate`` (which cannot be
    verified alone — the signature was computed over the full certificate
    document, not this subset).
    """
    found: list[EmbeddedCertificate] = []

    def walk(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            if _is_certificate_like(obj):
                found.append(
                    EmbeddedCertificate(
                        path=path, cert=obj, full=_is_full_certificate(obj)
                    )
                )
            for k, v in obj.items():
                walk(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for idx, v in enumerate(obj):
                walk(v, f"{path}[{idx}]")

    walk(record, _prefix)
    return found


__all__ = [
    "canonical_certificate_bytes",
    "key_fingerprint",
    "load_public_key",
    "verify_certificate",
    "fingerprint_matches",
    "find_embedded_certificates",
    "EmbeddedCertificate",
]
