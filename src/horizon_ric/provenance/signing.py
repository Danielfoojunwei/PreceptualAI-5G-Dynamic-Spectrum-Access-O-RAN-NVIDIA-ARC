"""Model provenance — cryptographic authenticity for promoted model artefacts.

The content-addressed artefact vault (``runtime.artefact_vault``) proves a model
binary is *byte-identical* to what was stored. It does NOT prove the model is
*trusted*: a poisoned-but-consistent ``.pt`` passes a hash check. Container
signing (cosign) signs the image, not the weights inside it.

This module closes that gap. A trainer signs ``(sha256(weights) || sha256(
training_manifest) || trainer_id)`` with an HSM-held RSA key (RSA-PSS / SHA-256,
the same primitive as ``security.hsm``). On promotion, the rApp re-derives the
digest from the on-disk bytes and verifies the signature against the trainer's
public key — so an unauthenticated or tampered model cannot be promoted, and the
provenance hash is threaded into the Shield's SafetyCertificate.

Verification is self-contained: it needs only the public key (DER), so a
deployer can verify without access to the signing HSM.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization import load_der_public_key

_PROV_VERSION = "horizon-prov-v1"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_digest(manifest: dict[str, Any]) -> str:
    """Deterministic SHA-256 over the canonical JSON of a training manifest."""
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return sha256_hex(canonical.encode("utf-8"))


def _signing_payload(weights_sha: str, manifest_sha: str, trainer_id: str) -> bytes:
    return f"{_PROV_VERSION}|{weights_sha}|{manifest_sha}|{trainer_id}".encode("utf-8")


@dataclass(frozen=True)
class ModelProvenance:
    """Signed authenticity record for one model artefact."""

    weights_sha256: str
    manifest_sha256: str
    trainer_id: str
    sig_alg: str  # "RSA-PSS-SHA256"
    signature_hex: str
    public_key_der_hex: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelProvenance":
        return cls(**d)


def sign_model(
    weights: bytes,
    *,
    trainer_id: str,
    training_manifest: dict[str, Any],
    hsm: Any,
    key_label: str = "model-signer",
) -> ModelProvenance:
    """Sign a model artefact with an HSM-held key.

    ``hsm`` is any :class:`horizon_ric.security.hsm.HSMBackend`. If ``key_label``
    does not yet exist it is generated. Returns a :class:`ModelProvenance`
    carrying the signature and the signer's public key (DER) for offline verify.
    """
    weights_sha = sha256_hex(weights)
    manifest_sha = manifest_digest(training_manifest)

    if key_label not in set(hsm.list_keys()):
        pub_handle, priv_handle = hsm.generate_keypair(key_label)
    else:
        pub_handle = f"pub:{key_label}".encode()
        priv_handle = f"priv:{key_label}".encode()

    payload = _signing_payload(weights_sha, manifest_sha, trainer_id)
    signature = hsm.sign(priv_handle, payload)
    pub_der = hsm.export_public(pub_handle)

    return ModelProvenance(
        weights_sha256=weights_sha,
        manifest_sha256=manifest_sha,
        trainer_id=trainer_id,
        sig_alg="RSA-PSS-SHA256",
        signature_hex=signature.hex(),
        public_key_der_hex=pub_der.hex(),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


class ProvenanceError(RuntimeError):
    """Raised when a model fails authenticity verification."""


def verify_model(
    weights: bytes,
    provenance: ModelProvenance,
    *,
    trusted_public_key_der: Optional[bytes] = None,
) -> bool:
    """Verify a model artefact against its provenance record.

    Two checks must both pass:
      1. ``sha256(weights)`` matches ``provenance.weights_sha256`` (integrity).
      2. the RSA-PSS signature verifies under the public key (authenticity).

    If ``trusted_public_key_der`` is supplied the signature is verified against
    *that* key and it must equal the key embedded in the provenance — this is
    how a deployer pins a known trainer and rejects a self-signed swap.
    """
    if sha256_hex(weights) != provenance.weights_sha256:
        return False

    embedded_der = bytes.fromhex(provenance.public_key_der_hex)
    if trusted_public_key_der is not None and trusted_public_key_der != embedded_der:
        return False

    pub = load_der_public_key(embedded_der)
    payload = _signing_payload(
        provenance.weights_sha256, provenance.manifest_sha256, provenance.trainer_id
    )
    try:
        pub.verify(
            bytes.fromhex(provenance.signature_hex),
            payload,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return True
    except Exception:
        return False


def verify_or_raise(
    weights: bytes,
    provenance: ModelProvenance,
    *,
    trusted_public_key_der: Optional[bytes] = None,
) -> None:
    """Verify; raise :class:`ProvenanceError` on failure (use at promotion)."""
    if not verify_model(weights, provenance, trusted_public_key_der=trusted_public_key_der):
        raise ProvenanceError(
            f"model artefact {provenance.weights_sha256[:12]}… failed provenance "
            "verification — refusing promotion."
        )


__all__ = [
    "ModelProvenance",
    "ProvenanceError",
    "sha256_hex",
    "manifest_digest",
    "sign_model",
    "verify_model",
    "verify_or_raise",
]
