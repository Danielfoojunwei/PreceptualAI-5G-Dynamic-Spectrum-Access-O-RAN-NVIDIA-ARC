"""Tests for model provenance signing / verification."""

from __future__ import annotations

from horizon_ric.provenance import (
    ModelProvenance,
    ProvenanceError,
    sign_model,
    verify_model,
    verify_or_raise,
)
from horizon_ric.security.hsm import InMemoryHSMBackend

WEIGHTS = b"\x13\x37neural-rx-weights" * 64
MANIFEST = {"dataset": "deepmimo-o1", "rows": 50000, "epochs": 40}


def _sign() -> tuple[ModelProvenance, InMemoryHSMBackend]:
    hsm = InMemoryHSMBackend()
    prov = sign_model(WEIGHTS, trainer_id="trainer-a", training_manifest=MANIFEST, hsm=hsm)
    return prov, hsm


def test_sign_then_verify_roundtrip():
    prov, _ = _sign()
    assert verify_model(WEIGHTS, prov) is True
    assert prov.sig_alg == "RSA-PSS-SHA256"
    assert len(prov.weights_sha256) == 64


def test_tampered_weights_fail_verification():
    prov, _ = _sign()
    assert verify_model(WEIGHTS + b"x", prov) is False


def test_tampered_manifest_breaks_signature():
    prov, _ = _sign()
    forged = ModelProvenance(
        weights_sha256=prov.weights_sha256,
        manifest_sha256="0" * 64,  # different manifest
        trainer_id=prov.trainer_id,
        sig_alg=prov.sig_alg,
        signature_hex=prov.signature_hex,
        public_key_der_hex=prov.public_key_der_hex,
        created_at=prov.created_at,
    )
    assert verify_model(WEIGHTS, forged) is False


def test_untrusted_signer_rejected_when_key_pinned():
    prov, _ = _sign()
    # An attacker re-signs the same weights with their own key.
    attacker_hsm = InMemoryHSMBackend()
    forged = sign_model(WEIGHTS, trainer_id="trainer-a", training_manifest=MANIFEST, hsm=attacker_hsm)
    trusted_der = bytes.fromhex(prov.public_key_der_hex)
    # The forged provenance verifies on its own, but not against the pinned key.
    assert verify_model(WEIGHTS, forged) is True
    assert verify_model(WEIGHTS, forged, trusted_public_key_der=trusted_der) is False
    assert verify_model(WEIGHTS, prov, trusted_public_key_der=trusted_der) is True


def test_verify_or_raise_on_promotion():
    prov, _ = _sign()
    verify_or_raise(WEIGHTS, prov)  # no raise
    try:
        verify_or_raise(WEIGHTS + b"!", prov)
        raised = False
    except ProvenanceError:
        raised = True
    assert raised


def test_provenance_dict_roundtrip():
    prov, _ = _sign()
    d = prov.to_dict()
    assert ModelProvenance.from_dict(d) == prov
