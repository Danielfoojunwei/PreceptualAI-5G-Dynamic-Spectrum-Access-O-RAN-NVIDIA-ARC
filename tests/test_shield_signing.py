"""Ed25519 SafetyCertificate signing — real keys, real signatures.

Round-trip: generate a key on disk, sign a certificate, verify; tamper with
the certificate → verification fails. Key management is explicit-only: a
missing key path is a hard error, and the CLI (`python -m
horizon_ric.shield.signing --generate`) is the only generation path.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from horizon_ric.shield.certificate import InvariantCheck, SafetyCertificate, utc_now_iso
from horizon_ric.shield.signing import (
    canonical_certificate_bytes,
    generate_signing_key,
    key_fingerprint,
    load_signing_key,
    sign_certificate,
    signed_certificate,
    verify_certificate,
)


def _certificate(decision_id: str = "dec-1", tx_power: float = 20.0) -> SafetyCertificate:
    return SafetyCertificate(
        decision_id=decision_id,
        issued_at=utc_now_iso(),
        loop_tier="non_rt",
        block="policy_emit",
        action_proposed={"tx_power_dBm": tx_power},
        action_safe={"tx_power_dBm": tx_power},
        invariants=[
            InvariantCheck(invariant_id="eirp_ceiling", satisfied=True, margin=9.0, unit="dB")
        ],
        corrections=[],
        violated_ids=[],
        projected=False,
        fallback_used=False,
        fallback_to=None,
        safe=True,
        emit_blocked=False,
    )


def test_generate_sign_verify_roundtrip(tmp_path: Path):
    key_path = tmp_path / "signing.pem"
    generate_signing_key(key_path)
    assert key_path.exists()
    # Key file is a real PEM Ed25519 private key with restrictive perms.
    assert (key_path.stat().st_mode & 0o777) == 0o600
    key = load_signing_key(key_path)
    assert isinstance(key, Ed25519PrivateKey)

    cert = _certificate()
    stamped = signed_certificate(cert, key)
    assert stamped.signature and stamped.signing_key_fingerprint
    assert stamped.signing_key_fingerprint == key_fingerprint(key.public_key())
    assert verify_certificate(stamped, key.public_key()) is True

    # The signature covers the canonical dump minus the signature fields, so
    # verification is stable whether or not the cert is already stamped.
    assert canonical_certificate_bytes(cert) == canonical_certificate_bytes(stamped)
    assert stamped.signature == sign_certificate(cert, key)


def test_tampered_certificate_fails_verification(tmp_path: Path):
    key = generate_signing_key(tmp_path / "signing.pem")
    stamped = signed_certificate(_certificate(), key)

    # Tamper with the certified content, keep the signature → must fail.
    tampered = dataclasses.replace(
        stamped, action_safe={"tx_power_dBm": 43.0}
    )
    assert verify_certificate(tampered, key.public_key()) is False

    # Tamper with the signature itself → must fail (including non-hex junk).
    bad_sig = dataclasses.replace(stamped, signature="00" * 64)
    assert verify_certificate(bad_sig, key.public_key()) is False
    junk_sig = dataclasses.replace(stamped, signature="not-hex")
    assert verify_certificate(junk_sig, key.public_key()) is False

    # Unsigned certificate never verifies.
    assert verify_certificate(_certificate(), key.public_key()) is False


def test_wrong_key_fails_verification(tmp_path: Path):
    key_a = generate_signing_key(tmp_path / "a.pem")
    key_b = generate_signing_key(tmp_path / "b.pem")
    stamped = signed_certificate(_certificate(), key_a)
    assert verify_certificate(stamped, key_b.public_key()) is False


def test_load_signing_key_missing_path_fails_loudly(tmp_path: Path):
    with pytest.raises(RuntimeError, match="cannot read Ed25519"):
        load_signing_key(tmp_path / "nope.pem")


def test_load_signing_key_rejects_non_ed25519(tmp_path: Path):
    from cryptography.hazmat.primitives.asymmetric import rsa

    rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = rsa_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    p = tmp_path / "rsa.pem"
    p.write_bytes(pem)
    with pytest.raises(ValueError, match="not an Ed25519"):
        load_signing_key(p)


def test_generate_refuses_overwrite(tmp_path: Path):
    p = tmp_path / "signing.pem"
    generate_signing_key(p)
    with pytest.raises(FileExistsError):
        generate_signing_key(p)


def test_cli_generate(tmp_path: Path):
    key_path = tmp_path / "cli.pem"
    proc = subprocess.run(
        [sys.executable, "-m", "horizon_ric.shield.signing", "--generate", str(key_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "generated Ed25519 signing key" in proc.stdout
    assert key_path.exists()
    # The CLI-generated key loads and signs.
    key = load_signing_key(key_path)
    stamped = signed_certificate(_certificate(), key)
    assert verify_certificate(stamped, key.public_key())
