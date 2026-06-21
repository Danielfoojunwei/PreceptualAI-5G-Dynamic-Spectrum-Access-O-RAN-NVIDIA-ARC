"""JWT manager tests — RS256 mint/verify, expiry, audience, rotation."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest
from jose import jwt as jose_jwt
from jose.exceptions import (
    ExpiredSignatureError,
    JWTClaimsError,
    JWTError,
)

from horizon_ric.security.jwt import JWTManager

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "test_jwt_signing.pem"


@pytest.fixture()
def signing_key(tmp_path: Path) -> Path:
    """Per-test copy of the test signing key so rotation tests can't
    pollute other tests' state."""
    dst = tmp_path / "signing.pem"
    dst.write_bytes(_FIXTURE.read_bytes())
    return dst


@pytest.fixture()
def jwt_manager(signing_key: Path) -> JWTManager:
    return JWTManager(
        signing_key_path=signing_key,
        issuer="horizon-ric-test",
        audience="horizon-ric-api",
    )


def _gen_rsa_key(path: Path) -> Path:
    """Generate a fresh RSA key via openssl for rotation tests."""
    subprocess.run(
        ["openssl", "genrsa", "-out", str(path), "2048"],
        check=True,
        capture_output=True,
    )
    return path


# ---------------------------------------------------------------------------
# 1. Round-trip
# ---------------------------------------------------------------------------
def test_mint_verify_roundtrip(jwt_manager: JWTManager) -> None:
    tok = jwt_manager.mint_token("alice", "default", ["operator"], ttl_seconds=120)
    claims = jwt_manager.verify_token(tok)
    assert claims["sub"] == "alice"
    assert claims["tenant"] == "default"
    assert claims["roles"] == ["operator"]
    assert claims["iss"] == "horizon-ric-test"
    assert claims["aud"] == "horizon-ric-api"
    assert "exp" in claims and "iat" in claims and "jti" in claims


# ---------------------------------------------------------------------------
# 2. Expiry
# ---------------------------------------------------------------------------
def test_expired_token_rejected(jwt_manager: JWTManager) -> None:
    tok = jwt_manager.mint_token("alice", "default", ["operator"], ttl_seconds=1)
    time.sleep(2)
    with pytest.raises((ExpiredSignatureError, JWTError)):
        jwt_manager.verify_token(tok)


# ---------------------------------------------------------------------------
# 3. Wrong audience
# ---------------------------------------------------------------------------
def test_wrong_audience_rejected(signing_key: Path) -> None:
    issuer_mgr = JWTManager(signing_key, "iss", "aud-A")
    tok = issuer_mgr.mint_token("alice", "default", ["operator"])
    verifier = JWTManager(signing_key, "iss", "aud-B")
    with pytest.raises((JWTClaimsError, JWTError)):
        verifier.verify_token(tok)


# ---------------------------------------------------------------------------
# 4. Tampered signature
# ---------------------------------------------------------------------------
def test_tampered_signature_rejected(jwt_manager: JWTManager) -> None:
    tok = jwt_manager.mint_token("alice", "default", ["operator"])
    # Flip a char in the MIDDLE of the signature segment so the decoded
    # bytes definitely change. (A swap at the very last char of base64url
    # may decode to the same bytes — those positions only carry 2-4
    # significant bits and 'A'<->'B' shares the same trailing-bit pattern.)
    head, body, sig = tok.split(".")
    mid = len(sig) // 2
    flipped = "A" if sig[mid] != "A" else "B"
    new_sig = sig[:mid] + flipped + sig[mid + 1 :]
    tampered = f"{head}.{body}.{new_sig}"
    with pytest.raises(JWTError):
        jwt_manager.verify_token(tampered)


# ---------------------------------------------------------------------------
# 5. Wrong issuer
# ---------------------------------------------------------------------------
def test_wrong_issuer_rejected(signing_key: Path) -> None:
    a = JWTManager(signing_key, "iss-A", "aud")
    b = JWTManager(signing_key, "iss-B", "aud")
    tok = a.mint_token("alice", "default", ["operator"])
    with pytest.raises((JWTClaimsError, JWTError)):
        b.verify_token(tok)


# ---------------------------------------------------------------------------
# 6. Rotation: new key signs; old key still verifies during overlap
# ---------------------------------------------------------------------------
def test_rotation_overlap(signing_key: Path, tmp_path: Path) -> None:
    mgr = JWTManager(
        signing_key, "iss", "aud", overlap_window_seconds=3600
    )
    old_token = mgr.mint_token("alice", "default", ["operator"], ttl_seconds=300)
    # Verify it works pre-rotation
    assert mgr.verify_token(old_token)["sub"] == "alice"

    new_key = _gen_rsa_key(tmp_path / "new.pem")
    mgr.rotate_signing_key(new_key)

    # Old token should STILL verify under the retired key.
    claims = mgr.verify_token(old_token)
    assert claims["sub"] == "alice"

    # New tokens are signed by the new key.
    new_token = mgr.mint_token("bob", "default", ["admin"], ttl_seconds=300)
    assert mgr.verify_token(new_token)["sub"] == "bob"

    # Two distinct keys should produce different tokens (sanity).
    assert old_token != new_token


# ---------------------------------------------------------------------------
# 7. Rotation: tokens signed by an unrelated key are rejected
# ---------------------------------------------------------------------------
def test_unrelated_key_rejected(signing_key: Path, tmp_path: Path) -> None:
    mgr = JWTManager(signing_key, "iss", "aud")
    other_key = _gen_rsa_key(tmp_path / "other.pem")
    other_pem = other_key.read_text()
    rogue = jose_jwt.encode(
        {
            "sub": "evil",
            "tenant": "default",
            "roles": ["admin"],
            "iss": "iss",
            "aud": "aud",
            "exp": int(time.time()) + 600,
            "iat": int(time.time()),
        },
        other_pem,
        algorithm="RS256",
    )
    with pytest.raises(JWTError):
        mgr.verify_token(rogue)


# ---------------------------------------------------------------------------
# 8. Missing required claims
# ---------------------------------------------------------------------------
def test_missing_claims_rejected(signing_key: Path) -> None:
    mgr = JWTManager(signing_key, "iss", "aud")
    pem = signing_key.read_text()
    # Mint a token directly via jose with no `tenant` claim.
    tok = jose_jwt.encode(
        {
            "sub": "alice",
            "roles": ["operator"],
            "iss": "iss",
            "aud": "aud",
            "exp": int(time.time()) + 600,
            "iat": int(time.time()),
        },
        pem,
        algorithm="RS256",
    )
    with pytest.raises(JWTError):
        mgr.verify_token(tok)


# ---------------------------------------------------------------------------
# 9. Empty subject is rejected at mint time
# ---------------------------------------------------------------------------
def test_empty_subject_rejected(jwt_manager: JWTManager) -> None:
    with pytest.raises(ValueError):
        jwt_manager.mint_token("", "default", ["operator"])


# ---------------------------------------------------------------------------
# 10. Empty tenant rejected at mint time
# ---------------------------------------------------------------------------
def test_empty_tenant_rejected(jwt_manager: JWTManager) -> None:
    with pytest.raises(ValueError):
        jwt_manager.mint_token("alice", "", ["operator"])
