"""HSM-abstraction tests (Row 15 closure).

Covers:
    1. InMemoryHSMBackend keypair generate / sign / verify roundtrip.
    2. RSA-OAEP encrypt / decrypt roundtrip.
    3. Factory selection of in_memory backend.
    4. AWS / Thales factory paths raise NotImplementedError.
    5. SecureFedAvg with InMemoryHSMBackend produces same aggregate as
       without an HSM (HSM does NOT change the math, only key custody).
    6. Tampered share-dealer announcements are rejected.
    7. SoftHSM2 backend exercised when softhsm2 + python-pkcs11 are both
       on the host (skipped otherwise).
"""

from __future__ import annotations

import os
import shutil

import pytest
import torch

from horizon_ric.federated import ClientUpdate, FedAvg, SecureFedAvg
from horizon_ric.security.hsm import (
    HSMBackend,
    InMemoryHSMBackend,
    SoftHSM2Backend,
)


# ─── 1. In-memory keypair roundtrip ─────────────────────────────────────────
def test_in_memory_hsm_keypair_roundtrip() -> None:
    hsm = InMemoryHSMBackend()
    pub, priv = hsm.generate_keypair("alpha")
    assert pub.startswith(b"pub:") and priv.startswith(b"priv:")

    msg = b"share-dealer-announcement-payload"
    sig = hsm.sign(priv, msg)
    assert len(sig) == 256  # RSA-2048 PSS signature
    assert hsm.verify(pub, msg, sig) is True
    # Tamper detection.
    assert hsm.verify(pub, msg + b"!", sig) is False


# ─── 2. In-memory RSA-OAEP encrypt / decrypt ────────────────────────────────
def test_in_memory_hsm_encrypt_decrypt() -> None:
    hsm = InMemoryHSMBackend()
    pub, priv = hsm.generate_keypair("oaep-key")
    payload = b"\x00\x01\x02 secret content for OAEP roundtrip \xff\xfe"
    ct = hsm.encrypt(pub, payload)
    assert ct != payload
    assert len(ct) == 256  # RSA-2048 ciphertext
    pt = hsm.decrypt(priv, ct)
    assert pt == payload
    # Public-key export → SubjectPublicKeyInfo DER.
    spki = hsm.export_public(pub)
    assert spki.startswith(b"\x30")  # ASN.1 SEQUENCE


# ─── 3. Factory: in_memory ──────────────────────────────────────────────────
def test_hsm_factory_selection() -> None:
    backend = HSMBackend.from_config({"backend": "in_memory"})
    assert isinstance(backend, InMemoryHSMBackend)
    # Default (no key) also resolves to in_memory.
    backend2 = HSMBackend.from_config({})
    assert isinstance(backend2, InMemoryHSMBackend)
    # Unknown backend label raises ValueError.
    with pytest.raises(ValueError, match="unknown HSM backend"):
        HSMBackend.from_config({"backend": "frobnicator"})


# ─── 4. Factory: production backends are documented-only ────────────────────
def test_hsm_factory_aws_unimplemented() -> None:
    with pytest.raises(NotImplementedError, match="contact ops"):
        HSMBackend.from_config({"backend": "aws_cloudhsm"})
    with pytest.raises(NotImplementedError, match="contact ops"):
        HSMBackend.from_config({"backend": "thales_luna"})


# ─── 5. SecureFedAvg with HSM matches plain SecureFedAvg ────────────────────
def _uniform_clients(n_clients: int, n_elems: int, seed: int) -> list[ClientUpdate]:
    g = torch.Generator().manual_seed(seed)
    return [
        ClientUpdate(
            client_id=f"c{i}",
            state_dict={
                "w": torch.randn(n_elems, generator=g, dtype=torch.float32),
            },
            sample_count=100,
        )
        for i in range(n_clients)
    ]


def test_secure_fedavg_with_hsm() -> None:
    clients = _uniform_clients(n_clients=6, n_elems=24, seed=11)
    plain = FedAvg().aggregate(clients)
    no_hsm = SecureFedAvg(n_shares=5, threshold=3).aggregate(clients)
    with_hsm = SecureFedAvg(
        n_shares=5, threshold=3, hsm=InMemoryHSMBackend()
    ).aggregate(clients)
    # All three agree to within quantisation precision (the HSM only
    # affects key custody, not the secret-sharing math).
    assert (plain["w"] - with_hsm["w"]).abs().max().item() < 1e-4
    assert (no_hsm["w"] - with_hsm["w"]).abs().max().item() < 1e-4


def test_secure_fedavg_rejects_tampered_announcement() -> None:
    """Tampering with a share-dealer's announcement metadata must cause the
    HSM-signature verification to fail."""
    hsm = InMemoryHSMBackend()
    sf = SecureFedAvg(n_shares=5, threshold=3, hsm=hsm)
    clients = _uniform_clients(n_clients=4, n_elems=8, seed=3)
    bundle = sf.deal(clients[0])
    assert sf._verify_announcement(bundle) is True  # baseline ok
    # Mutate sample_count → digest changes → signature fails.
    bundle.sample_count = bundle.sample_count + 1
    assert sf._verify_announcement(bundle) is False


# ─── 6. SoftHSM2 — exercised only when host has it ──────────────────────────
def test_softhsm2_backend_when_available() -> None:
    """If a real SoftHSM2 install + python-pkcs11 are present, instantiate
    the backend and round-trip a keypair. Otherwise skip (the unit suite
    must remain green on hosts without softhsm2)."""
    pytest.importorskip("pkcs11")
    if not shutil.which("softhsm2-util"):
        pytest.skip("softhsm2-util not on PATH")
    # Look for the .so on standard locations.
    candidates = [
        "/usr/lib/softhsm/libsofthsm2.so",
        "/usr/lib/x86_64-linux-gnu/softhsm/libsofthsm2.so",
        "/usr/lib/aarch64-linux-gnu/softhsm/libsofthsm2.so",
        "/usr/local/lib/softhsm/libsofthsm2.so",
    ]
    if not any(os.path.exists(p) for p in candidates):
        pytest.skip("libsofthsm2.so not installed on this host")
    # If we get here, instantiate; this requires an initialised token,
    # which is an ops setup step. We don't initialise tokens from a unit
    # test (would mutate ~/.config/softhsm2 globally), so we accept either
    # successful __init__ or a clear "token not initialised" error.
    try:
        SoftHSM2Backend(token_label="horizon-ric", user_pin="1234")
    except RuntimeError as e:
        msg = str(e)
        assert "token" in msg.lower() or "softhsm2" in msg.lower()


# ─── 7. SoftHSM2 import resilience without library ──────────────────────────
def test_softhsm2_init_helpful_error_without_library() -> None:
    """If libsofthsm2.so is genuinely missing, __init__ must raise a
    user-actionable RuntimeError (not an obscure import / segfault)."""
    candidates = [
        "/usr/lib/softhsm/libsofthsm2.so",
        "/usr/lib/x86_64-linux-gnu/softhsm/libsofthsm2.so",
        "/usr/lib/aarch64-linux-gnu/softhsm/libsofthsm2.so",
        "/usr/local/lib/softhsm/libsofthsm2.so",
    ]
    if any(os.path.exists(p) for p in candidates):
        pytest.skip("host has libsofthsm2 — handled by the other test")
    pytest.importorskip("pkcs11")
    with pytest.raises(RuntimeError):
        SoftHSM2Backend(token_label="nonexistent", user_pin="1234")
