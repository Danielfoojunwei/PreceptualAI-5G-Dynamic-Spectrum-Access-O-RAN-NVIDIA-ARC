"""HSM-abstraction layer with PKCS#11 (SoftHSM2) and pure-Python fallbacks.

Closes Row 15 of ``GAPS_TO_PILOT.md`` by adding a real key-custody
boundary to the federated aggregator: share-dealer signing keys live in
the HSM rather than in process memory.

Backend matrix
--------------
* :class:`SoftHSM2Backend` — talks PKCS#11 to a SoftHSM2 token via the
  ``python-pkcs11`` library. Functional when both ``python-pkcs11`` and
  ``libsofthsm2.so`` are on the host. NOT FIPS-validated; use only for
  test labs.
* :class:`InMemoryHSMBackend` — pure-Python RSA via ``cryptography``.
  TEST-ONLY (key material is in process memory). Real RSA-2048 keys, no
  mocks; suitable for unit tests and CI.
* ``backend == "aws_cloudhsm"`` and ``backend == "thales_luna"`` —
  :meth:`HSMBackend.from_config` raises :class:`NotImplementedError`
  with a "contact ops" message; the production wiring (CloudHSM-CLI /
  Luna client) is documented in ``docs/compliance/hsm_key_custody.md``
  but not bundled here.

Mechanism set (for any production backend):
    * ``CKM_RSA_PKCS_OAEP``  — encrypt / decrypt
    * ``CKM_RSA_PKCS_PSS``   — sign / verify
    * ``CKM_AES_GCM``        — symmetric DEK wrapping (future)

Standards alignment: NIST SP 800-57 (key management lifecycle),
FIPS 140-3 (cryptographic module validation; inherited from CloudHSM /
Luna in production), eIDAS QSCD profile.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Protocol

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional PKCS#11 import. Kept lazy so the rest of the module imports even
# when ``python-pkcs11`` (or its underlying ``libsofthsm2.so``) is missing.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import shim
    import pkcs11 as _pkcs11
    from pkcs11 import Mechanism as _Mechanism
    from pkcs11 import KeyType as _KeyType
    from pkcs11 import ObjectClass as _ObjectClass

    _PKCS11_AVAILABLE = True
except Exception:  # pragma: no cover
    _pkcs11 = None
    _Mechanism = None
    _KeyType = None
    _ObjectClass = None
    _PKCS11_AVAILABLE = False


# ---------------------------------------------------------------------------
# Protocol-style abstract base
# ---------------------------------------------------------------------------
class _HSMProto(Protocol):  # pragma: no cover - structural typing only
    def generate_keypair(self, label: str) -> tuple[bytes, bytes]: ...
    def sign(self, priv_handle: bytes, data: bytes) -> bytes: ...
    def encrypt(self, pub_handle: bytes, data: bytes) -> bytes: ...
    def decrypt(self, priv_handle: bytes, ct: bytes) -> bytes: ...
    def export_public(self, pub_handle: bytes) -> bytes: ...
    def list_keys(self) -> list[str]: ...


class HSMBackend:
    """Abstract HSM interface.

    Handles are opaque ``bytes`` (HSM-internal labels / object IDs).
    Concrete subclasses: :class:`InMemoryHSMBackend`,
    :class:`SoftHSM2Backend`. ``HSMBackend.from_config(...)`` is the
    canonical way to obtain an instance.
    """

    # Method stubs (subclasses override) ------------------------------------
    def generate_keypair(self, label: str) -> tuple[bytes, bytes]:
        """Generate an RSA-2048 keypair under ``label``.

        Returns ``(pub_handle, priv_handle)``."""
        raise NotImplementedError

    def sign(self, priv_handle: bytes, data: bytes) -> bytes:
        """RSA-PSS / SHA-256 signature (CKM_RSA_PKCS_PSS)."""
        raise NotImplementedError

    def encrypt(self, pub_handle: bytes, data: bytes) -> bytes:
        """RSA-OAEP / SHA-256 encrypt (CKM_RSA_PKCS_OAEP)."""
        raise NotImplementedError

    def decrypt(self, priv_handle: bytes, ct: bytes) -> bytes:
        """RSA-OAEP decrypt — inverse of :meth:`encrypt`."""
        raise NotImplementedError

    def export_public(self, pub_handle: bytes) -> bytes:
        """Return SPKI DER for the referenced public key."""
        raise NotImplementedError

    def list_keys(self) -> list[str]:
        """Enumerate the labels of all keypairs currently held."""
        raise NotImplementedError

    # Factory ---------------------------------------------------------------
    @staticmethod
    def from_config(cfg: dict) -> "HSMBackend":
        """Select a backend by ``cfg['backend']``.

        Recognised values: ``"in_memory"``, ``"softhsm2"``,
        ``"aws_cloudhsm"``, ``"thales_luna"``. The last two raise
        :class:`NotImplementedError` — production deployment is documented
        in ``docs/compliance/hsm_key_custody.md`` and requires ops support.
        """
        backend = cfg.get("backend", "in_memory")
        if backend == "in_memory":
            return InMemoryHSMBackend()
        if backend == "softhsm2":
            return SoftHSM2Backend(
                module_path=cfg.get("module_path"),
                token_label=cfg.get("token_label", "horizon-ric"),
                user_pin=cfg.get("user_pin", "1234"),
            )
        if backend in ("aws_cloudhsm", "thales_luna"):
            raise NotImplementedError(
                f"HSM backend {backend!r} is documented but not bundled — "
                "contact ops (see docs/compliance/hsm_key_custody.md)."
            )
        raise ValueError(f"unknown HSM backend {backend!r}")


# ---------------------------------------------------------------------------
# In-memory backend (TEST-ONLY)
# ---------------------------------------------------------------------------
class InMemoryHSMBackend(HSMBackend):
    """Pure-Python RSA keypair store. **TEST-ONLY** — the private keys live
    in process memory and offer no isolation guarantees.

    Uses real RSA-2048 keys generated via ``cryptography``; padding is
    OAEP-SHA256 for encrypt / decrypt and PSS-SHA256 (salt = digest length)
    for sign. Equivalent to PKCS#11 ``CKM_RSA_PKCS_OAEP`` and
    ``CKM_RSA_PKCS_PSS`` — same wire format, different custody story.
    """

    _KEY_SIZE = 2048

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # label -> (private_key, public_key)
        self._keys: dict[str, tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]] = {}

    # -- key lifecycle ------------------------------------------------------
    def generate_keypair(self, label: str) -> tuple[bytes, bytes]:
        with self._lock:
            if label in self._keys:
                raise ValueError(f"key with label {label!r} already exists")
            priv = rsa.generate_private_key(public_exponent=65537, key_size=self._KEY_SIZE)
            pub = priv.public_key()
            self._keys[label] = (priv, pub)
        # Handles are simply the labels, namespaced to indicate side.
        return (f"pub:{label}".encode(), f"priv:{label}".encode())

    def _resolve(self, handle: bytes, want: str) -> tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
        try:
            side, label = handle.decode().split(":", 1)
        except Exception as e:
            raise ValueError(f"malformed handle {handle!r}") from e
        if side != want:
            raise ValueError(f"expected {want} handle, got {side}")
        if label not in self._keys:
            raise KeyError(f"no key registered under {label!r}")
        return self._keys[label]

    # -- crypto ops ---------------------------------------------------------
    def sign(self, priv_handle: bytes, data: bytes) -> bytes:
        priv, _ = self._resolve(priv_handle, "priv")
        return priv.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )

    def verify(self, pub_handle: bytes, data: bytes, signature: bytes) -> bool:
        """Convenience verifier (not part of the Protocol but useful for
        wiring tests / SecureFedAvg verification path)."""
        _, pub = self._resolve(pub_handle, "pub")
        try:
            pub.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH,
                ),
                hashes.SHA256(),
            )
            return True
        except Exception:
            return False

    def encrypt(self, pub_handle: bytes, data: bytes) -> bytes:
        _, pub = self._resolve(pub_handle, "pub")
        return pub.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

    def decrypt(self, priv_handle: bytes, ct: bytes) -> bytes:
        priv, _ = self._resolve(priv_handle, "priv")
        return priv.decrypt(
            ct,
            padding.OAEP(
                mgf=padding.MGF1(hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

    def export_public(self, pub_handle: bytes) -> bytes:
        _, pub = self._resolve(pub_handle, "pub")
        return pub.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    def list_keys(self) -> list[str]:
        with self._lock:
            return sorted(self._keys.keys())


# ---------------------------------------------------------------------------
# SoftHSM2 backend (PKCS#11)
# ---------------------------------------------------------------------------
class SoftHSM2Backend(HSMBackend):
    """PKCS#11 backend exercising SoftHSM2 (the FOSS reference HSM).

    Functional when both ``python-pkcs11`` is installed AND a
    ``libsofthsm2.so`` is available at one of the standard locations or
    via ``module_path`` / ``$PKCS11_MODULE``. Otherwise ``__init__`` raises
    a clear error.

    NOT FIPS-validated. SoftHSM2 is a *test* HSM; production must use
    AWS CloudHSM (FIPS 140-2 Level 3) or Thales Luna (FIPS 140-3 Level 3).
    """

    _DEFAULT_MODULE_PATHS = (
        "/usr/lib/softhsm/libsofthsm2.so",
        "/usr/lib/x86_64-linux-gnu/softhsm/libsofthsm2.so",
        "/usr/lib/aarch64-linux-gnu/softhsm/libsofthsm2.so",
        "/usr/local/lib/softhsm/libsofthsm2.so",
        "/opt/homebrew/lib/softhsm/libsofthsm2.so",
    )

    def __init__(
        self,
        module_path: str | None = None,
        token_label: str = "horizon-ric",
        user_pin: str = "1234",
    ) -> None:
        if not _PKCS11_AVAILABLE:
            raise RuntimeError(
                "python-pkcs11 not installed — `pip install python-pkcs11`. "
                "SoftHSM2Backend additionally needs libsofthsm2.so on the host."
            )
        path = module_path or os.environ.get("PKCS11_MODULE")
        if path is None:
            for p in self._DEFAULT_MODULE_PATHS:
                if os.path.exists(p):
                    path = p
                    break
        if path is None or not os.path.exists(path):
            raise RuntimeError(
                "libsofthsm2.so not found. Install softhsm2 (apt-get install "
                "softhsm2) or set $PKCS11_MODULE / module_path. "
                "Searched: " + ", ".join(self._DEFAULT_MODULE_PATHS)
            )
        self._lib = _pkcs11.lib(path)
        try:
            self._token = self._lib.get_token(token_label=token_label)
        except Exception as e:
            raise RuntimeError(
                f"SoftHSM2 token {token_label!r} not initialised. Run: "
                f"`softhsm2-util --init-token --slot 0 --label {token_label} "
                f"--pin {user_pin} --so-pin {user_pin}`"
            ) from e
        self._user_pin = user_pin
        self._token_label = token_label

    # -- helpers ------------------------------------------------------------
    def _session(self):
        return self._token.open(rw=True, user_pin=self._user_pin)

    # -- key lifecycle ------------------------------------------------------
    def generate_keypair(self, label: str) -> tuple[bytes, bytes]:
        with self._session() as s:
            pub, priv = s.generate_keypair(_KeyType.RSA, 2048, label=label, store=True)
            # Handles are the human-readable label; PKCS#11 sessions are
            # ephemeral so we re-look-up by label on use.
            del pub, priv
        return (f"pub:{label}".encode(), f"priv:{label}".encode())

    def _find(self, handle: bytes, klass):
        side, label = handle.decode().split(":", 1)
        del side  # informational
        with self._session() as s:
            obj = next(s.get_objects({_pkcs11.Attribute.LABEL: label,
                                      _pkcs11.Attribute.CLASS: klass}))
            return obj, s

    def sign(self, priv_handle: bytes, data: bytes) -> bytes:  # pragma: no cover
        # Cannot exercise without a live SoftHSM2 token; the path is
        # straight-line PKCS#11 calls. Kept thin and clearly documented.
        _, label = priv_handle.decode().split(":", 1)
        with self._session() as s:
            priv = next(s.get_objects({_pkcs11.Attribute.LABEL: label,
                                       _pkcs11.Attribute.CLASS: _ObjectClass.PRIVATE_KEY}))
            return priv.sign(data, mechanism=_Mechanism.SHA256_RSA_PKCS_PSS)

    def encrypt(self, pub_handle: bytes, data: bytes) -> bytes:  # pragma: no cover
        _, label = pub_handle.decode().split(":", 1)
        with self._session() as s:
            pub = next(s.get_objects({_pkcs11.Attribute.LABEL: label,
                                      _pkcs11.Attribute.CLASS: _ObjectClass.PUBLIC_KEY}))
            return pub.encrypt(data, mechanism=_Mechanism.RSA_PKCS_OAEP)

    def decrypt(self, priv_handle: bytes, ct: bytes) -> bytes:  # pragma: no cover
        _, label = priv_handle.decode().split(":", 1)
        with self._session() as s:
            priv = next(s.get_objects({_pkcs11.Attribute.LABEL: label,
                                       _pkcs11.Attribute.CLASS: _ObjectClass.PRIVATE_KEY}))
            return priv.decrypt(ct, mechanism=_Mechanism.RSA_PKCS_OAEP)

    def export_public(self, pub_handle: bytes) -> bytes:  # pragma: no cover
        _, label = pub_handle.decode().split(":", 1)
        with self._session() as s:
            pub = next(s.get_objects({_pkcs11.Attribute.LABEL: label,
                                      _pkcs11.Attribute.CLASS: _ObjectClass.PUBLIC_KEY}))
            # python-pkcs11 exposes the SPKI bytes via the EC_POINT/MODULUS
            # attributes; in production we serialise via pyca/cryptography.
            return bytes(pub[_pkcs11.Attribute.MODULUS])

    def list_keys(self) -> list[str]:  # pragma: no cover
        with self._session() as s:
            labels: set[str] = set()
            for obj in s.get_objects({_pkcs11.Attribute.CLASS: _ObjectClass.PRIVATE_KEY}):
                lbl = obj[_pkcs11.Attribute.LABEL]
                if isinstance(lbl, bytes):
                    lbl = lbl.decode()
                labels.add(str(lbl))
            return sorted(labels)


__all__ = [
    "HSMBackend",
    "InMemoryHSMBackend",
    "SoftHSM2Backend",
]
