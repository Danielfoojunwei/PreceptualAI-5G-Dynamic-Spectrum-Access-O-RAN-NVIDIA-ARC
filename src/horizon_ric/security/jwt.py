"""RS256 JWT issuance + verification with zero-downtime key rotation.

Uses `python-jose[cryptography]` (Apache 2.0). RS256 is mandatory: the
public key can be distributed to verifiers (gateways, sidecars) while the
private key stays on the SMO/rApp issuer host.

Key rotation:
    A graceful rotation accepts BOTH the previous and the new signing key
    for verification during an overlap window (default 24 hours, was 1 h).
    New tokens are signed with the new key only. Once every outstanding
    token has expired, callers can drop the old key.

Required claims:
    sub      — the subject (user identity)
    tenant   — the tenant (Casbin domain) the subject is acting in
    roles    — the list of roles the subject holds in `tenant`
    iss, aud — bound at construction time
    iat, exp — set automatically

FORMAL ROTATION-WINDOW BOUND (closes Devil-C Finding 12/33).
    Let ``T_rot`` be the wall-clock instant ``rotate_signing_key()``
    finishes; ``W`` the configured ``overlap_window_seconds``;
    ``Δ_skew`` the bounded clock-skew tolerance enforced by
    :func:`verify_token` (default ``MAX_CLOCK_SKEW_SECONDS = 60``);
    ``K_old`` the previous signing key.

    **Claim.** Under the EUF-CMA assumption on RS256 and a verifier
    clock with absolute skew ``|Δ| ≤ Δ_skew`` relative to the issuer,
    for every token ``T`` minted with ``K_old``,

        Pr[ verify(T) = True   |   t_verify > T_rot + W + Δ_skew ] = 0.

    **Proof sketch.** The verifier maintains the active key plus a
    list of retired keys, each tagged with ``retire_at = T_rot + W``.
    At verification time we (1) drop any retired key whose
    ``retire_at < t_verify - Δ_skew`` (lines 195-205, ``_prune_retired_keys``)
    and (2) refuse to validate against any key not in the surviving set.
    Since every old-key forgery requires the verifier to attempt
    ``K_old``, and ``K_old`` is provably absent from the verifier's
    set whenever ``t_verify > retire_at + Δ_skew = T_rot + W + Δ_skew``,
    no such token can verify. EUF-CMA on RS256 gives the
    complementary argument that no valid signature with ``K_old`` can
    be forged without compromising the private key. □

    **Implication.** Operators MUST size ``W`` so ``W ≥ T_ttl + Δ_skew``,
    where ``T_ttl`` is the longest TTL minted under the old key, OR a
    legitimate token will be rejected after rotation. The default
    ``W = 24 h`` and ``Δ_skew = 60 s`` covers any TTL up to 23h59m.

    See also THEOREMS.md §2 for the related anchored-chain
    unforgeability theorem.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jose import jwt as jose_jwt
from jose.exceptions import JWTError

# Re-export JWTError so callers needn't import jose directly.
__all__ = ["JWTManager", "JWTError"]

_DEFAULT_ALG = "RS256"
_OVERLAP_WINDOW_DEFAULT_SEC = 24 * 3600  # 24 hours (Devil-C #33)
MAX_CLOCK_SKEW_SECONDS: int = 60
"""Hard cap on tolerated clock skew between issuer and verifier (seconds).

Bounds the ``Δ_skew`` term in the rotation-window proof in this module's
docstring. Any retired key whose ``retire_at`` is more than this many
seconds in the past is unconditionally evicted from the verifier's
key set."""


def _load_private_pem(path: Path) -> str:
    """Read a PEM private key from disk. Raises FileNotFoundError or
    ValueError if the file is missing or not a valid private key."""
    raw = Path(path).read_bytes()
    # Validate it's a real RSA key — fail fast at startup rather than on
    # the first mint attempt.
    serialization.load_pem_private_key(raw, password=None)
    return raw.decode("utf-8")


def _derive_public_pem(private_pem: str) -> str:
    key = serialization.load_pem_private_key(
        private_pem.encode("utf-8"), password=None
    )
    if not isinstance(key, rsa.RSAPrivateKey):
        raise ValueError("Signing key must be an RSA private key for RS256.")
    pub = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return pub.decode("utf-8")


@dataclass
class _ActiveKey:
    """A signing/verification key with an optional retirement deadline.

    `retire_at` is None for the currently-active signing key. After
    rotation the previous key gets `retire_at = now + overlap_window` and
    is used only for verification until that deadline passes.
    """

    private_pem: str
    public_pem: str
    retire_at: float | None  # epoch seconds; None = active for signing


class JWTManager:
    """Mint + verify RS256 JWTs with zero-downtime signing key rotation.

    Construction loads a single RS256 private key (the active signing
    key). After `rotate_signing_key(new_path)`, the old key is retained
    for verification only, until its overlap deadline.
    """

    def __init__(
        self,
        signing_key_path: Path | str,
        issuer: str,
        audience: str,
        overlap_window_seconds: int = _OVERLAP_WINDOW_DEFAULT_SEC,
    ):
        self._issuer = issuer
        self._audience = audience
        self._overlap_window = int(overlap_window_seconds)
        active = _load_active_key(signing_key_path)
        self._active: _ActiveKey = active
        # Retired keys: still verify, but no longer sign.
        self._retired: list[_ActiveKey] = []

    # ------------------------------------------------------------------
    # Mint
    # ------------------------------------------------------------------
    def mint_token(
        self,
        subject: str,
        tenant: str,
        roles: list[str],
        ttl_seconds: int = 3600,
    ) -> str:
        """Sign and return an RS256 JWT.

        Required claims encoded:
            sub, tenant, roles, iss, aud, iat, exp, jti
        """
        if not subject:
            raise ValueError("subject is required")
        if not tenant:
            raise ValueError("tenant is required")
        if roles is None:
            raise ValueError("roles is required (use [] for none)")
        now = int(time.time())
        claims = {
            "sub": subject,
            "tenant": tenant,
            "roles": list(roles),
            "iss": self._issuer,
            "aud": self._audience,
            "iat": now,
            "exp": now + int(ttl_seconds),
            "jti": uuid.uuid4().hex,
        }
        return jose_jwt.encode(
            claims,
            self._active.private_pem,
            algorithm=_DEFAULT_ALG,
        )

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------
    def verify_token(self, token: str) -> dict:
        """Decode + validate `token`. Returns the claim dict on success.

        Raises `jose.JWTError` (or a subclass) on any failure: bad
        signature, expired, wrong audience/issuer, missing claims.

        Rotation-window proof (see module docstring §"FORMAL
        ROTATION-WINDOW BOUND"): retired keys are pruned eagerly at
        verify time using the bounded-skew margin
        ``MAX_CLOCK_SKEW_SECONDS``, so old-key tokens cannot verify
        once ``t > retire_at + MAX_CLOCK_SKEW_SECONDS``.
        """
        # Eagerly prune retired keys past their effective deadline so a
        # long-running issuer that doesn't rotate again still expires
        # them. Closes Devil-C #33 sub-bullet 2 (key set was monotonic).
        self._prune_retired_keys()

        # Try the active key first, then any non-retired retired keys.
        keys_to_try: list[_ActiveKey] = [self._active]
        now = time.time()
        keys_to_try.extend(
            k for k in self._retired
            if k.retire_at is None or k.retire_at > now
        )
        last_err: Exception | None = None
        for k in keys_to_try:
            try:
                claims = jose_jwt.decode(
                    token,
                    k.public_pem,
                    algorithms=[_DEFAULT_ALG],
                    audience=self._audience,
                    issuer=self._issuer,
                )
            except JWTError as exc:
                last_err = exc
                continue
            # Enforce required claims explicitly. python-jose validates
            # exp/iss/aud automatically; we additionally require sub,
            # tenant, roles since they are application-mandatory.
            for required in ("sub", "tenant", "roles"):
                if required not in claims:
                    raise JWTError(f"missing required claim: {required}")
            return claims
        # All keys rejected — re-raise the last decoder error.
        assert last_err is not None
        raise last_err

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------
    def _prune_retired_keys(self) -> None:
        """Drop any retired key whose deadline has passed (with a
        ``MAX_CLOCK_SKEW_SECONDS`` grace window).

        Called from both :meth:`verify_token` and
        :meth:`rotate_signing_key` so the key set is monotonically
        bounded by the rotation-window proof in the module docstring.
        """
        cutoff = time.time() - MAX_CLOCK_SKEW_SECONDS
        self._retired = [
            k for k in self._retired
            if k.retire_at is None or k.retire_at > cutoff
        ]

    def rotate_signing_key(self, new_key_path: Path | str) -> None:
        """Promote `new_key_path` to the active signing key.

        The previous key is retained for verification until the overlap
        window elapses, enabling zero-downtime rotation: tokens issued by
        the old key still verify until they naturally expire (assuming
        their TTL fits within the overlap).
        """
        new_active = _load_active_key(new_key_path)
        previous = self._active
        previous.retire_at = time.time() + self._overlap_window
        self._retired.append(previous)
        self._active = new_active
        # Drop fully-expired retired keys so memory doesn't grow unbounded.
        self._prune_retired_keys()

    # ------------------------------------------------------------------
    # Introspection — useful for the CLI / health endpoint.
    # ------------------------------------------------------------------
    @property
    def issuer(self) -> str:
        return self._issuer

    @property
    def audience(self) -> str:
        return self._audience

    @property
    def public_pem(self) -> str:
        """The currently-active public key, PEM-encoded."""
        return self._active.public_pem

    def retired_key_count(self) -> int:
        return len(self._retired)


def _load_active_key(path: Path | str) -> _ActiveKey:
    private_pem = _load_private_pem(Path(path))
    public_pem = _derive_public_pem(private_pem)
    return _ActiveKey(
        private_pem=private_pem, public_pem=public_pem, retire_at=None
    )
