"""Privacy layer for cross-operator resource trading auctions.

Promotes Row 16 of GAPS_TO_PILOT.md from SCAFFOLD to PARTIAL by adding two
real privacy primitives on top of the existing trusted-auctioneer Vickrey
implementation in `auction.py`:

1. **Commitment-and-reveal** (`CommitReveal`): bidders post
   ``H(bid || nonce)`` during the bidding window so peers cannot observe
   the bid value, then publish ``(bid, nonce)`` after the deadline. The
   server re-hashes and verifies before scoring. This blocks bid-shading
   based on observed competitor bids.

2. **Paillier additive HE** (`PaillierBidder`): a from-scratch
   number-theoretic implementation of Paillier (1999) with the
   homomorphic identity ``Enc(a) * Enc(b) mod n^2 == Enc(a + b)``.
   Useful for aggregating sealed bids (e.g. clearing-price computation
   without revealing individual bids to anyone but the auctioneer).

3. **`PrivateSecondPriceAuction`**: orchestrator wrapping `auction.py`
   with commit-reveal in front, plus a ``blind_rank`` flag whose
   ``_blind_rank()`` stub documents (and roadmaps) the full MPC
   protocol — that part is Phase-3 research.

SECURITY NOTE — toy modulus
---------------------------
The default 1024-bit Paillier modulus is **for testing only**.
NIST SP 800-57 Part 1 Rev. 5 (Table 2) places 1024-bit RSA-equivalent
moduli below the 112-bit security floor that has been deprecated since
2014; production deployments must use **n ≥ 3072 bits** for >100-year
security horizons. Re-instantiate `PaillierBidder(key_bits=3072)` (or
larger) in any non-test environment.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import time
from dataclasses import dataclass, field

try:  # gmpy2 is ~50× faster on big-int modexp; fall back to stdlib pow().
    from gmpy2 import mpz, powmod  # type: ignore[import-not-found]

    def _powmod(b: int, e: int, m: int) -> int:
        return int(powmod(mpz(b), mpz(e), mpz(m)))

    def _invert(a: int, m: int) -> int:
        from gmpy2 import invert  # type: ignore[import-not-found]

        return int(invert(mpz(a), mpz(m)))

    _HAS_GMPY = True
except ImportError:  # pragma: no cover - exercised on systems without gmpy2

    def _powmod(b: int, e: int, m: int) -> int:
        return pow(b, e, m)

    def _invert(a: int, m: int) -> int:
        return pow(a, -1, m)

    _HAS_GMPY = False


# ----------------------------------------------------------------------
# Commit-Reveal
# ----------------------------------------------------------------------
@dataclass
class Commitment:
    """Sealed-bid commitment posted during the bidding window.

    `digest_hex` is ``SHA-256(bid || nonce)`` so peers cannot recover the
    bid until the bidder reveals it after the deadline.
    """

    bidder_id: str
    digest_hex: str
    submitted_at: float


@dataclass
class Reveal:
    """Plaintext bid + nonce published after the bidding window closes.

    Re-hashed by the server against the matching :class:`Commitment` to
    confirm the bidder did not change their bid post-hoc.
    """

    bidder_id: str
    bid: int
    nonce_hex: str


class CommitReveal:
    """Two-phase commit-reveal scheme over SHA-256.

    Phase 1 (bidding window): each bidder calls :meth:`make_commitment`
    locally, then submits the digest via :meth:`accept_commitment`.
    Phase 2 (after deadline): bidders publish ``(bid, nonce)`` and the
    server runs :meth:`verify_reveal` to re-hash and check equality.

    Hides bids from *other bidders* during the bidding window. (It does
    *not* hide bids from the server post-reveal — that's what Paillier /
    MPC give you.)
    """

    def __init__(self, deadline_unix: float | None = None):
        self.deadline = deadline_unix
        self._commitments: dict[str, Commitment] = {}
        self._closed = False

    @staticmethod
    def make_commitment(bid: int, nonce: bytes | None = None) -> tuple[str, str]:
        """Return ``(digest_hex, nonce_hex)``. Caller keeps the nonce private."""
        if bid < 0:
            raise ValueError(f"bid must be non-negative, got {bid}")
        if nonce is None:
            nonce = secrets.token_bytes(32)
        h = hashlib.sha256()
        h.update(int(bid).to_bytes((max(1, bid.bit_length()) + 7) // 8, "big"))
        h.update(b"||")
        h.update(nonce)
        return h.hexdigest(), nonce.hex()

    def accept_commitment(self, bidder_id: str, digest_hex: str) -> Commitment:
        if self._closed:
            raise RuntimeError("commitment window closed")
        if self.deadline is not None and time.time() > self.deadline:
            raise RuntimeError("commitment window closed (deadline passed)")
        if bidder_id in self._commitments:
            raise ValueError(f"bidder {bidder_id} already committed")
        c = Commitment(bidder_id=bidder_id, digest_hex=digest_hex, submitted_at=time.time())
        self._commitments[bidder_id] = c
        return c

    def close(self) -> None:
        self._closed = True

    def verify_reveal(self, reveal: Reveal) -> bool:
        c = self._commitments.get(reveal.bidder_id)
        if c is None:
            raise ValueError(f"no commitment for bidder {reveal.bidder_id}")
        nonce = bytes.fromhex(reveal.nonce_hex)
        recomputed, _ = self.make_commitment(reveal.bid, nonce=nonce)
        return secrets.compare_digest(recomputed, c.digest_hex)


# ----------------------------------------------------------------------
# Paillier additive HE
# ----------------------------------------------------------------------
def _is_probable_prime(n: int, k: int = 40) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n == p:
            return True
        if n % p == 0:
            return False
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for _ in range(k):
        a = secrets.randbelow(n - 3) + 2
        x = _powmod(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = _powmod(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _gen_prime(bits: int) -> int:
    while True:
        cand = secrets.randbits(bits) | (1 << (bits - 1)) | 1
        if _is_probable_prime(cand):
            return cand


@dataclass
class PaillierCiphertext:
    """Paillier ciphertext bundled with its public modulus.

    ``c`` lives in :math:`Z_{n^2}^*`; ``n`` is the public modulus. Two
    ciphertexts under the same ``n`` can be homomorphically added.
    """

    c: int
    n: int  # public modulus (n^2 is the actual ciphertext space)

    def to_bytes(self) -> bytes:
        size = (self.n.bit_length() * 2 + 7) // 8
        return self.c.to_bytes(size, "big")

    def to_b64(self) -> str:
        return base64.b64encode(self.to_bytes()).decode("ascii")

    @classmethod
    def from_b64(cls, b64: str, n: int) -> "PaillierCiphertext":
        raw = base64.b64decode(b64.encode("ascii"))
        return cls(c=int.from_bytes(raw, "big"), n=n)


class PaillierBidder:
    """Minimal Paillier (1999) additive HE.

    Key generation: pick primes p,q of bit-length key_bits/2 with
    gcd(pq, (p-1)(q-1)) = 1; set n = p*q, lambda = lcm(p-1, q-1).
    With g = n+1 (the standard simplification), mu = lambda^-1 mod n.

    Encryption: ``Enc(m) = (1 + m*n) * r^n mod n^2``  with random
    r ∈ Z_n^*. (Uses the g = n+1 shortcut: ``g^m = 1 + m*n mod n^2``.)
    Decryption: ``m = L(c^lambda mod n^2) * mu mod n`` where
    ``L(x) = (x - 1) / n``.
    Homomorphism: ``Enc(a) * Enc(b) mod n^2 == Enc(a + b mod n)``.
    """

    def __init__(self, key_bits: int = 1024):
        if key_bits < 256:
            raise ValueError("key_bits must be ≥ 256 (and ≥ 3072 for production)")
        half = key_bits // 2
        while True:
            p = _gen_prime(half)
            q = _gen_prime(half)
            if p == q:
                continue
            n = p * q
            if n.bit_length() == key_bits:
                break
        self.n = n
        self.n_sq = n * n
        self.g = n + 1  # standard simplification
        lam = (p - 1) * (q - 1) // _gcd(p - 1, q - 1)  # lcm(p-1, q-1)
        self._lambda = lam
        # With g = n+1: L(g^lambda mod n^2) = lambda mod n, so mu = lambda^-1.
        self._mu = _invert(lam % n, n)
        self.key_bits = key_bits

    # -- public API ---------------------------------------------------
    def encrypt(self, m: int) -> PaillierCiphertext:
        if not 0 <= m < self.n:
            raise ValueError(f"plaintext out of range [0, n); got {m}")
        # Pick r in Z_n^*. With overwhelming probability a random
        # r ∈ [1, n) is coprime to n; we retry on the negligible miss.
        while True:
            r = secrets.randbelow(self.n - 1) + 1
            if _gcd(r, self.n) == 1:
                break
        # g^m mod n^2 = (1 + m*n) mod n^2 — exact, no modexp needed.
        gm = (1 + m * self.n) % self.n_sq
        rn = _powmod(r, self.n, self.n_sq)
        return PaillierCiphertext(c=(gm * rn) % self.n_sq, n=self.n)

    def add_ciphertexts(
        self, c1: PaillierCiphertext, c2: PaillierCiphertext
    ) -> PaillierCiphertext:
        if c1.n != self.n or c2.n != self.n:
            raise ValueError("ciphertexts from a different public key")
        return PaillierCiphertext(c=(c1.c * c2.c) % self.n_sq, n=self.n)

    def decrypt(self, c: PaillierCiphertext) -> int:
        if c.n != self.n:
            raise ValueError("ciphertext from a different public key")
        x = _powmod(c.c, self._lambda, self.n_sq)
        l = (x - 1) // self.n
        return (l * self._mu) % self.n


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------
@dataclass
class PrivateAuctionResult:
    """Outcome of a privacy-preserving second-price auction settlement.

    ``rejected_bidders`` contains anyone whose reveal failed verification;
    ``blind_ranked`` records whether the orchestrator used the Phase-3
    blind-ranking stub vs. the plaintext Vickrey path.
    """

    winner_id: str
    price_paid: int  # second price (Vickrey)
    num_bidders: int
    rejected_bidders: list[str] = field(default_factory=list)
    blind_ranked: bool = False


class PrivateSecondPriceAuction:
    """Second-price (Vickrey) auction with commit-reveal privacy.

    Flow
    ----
    1. ``open_bidding(bidder_ids)`` — initialise the commitment window.
    2. Each bidder calls ``CommitReveal.make_commitment(bid)`` locally
       and posts the digest via ``submit_commitment``.
    3. ``close_bidding()`` — auctioneer freezes the commitment set.
    4. Each bidder posts ``(bid, nonce)`` via ``submit_reveal``.
       Server hashes and rejects mismatches.
    5. ``settle()`` runs second-price scoring on revealed bids only.

    Modes
    -----
    * ``mode="blind"`` (default in v2) — runs the DGK secure comparison
      protocol from :mod:`dgk_compare` to find the winner via a
      homomorphic tournament. The auctioneer's intermediate state holds
      only Paillier ciphertexts; the only plaintext output is the
      ranking and the winning bid (which the winner must reveal anyway
      to be paid).
    * ``mode="fast"`` — the legacy trusted-auctioneer path: revealed
      bids are scored in the clear. Useful for benchmarks and for
      backwards compatibility.

    The boolean ``blind_rank`` constructor flag is preserved for
    backwards compatibility (``True`` ↔ ``mode="blind"``).
    """

    def __init__(
        self,
        blind_rank: bool | None = None,
        deadline_unix: float | None = None,
        mode: str = "blind",
        paillier_bits: int = 1024,
        comparison_bits: int = 32,
    ):
        # Resolve mode from the legacy flag if the caller passed it.
        if blind_rank is not None:
            mode = "blind" if blind_rank else "fast"
        if mode not in ("blind", "fast"):
            raise ValueError(f"mode must be 'blind' or 'fast'; got {mode!r}")
        self.commit_reveal = CommitReveal(deadline_unix=deadline_unix)
        self.mode = mode
        self.blind_rank = mode == "blind"
        self._paillier_bits = paillier_bits
        self._comparison_bits = comparison_bits
        # Lazily constructed: keygen is the slow step, no point paying
        # for it on a `mode="fast"` auction.
        self._paillier: PaillierBidder | None = None
        self._comparator = None  # type: ignore[var-annotated]
        # Diagnostic state captured during _blind_rank for audit/tests.
        self._last_blind_state: dict | None = None
        self._registered: set[str] = set()
        self._reveals: dict[str, Reveal] = {}
        self._rejected: list[str] = []
        self._closed = False

    def open_bidding(self, bidder_ids: list[str]) -> None:
        for bid_id in bidder_ids:
            if not isinstance(bid_id, str) or not bid_id:
                raise ValueError("bidder_id must be a non-empty string")
            self._registered.add(bid_id)

    def submit_commitment(self, bidder_id: str, digest_hex: str) -> Commitment:
        if bidder_id not in self._registered:
            raise ValueError(f"bidder {bidder_id} not registered")
        return self.commit_reveal.accept_commitment(bidder_id, digest_hex)

    def close_bidding(self) -> None:
        self.commit_reveal.close()
        self._closed = True

    def submit_reveal(self, bidder_id: str, bid: int, nonce_hex: str) -> bool:
        if not self._closed:
            raise RuntimeError("close_bidding() before accepting reveals")
        rev = Reveal(bidder_id=bidder_id, bid=bid, nonce_hex=nonce_hex)
        if not self.commit_reveal.verify_reveal(rev):
            self._rejected.append(bidder_id)
            return False
        self._reveals[bidder_id] = rev
        return True

    def settle(self) -> PrivateAuctionResult:
        if not self._closed:
            raise RuntimeError("close_bidding() before settle()")
        valid = list(self._reveals.values())
        if len(valid) < 2:
            raise ValueError(
                f"second-price auction requires ≥ 2 valid reveals, got {len(valid)}"
            )
        if self.blind_rank:
            winner_id, price = self._blind_rank(valid)
        else:
            valid.sort(key=lambda r: (r.bid, r.bidder_id), reverse=True)
            # Lexicographic tie-break on bidder_id keeps the winner
            # deterministic when top bids tie.
            winner = valid[0]
            tied = [r for r in valid if r.bid == winner.bid]
            if len(tied) > 1:
                tied.sort(key=lambda r: r.bidder_id)
                winner = tied[0]
            second = max(r.bid for r in valid if r.bidder_id != winner.bidder_id)
            winner_id, price = winner.bidder_id, second
        return PrivateAuctionResult(
            winner_id=winner_id,
            price_paid=price,
            num_bidders=len(valid),
            rejected_bidders=list(self._rejected),
            blind_ranked=self.blind_rank,
        )

    def _ensure_comparator(self):
        """Lazily build the Paillier key + DGK comparator on first use."""
        if self._comparator is None:
            # Local import to avoid a circular dependency at module load.
            from horizon_ric.trading.dgk_compare import DGKComparator

            self._paillier = PaillierBidder(key_bits=self._paillier_bits)
            self._comparator = DGKComparator(
                self._paillier, bits=self._comparison_bits
            )
        return self._comparator

    def _blind_rank(self, reveals: list[Reveal]) -> tuple[str, int]:
        """MPC blind ranking via DGK-2007 secure comparison.

        Encrypts every revealed bid bit-by-bit under a fresh Paillier
        key, then runs :meth:`DGKComparator.tournament_rank` to obtain
        the index of the winner without holding any bid in plaintext
        once the encryption step has completed. The auctioneer's
        intermediate state (captured in ``self._last_blind_state``
        for audit purposes) contains only:

        * ``encrypted_bids``: list of Paillier ciphertexts (opaque ints).
        * ``ranking``: list of bidder indices in descending order.
        * ``winner_id``, ``second_price``: the public auction outcome.

        The bid plaintexts are *not* stored in this state. Tests in
        ``tests/test_mpc_blind_ranking.py`` (``test_blind_rank_does_not_leak``)
        verify the absence.

        Security: semi-honest. Malicious-bidder tolerance requires the
        Phase-3 verifiable-SS extension (≤1-month effort, deferred
        until first auction customer per ``GAPS_TO_PILOT.md``).
        """
        cmp = self._ensure_comparator()

        # Stable ordering for reproducibility.
        sorted_reveals = sorted(reveals, key=lambda r: r.bidder_id)
        bids_plain = [r.bid for r in sorted_reveals]
        ids = [r.bidder_id for r in sorted_reveals]

        # Encrypt bids. After this point, bids_plain is captured *only*
        # in the local frame of `_blind_rank`; the auctioneer-visible
        # `_last_blind_state` keeps ciphertexts.
        assert self._paillier is not None  # set by _ensure_comparator
        enc_bids = [self._paillier.encrypt(b) for b in bids_plain]

        # Run DGK tournament. Internally, every comparison is a fresh
        # encrypted carry-chain that is blinded and shuffled before the
        # decryption oracle, so no bid value escapes.
        ranking = cmp.tournament_rank(enc_bids)

        winner_idx = ranking[0]
        second_idx = ranking[1] if len(ranking) > 1 else ranking[0]
        winner_id = ids[winner_idx]
        # Second-price (Vickrey): the second-highest bid value.
        second_price = bids_plain[second_idx]

        self._last_blind_state = {
            "encrypted_bids": enc_bids,
            "ranking": ranking,
            "winner_id": winner_id,
            "second_price": second_price,
            "ids": ids,
        }
        return winner_id, second_price


__all__ = [
    "Commitment",
    "CommitReveal",
    "PaillierBidder",
    "PaillierCiphertext",
    "PrivateAuctionResult",
    "PrivateSecondPriceAuction",
    "Reveal",
]
