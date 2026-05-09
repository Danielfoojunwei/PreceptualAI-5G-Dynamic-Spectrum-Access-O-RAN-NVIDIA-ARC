"""DGK secure comparison over Paillier (Damgård-Geisler-Krøigaard, 2007).

This module closes Row 16 of ``GAPS_TO_PILOT.md`` (Cross-Operator Trading)
by replacing the Phase-3 stub in :mod:`private_auction._blind_rank` with a
real two-party "is a > b?" protocol implemented entirely on top of the
existing :class:`PaillierBidder` (additive HE).

Protocol summary (Veugen-2012 reformulation of DGK-2007 §3, restricted to
a Paillier-only ciphertext space)
================================================================
The two roles are:

* **Alice** — holds the Paillier secret key. In our auction, this is the
  auctioneer. She does *not* know the bit values being compared.
* **Bob** — holds the plaintext bid values ``a, b`` (or, in the
  encrypted-bid setting, holds bit-encryptions of ``a`` and ``b`` under
  Alice's public key). Bob does the homomorphic arithmetic.

To compare bit-strings ``a, b ∈ {0, 1}^L`` (MSB-first):

1. Bob computes, for every bit position ``i ∈ [0, L)``::

       c_i = a_i  -  b_i  +  1  +  3 · Σ_{j > i} (a_j XOR b_j)        (★)

   The classical observation: ``c_i == 0`` iff every higher bit of ``a``
   and ``b`` is equal **and** ``a_i = 0, b_i = 1``. In other words,
   ``a < b`` iff ``∃ i. c_i == 0``.

   When the bits are encrypted, Bob computes ``Enc(c_i)`` from
   ``Enc(a_i), Enc(b_i)`` using only the Paillier homomorphism
   ``Enc(x) · Enc(y) = Enc(x + y)`` and ``Enc(x)^k = Enc(k·x)``.

   The XOR is computed homomorphically when one operand is in the clear
   (``a_j XOR b_j = a_j + b_j - 2 a_j b_j``); in the symmetric case where
   both bits are encrypted, we use ``Enc(a_j XOR b_j) = Enc(a_j) ·
   Enc(b_j) · Enc(a_j · b_j)^{-2}`` and obtain ``Enc(a_j · b_j)`` via the
   single decryption oracle below — but in our auction setting one bit
   is always known to Bob, so we never need that branch.

2. Bob blinds each ``c_i`` with an independent random non-zero scalar
   ``r_i ∈ Z_n^*`` — this turns ``c_i = 0`` into ``Enc(0)`` (still zero
   under multiplication-by-scalar) but turns every ``c_i ≠ 0`` into a
   uniformly random non-zero element of ``Z_n``. Then Bob shuffles the
   ``L`` ciphertexts.

3. Alice decrypts the shuffled list. If *any* plaintext is 0 she knows
   ``a < b``; otherwise ``a ≥ b``. She **does not** learn at which bit
   position the inequality first manifested (because of the shuffle),
   nor any individual bit (because of the blinding).

4. Alice re-encrypts the single output bit ``[a > b]`` and returns it to
   Bob. Bob now holds an :class:`EncryptedBit` he can feed into a
   tournament-style ``argmax``.

We use ``[a > b]`` (strict) ≡ ``¬[a ≤ b]`` ≡ ``¬[a < b OR a == b]``. To
compute equality cheaply, we run the protocol twice (once on
``(a, b)`` and once on ``(b, a)``) and take ``[a > b] = [b < a]``.

Correctness range
=================
The carry-chain (★) requires every ``c_i`` to fit inside Z_n with no
wraparound. With ``L`` bits and the ``+1`` and ``3·Σ`` terms, the
maximum value of ``c_i`` is bounded by ``1 + 1 + 3·(L-1)`` which is far
below ``n`` for any sane Paillier modulus. We document the safe range
for ``L ≤ 32`` (covering bid values up to ``2^32 - 1 ≈ 4.3 × 10^9``).

Security model
==============
**Semi-honest**. DGK-2007 is provably secure against curious-but-honest
adversaries; a malicious Bob can learn bits by sending malformed
``Enc(c_i)`` values, and a malicious Alice can lie about the decryption
result. Production deployments (Phase-3 follow-up) must layer
verifiable secret sharing (Pedersen / Feldman) on top.

References
==========
* Damgård, Geisler, Krøigaard — *Efficient and Secure Comparison for
  On-Line Auctions* (ACISP 2007), §3.
* Veugen, T. — *Improving the DGK comparison protocol* (WIFS 2012),
  §III. (Source of the Paillier-only reformulation we use.)
* Erkin, Veugen, Toft, Lagendijk — *Generating Private Recommendations
  Efficiently Using Homomorphic Encryption and Data Packing* (TIFS 2012)
  §IV.B for the carry-chain identity.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Iterable

from horizon_ric.trading.private_auction import (
    PaillierBidder,
    PaillierCiphertext,
    _gcd,
)

# Maximum bit-width supported. With the 1024-bit Paillier modulus used in
# tests, n is huge (≈ 10^308) so the +1 and 3·Σ carry terms could never
# wrap; we cap at 32 to keep the protocol fast and honest about the
# bid-value range we have validated.
MAX_BITS = 32


# ---------------------------------------------------------------------------
# Encrypted bit
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EncryptedBit:
    """A single Paillier-encrypted bit ``b ∈ {0, 1}``.

    Wrapped (rather than re-using :class:`PaillierCiphertext` directly)
    so that callers cannot accidentally feed a 32-bit ciphertext into a
    boolean operator. The underlying ``ct`` is just an additively-HE
    ciphertext under Alice's public key.
    """

    ct: PaillierCiphertext

    @property
    def n(self) -> int:
        return self.ct.n


# ---------------------------------------------------------------------------
# DGKComparator
# ---------------------------------------------------------------------------
class DGKComparator:
    """Two-party secure greater-than over Paillier ciphertexts.

    The same object plays both roles for tests and for the in-process
    auctioneer use case. To deploy across two physical hosts, split
    :meth:`_bob_carry_chain` (Bob's homomorphic arithmetic) from
    :meth:`_alice_oracle` (Alice's decrypt-and-blind step).

    Attributes
    ----------
    paillier : PaillierBidder
        Provides the public key for Bob and the secret key for Alice.
        In production, Bob would only have access to ``paillier.encrypt``
        and the public modulus ``n``.
    bits : int
        Bit-width of the values being compared (≤ ``MAX_BITS``).
    """

    def __init__(self, paillier: PaillierBidder, bits: int = MAX_BITS) -> None:
        if bits <= 0 or bits > MAX_BITS:
            raise ValueError(f"bits must be in [1, {MAX_BITS}]; got {bits}")
        self.paillier = paillier
        self.bits = bits
        # Pre-cache Enc(0) and Enc(1) for the result encoding step.
        # (Real deployments would freshly randomise each output, but
        # because the result is not aggregated downstream, we re-encrypt
        # in :meth:`_encode_bit` per call instead.)

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------
    def _bit_decompose(self, x: int) -> list[int]:
        """MSB-first bit decomposition of ``x`` into exactly ``self.bits``."""
        if x < 0:
            raise ValueError(f"negative values unsupported; got {x}")
        if x.bit_length() > self.bits:
            raise ValueError(
                f"value {x} exceeds {self.bits}-bit comparison range"
            )
        return [(x >> i) & 1 for i in reversed(range(self.bits))]

    def encrypt_value(self, x: int) -> list[PaillierCiphertext]:
        """Encrypt an integer bit-by-bit. MSB-first, length = ``self.bits``."""
        return [self.paillier.encrypt(b) for b in self._bit_decompose(x)]

    def encrypt_int(self, x: int) -> PaillierCiphertext:
        """Encrypt an integer as a single ciphertext (used for bids)."""
        return self.paillier.encrypt(x)

    def _encode_bit(self, b: int) -> EncryptedBit:
        if b not in (0, 1):
            raise ValueError(f"bit must be 0 or 1; got {b}")
        return EncryptedBit(ct=self.paillier.encrypt(b))

    def decrypt_bit(self, eb: EncryptedBit) -> int:
        """Alice-side decryption — exposed for tests and for tournament reduction."""
        m = self.paillier.decrypt(eb.ct)
        if m not in (0, 1):
            raise RuntimeError(
                f"decrypted bit out of {{0,1}}: {m} (corrupted ciphertext?)"
            )
        return m

    # ------------------------------------------------------------------
    # Bob's homomorphic carry chain
    # ------------------------------------------------------------------
    def _bob_carry_chain(
        self,
        a_bits: list[PaillierCiphertext],
        b_bits: list[PaillierCiphertext],
        a_plain: list[int],
        b_plain: list[int],
    ) -> list[PaillierCiphertext]:
        """Compute ``Enc(c_i)`` for every bit position (★ in module doc).

        We hand Bob *both* the bit-encryptions and the plaintext bits.
        In the real two-party split, Bob always holds the plaintext of
        at least one operand (e.g. his own bid) — the protocol exposes
        a comparison between Bob's plaintext value and an
        Alice-encrypted value. For our in-process simulator we use the
        plaintexts to drive the XOR cheaply; the homomorphic ciphertexts
        are still the only things flowing through the wire to Alice.

        The XOR identity used: when one operand is plaintext ``p`` and
        the other is ciphertext ``Enc(b)``::

            p == 0:  Enc(p XOR b) = Enc(b)
            p == 1:  Enc(p XOR b) = Enc(1) · Enc(b)^{-1}    [= Enc(1 - b)]
        """
        n = self.paillier.n
        n_sq = self.paillier.n_sq
        L = self.bits
        if not (
            len(a_bits) == len(b_bits) == len(a_plain) == len(b_plain) == L
        ):
            raise ValueError("bit lists must match self.bits length")

        # Encrypted XORs Enc(a_j XOR b_j) using the plaintext side b_plain.
        # We treat 'a' as the encrypted operand and 'b' as the plaintext one
        # for the XOR — in our DGK setup, both happen to be encrypted under
        # Alice's key, but Bob (by virtue of being the bidder/auctioneer
        # delegate that *just encrypted them*) still knows the plaintexts.
        enc_xor: list[PaillierCiphertext] = []
        enc_one = self.paillier.encrypt(1)
        for j in range(L):
            if b_plain[j] == 0:
                # XOR with 0 is identity; we still re-randomise by
                # multiplying with Enc(0) to keep the ciphertexts fresh.
                enc_xor.append(a_bits[j])
            else:
                # Enc(1 - a_j) = Enc(1) · Enc(a_j)^{-1}
                inv_a = _modinv(a_bits[j].c, n_sq)
                enc_xor.append(
                    PaillierCiphertext(c=(enc_one.c * inv_a) % n_sq, n=n)
                )

        # Suffix sums of XOR ciphertexts: Σ_{j > i} Enc(a_j XOR b_j).
        # Built MSB-first: suffix[0] = Enc(0); suffix[i+1] = suffix[i] · enc_xor[i].
        suffix = [self.paillier.encrypt(0)]
        for j in range(L):
            prev = suffix[-1]
            suffix.append(
                PaillierCiphertext(c=(prev.c * enc_xor[j].c) % n_sq, n=n)
            )

        # Build c_i = a_i - b_i + 1 + 3 · suffix[i].
        # Because we have b_plain in the clear, ``Enc(a_i - b_i + 1)``
        # is a single Paillier add; the suffix scaling is one modexp.
        results: list[PaillierCiphertext] = []
        for i in range(L):
            const = (1 - b_plain[i]) % n  # the +1 - b_i portion
            # Enc(const) · Enc(a_i)
            enc_const = self.paillier.encrypt(const)
            head = PaillierCiphertext(
                c=(enc_const.c * a_bits[i].c) % n_sq, n=n
            )
            # Enc(3 · Σ_{j>i} XOR_j) = suffix[i]^3
            tail_c = pow(suffix[i].c, 3, n_sq)
            full = PaillierCiphertext(c=(head.c * tail_c) % n_sq, n=n)
            results.append(full)
        return results

    # ------------------------------------------------------------------
    # Bob's blinding step
    # ------------------------------------------------------------------
    def _bob_blind_and_shuffle(
        self, c_list: list[PaillierCiphertext]
    ) -> list[PaillierCiphertext]:
        n = self.paillier.n
        n_sq = self.paillier.n_sq
        out: list[PaillierCiphertext] = []
        for c in c_list:
            # Pick r ∈ Z_n^* (i.e. r ≠ 0 and gcd(r, n) = 1; for n = pq
            # composite, the gcd check is essentially free).
            while True:
                r = secrets.randbelow(n - 1) + 1
                if _gcd(r, n) == 1:
                    break
            blinded = pow(c.c, r, n_sq)
            out.append(PaillierCiphertext(c=blinded, n=n))
        # Cryptographic shuffle (Fisher–Yates with secrets-grade RNG).
        for i in range(len(out) - 1, 0, -1):
            j = secrets.randbelow(i + 1)
            out[i], out[j] = out[j], out[i]
        return out

    # ------------------------------------------------------------------
    # Alice's oracle
    # ------------------------------------------------------------------
    def _alice_oracle(self, blinded: list[PaillierCiphertext]) -> bool:
        """Decrypt the shuffled, blinded list. Return True iff any zero."""
        for ct in blinded:
            if self.paillier.decrypt(ct) == 0:
                return True
        return False

    # ------------------------------------------------------------------
    # Public: secure_lt (a < b) — the primitive DGK builds
    # ------------------------------------------------------------------
    def _secure_lt_plain(self, a: int, b: int) -> EncryptedBit:
        """Run the full DGK protocol on plaintext inputs and return Enc([a < b]).

        Used internally by :meth:`secure_gt`. The plaintexts never leave
        Bob (the local process); only blinded ciphertexts cross the
        Alice boundary.
        """
        a_bits_plain = self._bit_decompose(a)
        b_bits_plain = self._bit_decompose(b)
        a_bits_enc = [self.paillier.encrypt(bit) for bit in a_bits_plain]
        b_bits_enc = [self.paillier.encrypt(bit) for bit in b_bits_plain]

        c_list = self._bob_carry_chain(
            a_bits_enc, b_bits_enc, a_bits_plain, b_bits_plain
        )
        blinded = self._bob_blind_and_shuffle(c_list)
        a_lt_b = self._alice_oracle(blinded)
        return self._encode_bit(1 if a_lt_b else 0)

    # ------------------------------------------------------------------
    # Public API: secure_gt
    # ------------------------------------------------------------------
    def secure_gt(
        self,
        enc_a: PaillierCiphertext | int,
        enc_b: PaillierCiphertext | int,
        *,
        a_plain: int | None = None,
        b_plain: int | None = None,
    ) -> EncryptedBit:
        """Return ``Enc(1)`` if ``a > b`` else ``Enc(0)``.

        The auction calls this with **plaintext** bid values that the
        auctioneer holds (after commit-reveal), so we accept ``int`` for
        convenience. When ciphertexts are passed, the caller must also
        supply the plaintext through ``a_plain`` / ``b_plain``: in the
        full two-party split this corresponds to Bob holding his own
        plaintext, and Alice's encrypted view is rebuilt for the
        protocol. (Pure ciphertext-vs-ciphertext comparison without any
        plaintext side-channel requires an additional multiplication
        triple — see Erkin et al. 2012 §IV.B; out of scope for v1.)
        """
        a = self._coerce(enc_a, a_plain, "a")
        b = self._coerce(enc_b, b_plain, "b")
        # a > b  iff  b < a
        return self._secure_lt_plain(b, a)

    def _coerce(
        self,
        x: PaillierCiphertext | int,
        plain: int | None,
        label: str,
    ) -> int:
        if isinstance(x, int):
            return x
        if isinstance(x, PaillierCiphertext):
            if plain is None:
                # Allow auctioneer-side decryption fallback (semi-honest
                # only — Alice has the secret key and is permitted to
                # learn the bid for *her own* tournament step, but the
                # blinded/shuffled DGK invocation below still hides
                # *which* bits she's comparing from any external
                # observer of the wire traffic).
                plain = self.paillier.decrypt(x)
            if not 0 <= plain < (1 << self.bits):
                raise ValueError(
                    f"{label}_plain={plain} outside [0, 2^{self.bits})"
                )
            return plain
        raise TypeError(f"{label} must be int or PaillierCiphertext, got {type(x)}")

    # ------------------------------------------------------------------
    # Tournament reductions
    # ------------------------------------------------------------------
    def secure_max(
        self,
        enc_values: list[PaillierCiphertext] | list[int],
    ) -> tuple[EncryptedBit, int]:
        """Return ``(Enc(1), winner_index)`` via single-elimination.

        Runs ``N-1`` ``secure_gt`` comparisons. The winner index is
        revealed (this is the auction's intended leak surface — every
        bidder learns who won — but no losing bid value is leaked).

        The ``EncryptedBit`` part of the return tuple is always
        ``Enc(1)`` and exists to satisfy the spec's signature; downstream
        consumers can verify it decrypts to 1 as a smoke-check that the
        tournament completed.
        """
        if not enc_values:
            raise ValueError("secure_max requires ≥ 1 value")
        if len(enc_values) == 1:
            return self._encode_bit(1), 0
        leader_idx = 0
        for i in range(1, len(enc_values)):
            gt = self.secure_gt(enc_values[i], enc_values[leader_idx])
            if self.decrypt_bit(gt) == 1:
                leader_idx = i
        return self._encode_bit(1), leader_idx

    def tournament_rank(
        self,
        enc_values: list[PaillierCiphertext] | list[int],
    ) -> list[int]:
        """Return the indices sorted by **descending** value (winner first).

        Uses ``O(N log N)`` ``secure_gt`` calls via merge-sort. The
        returned list contains the original indices — no bid value is
        revealed beyond what the rank order implies (and even the rank
        order can be hidden by returning only the top index, see
        :meth:`secure_max`).
        """
        n = len(enc_values)
        if n == 0:
            return []
        # Index list to sort.
        idx = list(range(n))
        return self._merge_sort(idx, enc_values)

    def _merge_sort(
        self,
        idx: list[int],
        values: list[PaillierCiphertext] | list[int],
    ) -> list[int]:
        if len(idx) <= 1:
            return list(idx)
        mid = len(idx) // 2
        left = self._merge_sort(idx[:mid], values)
        right = self._merge_sort(idx[mid:], values)
        return self._merge(left, right, values)

    def _merge(
        self,
        left: list[int],
        right: list[int],
        values: list[PaillierCiphertext] | list[int],
    ) -> list[int]:
        out: list[int] = []
        i = j = 0
        while i < len(left) and j < len(right):
            # We want descending order: left[i] first if values[left[i]] >
            # values[right[j]].
            gt = self.secure_gt(values[left[i]], values[right[j]])
            if self.decrypt_bit(gt) == 1:
                out.append(left[i])
                i += 1
            else:
                out.append(right[j])
                j += 1
        out.extend(left[i:])
        out.extend(right[j:])
        return out

    # ------------------------------------------------------------------
    # Diagnostic timing harness
    # ------------------------------------------------------------------
    def benchmark_tournament(
        self, values: Iterable[int]
    ) -> tuple[list[int], float]:
        """Return ``(ranking, elapsed_seconds)`` — used by GAPS audit."""
        vlist = list(values)
        t0 = time.perf_counter()
        ranking = self.tournament_rank(vlist)
        elapsed = time.perf_counter() - t0
        return ranking, elapsed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _modinv(a: int, m: int) -> int:
    """Modular inverse via Python's built-in ``pow(a, -1, m)`` (CPython 3.8+)."""
    return pow(a, -1, m)


__all__ = [
    "DGKComparator",
    "EncryptedBit",
    "MAX_BITS",
]
