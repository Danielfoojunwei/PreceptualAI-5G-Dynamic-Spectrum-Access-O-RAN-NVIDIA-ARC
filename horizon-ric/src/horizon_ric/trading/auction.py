"""Sealed-bid second-price auction with Paillier homomorphic encryption.

The protocol shipped here:

    Setup
        Auctioneer generates a Paillier (pk, sk) keypair and publishes pk.

    Bid
        Each Bidder enters an integer bid (units = whatever the auction
        is denominated in — milli-credits / cents / PRBs, anything additive).
        Bidder.encrypt(value) returns an EncryptedBid carrying ciphertext
        keyed by `bidder_id` so the auctioneer can identify the winner.

    Reveal
        Auctioneer collects EncryptedBids, decrypts EACH (single-step
        protocol — equivalent to a trusted clearinghouse). Returns the
        winner identity and the second-highest bid as the price paid.

This is intentionally the *trusted-auctioneer* variant — the simplest
useful baseline. The fully decentralised Brandt-2006 protocol replaces
the auctioneer with a multi-round MPC; we leave that to a Phase-3
deliverable. The interface is identical: callers swap the implementation
without changing call sites.

Privacy guarantees of the trusted-auctioneer variant:
    * No bidder can observe another bidder's ciphertext value.
    * The auctioneer learns the bids only after reveal — the plaintext
      bid is *never* transmitted from the bidder.
    * Encrypted ciphertexts are randomised (Paillier IND-CPA), so a
      passive observer of the bidding channel learns nothing about bid
      ordering or magnitude before reveal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from phe import EncryptedNumber, paillier


@dataclass
class EncryptedBid:
    """One bidder's sealed bid."""

    bidder_id: str
    ciphertext: EncryptedNumber

    def __post_init__(self) -> None:
        if not isinstance(self.bidder_id, str) or not self.bidder_id:
            raise ValueError("bidder_id must be a non-empty string")


@dataclass
class SecondPriceResult:
    """Auction outcome (Vickrey)."""

    winner_id: str
    winning_bid: int
    price_paid: int
    """Second-highest bid; the price the winner actually pays in Vickrey."""
    num_bidders: int
    losing_bids: dict[str, int] = field(default_factory=dict)
    """All non-winning bids — only revealed under the trusted-auctioneer
    variant. The decentralised future implementation will keep these
    private."""


class Auctioneer:
    """Generates a Paillier keypair and runs the reveal step."""

    def __init__(self, key_length_bits: int = 2048):
        if key_length_bits < 1024:
            raise ValueError(
                "key_length_bits should be ≥ 1024 for any production use; "
                f"got {key_length_bits}"
            )
        self.public_key, self._private_key = paillier.generate_paillier_keypair(
            n_length=key_length_bits,
        )

    @property
    def public_key_serialized(self) -> dict[str, Any]:
        """Public key as a JSON-serialisable dict for distribution to bidders."""
        return {"n": str(self.public_key.n)}

    def reveal(self, bids: list[EncryptedBid]) -> SecondPriceResult:
        """Decrypt all bids and pick the Vickrey winner.

        Raises ValueError on fewer than two distinct bidders (Vickrey
        requires a second price).
        """
        if len(bids) < 2:
            raise ValueError(
                "Vickrey auction requires at least two bidders; got "
                f"{len(bids)}"
            )
        seen: set[str] = set()
        plaintext: list[tuple[str, int]] = []
        for bid in bids:
            if bid.bidder_id in seen:
                raise ValueError(f"duplicate bidder_id {bid.bidder_id}")
            seen.add(bid.bidder_id)
            value = self._private_key.decrypt(bid.ciphertext)
            if value < 0:
                raise ValueError(
                    f"bidder {bid.bidder_id} submitted negative bid {value}"
                )
            plaintext.append((bid.bidder_id, int(value)))

        plaintext.sort(key=lambda t: t[1], reverse=True)
        winner_id, winner_bid = plaintext[0]
        second_bid = plaintext[1][1]
        if winner_bid == second_bid:
            # Tie-breaking: lexicographic on bidder_id makes the result
            # deterministic and verifiable by external auditors.
            top_bidders = [b for b, v in plaintext if v == winner_bid]
            winner_id = sorted(top_bidders)[0]

        losing = {bid_id: v for bid_id, v in plaintext if bid_id != winner_id}
        return SecondPriceResult(
            winner_id=winner_id,
            winning_bid=winner_bid,
            price_paid=second_bid,
            num_bidders=len(plaintext),
            losing_bids=losing,
        )


class Bidder:
    """Encrypts a bid against the auctioneer's public key."""

    def __init__(self, bidder_id: str, public_key: paillier.PaillierPublicKey):
        if not isinstance(bidder_id, str) or not bidder_id:
            raise ValueError("bidder_id must be a non-empty string")
        self.bidder_id = bidder_id
        self.public_key = public_key

    def encrypt(self, bid_value: int) -> EncryptedBid:
        if bid_value < 0:
            raise ValueError(f"bid_value must be ≥ 0, got {bid_value}")
        ct = self.public_key.encrypt(int(bid_value))
        return EncryptedBid(bidder_id=self.bidder_id, ciphertext=ct)


@dataclass
class SealedBidAuction:
    """High-level orchestration: spin up an auctioneer, register bidders,
    collect encrypted bids, run reveal."""

    auctioneer: Auctioneer = field(default_factory=Auctioneer)
    _bidders: dict[str, Bidder] = field(default_factory=dict)
    _bids: list[EncryptedBid] = field(default_factory=list)

    def register_bidder(self, bidder_id: str) -> Bidder:
        if bidder_id in self._bidders:
            raise ValueError(f"bidder {bidder_id} already registered")
        bidder = Bidder(bidder_id, self.auctioneer.public_key)
        self._bidders[bidder_id] = bidder
        return bidder

    def submit(self, bid: EncryptedBid) -> None:
        if bid.bidder_id not in self._bidders:
            raise ValueError(f"bidder {bid.bidder_id} not registered")
        if any(b.bidder_id == bid.bidder_id for b in self._bids):
            raise ValueError(f"bidder {bid.bidder_id} already submitted")
        self._bids.append(bid)

    def settle(self) -> SecondPriceResult:
        result = self.auctioneer.reveal(self._bids)
        self._bids.clear()
        return result


__all__ = [
    "Auctioneer",
    "Bidder",
    "EncryptedBid",
    "SealedBidAuction",
    "SecondPriceResult",
]
