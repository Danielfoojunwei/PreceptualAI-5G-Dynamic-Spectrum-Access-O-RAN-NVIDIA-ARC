"""Tests for src/horizon_ric/trading/private_auction.py.

Covers:
    * commit-reveal hash round-trip
    * tampering detection
    * Paillier additive homomorphism
    * ciphertext base64 wire format
    * end-to-end private second-price auction
"""

from __future__ import annotations

import base64

import pytest

from horizon_ric.trading.private_auction import (
    CommitReveal,
    PaillierBidder,
    PaillierCiphertext,
    PrivateSecondPriceAuction,
    Reveal,
)

# Module-scoped Paillier key — keygen at 1024 bits costs ~0.5–1 s and we
# want every test in the file to share one key so the suite stays under
# a couple of seconds.
_KEY = PaillierBidder(key_bits=1024)


# ---------------------------------------------------------------------
# Commit-Reveal
# ---------------------------------------------------------------------
def test_commit_reveal_round_trip() -> None:
    cr = CommitReveal()
    digest, nonce = CommitReveal.make_commitment(bid=1234)
    cr.accept_commitment("op-A", digest)
    cr.close()
    assert cr.verify_reveal(Reveal(bidder_id="op-A", bid=1234, nonce_hex=nonce))


def test_commit_reveal_tampering_detected() -> None:
    cr = CommitReveal()
    digest, nonce = CommitReveal.make_commitment(bid=500)
    cr.accept_commitment("op-B", digest)
    cr.close()

    # Changing the bid invalidates the commitment.
    assert not cr.verify_reveal(Reveal(bidder_id="op-B", bid=600, nonce_hex=nonce))
    # Changing the nonce also invalidates it.
    bogus_nonce = "00" * 32
    assert not cr.verify_reveal(Reveal(bidder_id="op-B", bid=500, nonce_hex=bogus_nonce))


def test_commit_reveal_unknown_bidder_raises() -> None:
    cr = CommitReveal()
    cr.close()
    with pytest.raises(ValueError):
        cr.verify_reveal(Reveal(bidder_id="ghost", bid=1, nonce_hex="ab" * 32))


# ---------------------------------------------------------------------
# Paillier
# ---------------------------------------------------------------------
def test_paillier_homomorphic_add() -> None:
    a, b = 17, 25
    ca = _KEY.encrypt(a)
    cb = _KEY.encrypt(b)
    csum = _KEY.add_ciphertexts(ca, cb)
    assert _KEY.decrypt(csum) == a + b

    # Multi-add: aggregating five sealed bids.
    bids = [13, 71, 4, 99, 200]
    acc = _KEY.encrypt(bids[0])
    for v in bids[1:]:
        acc = _KEY.add_ciphertexts(acc, _KEY.encrypt(v))
    assert _KEY.decrypt(acc) == sum(bids)


def test_paillier_decrypt_round_trip() -> None:
    for m in (0, 1, 42, 2**100, _KEY.n - 1):
        assert _KEY.decrypt(_KEY.encrypt(m)) == m


def test_paillier_serialize() -> None:
    c = _KEY.encrypt(98765)
    b64 = c.to_b64()
    # Pure ASCII, valid base64.
    assert isinstance(b64, str)
    base64.b64decode(b64.encode("ascii"))  # no exception
    # Round-trip through the wire format preserves the plaintext.
    rehydrated = PaillierCiphertext.from_b64(b64, n=_KEY.n)
    assert _KEY.decrypt(rehydrated) == 98765


def test_paillier_rejects_out_of_range_plaintext() -> None:
    with pytest.raises(ValueError):
        _KEY.encrypt(-1)
    with pytest.raises(ValueError):
        _KEY.encrypt(_KEY.n)


# ---------------------------------------------------------------------
# Private second-price auction
# ---------------------------------------------------------------------
def test_private_second_price_winner() -> None:
    bids = {"opA": 100, "opB": 250, "opC": 175, "opD": 300, "opE": 220}
    auc = PrivateSecondPriceAuction()
    auc.open_bidding(list(bids))

    # Phase 1: commit.
    nonces: dict[str, str] = {}
    for bid_id, value in bids.items():
        digest, nonce = CommitReveal.make_commitment(bid=value)
        auc.submit_commitment(bid_id, digest)
        nonces[bid_id] = nonce

    auc.close_bidding()

    # Phase 2: reveal.
    for bid_id, value in bids.items():
        assert auc.submit_reveal(bid_id, value, nonces[bid_id])

    result = auc.settle()
    assert result.winner_id == "opD"  # highest bid
    assert result.price_paid == 250  # second-highest (Vickrey)
    assert result.num_bidders == 5
    assert result.rejected_bidders == []


def test_private_auction_rejects_tampered_reveal() -> None:
    bids = {"x": 10, "y": 20}
    auc = PrivateSecondPriceAuction()
    auc.open_bidding(list(bids))

    nonces: dict[str, str] = {}
    for bid_id, value in bids.items():
        digest, nonce = CommitReveal.make_commitment(bid=value)
        auc.submit_commitment(bid_id, digest)
        nonces[bid_id] = nonce
    auc.close_bidding()

    # x tries to inflate its bid post-deadline; verification rejects it.
    assert not auc.submit_reveal("x", 999, nonces["x"])
    assert auc.submit_reveal("y", 20, nonces["y"])
    # Only one valid reveal — Vickrey can't settle.
    with pytest.raises(ValueError):
        auc.settle()


def test_private_auction_blind_rank_flag_documented() -> None:
    bids = {"a": 5, "b": 9, "c": 7}
    auc = PrivateSecondPriceAuction(blind_rank=True)
    auc.open_bidding(list(bids))
    nonces: dict[str, str] = {}
    for bid_id, value in bids.items():
        digest, nonce = CommitReveal.make_commitment(bid=value)
        auc.submit_commitment(bid_id, digest)
        nonces[bid_id] = nonce
    auc.close_bidding()
    for bid_id, value in bids.items():
        auc.submit_reveal(bid_id, value, nonces[bid_id])
    result = auc.settle()
    # Stub falls back to trusted-auctioneer scoring; the audit flag
    # records that blind-rank was *requested*.
    assert result.blind_ranked is True
    assert result.winner_id == "b"
    assert result.price_paid == 7
