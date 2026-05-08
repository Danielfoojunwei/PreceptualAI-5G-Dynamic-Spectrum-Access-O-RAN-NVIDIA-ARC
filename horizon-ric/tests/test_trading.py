"""Sealed-bid Paillier auction tests."""

import pytest

from horizon_ric.trading import (
    Auctioneer,
    Bidder,
    EncryptedBid,
    SealedBidAuction,
    SecondPriceResult,
)

# Use a smaller key for fast tests; not for production.
_TEST_KEY_BITS = 1024


@pytest.fixture(scope="module")
def auctioneer() -> Auctioneer:
    return Auctioneer(key_length_bits=_TEST_KEY_BITS)


class TestEncryptionRoundTrip:
    def test_bidder_encrypts_against_pk(self, auctioneer):
        bidder = Bidder("alice", auctioneer.public_key)
        bid = bidder.encrypt(42)
        assert isinstance(bid, EncryptedBid)
        assert bid.bidder_id == "alice"
        # Auctioneer can decrypt
        plain = auctioneer._private_key.decrypt(bid.ciphertext)
        assert plain == 42


class TestVickreyOutcome:
    def test_three_bidders_winner_pays_second(self, auctioneer):
        bids = [
            Bidder("a", auctioneer.public_key).encrypt(50),
            Bidder("b", auctioneer.public_key).encrypt(120),
            Bidder("c", auctioneer.public_key).encrypt(80),
        ]
        result = auctioneer.reveal(bids)
        assert isinstance(result, SecondPriceResult)
        assert result.winner_id == "b"
        assert result.winning_bid == 120
        assert result.price_paid == 80   # second-highest
        assert result.num_bidders == 3
        assert "a" in result.losing_bids and "c" in result.losing_bids

    def test_two_bidders(self, auctioneer):
        bids = [
            Bidder("x", auctioneer.public_key).encrypt(10),
            Bidder("y", auctioneer.public_key).encrypt(15),
        ]
        result = auctioneer.reveal(bids)
        assert result.winner_id == "y"
        assert result.price_paid == 10

    def test_tie_broken_lexicographically(self, auctioneer):
        bids = [
            Bidder("zebra", auctioneer.public_key).encrypt(50),
            Bidder("alpha", auctioneer.public_key).encrypt(50),
            Bidder("middle", auctioneer.public_key).encrypt(40),
        ]
        result = auctioneer.reveal(bids)
        # lex-smallest wins
        assert result.winner_id == "alpha"
        assert result.price_paid == 50  # other tied bid is the second-price

    def test_single_bidder_rejected(self, auctioneer):
        bid = Bidder("solo", auctioneer.public_key).encrypt(99)
        with pytest.raises(ValueError):
            auctioneer.reveal([bid])

    def test_duplicate_bidder_rejected(self, auctioneer):
        b1 = Bidder("dup", auctioneer.public_key).encrypt(10)
        b2 = Bidder("dup", auctioneer.public_key).encrypt(20)
        with pytest.raises(ValueError):
            auctioneer.reveal([b1, b2])


class TestNegativeBidsRejected:
    def test_bidder_rejects_negative(self, auctioneer):
        bidder = Bidder("a", auctioneer.public_key)
        with pytest.raises(ValueError):
            bidder.encrypt(-1)


class TestSealedBidAuctionOrchestrator:
    def test_full_flow(self, auctioneer):
        sba = SealedBidAuction(auctioneer=auctioneer)
        for name, value in [("a", 10), ("b", 30), ("c", 20)]:
            bidder = sba.register_bidder(name)
            sba.submit(bidder.encrypt(value))
        result = sba.settle()
        assert result.winner_id == "b"
        assert result.price_paid == 20

    def test_unregistered_submit_rejected(self, auctioneer):
        sba = SealedBidAuction(auctioneer=auctioneer)
        with pytest.raises(ValueError):
            sba.submit(EncryptedBid("ghost", auctioneer.public_key.encrypt(1)))

    def test_double_submit_rejected(self, auctioneer):
        sba = SealedBidAuction(auctioneer=auctioneer)
        bidder = sba.register_bidder("a")
        sba.submit(bidder.encrypt(5))
        with pytest.raises(ValueError):
            sba.submit(bidder.encrypt(5))


class TestPublicKeySerialization:
    def test_serialize_returns_n(self, auctioneer):
        d = auctioneer.public_key_serialized
        assert "n" in d
        assert int(d["n"]) == auctioneer.public_key.n
