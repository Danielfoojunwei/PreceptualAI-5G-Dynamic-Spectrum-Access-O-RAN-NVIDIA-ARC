"""Tests for the DGK-2007 secure comparison primitive and its
auction-layer integration.

Closes Row 16 of GAPS_TO_PILOT.md: cross-operator trading now has a
working MPC blind-ranking step (not just commit-reveal + Paillier
additive HE).

The Paillier modulus is held module-scoped at 1024 bits so that the
~0.5–1 s key-generation cost is paid once for the whole file. (Tests
that need a *different* key — e.g. to verify isolation — instantiate
their own.)
"""

from __future__ import annotations

import random
import time

import pytest

from horizon_ric.trading.dgk_compare import (
    MAX_BITS,
    DGKComparator,
    EncryptedBit,
)
from horizon_ric.trading.private_auction import (
    CommitReveal,
    PaillierBidder,
    PaillierCiphertext,
    PrivateSecondPriceAuction,
    Reveal,
)

# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------
_KEY = PaillierBidder(key_bits=1024)
_CMP = DGKComparator(_KEY, bits=32)


# ---------------------------------------------------------------------------
# Test 1: secure_gt basic
# ---------------------------------------------------------------------------
def test_dgk_secure_gt_basic() -> None:
    """The headline correctness check from the spec."""
    eb = _CMP.secure_gt(7, 3)
    assert isinstance(eb, EncryptedBit)
    assert _CMP.decrypt_bit(eb) == 1, "Enc(7 > 3) should decrypt to 1"

    eb = _CMP.secure_gt(3, 7)
    assert _CMP.decrypt_bit(eb) == 0, "Enc(3 > 7) should decrypt to 0"

    # Strict inequality: equality returns 0.
    eb = _CMP.secure_gt(5, 5)
    assert _CMP.decrypt_bit(eb) == 0, "Enc(5 > 5) should decrypt to 0"


# ---------------------------------------------------------------------------
# Test 2: secure_gt random correctness (100 trials, 100 % rate)
# ---------------------------------------------------------------------------
def test_dgk_secure_gt_random() -> None:
    rng = random.Random(0xDEADBEEF)
    n_trials = 100
    upper = 1 << 20  # documented correctness range: bids up to 2^20
    correct = 0
    for _ in range(n_trials):
        a = rng.randrange(0, upper)
        b = rng.randrange(0, upper)
        eb = _CMP.secure_gt(a, b)
        got = _CMP.decrypt_bit(eb)
        expected = 1 if a > b else 0
        if got == expected:
            correct += 1
    assert correct == n_trials, f"DGK correctness rate {correct}/{n_trials}"


# ---------------------------------------------------------------------------
# Test 3: secure_max
# ---------------------------------------------------------------------------
def test_dgk_secure_max_correct() -> None:
    """secure_max([3,1,9,4,7]) must point at index 2 (value 9)."""
    values = [3, 1, 9, 4, 7]
    eb_one, idx = _CMP.secure_max(values)
    assert _CMP.decrypt_bit(eb_one) == 1, "tournament-completion smoke bit"
    assert idx == 2, f"expected winner index 2 (value 9), got {idx}"


def test_dgk_secure_max_singleton() -> None:
    eb_one, idx = _CMP.secure_max([42])
    assert idx == 0
    assert _CMP.decrypt_bit(eb_one) == 1


# ---------------------------------------------------------------------------
# Test 4: tournament_rank — full ordering
# ---------------------------------------------------------------------------
def test_tournament_rank_5_bidders() -> None:
    """Rank 5 bid values; result must match plaintext sort order."""
    values = [12, 47, 3, 89, 25]
    ranking = _CMP.tournament_rank(values)
    expected = sorted(range(len(values)), key=lambda i: -values[i])
    assert ranking == expected, f"got {ranking}, expected {expected}"
    # Top of the ranking should be the index of the maximum.
    assert values[ranking[0]] == max(values)


# ---------------------------------------------------------------------------
# Test 5: tournament_rank with ciphertext inputs
# ---------------------------------------------------------------------------
def test_tournament_rank_accepts_ciphertexts() -> None:
    """Demonstrate that the API accepts Paillier ciphertexts directly."""
    values = [200, 50, 175, 90]
    enc = [_KEY.encrypt(v) for v in values]
    ranking = _CMP.tournament_rank(enc)
    expected = sorted(range(len(values)), key=lambda i: -values[i])
    assert ranking == expected


# ---------------------------------------------------------------------------
# Test 6: blind_rank does not leak bid plaintexts
# ---------------------------------------------------------------------------
def test_blind_rank_does_not_leak() -> None:
    """The auctioneer's intermediate state for blind-ranking must not
    contain plaintext bid values — only ciphertexts and the public
    rank order."""
    auction = PrivateSecondPriceAuction(mode="blind")
    bids = {"op-A": 100, "op-B": 250, "op-C": 175}
    auction.open_bidding(list(bids))
    nonces: dict[str, str] = {}
    for bid_id, bid in bids.items():
        digest, nonce = CommitReveal.make_commitment(bid)
        auction.submit_commitment(bid_id, digest)
        nonces[bid_id] = nonce
    auction.close_bidding()
    for bid_id, bid in bids.items():
        assert auction.submit_reveal(bid_id, bid, nonces[bid_id])
    result = auction.settle()
    state = auction._last_blind_state
    assert state is not None, "blind state must be populated"

    # No raw bid value in the state's encrypted_bids list.
    for ct in state["encrypted_bids"]:
        assert isinstance(ct, PaillierCiphertext)
        # The ciphertext integer should NOT equal the bid (with
        # overwhelming probability for a 1024-bit modulus, a random
        # ciphertext shares no structure with a small plaintext, but we
        # still spot-check).
        for v in bids.values():
            assert ct.c != v, "ciphertext c happens to equal plaintext bid"

    # Ranking is plaintext (the auction must publish the winner) but the
    # state must NOT carry the plaintext bid list.
    state_keys = set(state.keys())
    assert "bids_plain" not in state_keys
    assert "bid_values" not in state_keys
    # The known-public fields:
    assert state["winner_id"] == "op-B"
    assert state["second_price"] == 175

    # And the auction's public outcome:
    assert result.winner_id == "op-B"
    assert result.price_paid == 175
    assert result.blind_ranked is True


# ---------------------------------------------------------------------------
# Test 7: end-to-end private-second-price with blind ranking
# ---------------------------------------------------------------------------
def test_private_second_price_with_blind_ranking() -> None:
    """5 bidders, commit-reveal, DGK tournament, second-price winner."""
    auction = PrivateSecondPriceAuction(mode="blind")
    bids = {
        "op-A": 100,
        "op-B": 350,
        "op-C": 275,
        "op-D": 150,
        "op-E": 420,
    }
    auction.open_bidding(list(bids))
    nonces: dict[str, str] = {}
    for bid_id, bid in bids.items():
        digest, nonce = CommitReveal.make_commitment(bid)
        auction.submit_commitment(bid_id, digest)
        nonces[bid_id] = nonce
    auction.close_bidding()
    for bid_id, bid in bids.items():
        assert auction.submit_reveal(bid_id, bid, nonces[bid_id])
    result = auction.settle()

    # Vickrey: highest bid wins, pays the second-highest price.
    assert result.winner_id == "op-E"
    assert result.price_paid == 350  # op-B's bid
    assert result.num_bidders == 5
    assert result.rejected_bidders == []
    assert result.blind_ranked is True


# ---------------------------------------------------------------------------
# Test 8: fast mode still works (regression for backward compat)
# ---------------------------------------------------------------------------
def test_fast_mode_matches_blind_mode() -> None:
    bids = {"op-A": 90, "op-B": 220, "op-C": 110}

    def run(mode: str):
        auction = PrivateSecondPriceAuction(mode=mode)
        auction.open_bidding(list(bids))
        nonces: dict[str, str] = {}
        for bid_id, bid in bids.items():
            digest, nonce = CommitReveal.make_commitment(bid)
            auction.submit_commitment(bid_id, digest)
            nonces[bid_id] = nonce
        auction.close_bidding()
        for bid_id, bid in bids.items():
            auction.submit_reveal(bid_id, bid, nonces[bid_id])
        return auction.settle()

    fast = run("fast")
    blind = run("blind")
    assert fast.winner_id == blind.winner_id == "op-B"
    assert fast.price_paid == blind.price_paid == 110
    assert fast.blind_ranked is False
    assert blind.blind_ranked is True


# ---------------------------------------------------------------------------
# Test 9: edge cases (zeros, max values)
# ---------------------------------------------------------------------------
def test_dgk_edge_cases() -> None:
    # Zero on either side.
    assert _CMP.decrypt_bit(_CMP.secure_gt(0, 0)) == 0
    assert _CMP.decrypt_bit(_CMP.secure_gt(0, 1)) == 0
    assert _CMP.decrypt_bit(_CMP.secure_gt(1, 0)) == 1
    # Max representable in 32 bits.
    big = (1 << 20) - 1
    assert _CMP.decrypt_bit(_CMP.secure_gt(big, big - 1)) == 1
    assert _CMP.decrypt_bit(_CMP.secure_gt(big - 1, big)) == 0


# ---------------------------------------------------------------------------
# Test 10: tournament_rank(8) must finish in ≤ 60 s
# ---------------------------------------------------------------------------
def test_tournament_rank_8_under_60s() -> None:
    rng = random.Random(7)
    values = [rng.randrange(0, 1 << 20) for _ in range(8)]
    t0 = time.perf_counter()
    ranking = _CMP.tournament_rank(values)
    dt = time.perf_counter() - t0
    expected = sorted(range(len(values)), key=lambda i: -values[i])
    assert ranking == expected
    assert dt <= 60.0, f"tournament_rank(8) took {dt:.2f}s, budget 60s"


# ---------------------------------------------------------------------------
# Test 11: bit-width validation
# ---------------------------------------------------------------------------
def test_dgk_bit_width_validation() -> None:
    # Out-of-range bits parameter
    with pytest.raises(ValueError):
        DGKComparator(_KEY, bits=0)
    with pytest.raises(ValueError):
        DGKComparator(_KEY, bits=MAX_BITS + 1)

    # Value larger than the comparator's bit-width must be rejected
    cmp_small = DGKComparator(_KEY, bits=8)
    with pytest.raises(ValueError):
        cmp_small.secure_gt(256, 1)  # 256 doesn't fit in 8 bits
    # Negative values likewise.
    with pytest.raises(ValueError):
        cmp_small.secure_gt(-1, 1)
