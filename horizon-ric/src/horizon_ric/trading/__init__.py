"""Cross-operator resource trading (PARADIGMS.md §H3).

A privacy-preserving sealed-bid second-price (Vickrey) auction. Each
operator encrypts their bid with the auctioneer's Paillier public key;
the auctioneer can compute aggregate statistics under encryption (sums,
ranks via comparison protocols) without ever seeing plaintext bids
except those of the winning bidder. This implements the §H3 promise
that operators trade resources without exposing pricing strategy.

Backed by `phe` (python-paillier, Apache 2.0) — pure Python, no native
deps; fast enough for the few hundred bids/round we expect.

References:
    Paillier P. *Public-Key Cryptosystems Based on Composite Degree
        Residuosity Classes*, EUROCRYPT 1999.
    Vickrey W. *Counterspeculation, Auctions, and Competitive Sealed
        Tenders*, J. Finance 1961.
    Brandt F. *How to obtain full privacy in auctions*, IJIS 2006 — the
        canonical privacy-preserving auction reference.
"""

from horizon_ric.trading.auction import (
    Auctioneer,
    Bidder,
    EncryptedBid,
    SealedBidAuction,
    SecondPriceResult,
)
from horizon_ric.trading.dgk_compare import DGKComparator, EncryptedBit
from horizon_ric.trading.private_auction import (
    CommitReveal,
    PaillierBidder,
    PaillierCiphertext,
    PrivateAuctionResult,
    PrivateSecondPriceAuction,
)

__all__ = [
    "Auctioneer",
    "Bidder",
    "CommitReveal",
    "DGKComparator",
    "EncryptedBid",
    "EncryptedBit",
    "PaillierBidder",
    "PaillierCiphertext",
    "PrivateAuctionResult",
    "PrivateSecondPriceAuction",
    "SealedBidAuction",
    "SecondPriceResult",
]
