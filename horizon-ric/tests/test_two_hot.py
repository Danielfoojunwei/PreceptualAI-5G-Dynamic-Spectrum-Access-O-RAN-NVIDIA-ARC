"""Two-hot encode/decode round-trip + shape tests."""

import pytest
import torch

from horizon_ric.heads._two_hot import (
    make_bins,
    symexp,
    symlog,
    two_hot,
    two_hot_decode,
    two_hot_loss,
)


class TestSymlogSymexp:
    def test_inverse(self):
        x = torch.linspace(-100, 100, 21)
        torch.testing.assert_close(symexp(symlog(x)), x, atol=1e-4, rtol=1e-4)

    def test_zero(self):
        assert symlog(torch.tensor(0.0)).item() == 0.0
        assert symexp(torch.tensor(0.0)).item() == 0.0


class TestTwoHotBinsContract:
    def test_rejects_too_few_bins(self):
        with pytest.raises(ValueError):
            make_bins(num_bins=2)

    def test_rejects_inverted_range(self):
        with pytest.raises(ValueError):
            make_bins(num_bins=11, low=5.0, high=-5.0)

    def test_rejects_2d_bins(self):
        bad = torch.zeros(3, 4)
        with pytest.raises(ValueError):
            two_hot(torch.zeros(2), bad)


class TestRoundTripSymlog:
    def test_scalar_round_trip(self):
        bins = make_bins(num_bins=255, low=-10.0, high=10.0)
        target = torch.tensor([0.0, 1.0, 10.0, 100.0, -50.0])
        # one-hot-style logits: spike at the right bin
        dist = two_hot(target, bins, apply_symlog=True)
        # log(p) → softmax(log(p)) ≈ p in expectation; use log directly
        decoded = two_hot_decode(torch.log(dist + 1e-12), bins, apply_symlog=True)
        torch.testing.assert_close(decoded, target, atol=0.5, rtol=0.05)


class TestRoundTripLinear:
    def test_logit_round_trip_no_warp(self):
        # Bins ARE in logit space; no symlog warp.
        bins = make_bins(num_bins=51, low=-7.0, high=7.0)
        target = torch.tensor([-5.0, -1.0, 0.0, 2.0, 6.0])
        dist = two_hot(target, bins, apply_symlog=False)
        decoded = two_hot_decode(
            torch.log(dist + 1e-12), bins, apply_symlog=False
        )
        torch.testing.assert_close(decoded, target, atol=0.5, rtol=0.05)


class TestSequenceShape:
    def test_two_hot_supports_BT_input(self):
        bins = make_bins(num_bins=21, low=-5.0, high=5.0)
        target = torch.tensor([[0.0, 1.0, 2.0], [-3.0, 0.5, 4.0]])  # (2, 3)
        dist = two_hot(target, bins, apply_symlog=False)
        assert dist.shape == (2, 3, 21)
        # Each (B, T) slice should sum to ~1
        sums = dist.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)

    def test_decode_supports_BT_input(self):
        bins = make_bins(num_bins=21, low=-5.0, high=5.0)
        # Random logits, just check shape
        logits = torch.randn(4, 6, 21)
        decoded = two_hot_decode(logits, bins, apply_symlog=False)
        assert decoded.shape == (4, 6)


class TestLoss:
    def test_loss_finite_and_positive(self):
        bins = make_bins(num_bins=31, low=-10.0, high=10.0)
        logits = torch.randn(8, 31)
        target = torch.randn(8) * 3.0
        loss = two_hot_loss(logits, target, bins, apply_symlog=True)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_loss_grads_flow(self):
        bins = make_bins(num_bins=11, low=-5.0, high=5.0)
        logits = torch.randn(4, 11, requires_grad=True)
        target = torch.tensor([1.0, -1.0, 2.0, 0.0])
        loss = two_hot_loss(logits, target, bins, apply_symlog=False)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum().item() > 0
