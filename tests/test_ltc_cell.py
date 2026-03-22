"""Tests for the Liquid Time-Constant (LTC) cell."""

import torch

from spectrai.core.ltc_cell import LTCCell


class TestLTCCellForwardShape:
    """Verify output dimensionality of a single LTC step."""

    def test_forward_shape(self, device):
        input_dim, hidden_dim, batch = 12, 32, 4
        cell = LTCCell(input_dim, hidden_dim).to(device)

        x = torch.randn(batch, input_dim, device=device)
        h = torch.zeros(batch, hidden_dim, device=device)

        h_new = cell(x, h)

        assert h_new.shape == (batch, hidden_dim)


class TestTimeConstantPositive:
    """The effective time constant tau must always be strictly positive."""

    def test_time_constant_positive(self, device):
        input_dim, hidden_dim = 8, 16
        cell = LTCCell(input_dim, hidden_dim).to(device)

        # Wide range of inputs including large negatives
        for _ in range(5):
            x = torch.randn(16, input_dim, device=device) * 10.0
            tau = cell.tau_base + torch.nn.functional.softplus(cell.W_tau(x))
            assert (tau > 0).all(), "tau must be strictly positive everywhere"


class TestGradientFlow:
    """Gradients must flow through the LTC cell to enable learning."""

    def test_gradient_flow(self, device):
        input_dim, hidden_dim = 8, 16
        cell = LTCCell(input_dim, hidden_dim).to(device)

        x = torch.randn(4, input_dim, device=device)
        h = torch.randn(4, hidden_dim, device=device, requires_grad=True)

        h_new = cell(x, h)
        loss = h_new.sum()
        loss.backward()

        # Verify gradients exist on all learnable parameters
        for name, param in cell.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

        # Verify gradient flows back through hidden state
        assert h.grad is not None, "No gradient through hidden state"
