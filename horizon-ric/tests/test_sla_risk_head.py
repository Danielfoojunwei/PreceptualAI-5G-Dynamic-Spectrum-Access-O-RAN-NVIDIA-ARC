"""SLA risk head — two-hot symlog regression smoke tests."""

import torch

from horizon_ric.heads.sla_risk import SLARiskConfig, SLARiskHead


class TestSLARiskHead:
    def test_forward_shape(self):
        head = SLARiskHead(SLARiskConfig(latent_dim=128, hidden_dim=64))
        z = torch.randn(8, 128)
        out = head(z)
        assert set(out.keys()) == {"h_30s", "h_60s", "h_300s"}
        for v in out.values():
            assert v.shape == (8,)

    def test_predictions_in_unit_interval(self):
        head = SLARiskHead(SLARiskConfig(latent_dim=128, hidden_dim=64))
        z = torch.randn(16, 128)
        out = head(z)
        for k, v in out.items():
            assert torch.all(v >= 0.0), f"{k} below 0"
            assert torch.all(v <= 1.0), f"{k} above 1"

    def test_loss_is_finite(self):
        head = SLARiskHead(SLARiskConfig(latent_dim=64, hidden_dim=32))
        z = torch.randn(4, 64)
        targets = {
            "h_30s": torch.rand(4),
            "h_60s": torch.rand(4),
            "h_300s": torch.rand(4),
        }
        loss = head.loss(z, targets)
        assert torch.isfinite(loss)
        assert loss.item() > 0  # CE loss should be positive

    def test_gradient_flows(self):
        head = SLARiskHead(SLARiskConfig(latent_dim=32, hidden_dim=16))
        z = torch.randn(2, 32, requires_grad=True)
        targets = {"h_30s": torch.rand(2)}
        loss = head.loss(z, targets)
        loss.backward()
        # At least one head's weight should have a non-zero gradient
        any_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in head.parameters()
        )
        assert any_grad

    def test_calibration_summary_returns_per_horizon(self):
        head = SLARiskHead(SLARiskConfig(latent_dim=32, hidden_dim=16))
        # Construct simple synthetic predicted/observed
        predicted = {
            "h_30s": torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9]),
        }
        observed = {
            "h_30s": torch.tensor([0, 0, 1, 1, 1], dtype=torch.float32),
        }
        ece = head.calibration_summary(predicted, observed, n_bins=5)
        assert "h_30s" in ece
        assert 0.0 <= ece["h_30s"] <= 1.0
