"""Tests for the multi-layer LTC sequence encoder."""

import pytest
import torch

from preceptualai.core.ltc_encoder import LTCEncoder


class TestOutputShape:
    """Encoder should map (B, T, F) -> (B, latent_dim)."""

    def test_output_shape(self, device):
        B, T, F = 4, 8, 12
        latent_dim = 24
        encoder = LTCEncoder(
            input_dim=F, hidden_dim=32, latent_dim=latent_dim, num_layers=1
        ).to(device)

        x = torch.randn(B, T, F, device=device)
        z = encoder(x)

        assert z.shape == (B, latent_dim)


class TestDifferentSequenceLengths:
    """Encoder must handle variable-length sequences since it processes step by step."""

    @pytest.mark.parametrize("T", [1, 4, 16, 64])
    def test_different_sequence_lengths(self, device, T):
        B, F = 2, 12
        encoder = LTCEncoder(
            input_dim=F, hidden_dim=32, latent_dim=32, num_layers=1
        ).to(device)

        x = torch.randn(B, T, F, device=device)
        z = encoder(x)

        assert z.shape == (B, 32)


class TestMultiLayer:
    """Stacking multiple LTC layers should work without error."""

    @pytest.mark.parametrize("num_layers", [1, 2, 4])
    def test_multi_layer(self, device, num_layers):
        B, T, F = 3, 8, 12
        latent_dim = 16
        encoder = LTCEncoder(
            input_dim=F,
            hidden_dim=32,
            latent_dim=latent_dim,
            num_layers=num_layers,
        ).to(device)

        x = torch.randn(B, T, F, device=device)
        z = encoder(x)

        assert z.shape == (B, latent_dim)
        assert len(encoder.cells) == num_layers
        assert len(encoder.layer_norms) == num_layers
