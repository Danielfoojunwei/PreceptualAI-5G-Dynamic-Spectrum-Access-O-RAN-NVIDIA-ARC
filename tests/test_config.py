"""Tests for the SpectralConfig schema."""

import pytest

from preceptualai.config import SpectralConfig


class TestDefaultConfig:
    """SpectralConfig should be constructable with minimal / no parameters."""

    def test_default_config(self):
        cfg = SpectralConfig()

        assert cfg.environment.num_channels == 10
        assert cfg.environment.sequence_length == 16
        assert cfg.environment.num_features == 3
        assert cfg.encoder.hidden_dim == 128
        assert cfg.encoder.latent_dim == 128
        assert cfg.encoder.num_layers == 2
        assert cfg.agent.lr == pytest.approx(3e-4)
        assert cfg.agent.gamma == pytest.approx(0.99)
        assert cfg.device == "auto"
        assert cfg.seed == 42

    def test_override_params(self):
        cfg = SpectralConfig(
            environment={"num_channels": 20},
            encoder={"hidden_dim": 64},
            device="cpu",
        )
        assert cfg.environment.num_channels == 20
        assert cfg.encoder.hidden_dim == 64
        assert cfg.device == "cpu"


class TestValidationErrors:
    """Invalid parameter ranges should raise ValidationError."""

    def test_invalid_device(self):
        with pytest.raises(Exception):
            SpectralConfig(device="tpu")

    def test_negative_channels(self):
        with pytest.raises(Exception):
            SpectralConfig(environment={"num_channels": 0})

    def test_gamma_out_of_range(self):
        with pytest.raises(Exception):
            SpectralConfig(agent={"gamma": 1.5})

    def test_negative_lr(self):
        with pytest.raises(Exception):
            SpectralConfig(agent={"lr": -0.001})

    def test_channels_too_large(self):
        with pytest.raises(Exception):
            SpectralConfig(environment={"num_channels": 999})


class TestYamlLoading:
    """Configuration should be loadable from a YAML string."""

    def test_yaml_loading(self):
        yaml_str = """
environment:
  num_channels: 5
  sequence_length: 8
encoder:
  hidden_dim: 64
  num_layers: 3
agent:
  lr: 0.001
  batch_size: 128
device: cpu
seed: 123
"""
        cfg = SpectralConfig.from_yaml(yaml_str)

        assert cfg.environment.num_channels == 5
        assert cfg.environment.sequence_length == 8
        assert cfg.encoder.hidden_dim == 64
        assert cfg.encoder.num_layers == 3
        assert cfg.agent.lr == pytest.approx(0.001)
        assert cfg.agent.batch_size == 128
        assert cfg.device == "cpu"
        assert cfg.seed == 123

    def test_empty_yaml(self):
        cfg = SpectralConfig.from_yaml("")
        assert cfg.environment.num_channels == 10  # default

    def test_partial_yaml(self):
        cfg = SpectralConfig.from_yaml("seed: 99\n")
        assert cfg.seed == 99
        assert cfg.environment.num_channels == 10  # default
