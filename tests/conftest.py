"""
Shared pytest fixtures for PreceptualAI test suite.

Provides lightweight, CPU-only fixtures suitable for CI environments
without GPU hardware.
"""

import pytest
import torch

from preceptualai.agent.sac_ltc import SACLTCAgent
from preceptualai.env.sim import SimulatedDSAEnv

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

@pytest.fixture
def device() -> torch.device:
    """CPU device for deterministic, CI-friendly tests."""
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Environment configuration (small / fast)
# ---------------------------------------------------------------------------

@pytest.fixture
def small_env_config() -> dict:
    """Minimal environment config for fast test execution."""
    return {
        "num_channels": 4,
        "sequence_length": 8,
        "num_features": 3,
    }


# ---------------------------------------------------------------------------
# Simulated environment
# ---------------------------------------------------------------------------

@pytest.fixture
def sim_env(small_env_config) -> SimulatedDSAEnv:
    """Small SimulatedDSAEnv instance for testing."""
    return SimulatedDSAEnv(
        num_channels=small_env_config["num_channels"],
        sequence_length=small_env_config["sequence_length"],
        num_features=small_env_config["num_features"],
        max_steps=50,
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@pytest.fixture
def small_agent(small_env_config, device) -> SACLTCAgent:
    """
    Lightweight SACLTCAgent for unit tests.

    Uses minimal dimensions so tests run in < 1 s on CPU.
    """
    num_channels = small_env_config["num_channels"]
    num_features = small_env_config["num_features"]
    sequence_length = small_env_config["sequence_length"]

    input_dim = num_channels * num_features
    state_shape = (sequence_length, input_dim)

    return SACLTCAgent(
        state_shape=state_shape,
        num_actions=num_channels,
        input_dim=input_dim,
        device=device,
        hidden_dim=32,
        latent_dim=32,
        num_layers=1,
        buffer_size=100,
        batch_size=8,
        learning_starts=10,
    )
