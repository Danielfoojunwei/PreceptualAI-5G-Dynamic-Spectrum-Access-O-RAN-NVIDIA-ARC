"""Tests for the LTC Actor and Critic networks."""

import torch

from preceptualai.core.actor import LTCActor
from preceptualai.core.critic import LTCCritic
from preceptualai.core.ltc_encoder import LTCEncoder


def _make_encoder(device, input_dim=12, hidden_dim=32, latent_dim=32):
    return LTCEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=1,
    ).to(device)


class TestActorOutputSumsToOne:
    """Softmax policy probabilities must sum to 1 for every sample."""

    def test_actor_output_sums_to_one(self, device):
        num_actions = 4
        encoder = _make_encoder(device)
        actor = LTCActor(encoder, num_actions).to(device)

        x = torch.randn(8, 6, 12, device=device)  # (B, T, F)
        probs = actor(x)

        assert probs.shape == (8, num_actions)
        # Each row should sum to ~1.0
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
        # All probabilities non-negative
        assert (probs >= 0).all()


class TestActorDeterministicAction:
    """Deterministic action selection should return the argmax."""

    def test_actor_deterministic_action(self, device):
        num_actions = 4
        encoder = _make_encoder(device)
        actor = LTCActor(encoder, num_actions).to(device)

        x = torch.randn(1, 6, 12, device=device)
        action, probs = actor.get_action(x, deterministic=True)

        expected = probs.argmax(dim=-1).item()
        assert action == expected
        assert isinstance(action, int)


class TestCriticOutputShape:
    """Critic should output Q-values for every action."""

    def test_critic_output_shape(self, device):
        num_actions = 4
        encoder = _make_encoder(device)
        critic = LTCCritic(encoder, num_actions).to(device)

        x = torch.randn(8, 6, 12, device=device)
        q_values = critic(x)

        assert q_values.shape == (8, num_actions)
